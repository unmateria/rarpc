#include "common.cuh"
#include "sha256_device.cuh"

// ════════════════════════════════════════════════════════════════
//  RAR5 cracking kernel — PBKDF2-HMAC-SHA256 + password check
//
//  Real WinRAR PswCheck formula (verified empirically):
//    buf   = PBKDF2-SHA256(utf8_password, salt, (1<<iter_count) + 32, 32)
//    xor8  = fold buf[32] into 8 bytes: xor8[i%8] ^= buf[i]
//    init_v stored in archive header = xor8
//    Accepted if xor8 == stored init_v  (i.e., psw_check_data[0..8])
//
//  Optimisations vs. naive implementation:
//    • HMAC midstates (H_inner, H_outer) precomputed once per password,
//      carried through the hot loop as sha256_h8_t (named fields in regs).
//    • sha256_compress_bv takes state BY VALUE — NVCC keeps h[] entirely in
//      registers instead of promoting to local memory (the PTX previously
//      emitted cvta.to.local → ld.local for every iteration).
//    • sha256_blk_t by-value: the 16-word message block is also a struct
//      with named fields → register-resident across the call boundary.
//    • Ada (SM_89) ISA: LOP3.LUT for Ch/Maj/XOR3, SHF.R.WRAP for rotates,
//      IADD3 auto-emitted by ptxas for 3-operand add chains.
//    • __forceinline__ + no __launch_bounds__ → nvcc picks 64 regs/thread,
//      zero spill, 4 blocks/SM (1024 threads/SM = 32 warps).
//    • ILP=2 experiment (commit-and-revert 2026-04-14): doubling register
//      state to interleave 2 candidates per thread produced no measurable
//      speedup on RTX 4060 Ti — the kernel is compute-bound, not
//      latency-hiding-bound. 32 warps × 4 scheduler slots already saturate.
// ════════════════════════════════════════════════════════════════

// No __launch_bounds__ — measured with Nsight Compute (2026-04-14):
//   * ALU utilization: 95.8% → kernel is ALU-bound, not latency-bound
//   * Achieved occupancy: ~50% at 1M batch (theoretical 66.7%)
//   * Experiment __launch_bounds__(256, 5): 48 regs → 72B/88B spill.
//     The spill hits the PBKDF2 hot loop, saturating LSU. Net regression
//     (68 → 65 KH/s). The ALU ceiling is the fundamental constraint;
//     trading compute for memory traffic doesn't help here.
extern "C"
__global__
void rar5_crack(
    const uint8_t * __restrict__ passwords,
    const int32_t * __restrict__ pw_lengths,
    int32_t         num_passwords,
    const uint8_t * __restrict__ salt,           // 16 bytes
    int32_t         iter_count,
    const uint8_t * __restrict__ psw_check_data, // 12 bytes: InitV[8] + PswCheck[4]
    int32_t       * __restrict__ result
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_passwords) return;

    const uint8_t *pw    = passwords + (size_t)tid * MAX_PW_BYTES;
    uint32_t       pwlen = (uint32_t)pw_lengths[tid];
    uint32_t       iters = (1u << (uint32_t)iter_count) + 32u;

    // Step 1: PBKDF2 → T as uint32[8] (no final byte conversion)
    uint32_t dk_w[8];
    pbkdf2_sha256_rar5_u32(pw, pwlen, salt, 16, iters, dk_w);

    // Step 2: XOR-compress uint32[8] → 8 bytes (init_v)
    // dk_w[i] is big-endian: high byte first.
    // xor8[i%8] ^= each byte of dk_w (4 bytes per word × 8 words = 32 bytes total)
    uint8_t xor8[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 8; i++) {
        uint32_t w = dk_w[i];
        uint8_t  b0 = (w >> 24) & 0xff;
        uint8_t  b1 = (w >> 16) & 0xff;
        uint8_t  b2 = (w >>  8) & 0xff;
        uint8_t  b3 =  w        & 0xff;
        int base = (i * 4) & 7;
        xor8[base]        ^= b0;
        xor8[(base+1)&7]  ^= b1;
        xor8[(base+2)&7]  ^= b2;
        xor8[(base+3)&7]  ^= b3;
    }

    // Step 3: compare xor8 with stored InitV (psw_check_data[0..8])
    if (memcmp_ct(xor8, psw_check_data, 8)) {
        atomicCAS(result, NO_MATCH, tid);
    }
}
