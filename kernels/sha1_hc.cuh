// Optimized SHA-1 primitives for NVCC (RAR3 KDF hot path).
//
// Key differences from sha1_device.cuh:
//   * 80 named message-schedule scalars (w00_t..w4f_t) — no circular buffer
//     indexing, everything resolves to registers.
//   * Block held as w0[4]/w1[4]/w2[4]/w3[4] (NVCC with __forceinline__ keeps
//     these in registers when no pointer-to-array is taken externally).
//   * SHA1_STEP_S is a macro — no call overhead, message schedule stays in
//     the caller's register frame.
//
// Based on published SHA-1 optimizations for GPU (MIT).

#pragma once
#include <cstdint>
#include "common.cuh"

#define SHA1M_A 0x67452301u
#define SHA1M_B 0xefcdab89u
#define SHA1M_C 0x98badcfeu
#define SHA1M_D 0x10325476u
#define SHA1M_E 0xc3d2e1f0u

#define SHA1C00 0x5a827999u
#define SHA1C01 0x6ed9eba1u
#define SHA1C02 0x8f1bbcdcu
#define SHA1C03 0xca62c1d6u

#define SHA1_F0(x,y,z)  ((z) ^ ((x) & ((y) ^ (z))))
#define SHA1_F1(x,y,z)  ((x) ^ (y) ^ (z))
#define SHA1_F2(x,y,z)  (((x) & (y)) | ((z) & ((x) ^ (y))))
#define SHA1_F0o(x,y,z) SHA1_F0(x,y,z)
#define SHA1_F2o(x,y,z) SHA1_F2(x,y,z)

__device__ __forceinline__ uint32_t hc_rotl32_S(uint32_t a, int n) {
    return (a << n) | (a >> (32 - n));
}
__device__ __forceinline__ uint32_t hc_swap32_S(uint32_t v) {
    return __byte_perm(v, 0, 0x0123);
}
__device__ __forceinline__ uint32_t hc_add3_S(uint32_t a, uint32_t b, uint32_t c) {
    return a + b + c;
}
// hc_funnelshift_r(b, a, c_mod_4 * 8)
//   = ((a << 32) | b) >> (c_mod_4 * 8), low 32 bits.
// When c_mod_4 == 0, this returns b (NOT a).
__device__ __forceinline__ uint32_t hc_bytealign_be_S(uint32_t a, uint32_t b, int c) {
    int n = (c & 3) * 8;
    if (n == 0) return b;
    return (a << (32 - n)) | (b >> n);
}

#define SHA1_STEP_S(f, a, b, c, d, e, x)                   \
    do {                                                    \
        e += K;                                             \
        e  = hc_add3_S(e, (x), f((b), (c), (d)));           \
        e += hc_rotl32_S((a), 5u);                          \
        (b) = hc_rotl32_S((b), 30u);                        \
    } while (0)

// sha1_transform_hc: 80-scalar unrolled SHA-1 block transform.
// w0..w3 are the 4×4 u32 words of the 64-byte block (big-endian).
// digest[5] updated in place.
//
__device__ __forceinline__
void sha1_transform_hc(
    const uint32_t w0[4], const uint32_t w1[4],
    const uint32_t w2[4], const uint32_t w3[4],
    uint32_t digest[5])
{
    uint32_t a = digest[0];
    uint32_t b = digest[1];
    uint32_t c = digest[2];
    uint32_t d = digest[3];
    uint32_t e = digest[4];

    uint32_t w00_t = w0[0], w01_t = w0[1], w02_t = w0[2], w03_t = w0[3];
    uint32_t w04_t = w1[0], w05_t = w1[1], w06_t = w1[2], w07_t = w1[3];
    uint32_t w08_t = w2[0], w09_t = w2[1], w0a_t = w2[2], w0b_t = w2[3];
    uint32_t w0c_t = w3[0], w0d_t = w3[1], w0e_t = w3[2], w0f_t = w3[3];

    #define K SHA1C00
    SHA1_STEP_S(SHA1_F0o, a, b, c, d, e, w00_t);
    SHA1_STEP_S(SHA1_F0o, e, a, b, c, d, w01_t);
    SHA1_STEP_S(SHA1_F0o, d, e, a, b, c, w02_t);
    SHA1_STEP_S(SHA1_F0o, c, d, e, a, b, w03_t);
    SHA1_STEP_S(SHA1_F0o, b, c, d, e, a, w04_t);
    SHA1_STEP_S(SHA1_F0o, a, b, c, d, e, w05_t);
    SHA1_STEP_S(SHA1_F0o, e, a, b, c, d, w06_t);
    SHA1_STEP_S(SHA1_F0o, d, e, a, b, c, w07_t);
    SHA1_STEP_S(SHA1_F0o, c, d, e, a, b, w08_t);
    SHA1_STEP_S(SHA1_F0o, b, c, d, e, a, w09_t);
    SHA1_STEP_S(SHA1_F0o, a, b, c, d, e, w0a_t);
    SHA1_STEP_S(SHA1_F0o, e, a, b, c, d, w0b_t);
    SHA1_STEP_S(SHA1_F0o, d, e, a, b, c, w0c_t);
    SHA1_STEP_S(SHA1_F0o, c, d, e, a, b, w0d_t);
    SHA1_STEP_S(SHA1_F0o, b, c, d, e, a, w0e_t);
    SHA1_STEP_S(SHA1_F0o, a, b, c, d, e, w0f_t);

    uint32_t w10_t = hc_rotl32_S(w0d_t ^ w08_t ^ w02_t ^ w00_t, 1u); SHA1_STEP_S(SHA1_F0o, e, a, b, c, d, w10_t);
    uint32_t w11_t = hc_rotl32_S(w0e_t ^ w09_t ^ w03_t ^ w01_t, 1u); SHA1_STEP_S(SHA1_F0o, d, e, a, b, c, w11_t);
    uint32_t w12_t = hc_rotl32_S(w0f_t ^ w0a_t ^ w04_t ^ w02_t, 1u); SHA1_STEP_S(SHA1_F0o, c, d, e, a, b, w12_t);
    uint32_t w13_t = hc_rotl32_S(w10_t ^ w0b_t ^ w05_t ^ w03_t, 1u); SHA1_STEP_S(SHA1_F0o, b, c, d, e, a, w13_t);

    #undef K
    #define K SHA1C01
    uint32_t w14_t = hc_rotl32_S(w11_t ^ w0c_t ^ w06_t ^ w04_t, 1u); SHA1_STEP_S(SHA1_F1, a, b, c, d, e, w14_t);
    uint32_t w15_t = hc_rotl32_S(w12_t ^ w0d_t ^ w07_t ^ w05_t, 1u); SHA1_STEP_S(SHA1_F1, e, a, b, c, d, w15_t);
    uint32_t w16_t = hc_rotl32_S(w13_t ^ w0e_t ^ w08_t ^ w06_t, 1u); SHA1_STEP_S(SHA1_F1, d, e, a, b, c, w16_t);
    uint32_t w17_t = hc_rotl32_S(w14_t ^ w0f_t ^ w09_t ^ w07_t, 1u); SHA1_STEP_S(SHA1_F1, c, d, e, a, b, w17_t);
    uint32_t w18_t = hc_rotl32_S(w15_t ^ w10_t ^ w0a_t ^ w08_t, 1u); SHA1_STEP_S(SHA1_F1, b, c, d, e, a, w18_t);
    uint32_t w19_t = hc_rotl32_S(w16_t ^ w11_t ^ w0b_t ^ w09_t, 1u); SHA1_STEP_S(SHA1_F1, a, b, c, d, e, w19_t);
    uint32_t w1a_t = hc_rotl32_S(w17_t ^ w12_t ^ w0c_t ^ w0a_t, 1u); SHA1_STEP_S(SHA1_F1, e, a, b, c, d, w1a_t);
    uint32_t w1b_t = hc_rotl32_S(w18_t ^ w13_t ^ w0d_t ^ w0b_t, 1u); SHA1_STEP_S(SHA1_F1, d, e, a, b, c, w1b_t);
    uint32_t w1c_t = hc_rotl32_S(w19_t ^ w14_t ^ w0e_t ^ w0c_t, 1u); SHA1_STEP_S(SHA1_F1, c, d, e, a, b, w1c_t);
    uint32_t w1d_t = hc_rotl32_S(w1a_t ^ w15_t ^ w0f_t ^ w0d_t, 1u); SHA1_STEP_S(SHA1_F1, b, c, d, e, a, w1d_t);
    uint32_t w1e_t = hc_rotl32_S(w1b_t ^ w16_t ^ w10_t ^ w0e_t, 1u); SHA1_STEP_S(SHA1_F1, a, b, c, d, e, w1e_t);
    uint32_t w1f_t = hc_rotl32_S(w1c_t ^ w17_t ^ w11_t ^ w0f_t, 1u); SHA1_STEP_S(SHA1_F1, e, a, b, c, d, w1f_t);
    uint32_t w20_t = hc_rotl32_S(w1d_t ^ w18_t ^ w12_t ^ w10_t, 1u); SHA1_STEP_S(SHA1_F1, d, e, a, b, c, w20_t);
    uint32_t w21_t = hc_rotl32_S(w1e_t ^ w19_t ^ w13_t ^ w11_t, 1u); SHA1_STEP_S(SHA1_F1, c, d, e, a, b, w21_t);
    uint32_t w22_t = hc_rotl32_S(w1f_t ^ w1a_t ^ w14_t ^ w12_t, 1u); SHA1_STEP_S(SHA1_F1, b, c, d, e, a, w22_t);
    uint32_t w23_t = hc_rotl32_S(w20_t ^ w1b_t ^ w15_t ^ w13_t, 1u); SHA1_STEP_S(SHA1_F1, a, b, c, d, e, w23_t);
    uint32_t w24_t = hc_rotl32_S(w21_t ^ w1c_t ^ w16_t ^ w14_t, 1u); SHA1_STEP_S(SHA1_F1, e, a, b, c, d, w24_t);
    uint32_t w25_t = hc_rotl32_S(w22_t ^ w1d_t ^ w17_t ^ w15_t, 1u); SHA1_STEP_S(SHA1_F1, d, e, a, b, c, w25_t);
    uint32_t w26_t = hc_rotl32_S(w23_t ^ w1e_t ^ w18_t ^ w16_t, 1u); SHA1_STEP_S(SHA1_F1, c, d, e, a, b, w26_t);
    uint32_t w27_t = hc_rotl32_S(w24_t ^ w1f_t ^ w19_t ^ w17_t, 1u); SHA1_STEP_S(SHA1_F1, b, c, d, e, a, w27_t);

    #undef K
    #define K SHA1C02
    uint32_t w28_t = hc_rotl32_S(w25_t ^ w20_t ^ w1a_t ^ w18_t, 1u); SHA1_STEP_S(SHA1_F2o, a, b, c, d, e, w28_t);
    uint32_t w29_t = hc_rotl32_S(w26_t ^ w21_t ^ w1b_t ^ w19_t, 1u); SHA1_STEP_S(SHA1_F2o, e, a, b, c, d, w29_t);
    uint32_t w2a_t = hc_rotl32_S(w27_t ^ w22_t ^ w1c_t ^ w1a_t, 1u); SHA1_STEP_S(SHA1_F2o, d, e, a, b, c, w2a_t);
    uint32_t w2b_t = hc_rotl32_S(w28_t ^ w23_t ^ w1d_t ^ w1b_t, 1u); SHA1_STEP_S(SHA1_F2o, c, d, e, a, b, w2b_t);
    uint32_t w2c_t = hc_rotl32_S(w29_t ^ w24_t ^ w1e_t ^ w1c_t, 1u); SHA1_STEP_S(SHA1_F2o, b, c, d, e, a, w2c_t);
    uint32_t w2d_t = hc_rotl32_S(w2a_t ^ w25_t ^ w1f_t ^ w1d_t, 1u); SHA1_STEP_S(SHA1_F2o, a, b, c, d, e, w2d_t);
    uint32_t w2e_t = hc_rotl32_S(w2b_t ^ w26_t ^ w20_t ^ w1e_t, 1u); SHA1_STEP_S(SHA1_F2o, e, a, b, c, d, w2e_t);
    uint32_t w2f_t = hc_rotl32_S(w2c_t ^ w27_t ^ w21_t ^ w1f_t, 1u); SHA1_STEP_S(SHA1_F2o, d, e, a, b, c, w2f_t);
    uint32_t w30_t = hc_rotl32_S(w2d_t ^ w28_t ^ w22_t ^ w20_t, 1u); SHA1_STEP_S(SHA1_F2o, c, d, e, a, b, w30_t);
    uint32_t w31_t = hc_rotl32_S(w2e_t ^ w29_t ^ w23_t ^ w21_t, 1u); SHA1_STEP_S(SHA1_F2o, b, c, d, e, a, w31_t);
    uint32_t w32_t = hc_rotl32_S(w2f_t ^ w2a_t ^ w24_t ^ w22_t, 1u); SHA1_STEP_S(SHA1_F2o, a, b, c, d, e, w32_t);
    uint32_t w33_t = hc_rotl32_S(w30_t ^ w2b_t ^ w25_t ^ w23_t, 1u); SHA1_STEP_S(SHA1_F2o, e, a, b, c, d, w33_t);
    uint32_t w34_t = hc_rotl32_S(w31_t ^ w2c_t ^ w26_t ^ w24_t, 1u); SHA1_STEP_S(SHA1_F2o, d, e, a, b, c, w34_t);
    uint32_t w35_t = hc_rotl32_S(w32_t ^ w2d_t ^ w27_t ^ w25_t, 1u); SHA1_STEP_S(SHA1_F2o, c, d, e, a, b, w35_t);
    uint32_t w36_t = hc_rotl32_S(w33_t ^ w2e_t ^ w28_t ^ w26_t, 1u); SHA1_STEP_S(SHA1_F2o, b, c, d, e, a, w36_t);
    uint32_t w37_t = hc_rotl32_S(w34_t ^ w2f_t ^ w29_t ^ w27_t, 1u); SHA1_STEP_S(SHA1_F2o, a, b, c, d, e, w37_t);
    uint32_t w38_t = hc_rotl32_S(w35_t ^ w30_t ^ w2a_t ^ w28_t, 1u); SHA1_STEP_S(SHA1_F2o, e, a, b, c, d, w38_t);
    uint32_t w39_t = hc_rotl32_S(w36_t ^ w31_t ^ w2b_t ^ w29_t, 1u); SHA1_STEP_S(SHA1_F2o, d, e, a, b, c, w39_t);
    uint32_t w3a_t = hc_rotl32_S(w37_t ^ w32_t ^ w2c_t ^ w2a_t, 1u); SHA1_STEP_S(SHA1_F2o, c, d, e, a, b, w3a_t);
    uint32_t w3b_t = hc_rotl32_S(w38_t ^ w33_t ^ w2d_t ^ w2b_t, 1u); SHA1_STEP_S(SHA1_F2o, b, c, d, e, a, w3b_t);

    #undef K
    #define K SHA1C03
    uint32_t w3c_t = hc_rotl32_S(w39_t ^ w34_t ^ w2e_t ^ w2c_t, 1u); SHA1_STEP_S(SHA1_F1, a, b, c, d, e, w3c_t);
    uint32_t w3d_t = hc_rotl32_S(w3a_t ^ w35_t ^ w2f_t ^ w2d_t, 1u); SHA1_STEP_S(SHA1_F1, e, a, b, c, d, w3d_t);
    uint32_t w3e_t = hc_rotl32_S(w3b_t ^ w36_t ^ w30_t ^ w2e_t, 1u); SHA1_STEP_S(SHA1_F1, d, e, a, b, c, w3e_t);
    uint32_t w3f_t = hc_rotl32_S(w3c_t ^ w37_t ^ w31_t ^ w2f_t, 1u); SHA1_STEP_S(SHA1_F1, c, d, e, a, b, w3f_t);
    uint32_t w40_t = hc_rotl32_S(w3d_t ^ w38_t ^ w32_t ^ w30_t, 1u); SHA1_STEP_S(SHA1_F1, b, c, d, e, a, w40_t);
    uint32_t w41_t = hc_rotl32_S(w3e_t ^ w39_t ^ w33_t ^ w31_t, 1u); SHA1_STEP_S(SHA1_F1, a, b, c, d, e, w41_t);
    uint32_t w42_t = hc_rotl32_S(w3f_t ^ w3a_t ^ w34_t ^ w32_t, 1u); SHA1_STEP_S(SHA1_F1, e, a, b, c, d, w42_t);
    uint32_t w43_t = hc_rotl32_S(w40_t ^ w3b_t ^ w35_t ^ w33_t, 1u); SHA1_STEP_S(SHA1_F1, d, e, a, b, c, w43_t);
    uint32_t w44_t = hc_rotl32_S(w41_t ^ w3c_t ^ w36_t ^ w34_t, 1u); SHA1_STEP_S(SHA1_F1, c, d, e, a, b, w44_t);
    uint32_t w45_t = hc_rotl32_S(w42_t ^ w3d_t ^ w37_t ^ w35_t, 1u); SHA1_STEP_S(SHA1_F1, b, c, d, e, a, w45_t);
    uint32_t w46_t = hc_rotl32_S(w43_t ^ w3e_t ^ w38_t ^ w36_t, 1u); SHA1_STEP_S(SHA1_F1, a, b, c, d, e, w46_t);
    uint32_t w47_t = hc_rotl32_S(w44_t ^ w3f_t ^ w39_t ^ w37_t, 1u); SHA1_STEP_S(SHA1_F1, e, a, b, c, d, w47_t);
    uint32_t w48_t = hc_rotl32_S(w45_t ^ w40_t ^ w3a_t ^ w38_t, 1u); SHA1_STEP_S(SHA1_F1, d, e, a, b, c, w48_t);
    uint32_t w49_t = hc_rotl32_S(w46_t ^ w41_t ^ w3b_t ^ w39_t, 1u); SHA1_STEP_S(SHA1_F1, c, d, e, a, b, w49_t);
    uint32_t w4a_t = hc_rotl32_S(w47_t ^ w42_t ^ w3c_t ^ w3a_t, 1u); SHA1_STEP_S(SHA1_F1, b, c, d, e, a, w4a_t);
    uint32_t w4b_t = hc_rotl32_S(w48_t ^ w43_t ^ w3d_t ^ w3b_t, 1u); SHA1_STEP_S(SHA1_F1, a, b, c, d, e, w4b_t);
    uint32_t w4c_t = hc_rotl32_S(w49_t ^ w44_t ^ w3e_t ^ w3c_t, 1u); SHA1_STEP_S(SHA1_F1, e, a, b, c, d, w4c_t);
    uint32_t w4d_t = hc_rotl32_S(w4a_t ^ w45_t ^ w3f_t ^ w3d_t, 1u); SHA1_STEP_S(SHA1_F1, d, e, a, b, c, w4d_t);
    uint32_t w4e_t = hc_rotl32_S(w4b_t ^ w46_t ^ w40_t ^ w3e_t, 1u); SHA1_STEP_S(SHA1_F1, c, d, e, a, b, w4e_t);
    uint32_t w4f_t = hc_rotl32_S(w4c_t ^ w47_t ^ w41_t ^ w3f_t, 1u); SHA1_STEP_S(SHA1_F1, b, c, d, e, a, w4f_t);
    #undef K

    digest[0] += a; digest[1] += b; digest[2] += c; digest[3] += d; digest[4] += e;
}
