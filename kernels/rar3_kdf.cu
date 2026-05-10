#include "common.cuh"
#include "sha1_hc.cuh"
#include "sha1_hc_switch.cuh"
#include "aes_device.cuh"

// ════════════════════════════════════════════════════════════════
//  RAR3 cracking kernel (mode 12500).
//
//  Design:
//    * Per-thread `w[66]` buffer: pw (UTF-16LE, byte-swapped to big-endian
//      u32) + 8-byte salt, composed ONCE at init. Never rebuilt in the loop.
//    * Running SHA-1 block held as w0[4]/w1[4]/w2[4]/w3[4] — nvcc keeps
//      these in registers when all helpers are __forceinline__.
//    * `sha1_update_rar29_inline`: copies pw_salt_len bytes from w[] into
//      the current block, transforms on overflow. For pw_salt_len ≤ 64
//      (pw_len ≤ 56 UTF-16 bytes = 28 ASCII chars) it's a single branch.
//    * `memcat8c_be_inline`: appends the 3-byte big-endian counter via
//      bytealign + switch(div) — single transform if block fills.
//    * Sample fusion: every 16384 iters, clone ctx to local scalars and
//      finalise to extract one key/IV byte without touching running ctx.
//
//  Local memory footprint: w[66] u32 = 264 B/thread. Sequential reads,
//  cached in L1/L2.
// ════════════════════════════════════════════════════════════════

#define RAR3_ITERS      0x40000u   // 262144
#define RAR3_SAMPLE_INT 0x4000u    // 16384

// ── Helper: put 4 u32 BE words into the 16-word block at arbitrary byte ──
// Not used in the hot path (handled inline for pw_salt_len ≤ 64); kept as
// reference / possible slow path for pw_len > 28 case.

// ── Running-block accessor pack ─────────────────────────────────
// A "ctx" is 16 u32 (the current 64-byte block) + h[5] + len.
// We keep the block as 4 arrays of 4 for compatibility with the transform
// signature, plus a running len for position tracking.

// `len` is bytes absorbed so far (includes all compressed blocks). The
// position in the current (unflushed) block is `(len & 63)`.

// Feed pw_salt_bytes from w[] into the block. For pw_salt_len ≤ 64 this is
// a single sha1_update_64_hc call on the first 16 aligned words of w[].
// For longer pw_salt_len (> 64 bytes), loop in 64-byte chunks. Used only
// for the hot path — no w[] writeback (sha1_update_rar29 writes back only
// when len > 64 and transformed bytes overlap; we loop instead, which
// avoids the writeback but costs more transforms. For
// typical pw_len ≤ 28 bytes UTF-16 the single-block path dominates.)
__device__ __forceinline__
void sha1_feed_prefix(
    uint32_t h[5],
    uint32_t ctx_w0[4], uint32_t ctx_w1[4], uint32_t ctx_w2[4], uint32_t ctx_w3[4],
    uint32_t &ctx_len,
    const uint32_t* w_src,
    uint32_t pw_salt_len)
{
    uint32_t pos = 0;
    while (pos < pw_salt_len) {
        uint32_t chunk = pw_salt_len - pos;
        if (chunk > 64u) chunk = 64u;
        uint32_t w0[4], w1[4], w2[4], w3[4];
        uint32_t src = pos >> 2;
        w0[0] = w_src[src +  0]; w0[1] = w_src[src +  1];
        w0[2] = w_src[src +  2]; w0[3] = w_src[src +  3];
        w1[0] = w_src[src +  4]; w1[1] = w_src[src +  5];
        w1[2] = w_src[src +  6]; w1[3] = w_src[src +  7];
        w2[0] = w_src[src +  8]; w2[1] = w_src[src +  9];
        w2[2] = w_src[src + 10]; w2[3] = w_src[src + 11];
        w3[0] = w_src[src + 12]; w3[1] = w_src[src + 13];
        w3[2] = w_src[src + 14]; w3[3] = w_src[src + 15];
        sha1_update_64_hc(h, ctx_w0, ctx_w1, ctx_w2, ctx_w3, ctx_len, w0, w1, w2, w3, chunk);
        pos += chunk;
    }
}

#if 0  // legacy body — kept out of build
static void _legacy_unused(uint32_t ctx_len, const uint32_t* w_src, uint32_t pw_salt_len) {
    uint32_t h[5]={0}; uint32_t w0[4]={0}, w1[4]={0}, w2[4]={0}, w3[4]={0};
    // Current position in the block
    uint32_t pos  = ctx_len & 63u;
    uint32_t left = 64u - pos;

    // Number of full BE u32 words to feed. pw_salt_len is in BYTES.
    // The source w_src[] is zero-padded at tail, so we can always read
    // whole u32 words. Bytes beyond pw_salt_len within the last word are
    // guaranteed zero by the init builder.
    uint32_t nbytes = pw_salt_len;

    // If the block has enough room (pos + nbytes ≤ 64), no transform needed.
    // Otherwise: fill up the block, transform, then continue with remaining.
    if (pos == 0) {
        // Aligned: feed words directly.
        if (nbytes > left) {
            // fill 64 bytes = 16 u32
            w0[0] = w_src[0];  w0[1] = w_src[1];  w0[2] = w_src[2];  w0[3] = w_src[3];
            w1[0] = w_src[4];  w1[1] = w_src[5];  w1[2] = w_src[6];  w1[3] = w_src[7];
            w2[0] = w_src[8];  w2[1] = w_src[9];  w2[2] = w_src[10]; w2[3] = w_src[11];
            w3[0] = w_src[12]; w3[1] = w_src[13]; w3[2] = w_src[14]; w3[3] = w_src[15];
            sha1_transform_hc(w0, w1, w2, w3, h);
            // remaining bytes go into fresh block starting at pos=0
            uint32_t tail = nbytes - 64u;
            // zero block
            w0[0]=0; w0[1]=0; w0[2]=0; w0[3]=0;
            w1[0]=0; w1[1]=0; w1[2]=0; w1[3]=0;
            w2[0]=0; w2[1]=0; w2[2]=0; w2[3]=0;
            w3[0]=0; w3[1]=0; w3[2]=0; w3[3]=0;
            // copy tail words
            const uint32_t tw = (tail + 3u) >> 2;
            #pragma unroll
            for (uint32_t k = 0; k < 16; ++k) {
                if (k < tw) {
                    uint32_t v = w_src[16 + k];
                    if      (k < 4)  w0[k]     = v;
                    else if (k < 8)  w1[k - 4] = v;
                    else if (k < 12) w2[k - 8] = v;
                    else             w3[k - 12]= v;
                }
            }
            ctx_len += pw_salt_len;
        } else {
            // Fits entirely. Just write nbytes rounded-up words into block.
            // Block positions 0..nbytes-1 overwritten; must not clobber
            // already-present tail (there isn't any when pos==0 + nbytes≤64).
            const uint32_t tw = (nbytes + 3u) >> 2;
            #pragma unroll
            for (uint32_t k = 0; k < 16; ++k) {
                if (k < tw) {
                    uint32_t v = w_src[k];
                    if      (k < 4)  w0[k]     = v;
                    else if (k < 8)  w1[k - 4] = v;
                    else if (k < 12) w2[k - 8] = v;
                    else             w3[k - 12]= v;
                }
            }
            ctx_len += nbytes;
        }
        return;
    }

    // Misaligned: we need to bit-shift w_src[] into the block at byte offset `pos`.
    uint32_t off = pos & 3u;           // byte offset within a u32 (0..3)
    uint32_t widx = pos >> 2;          // word index in block (0..15)
    uint32_t shift = off * 8u;         // bits to shift

    // Iterate over source words; each emits up to 2 destination words (due to shift).
    const uint32_t nsrc = (nbytes + 3u) >> 2;

    // Accumulator of carry bytes bleeding into next dest word.
    uint32_t carry = 0;
    bool block_flushed = false;

    // Small helper macro to access block as linear 16-word view via pointer.
    // nvcc inlines this into direct register moves when widx is compile-time
    // constant. Here widx is runtime so it falls back to local-memory, but
    // only 16 cases — keep occupancy by keeping branches short.
    #define SET_BLK(i, v) do {                                                   \
        uint32_t _i = (i) & 15u;                                                 \
        if      (_i <  4) { w0[_i]       = (w0[_i]       & ((_i==widx && shift)? \
                          (0xffffffffu << (32u - shift)) : 0u)) | (v); }         \
        else if (_i <  8) { w1[_i -  4]  = (v); }                                \
        else if (_i < 12) { w2[_i -  8]  = (v); }                                \
        else              { w3[_i - 12]  = (v); }                                \
    } while (0)
    (void)carry; (void)block_flushed;
    #undef SET_BLK

    // Slow / fallback path for the misaligned case. We materialise the block
    // as 16 scalars, shift-merge, then scatter back. This only fires when
    // pos % 4 != 0 which in RAR3 happens when pw_salt_len % 3 shifts the
    // cursor. The hot path (aligned) handles ≥99% of iterations.
    uint32_t blk[16];
    blk[ 0]=w0[0]; blk[ 1]=w0[1]; blk[ 2]=w0[2]; blk[ 3]=w0[3];
    blk[ 4]=w1[0]; blk[ 5]=w1[1]; blk[ 6]=w1[2]; blk[ 7]=w1[3];
    blk[ 8]=w2[0]; blk[ 9]=w2[1]; blk[10]=w2[2]; blk[11]=w2[3];
    blk[12]=w3[0]; blk[13]=w3[1]; blk[14]=w3[2]; blk[15]=w3[3];

    uint32_t cur = widx;
    uint32_t prev_tail = 0;
    uint32_t consumed = 0;

    for (uint32_t k = 0; k <= nsrc; ++k) {
        uint32_t src = (k < nsrc) ? w_src[k] : 0u;
        // High part goes into the current word at the LSB side (shift out by `shift`).
        // Low part goes into the next word at the MSB side.
        uint32_t high = prev_tail | (shift ? (src >> shift) : src);
        uint32_t low  = shift ? (src << (32u - shift)) : 0u;

        if (k == 0 && shift) {
            // First write: preserve high (32-shift) bits of blk[cur]
            uint32_t mask_hi = 0xffffffffu << (32u - shift);
            blk[cur] = (blk[cur] & mask_hi) | high;
        } else if (k > 0) {
            blk[cur] = high;
        }
        prev_tail = low;

        // Advance
        cur++;
        if (cur == 16) {
            // scatter back & flush
            w0[0]=blk[ 0]; w0[1]=blk[ 1]; w0[2]=blk[ 2]; w0[3]=blk[ 3];
            w1[0]=blk[ 4]; w1[1]=blk[ 5]; w1[2]=blk[ 6]; w1[3]=blk[ 7];
            w2[0]=blk[ 8]; w2[1]=blk[ 9]; w2[2]=blk[10]; w2[3]=blk[11];
            w3[0]=blk[12]; w3[1]=blk[13]; w3[2]=blk[14]; w3[3]=blk[15];
            sha1_transform_hc(w0, w1, w2, w3, h);
            #pragma unroll
            for (int i = 0; i < 16; ++i) blk[i] = 0;
            cur = 0;
        }

        consumed += 4u;
        if (consumed >= nbytes + (shift ? 4u : 0u)) break;
    }

    // scatter back
    w0[0]=blk[ 0]; w0[1]=blk[ 1]; w0[2]=blk[ 2]; w0[3]=blk[ 3];
    w1[0]=blk[ 4]; w1[1]=blk[ 5]; w1[2]=blk[ 6]; w1[3]=blk[ 7];
    w2[0]=blk[ 8]; w2[1]=blk[ 9]; w2[2]=blk[10]; w2[3]=blk[11];
    w3[0]=blk[12]; w3[1]=blk[13]; w3[2]=blk[14]; w3[3]=blk[15];

    ctx_len += pw_salt_len;
}
#endif // legacy body

// Append 3-byte big-endian counter to the current block; transform if full.
// `append` must already be in BE byte order (LSB of u32 is last appended byte).
// append = hc_swap32_S(counter_u32) puts the 3 LS bytes at the TOP of the
// u32; bytealign_be then shifts them into place.
//
// The switch is keyed off `div = (len & 63) / 4` = current word index.
__device__ __forceinline__
void memcat8c_be_inline(
    uint32_t h[5],
    uint32_t w0[4], uint32_t w1[4], uint32_t w2[4], uint32_t w3[4],
    uint32_t ctx_len,
    uint32_t append)
{
    const uint32_t func_len = ctx_len & 63u;
    const uint32_t div      = func_len >> 2;

    // tmp0: portion that fits in the current word (possibly zero bytes if
    //       the 3-byte append straddles a word boundary)
    // tmp1: carry bytes into the next word
    const uint32_t tmp0 = hc_bytealign_be_S(0u,      append, (int)func_len);
    const uint32_t tmp1 = hc_bytealign_be_S(append, 0u,      (int)func_len);

    uint32_t carry = 0u;

    switch (div) {
        case  0: w0[0] |= tmp0; w0[1]  = tmp1; break;
        case  1: w0[1] |= tmp0; w0[2]  = tmp1; break;
        case  2: w0[2] |= tmp0; w0[3]  = tmp1; break;
        case  3: w0[3] |= tmp0; w1[0]  = tmp1; break;
        case  4: w1[0] |= tmp0; w1[1]  = tmp1; break;
        case  5: w1[1] |= tmp0; w1[2]  = tmp1; break;
        case  6: w1[2] |= tmp0; w1[3]  = tmp1; break;
        case  7: w1[3] |= tmp0; w2[0]  = tmp1; break;
        case  8: w2[0] |= tmp0; w2[1]  = tmp1; break;
        case  9: w2[1] |= tmp0; w2[2]  = tmp1; break;
        case 10: w2[2] |= tmp0; w2[3]  = tmp1; break;
        case 11: w2[3] |= tmp0; w3[0]  = tmp1; break;
        case 12: w3[0] |= tmp0; w3[1]  = tmp1; break;
        case 13: w3[1] |= tmp0; w3[2]  = tmp1; break;
        case 14: w3[2] |= tmp0; w3[3]  = tmp1; break;
        default: w3[3] |= tmp0; carry  = tmp1; break; // case 15
    }

    const uint32_t new_len = func_len + 3u;
    if (new_len >= 64u) {
        sha1_transform_hc(w0, w1, w2, w3, h);
        w0[0] = carry; w0[1] = 0; w0[2] = 0; w0[3] = 0;
        w1[0] = 0;     w1[1] = 0; w1[2] = 0; w1[3] = 0;
        w2[0] = 0;     w2[1] = 0; w2[2] = 0; w2[3] = 0;
        w3[0] = 0;     w3[1] = 0; w3[2] = 0; w3[3] = 0;
    }
}

// Clone-and-finalise to extract the first digest byte. Does NOT mutate the
// caller's running h / w0..w3 / ctx_len. Used for key/IV byte sampling.
__device__ __forceinline__
uint8_t sha1_sample_byte(
    const uint32_t h_in[5],
    const uint32_t w0_in[4], const uint32_t w1_in[4],
    const uint32_t w2_in[4], const uint32_t w3_in[4],
    uint32_t ctx_len)
{
    uint32_t h[5] = { h_in[0], h_in[1], h_in[2], h_in[3], h_in[4] };
    uint32_t w0[4] = { w0_in[0], w0_in[1], w0_in[2], w0_in[3] };
    uint32_t w1[4] = { w1_in[0], w1_in[1], w1_in[2], w1_in[3] };
    uint32_t w2[4] = { w2_in[0], w2_in[1], w2_in[2], w2_in[3] };
    uint32_t w3[4] = { w3_in[0], w3_in[1], w3_in[2], w3_in[3] };

    const uint32_t func_len = ctx_len & 63u;
    const uint32_t div      = func_len >> 2;
    const uint32_t off      = (func_len & 3u) * 8u;
    const uint32_t pad      = off ? (0x80u << (24u - off)) : 0x80000000u;

    switch (div) {
        case  0: w0[0] |= pad; break; case  1: w0[1] |= pad; break;
        case  2: w0[2] |= pad; break; case  3: w0[3] |= pad; break;
        case  4: w1[0] |= pad; break; case  5: w1[1] |= pad; break;
        case  6: w1[2] |= pad; break; case  7: w1[3] |= pad; break;
        case  8: w2[0] |= pad; break; case  9: w2[1] |= pad; break;
        case 10: w2[2] |= pad; break; case 11: w2[3] |= pad; break;
        case 12: w3[0] |= pad; break; case 13: w3[1] |= pad; break;
        case 14: w3[2] |= pad; break; default: w3[3] |= pad; break;
    }

    // If no room for 8-byte bit count, flush and start new block.
    if (func_len >= 56u) {
        sha1_transform_hc(w0, w1, w2, w3, h);
        w0[0]=0; w0[1]=0; w0[2]=0; w0[3]=0;
        w1[0]=0; w1[1]=0; w1[2]=0; w1[3]=0;
        w2[0]=0; w2[1]=0; w2[2]=0; w2[3]=0;
        w3[0]=0; w3[1]=0; w3[2]=0; w3[3]=0;
    }

    uint64_t bits = (uint64_t)ctx_len * 8ull;
    w3[2] = (uint32_t)(bits >> 32);
    w3[3] = (uint32_t)(bits & 0xffffffffu);
    sha1_transform_hc(w0, w1, w2, w3, h);

    return (uint8_t)(h[0] >> 24);
}

// ── RAR3 key derivation (one thread, one password) ──────────────
__device__ static void rar3_kdf(
    const uint8_t *utf16pw, uint32_t pwlen_bytes,  // pwlen_bytes ≤ 64 recommended
    const uint8_t *salt,                           // 8 bytes
    uint8_t key[16], uint8_t iv[16])
{
    const uint32_t pw_salt_len = pwlen_bytes + 8u;

    // ── 1. Build w[66] in BE u32 words ────────────────────────
    // Up to 256 pw bytes + 8 salt = 264 bytes = 66 u32. We only use the
    // first ceil((pwlen_bytes+8)/4) words; the rest remain zero.
    uint32_t w[66];
    #pragma unroll
    for (int i = 0; i < 66; ++i) w[i] = 0u;

    // Load pw as little-endian u32 words, swap to big-endian. The host
    // packs UTF-16LE bytes so that 4 consecutive bytes form the natural
    // byte order; we swap to match SHA-1 big-endian word layout.
    // For odd byte counts we handle a partial tail word.
    const uint32_t pw_words = pwlen_bytes >> 2;
    const uint32_t pw_tail  = pwlen_bytes &  3u;
    #pragma unroll 16
    for (uint32_t i = 0; i < 64; ++i) {  // up to 256 pw bytes = 64 words
        if (i < pw_words) {
            const uint8_t *p = utf16pw + i * 4u;
            uint32_t le = ((uint32_t)p[0])        | ((uint32_t)p[1] << 8)
                        | ((uint32_t)p[2] << 16)  | ((uint32_t)p[3] << 24);
            w[i] = hc_swap32_S(le);
        }
    }
    if (pw_tail) {
        uint32_t le = 0;
        const uint8_t *p = utf16pw + pw_words * 4u;
        for (uint32_t j = 0; j < pw_tail; ++j) {
            le |= ((uint32_t)p[j]) << (j * 8u);
        }
        w[pw_words] = hc_swap32_S(le);
    }

    // Append salt (8 bytes = 2 BE u32 words) at position pwlen_bytes.
    // If pwlen_bytes % 4 == 0 → clean word append at word index pw_words.
    // Otherwise shift-merge into w[pw_words] and w[pw_words+1].
    uint32_t salt0 = ((uint32_t)salt[0] << 24) | ((uint32_t)salt[1] << 16)
                   | ((uint32_t)salt[2] <<  8) |  (uint32_t)salt[3];
    uint32_t salt1 = ((uint32_t)salt[4] << 24) | ((uint32_t)salt[5] << 16)
                   | ((uint32_t)salt[6] <<  8) |  (uint32_t)salt[7];
    uint32_t salt2 = 0u;

    if (pw_tail != 0u) {
        uint32_t shift = pw_tail * 8u;            // bytes-to-bits
        uint32_t inv   = 32u - shift;
        salt2 =              (salt1 << inv);
        salt1 = (salt1 >> shift) | (salt0 << inv);
        salt0 = (salt0 >> shift);
    }
    w[pw_words + 0] |= salt0;
    w[pw_words + 1]  = salt1;
    w[pw_words + 2]  = salt2;

    // ── 2. Initialise running SHA-1 ctx ──────────────────────
    uint32_t h[5] = { SHA1M_A, SHA1M_B, SHA1M_C, SHA1M_D, SHA1M_E };
    uint32_t w0[4] = {0,0,0,0};
    uint32_t w1[4] = {0,0,0,0};
    uint32_t w2[4] = {0,0,0,0};
    uint32_t w3[4] = {0,0,0,0};
    uint32_t ctx_len = 0u;

    uint32_t key_idx = 0u, iv_idx = 0u;

    // ── 3. Main loop: 262144 iters ───────────────────────────
    for (uint32_t i = 0; i < RAR3_ITERS; ++i) {
        sha1_feed_prefix(h, w0, w1, w2, w3, ctx_len, w, pw_salt_len);

        // UnRAR appends 3 counter bytes LSB-first: [i&0xff, (i>>8)&0xff, (i>>16)&0xff].
        // memcat8c_be writes the TOP 3 BE bytes of `append`. So we need byte 0 of
        // the counter at [31:24] of append. That's exactly hc_swap32_S(i): for
        // i=0x010203 → 0x03020100 → top bytes [03,02,01] = (i>>16)|(i>>8)|(i&ff).
        // Wait — that reverses the order. We actually want [i&ff, (i>>8)&ff, (i>>16)&ff]
        // at [31:24, 23:16, 15:8]. That is ((i&ff)<<24) | (((i>>8)&ff)<<16) | (((i>>16)&ff)<<8).
        // Equivalent: swap the low 3 bytes end-to-end. hc_swap32_S(i) does exactly this
        // (swaps 4 bytes; byte 3 of i is always 0 for i < 2^24 so it lands at [7:0]=0,
        // which is what we want).
        uint32_t append = hc_swap32_S(i);
        memcat8c_be_inline(h, w0, w1, w2, w3, ctx_len, append);
        ctx_len += 3u;

        // Sample every RAR3_SAMPLE_INT iters:
        // Key byte n at i = n*16384 + 16383
        // IV  byte n at i = n*16384 +  8191
        uint32_t imod = i & (RAR3_SAMPLE_INT - 1u);
        if (imod == (RAR3_SAMPLE_INT - 1u) && key_idx < 16u) {
            key[key_idx++] = sha1_sample_byte(h, w0, w1, w2, w3, ctx_len);
        } else if (imod == (RAR3_SAMPLE_INT / 2u - 1u) && iv_idx < 16u) {
            iv[iv_idx++] = sha1_sample_byte(h, w0, w1, w2, w3, ctx_len);
        }
    }
}

// ── CRC32 (IEEE 802.3 / zip polynomial) ──────────────────────
__device__ __forceinline__
uint32_t rar3_crc32(const uint8_t *data, int len)
{
    uint32_t crc = 0xFFFFFFFFu;
    for (int i = 0; i < len; i++) {
        crc ^= data[i];
        for (int b = 0; b < 8; b++)
            crc = (crc >> 1) ^ (0xEDB88320u & (uint32_t)(-(int32_t)(crc & 1)));
    }
    return crc ^ 0xFFFFFFFFu;
}

// ── Kernel entry point ──────────────────────────────────────
extern "C"
__launch_bounds__(128, 6)
__global__ void rar3_crack(
    const uint8_t * __restrict__ passwords_utf16,
    const int32_t * __restrict__ pw_lengths_u16,
    int32_t         num_passwords,
    const uint8_t * __restrict__ salt,          // 8 bytes
    const uint8_t * __restrict__ enc_check,     // 16 bytes
    int32_t         check_mode,
    int32_t         head_type,
    uint32_t        file_crc,
    int32_t         pack_size,
    int32_t       * __restrict__ result)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_passwords) return;

    const uint8_t *pw = passwords_utf16 + (size_t)tid * MAX_PW_BYTES;
    uint32_t pwlen   = (uint32_t)pw_lengths_u16[tid];

    uint8_t key[16], iv[16];
    rar3_kdf(pw, pwlen, salt, key, iv);

    aes_ctx_t aes;
    aes128_key_expand(&aes, key);

    uint8_t block[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) block[i] = enc_check[i];
    aes_cbc_decrypt(&aes, block, 1, iv);

    bool match = false;
    if (check_mode == 0) {
        uint8_t b2 = block[2];
        match = (b2 == 0x72 || b2 == 0x73 || b2 == 0x74 ||
                 b2 == 0x75 || b2 == 0x76 || b2 == 0x7a || b2 == 0x7b);
        (void)head_type;
    } else if (check_mode == 1) {
        uint32_t computed = rar3_crc32(block, pack_size);
        match = (computed == file_crc);
    } else {
        uint8_t b0 = block[0];
        match = (b0 >= 0x72 && b0 <= 0x7b);
    }

    if (match) atomicCAS(result, NO_MATCH, tid);
}

// ════════════════════════════════════════════════════════════════
//  Kernel-split variant (init / loop / comp).
//
//  Splits the 262144-iteration monolithic kernel into:
//    rar3_init  — build w[66], init SHA-1 state → tmps[gid]
//    rar3_loop  — 16384 iters per launch (host launches 16×)
//    rar3_comp  — AES decrypt + check
//
//  By reducing per-kernel register pressure, this allows higher
//  occupancy and better latency hiding for SHA-1 stall_wait.
// ════════════════════════════════════════════════════════════════

struct rar3_tmps {
    uint32_t h[5];
    uint32_t w0[4], w1[4], w2[4], w3[4];
    uint32_t ctx_len;
    uint32_t w[66];
    uint32_t pw_salt_len;
    uint8_t  key[16];
    uint8_t  iv[16];
    uint32_t key_idx;
    uint32_t iv_idx;
};
// sizeof = 5*4 + 16*4 + 4 + 66*4 + 4 + 32 + 8 = 396 bytes

extern "C"
__launch_bounds__(256, 4)
__global__ void rar3_init(
    const uint8_t * __restrict__ passwords_utf16,
    const int32_t * __restrict__ pw_lengths_u16,
    int32_t         num_passwords,
    const uint8_t * __restrict__ salt,
    rar3_tmps     * __restrict__ tmps)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_passwords) return;

    const uint8_t *pw = passwords_utf16 + (size_t)gid * MAX_PW_BYTES;
    uint32_t pwlen = (uint32_t)pw_lengths_u16[gid];
    uint32_t pw_salt_len = pwlen + 8u;

    // Build w[66]
    uint32_t w[66];
    #pragma unroll
    for (int i = 0; i < 66; ++i) w[i] = 0u;

    const uint32_t pw_words = pwlen >> 2;
    const uint32_t pw_tail  = pwlen &  3u;
    #pragma unroll 16
    for (uint32_t i = 0; i < 64; ++i) {
        if (i < pw_words) {
            const uint8_t *p = pw + i * 4u;
            uint32_t le = ((uint32_t)p[0])        | ((uint32_t)p[1] << 8)
                        | ((uint32_t)p[2] << 16)  | ((uint32_t)p[3] << 24);
            w[i] = hc_swap32_S(le);
        }
    }
    if (pw_tail) {
        uint32_t le = 0;
        const uint8_t *p = pw + pw_words * 4u;
        for (uint32_t j = 0; j < pw_tail; ++j)
            le |= ((uint32_t)p[j]) << (j * 8u);
        w[pw_words] = hc_swap32_S(le);
    }

    uint32_t salt0 = ((uint32_t)salt[0] << 24) | ((uint32_t)salt[1] << 16)
                   | ((uint32_t)salt[2] <<  8) |  (uint32_t)salt[3];
    uint32_t salt1 = ((uint32_t)salt[4] << 24) | ((uint32_t)salt[5] << 16)
                   | ((uint32_t)salt[6] <<  8) |  (uint32_t)salt[7];
    uint32_t salt2 = 0u;
    if (pw_tail != 0u) {
        uint32_t shift = pw_tail * 8u;
        uint32_t inv   = 32u - shift;
        salt2 = (salt1 << inv);
        salt1 = (salt1 >> shift) | (salt0 << inv);
        salt0 = (salt0 >> shift);
    }
    w[pw_words + 0] |= salt0;
    w[pw_words + 1]  = salt1;
    w[pw_words + 2]  = salt2;

    // Store to global
    #pragma unroll
    for (int i = 0; i < 66; ++i) tmps[gid].w[i] = w[i];
    tmps[gid].pw_salt_len = pw_salt_len;

    tmps[gid].h[0] = SHA1M_A; tmps[gid].h[1] = SHA1M_B;
    tmps[gid].h[2] = SHA1M_C; tmps[gid].h[3] = SHA1M_D;
    tmps[gid].h[4] = SHA1M_E;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        tmps[gid].w0[i] = 0; tmps[gid].w1[i] = 0;
        tmps[gid].w2[i] = 0; tmps[gid].w3[i] = 0;
    }
    tmps[gid].ctx_len = 0;
    tmps[gid].key_idx = 0;
    tmps[gid].iv_idx  = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) { tmps[gid].key[i] = 0; tmps[gid].iv[i] = 0; }
}

extern "C"
__launch_bounds__(128, 8)
__global__ void rar3_loop(
    rar3_tmps * __restrict__ tmps,
    int32_t     num_passwords,
    uint32_t    loop_pos,
    uint32_t    loop_cnt)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_passwords) return;

    // Load frozen w[] to local memory (once per launch)
    uint32_t w[66];
    #pragma unroll
    for (int i = 0; i < 66; ++i) w[i] = tmps[gid].w[i];
    uint32_t pw_salt_len = tmps[gid].pw_salt_len;

    // Load mutable state to registers
    uint32_t h[5];
    h[0] = tmps[gid].h[0]; h[1] = tmps[gid].h[1]; h[2] = tmps[gid].h[2];
    h[3] = tmps[gid].h[3]; h[4] = tmps[gid].h[4];
    uint32_t w0[4], w1[4], w2[4], w3[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        w0[k] = tmps[gid].w0[k]; w1[k] = tmps[gid].w1[k];
        w2[k] = tmps[gid].w2[k]; w3[k] = tmps[gid].w3[k];
    }
    uint32_t ctx_len = tmps[gid].ctx_len;
    uint32_t key_idx = tmps[gid].key_idx;
    uint32_t iv_idx  = tmps[gid].iv_idx;

    for (uint32_t j = 0; j < loop_cnt; ++j) {
        uint32_t i = loop_pos + j;

        sha1_feed_prefix(h, w0, w1, w2, w3, ctx_len, w, pw_salt_len);

        uint32_t append = hc_swap32_S(i);
        memcat8c_be_inline(h, w0, w1, w2, w3, ctx_len, append);
        ctx_len += 3u;

        uint32_t imod = i & (RAR3_SAMPLE_INT - 1u);
        if (imod == (RAR3_SAMPLE_INT - 1u) && key_idx < 16u) {
            tmps[gid].key[key_idx] = sha1_sample_byte(h, w0, w1, w2, w3, ctx_len);
            key_idx++;
        } else if (imod == (RAR3_SAMPLE_INT / 2u - 1u) && iv_idx < 16u) {
            tmps[gid].iv[iv_idx] = sha1_sample_byte(h, w0, w1, w2, w3, ctx_len);
            iv_idx++;
        }
    }

    // Store state back
    tmps[gid].h[0] = h[0]; tmps[gid].h[1] = h[1]; tmps[gid].h[2] = h[2];
    tmps[gid].h[3] = h[3]; tmps[gid].h[4] = h[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        tmps[gid].w0[k] = w0[k]; tmps[gid].w1[k] = w1[k];
        tmps[gid].w2[k] = w2[k]; tmps[gid].w3[k] = w3[k];
    }
    tmps[gid].ctx_len = ctx_len;
    tmps[gid].key_idx = key_idx;
    tmps[gid].iv_idx  = iv_idx;
}

// ════════════════════════════════════════════════════════════════
//  ILP=2 loop kernel: each thread processes TWO passwords.
//  Thread tid handles passwords gid_a = 2*tid and gid_b = 2*tid+1.
//  The two SHA-1 chains are interleaved so that dependency stalls
//  from one chain get filled by instructions from the other.
//  Host launches with grid = ceil(num_passwords / 2 / BLOCK).
// ════════════════════════════════════════════════════════════════
extern "C"
__launch_bounds__(64, 4)
__global__ void rar3_loop_ilp2(
    rar3_tmps * __restrict__ tmps,
    int32_t     num_passwords,
    uint32_t    loop_pos,
    uint32_t    loop_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gid_a = tid * 2;
    int gid_b = tid * 2 + 1;

    if (gid_a >= num_passwords) return;
    bool has_b = (gid_b < num_passwords);

    // ── Load frozen w[] for both passwords ──
    uint32_t w_a[66];
    #pragma unroll
    for (int i = 0; i < 66; ++i) w_a[i] = tmps[gid_a].w[i];
    uint32_t pw_salt_len_a = tmps[gid_a].pw_salt_len;

    uint32_t w_b[66];
    if (has_b) {
        #pragma unroll
        for (int i = 0; i < 66; ++i) w_b[i] = tmps[gid_b].w[i];
    } else {
        #pragma unroll
        for (int i = 0; i < 66; ++i) w_b[i] = 0;
    }
    uint32_t pw_salt_len_b = has_b ? tmps[gid_b].pw_salt_len : 0u;

    // ── Load mutable state A ──
    uint32_t h_a[5];
    h_a[0] = tmps[gid_a].h[0]; h_a[1] = tmps[gid_a].h[1]; h_a[2] = tmps[gid_a].h[2];
    h_a[3] = tmps[gid_a].h[3]; h_a[4] = tmps[gid_a].h[4];
    uint32_t w0_a[4], w1_a[4], w2_a[4], w3_a[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        w0_a[k] = tmps[gid_a].w0[k]; w1_a[k] = tmps[gid_a].w1[k];
        w2_a[k] = tmps[gid_a].w2[k]; w3_a[k] = tmps[gid_a].w3[k];
    }
    uint32_t ctx_len_a = tmps[gid_a].ctx_len;
    uint32_t key_idx_a = tmps[gid_a].key_idx;
    uint32_t iv_idx_a  = tmps[gid_a].iv_idx;

    // ── Load mutable state B ──
    uint32_t h_b[5];
    uint32_t w0_b[4], w1_b[4], w2_b[4], w3_b[4];
    uint32_t ctx_len_b, key_idx_b, iv_idx_b;
    if (has_b) {
        h_b[0] = tmps[gid_b].h[0]; h_b[1] = tmps[gid_b].h[1]; h_b[2] = tmps[gid_b].h[2];
        h_b[3] = tmps[gid_b].h[3]; h_b[4] = tmps[gid_b].h[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            w0_b[k] = tmps[gid_b].w0[k]; w1_b[k] = tmps[gid_b].w1[k];
            w2_b[k] = tmps[gid_b].w2[k]; w3_b[k] = tmps[gid_b].w3[k];
        }
        ctx_len_b = tmps[gid_b].ctx_len;
        key_idx_b = tmps[gid_b].key_idx;
        iv_idx_b  = tmps[gid_b].iv_idx;
    } else {
        h_b[0] = SHA1M_A; h_b[1] = SHA1M_B; h_b[2] = SHA1M_C;
        h_b[3] = SHA1M_D; h_b[4] = SHA1M_E;
        #pragma unroll
        for (int k = 0; k < 4; ++k) { w0_b[k]=0; w1_b[k]=0; w2_b[k]=0; w3_b[k]=0; }
        ctx_len_b = 0; key_idx_b = 0; iv_idx_b = 0;
    }

    // ── Main loop: interleave A and B ──
    for (uint32_t j = 0; j < loop_cnt; ++j) {
        uint32_t i = loop_pos + j;
        uint32_t append = hc_swap32_S(i);

        // Feed prefix A then B (compiler interleaves independent instructions)
        sha1_feed_prefix(h_a, w0_a, w1_a, w2_a, w3_a, ctx_len_a, w_a, pw_salt_len_a);
        sha1_feed_prefix(h_b, w0_b, w1_b, w2_b, w3_b, ctx_len_b, w_b, pw_salt_len_b);

        // Append counter A then B
        memcat8c_be_inline(h_a, w0_a, w1_a, w2_a, w3_a, ctx_len_a, append);
        memcat8c_be_inline(h_b, w0_b, w1_b, w2_b, w3_b, ctx_len_b, append);
        ctx_len_a += 3u;
        ctx_len_b += 3u;

        // Sample A
        uint32_t imod = i & (RAR3_SAMPLE_INT - 1u);
        if (imod == (RAR3_SAMPLE_INT - 1u) && key_idx_a < 16u) {
            tmps[gid_a].key[key_idx_a] = sha1_sample_byte(h_a, w0_a, w1_a, w2_a, w3_a, ctx_len_a);
            key_idx_a++;
        } else if (imod == (RAR3_SAMPLE_INT / 2u - 1u) && iv_idx_a < 16u) {
            tmps[gid_a].iv[iv_idx_a] = sha1_sample_byte(h_a, w0_a, w1_a, w2_a, w3_a, ctx_len_a);
            iv_idx_a++;
        }

        // Sample B
        if (has_b) {
            if (imod == (RAR3_SAMPLE_INT - 1u) && key_idx_b < 16u) {
                tmps[gid_b].key[key_idx_b] = sha1_sample_byte(h_b, w0_b, w1_b, w2_b, w3_b, ctx_len_b);
                key_idx_b++;
            } else if (imod == (RAR3_SAMPLE_INT / 2u - 1u) && iv_idx_b < 16u) {
                tmps[gid_b].iv[iv_idx_b] = sha1_sample_byte(h_b, w0_b, w1_b, w2_b, w3_b, ctx_len_b);
                iv_idx_b++;
            }
        }
    }

    // ── Store state back A ──
    tmps[gid_a].h[0] = h_a[0]; tmps[gid_a].h[1] = h_a[1]; tmps[gid_a].h[2] = h_a[2];
    tmps[gid_a].h[3] = h_a[3]; tmps[gid_a].h[4] = h_a[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        tmps[gid_a].w0[k] = w0_a[k]; tmps[gid_a].w1[k] = w1_a[k];
        tmps[gid_a].w2[k] = w2_a[k]; tmps[gid_a].w3[k] = w3_a[k];
    }
    tmps[gid_a].ctx_len = ctx_len_a;
    tmps[gid_a].key_idx = key_idx_a;
    tmps[gid_a].iv_idx  = iv_idx_a;

    // ── Store state back B ──
    if (has_b) {
        tmps[gid_b].h[0] = h_b[0]; tmps[gid_b].h[1] = h_b[1]; tmps[gid_b].h[2] = h_b[2];
        tmps[gid_b].h[3] = h_b[3]; tmps[gid_b].h[4] = h_b[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            tmps[gid_b].w0[k] = w0_b[k]; tmps[gid_b].w1[k] = w1_b[k];
            tmps[gid_b].w2[k] = w2_b[k]; tmps[gid_b].w3[k] = w3_b[k];
        }
        tmps[gid_b].ctx_len = ctx_len_b;
        tmps[gid_b].key_idx = key_idx_b;
        tmps[gid_b].iv_idx  = iv_idx_b;
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Largeblock loop kernel.
//
//  Key idea: precompute a "largeblock" = 64 consecutive reps of [pw | salt |
//  3-byte-counter-slot] packed back-to-back in local memory.  64 reps ×
//  p3 bytes/rep = p3 × 64 bytes = p3 complete SHA-1 blocks, so every group
//  of 64 iterations consumes exactly p3 unconditional sha1_transform calls
//  (no block-filling conditional, no switch_buf_carry).
//
//  Counter patching: a while-loop patches all counters that land in the
//  current block (usually 2-4 per block for typical passwords).  The counter
//  high byte is constant per batch and pre-filled in init; only low/mid bytes
//  are written each outer iteration.  A 16-case kd switch + 4-case km
//  if-chain replaces the 63-case switch_buf_carry_be_full.
//
//  Key/IV sampling: both sampling points (iters = multiple of 8192 or 16384)
//  fall on clean SHA-1 block boundaries (verified: 8192*p3 and 16384*p3 are
//  always multiples of 64).  sha1_sample_byte is called with zero w0..w3.
//
//  Launch config: __launch_bounds__(256, 4).
// ════════════════════════════════════════════════════════════════════════════

#define LB_ELEMS ((40 + 8 + 3) * 16)   // 816 u32s max largeblock

extern "C"
__launch_bounds__(256, 4)
__global__ void rar3_loop_lb(
    rar3_tmps * __restrict__ tmps,
    int32_t     num_passwords,
    uint32_t    loop_pos,
    uint32_t    loop_cnt)   // always 16384
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_passwords) return;

    const uint32_t pw_salt_len = tmps[gid].pw_salt_len;
    const uint32_t p3 = pw_salt_len + 3u;  // bytes per iter = blocks per 64-iter group

    // Load pw+salt (≤12 words = ≤48 bytes) from global tmps
    const uint32_t wg_words = (pw_salt_len + 3u) >> 2;
    uint32_t wg[12];
    for (uint32_t i = 0; i < 12u; ++i)
        wg[i] = (i < wg_words) ? tmps[gid].w[i] : 0u;

    // Load running SHA-1 digest
    uint32_t h[5];
    h[0]=tmps[gid].h[0]; h[1]=tmps[gid].h[1]; h[2]=tmps[gid].h[2];
    h[3]=tmps[gid].h[3]; h[4]=tmps[gid].h[4];
    uint32_t key_idx = tmps[gid].key_idx;
    uint32_t iv_idx  = tmps[gid].iv_idx;

    // ── Build largeblock (once per kernel call) ────────────────────────────
    // 64 reps of [pw_bytes | salt_bytes | 3_counter_bytes].
    // Counter bytes are zeroed; the high byte (constant per batch) is pre-filled.
    // Only zero the words actually used (p3*16), not the full LB_ELEMS.
    uint32_t largeblock[LB_ELEMS];
    const uint32_t lb_used = p3 << 4u;  // p3 * 16
    for (uint32_t i = 0; i < lb_used; ++i) largeblock[i] = 0u;

    const uint8_t ctr_high = (uint8_t)((loop_pos >> 16) & 0xFFu);
    uint32_t p = 0u;
    for (uint32_t rep = 0; rep < 64u; ++rep) {
        for (uint32_t bi = 0; bi < pw_salt_len; ++bi, ++p) {
            uint8_t b = (uint8_t)(wg[bi >> 2] >> (24u - (bi & 3u) * 8u));
            ((uint8_t*)largeblock)[p ^ 3u] = b;
        }
        ((uint8_t*)largeblock)[(p + 2u) ^ 3u] = ctr_high;
        p += 3u;
    }

    // ── Shared zero block for sha1_sample_byte ─────────────────────────────
    const uint32_t zero4[4] = {0u, 0u, 0u, 0u};

    // ── Main loop: 256 groups × p3 SHA-1 transforms ───────────────────────
    uint32_t iter = loop_pos;

    for (uint32_t outer = 0u; outer < 256u; ++outer) {
        uint32_t carry = 0u;
        uint32_t k = pw_salt_len;  // counter byte-offset within current block

        for (uint32_t j = 0u; j < p3; ++j) {
            const uint32_t j16 = j << 4u;

            uint32_t w0[4], w1[4], w2[4], w3[4];
            uint32_t wex = 0u;

            w0[0] = largeblock[j16 +  0] | carry;
            w0[1] = largeblock[j16 +  1];
            w0[2] = largeblock[j16 +  2];
            w0[3] = largeblock[j16 +  3];
            w1[0] = largeblock[j16 +  4];
            w1[1] = largeblock[j16 +  5];
            w1[2] = largeblock[j16 +  6];
            w1[3] = largeblock[j16 +  7];
            w2[0] = largeblock[j16 +  8];
            w2[1] = largeblock[j16 +  9];
            w2[2] = largeblock[j16 + 10];
            w2[3] = largeblock[j16 + 11];
            w3[0] = largeblock[j16 + 12];
            w3[1] = largeblock[j16 + 13];
            w3[2] = largeblock[j16 + 14];
            w3[3] = largeblock[j16 + 15];

            // Patch all counters whose byte-offset k falls in this block
            while (k < 64u) {
                const uint32_t iter_s = hc_swap32_S(iter);
                const uint32_t kd = k >> 2u;
                const uint32_t km = k & 3u;

                uint32_t m0, m1, t0, t1;
                if      (km == 0u) { t0 = iter_s;         t1 = 0u;           m0 = 0x0000ffffu; m1 = 0xffffffffu; }
                else if (km == 1u) { t0 = iter_s >>  8u;  t1 = 0u;           m0 = 0xff0000ffu; m1 = 0xffffffffu; }
                else if (km == 2u) { t0 = iter_s >> 16u;  t1 = 0u;           m0 = 0xffff0000u; m1 = 0xffffffffu; }
                else               { t0 = iter_s >> 24u;  t1 = iter_s << 8u; m0 = 0xffffff00u; m1 = 0x00ffffffu; }

                switch (kd) {
                    case  0: w0[0]=(w0[0]&m0)|t0; w0[1]=(w0[1]&m1)|t1; break;
                    case  1: w0[1]=(w0[1]&m0)|t0; w0[2]=(w0[2]&m1)|t1; break;
                    case  2: w0[2]=(w0[2]&m0)|t0; w0[3]=(w0[3]&m1)|t1; break;
                    case  3: w0[3]=(w0[3]&m0)|t0; w1[0]=(w1[0]&m1)|t1; break;
                    case  4: w1[0]=(w1[0]&m0)|t0; w1[1]=(w1[1]&m1)|t1; break;
                    case  5: w1[1]=(w1[1]&m0)|t0; w1[2]=(w1[2]&m1)|t1; break;
                    case  6: w1[2]=(w1[2]&m0)|t0; w1[3]=(w1[3]&m1)|t1; break;
                    case  7: w1[3]=(w1[3]&m0)|t0; w2[0]=(w2[0]&m1)|t1; break;
                    case  8: w2[0]=(w2[0]&m0)|t0; w2[1]=(w2[1]&m1)|t1; break;
                    case  9: w2[1]=(w2[1]&m0)|t0; w2[2]=(w2[2]&m1)|t1; break;
                    case 10: w2[2]=(w2[2]&m0)|t0; w2[3]=(w2[3]&m1)|t1; break;
                    case 11: w2[3]=(w2[3]&m0)|t0; w3[0]=(w3[0]&m1)|t1; break;
                    case 12: w3[0]=(w3[0]&m0)|t0; w3[1]=(w3[1]&m1)|t1; break;
                    case 13: w3[1]=(w3[1]&m0)|t0; w3[2]=(w3[2]&m1)|t1; break;
                    case 14: w3[2]=(w3[2]&m0)|t0; w3[3]=(w3[3]&m1)|t1; break;
                    case 15: w3[3]=(w3[3]&m0)|t0; wex  =(wex   &m1)|t1; break;
                }

                ++iter;
                k += p3;
            }

            sha1_transform_hc(w0, w1, w2, w3, h);

            k &= 63u;
            carry = wex;
        }

        // IV sample at batch midpoint (after outer 127 = iters 0..8191 of batch)
        if (outer == 127u && iv_idx < 16u) {
            const uint32_t ctx = (loop_pos + 8192u) * p3;
            tmps[gid].iv[iv_idx++] = sha1_sample_byte(h, zero4, zero4, zero4, zero4, ctx);
        }
    }

    // Key sample at batch end (after all 16384 iters)
    if (key_idx < 16u) {
        const uint32_t ctx = (loop_pos + 16384u) * p3;
        tmps[gid].key[key_idx++] = sha1_sample_byte(h, zero4, zero4, zero4, zero4, ctx);
    }

    // Persist state — context is at a clean block boundary (w0..w3 = 0)
    tmps[gid].h[0]=h[0]; tmps[gid].h[1]=h[1]; tmps[gid].h[2]=h[2];
    tmps[gid].h[3]=h[3]; tmps[gid].h[4]=h[4];
    for (int ki = 0; ki < 4; ++ki) {
        tmps[gid].w0[ki]=0u; tmps[gid].w1[ki]=0u;
        tmps[gid].w2[ki]=0u; tmps[gid].w3[ki]=0u;
    }
    tmps[gid].ctx_len = (loop_pos + 16384u) * p3;
    tmps[gid].key_idx = key_idx;
    tmps[gid].iv_idx  = iv_idx;
}

extern "C"
__launch_bounds__(256, 4)
__global__ void rar3_comp(
    const rar3_tmps * __restrict__ tmps,
    int32_t         num_passwords,
    const uint8_t * __restrict__ enc_check,
    int32_t         check_mode,
    int32_t         head_type,
    uint32_t        file_crc,
    int32_t         pack_size,
    int32_t       * __restrict__ result)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_passwords) return;

    uint8_t key[16], iv[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) { key[i] = tmps[gid].key[i]; iv[i] = tmps[gid].iv[i]; }

    aes_ctx_t aes;
    aes128_key_expand(&aes, key);

    uint8_t block[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) block[i] = enc_check[i];
    aes_cbc_decrypt(&aes, block, 1, iv);

    bool match = false;
    if (check_mode == 0) {
        uint8_t b2 = block[2];
        match = (b2 == 0x72 || b2 == 0x73 || b2 == 0x74 ||
                 b2 == 0x75 || b2 == 0x76 || b2 == 0x7a || b2 == 0x7b);
    } else if (check_mode == 1) {
        uint32_t computed = rar3_crc32(block, pack_size);
        match = (computed == file_crc);
    } else {
        uint8_t b0 = block[0];
        match = (b0 >= 0x72 && b0 <= 0x7b);
    }

    if (match) atomicCAS(result, NO_MATCH, gid);
}

// Debug kernel: dump key[16]+iv[16] per thread for parity testing vs CPU.
// d_out layout: tid * 32 → [key[0..16], iv[0..16]]
extern "C"
__global__ void rar3_kdf_dump(
    const uint8_t * __restrict__ passwords_utf16,
    const int32_t * __restrict__ pw_lengths_u16,
    int32_t         num_passwords,
    const uint8_t * __restrict__ salt,
    uint8_t       * __restrict__ d_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_passwords) return;

    const uint8_t *pw = passwords_utf16 + (size_t)tid * MAX_PW_BYTES;
    uint32_t pwlen    = (uint32_t)pw_lengths_u16[tid];

    uint8_t key[16], iv[16];
    rar3_kdf(pw, pwlen, salt, key, iv);

    uint8_t *out = d_out + (size_t)tid * 32;
    #pragma unroll
    for (int i = 0; i < 16; ++i) out[i]      = key[i];
    #pragma unroll
    for (int i = 0; i < 16; ++i) out[16 + i] = iv[i];
}
