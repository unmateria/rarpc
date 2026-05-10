#pragma once
#include "common.cuh"

// ────────────────────────────────────────────────────────────
//  SHA-256 device implementation (for RAR5 PBKDF2-HMAC-SHA256)
// ────────────────────────────────────────────────────────────

// SHA-256 Ch / Maj / 3-way XOR via LOP3.LUT (1 instruction each on SM_70+).
//
//   Ch(e,f,g)  = (e & f) ^ (~e & g)          truth table = 0xCA
//   Maj(a,b,c) = (a & b) ^ (a & c) ^ (b & c) truth table = 0xE8
//   XOR3(a,b,c) = a ^ b ^ c                   truth table = 0x96
//
// Verified tables (bit i = function value for the i-th (A,B,C) triple):
//   Ch : 11001010 = 0xCA
//   Maj: majority of 3 → 11101000 = 0xE8
//   XOR3: parity     → 10010110 = 0x96
__device__ __forceinline__ uint32_t sha256_ch_lop3(uint32_t e, uint32_t f, uint32_t g) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(r) : "r"(e), "r"(f), "r"(g));
    return r;
}

__device__ __forceinline__ uint32_t sha256_maj_lop3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

// 3-way XOR folded into a single LOP3. Used for S0/S1/s0/s1 in SHA-256.
__device__ __forceinline__ uint32_t lop3_xor3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

__device__ __constant__ uint32_t SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

typedef struct {
    uint32_t h[8];
    uint64_t total;     // bits
    uint8_t  buf[64];
    uint32_t buflen;
} sha256_ctx_t;

__device__ __forceinline__
void sha256_init(sha256_ctx_t *ctx) {
    ctx->h[0] = 0x6a09e667u;
    ctx->h[1] = 0xbb67ae85u;
    ctx->h[2] = 0x3c6ef372u;
    ctx->h[3] = 0xa54ff53au;
    ctx->h[4] = 0x510e527fu;
    ctx->h[5] = 0x9b05688cu;
    ctx->h[6] = 0x1f83d9abu;
    ctx->h[7] = 0x5be0cd19u;
    ctx->total  = 0;
    ctx->buflen = 0;
}

__device__ static void sha256_compress(uint32_t h[8], const uint8_t block[64]) {
    uint32_t w[64];
    #pragma unroll 16
    for (int i = 0; i < 16; i++) w[i] = load_be32(block + i * 4);
    #pragma unroll 48
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROTR32(w[i-15], 7) ^ ROTR32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = ROTR32(w[i-2], 17) ^ ROTR32(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a=h[0], b=h[1], c=h[2], d=h[3],
             e=h[4], f=h[5], g=h[6], hh=h[7];

    #pragma unroll 64
    for (int i = 0; i < 64; i++) {
        uint32_t S1  = ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25);
        uint32_t ch  = sha256_ch_lop3(e, f, g);
        uint32_t tmp1 = hh + S1 + ch + SHA256_K[i] + w[i];
        uint32_t S0  = ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22);
        uint32_t maj = sha256_maj_lop3(a, b, c);
        uint32_t tmp2 = S0 + maj;
        hh = g; g = f; f = e; e = d + tmp1;
        d  = c; c = b; b = a; a = tmp1 + tmp2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;
    h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}

__device__ __forceinline__
void sha256_update(sha256_ctx_t *ctx, const uint8_t *data, uint32_t len) {
    ctx->total += (uint64_t)len * 8;
    uint32_t left = ctx->buflen;
    uint32_t fill = 64 - left;

    if (left && len >= fill) {
        for (uint32_t i = 0; i < fill; i++) ctx->buf[left+i] = data[i];
        sha256_compress(ctx->h, ctx->buf);
        data += fill; len -= fill; left = 0; ctx->buflen = 0;
    }
    while (len >= 64) {
        sha256_compress(ctx->h, data);
        data += 64; len -= 64;
    }
    for (uint32_t i = 0; i < len; i++) ctx->buf[left+i] = data[i];
    ctx->buflen = left + len;
}

__device__ __forceinline__
void sha256_final(sha256_ctx_t *ctx, uint8_t digest[32]) {
    uint64_t bits = ctx->total;
    uint32_t last = ctx->buflen;
    uint32_t padn = (last < 56) ? (56 - last) : (120 - last);

    uint8_t pad[64];
    pad[0] = 0x80;
    for (uint32_t i = 1; i < padn; i++) pad[i] = 0;
    pad[padn]   = (bits >> 56) & 0xff;
    pad[padn+1] = (bits >> 48) & 0xff;
    pad[padn+2] = (bits >> 40) & 0xff;
    pad[padn+3] = (bits >> 32) & 0xff;
    pad[padn+4] = (bits >> 24) & 0xff;
    pad[padn+5] = (bits >> 16) & 0xff;
    pad[padn+6] = (bits >>  8) & 0xff;
    pad[padn+7] =  bits        & 0xff;

    sha256_update(ctx, pad, padn + 8);

    for (int i = 0; i < 8; i++) store_be32(digest + i*4, ctx->h[i]);
}

// ── HMAC-SHA256 ──────────────────────────────────────────────

typedef struct {
    sha256_ctx_t inner;
    sha256_ctx_t outer;
} hmac_sha256_ctx_t;

__device__ __forceinline__
void hmac_sha256_init(hmac_sha256_ctx_t *ctx,
                      const uint8_t *key, uint32_t keylen)
{
    uint8_t k[64];
    // If key > 64 bytes, hash it first
    if (keylen > 64) {
        sha256_ctx_t tmp;
        sha256_init(&tmp);
        sha256_update(&tmp, key, keylen);
        sha256_final(&tmp, k);
        for (int i = 32; i < 64; i++) k[i] = 0;
    } else {
        for (uint32_t i = 0; i < keylen; i++) k[i] = key[i];
        for (uint32_t i = keylen; i < 64; i++) k[i] = 0;
    }

    uint8_t ipad[64], opad[64];
    for (int i = 0; i < 64; i++) {
        ipad[i] = k[i] ^ 0x36;
        opad[i] = k[i] ^ 0x5c;
    }

    sha256_init(&ctx->inner);
    sha256_update(&ctx->inner, ipad, 64);

    sha256_init(&ctx->outer);
    sha256_update(&ctx->outer, opad, 64);
}

__device__ __forceinline__
void hmac_sha256_update(hmac_sha256_ctx_t *ctx,
                        const uint8_t *data, uint32_t len)
{
    sha256_update(&ctx->inner, data, len);
}

__device__ __forceinline__
void hmac_sha256_final(hmac_sha256_ctx_t *ctx, uint8_t mac[32])
{
    uint8_t inner_hash[32];
    sha256_final(&ctx->inner, inner_hash);
    sha256_update(&ctx->outer, inner_hash, 32);
    sha256_final(&ctx->outer, mac);
}

// Convenience: single-call HMAC-SHA256
__device__ __forceinline__
void hmac_sha256(const uint8_t *key, uint32_t keylen,
                 const uint8_t *data, uint32_t datalen,
                 uint8_t mac[32])
{
    hmac_sha256_ctx_t ctx;
    hmac_sha256_init(&ctx, key, keylen);
    hmac_sha256_update(&ctx, data, datalen);
    hmac_sha256_final(&ctx, mac);
}

// ── SHA256 word-domain compression ─────────────────────────
// Like sha256_compress but input and state are uint32[8/16] throughout.
// Avoids byte↔uint32 conversions in the hot PBKDF2 loop.
//
// __noinline__ keeps this in its own register frame (~29 regs: w[16]+abcdefgh+temps)
// instead of inflating the PBKDF2 hot-loop frame. Higher occupancy → better
// warp-level latency hiding → higher throughput.
//
// Kept as the by-pointer variant for callers that already hold state in local
// memory (sha256_update byte path). The hot PBKDF2 loop uses the by-value
// variant below (sha256_compress_bv) to keep h[] in registers.
__device__ __noinline__
void sha256_compress_words(uint32_t h[8], const uint32_t blk[16])
{
    uint32_t w[16];
    #pragma unroll 16
    for (int i = 0; i < 16; i++) w[i] = blk[i];

    uint32_t a=h[0], b=h[1], c=h[2], d=h[3],
             e=h[4], f=h[5], g=h[6], hh=h[7];

    // Full unroll: every i is a compile-time constant, so (i+1)&15 etc.
    // are also constants → w[] stays in registers, not local memory.
    #pragma unroll 64
    for (int i = 0; i < 64; i++) {
        uint32_t wi = w[i & 15];
        if (i >= 16) {
            // w[i-15] → slot (i+1)&15, w[i-2] → (i+14)&15,
            // w[i-7]  → (i+9)&15,     w[i-16] → i&15 (overwritten)
            uint32_t s0 = lop3_xor3(ROTR32(w[(i+ 1)&15],  7), ROTR32(w[(i+ 1)&15], 18), w[(i+ 1)&15] >>  3);
            uint32_t s1 = lop3_xor3(ROTR32(w[(i+14)&15], 17), ROTR32(w[(i+14)&15], 19), w[(i+14)&15] >> 10);
            wi = w[i&15] + s0 + w[(i+9)&15] + s1;
            w[i & 15] = wi;
        }
        uint32_t S1   = lop3_xor3(ROTR32(e,  6), ROTR32(e, 11), ROTR32(e, 25));
        uint32_t ch   = sha256_ch_lop3(e, f, g);
        uint32_t tmp1 = hh + S1 + ch + SHA256_K[i] + wi;
        uint32_t S0   = lop3_xor3(ROTR32(a,  2), ROTR32(a, 13), ROTR32(a, 22));
        uint32_t maj  = sha256_maj_lop3(a, b, c);
        uint32_t tmp2 = S0 + maj;
        hh=g; g=f; f=e; e=d+tmp1; d=c; c=b; b=a; a=tmp1+tmp2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;
    h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}

// ── By-value SHA-256 compression ────────────────────────────
//
// Critical optimisation for the PBKDF2 hot loop.
//
// Before: sha256_compress_words(uint32_t h[8], const uint32_t blk[16])
//   NVCC can't prove h/blk don't escape → promotes them to local memory.
//   The PTX shows `cvta.to.local.u64` + `ld.local.u32` — h[8] lives in
//   local memory at ~32 bytes/thread, hit on every one of the ~65 600
//   calls per password. Identical pattern to the RAR3 sha1_ctx_t bug.
//
// After: pass hin by-value (sha256_h8_t with named fields), return by-value.
//   NVCC allocates h0..h7 in registers in the caller; no local traffic for
//   state between calls. blk[] still goes via pointer (caller's local array)
//   but that's a single 64-byte chunk reused across iterations — much less
//   costly than the per-call state round-trip.
typedef struct { uint32_t h0,h1,h2,h3,h4,h5,h6,h7; } sha256_h8_t;

// 16-word block carried through parameters as named fields so NVCC keeps it
// entirely in registers on both sides of the call boundary (no local traffic).
typedef struct {
    uint32_t w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15;
} sha256_blk_t;

__device__ __forceinline__ sha256_h8_t sha256_iv_bv() {
    sha256_h8_t iv = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    return iv;
}

__device__ __forceinline__
sha256_h8_t sha256_compress_bv(sha256_h8_t hin, sha256_blk_t bin)
{
    // Copy the 16 input words into a circular buffer. Full unroll below
    // makes every `(i+k)&15` index a compile-time constant → w[] stays in
    // registers (no local memory).
    uint32_t w[16] = {
        bin.w0,  bin.w1,  bin.w2,  bin.w3,
        bin.w4,  bin.w5,  bin.w6,  bin.w7,
        bin.w8,  bin.w9,  bin.w10, bin.w11,
        bin.w12, bin.w13, bin.w14, bin.w15,
    };

    uint32_t a=hin.h0, b=hin.h1, c=hin.h2, d=hin.h3,
             e=hin.h4, f=hin.h5, g=hin.h6, hh=hin.h7;

    #pragma unroll 64
    for (int i = 0; i < 64; i++) {
        uint32_t wi = w[i & 15];
        if (i >= 16) {
            uint32_t s0 = lop3_xor3(ROTR32(w[(i+ 1)&15],  7), ROTR32(w[(i+ 1)&15], 18), w[(i+ 1)&15] >>  3);
            uint32_t s1 = lop3_xor3(ROTR32(w[(i+14)&15], 17), ROTR32(w[(i+14)&15], 19), w[(i+14)&15] >> 10);
            wi = w[i&15] + s0 + w[(i+9)&15] + s1;
            w[i & 15] = wi;
        }
        uint32_t S1   = lop3_xor3(ROTR32(e,  6), ROTR32(e, 11), ROTR32(e, 25));
        uint32_t ch   = sha256_ch_lop3(e, f, g);
        uint32_t tmp1 = hh + S1 + ch + SHA256_K[i] + wi;
        uint32_t S0   = lop3_xor3(ROTR32(a,  2), ROTR32(a, 13), ROTR32(a, 22));
        uint32_t maj  = sha256_maj_lop3(a, b, c);
        uint32_t tmp2 = S0 + maj;
        hh=g; g=f; f=e; e=d+tmp1; d=c; c=b; b=a; a=tmp1+tmp2;
    }
    sha256_h8_t hout = {
        hin.h0+a, hin.h1+b, hin.h2+c, hin.h3+d,
        hin.h4+e, hin.h5+f, hin.h6+g, hin.h7+hh,
    };
    return hout;
}

// ── PBKDF2-HMAC-SHA256 (RAR5-optimised) ──────────────────────
//
// Key insight (from unrar source analysis):
//   • For RAR5: dklen=32, only PBKDF2 block T_1 is ever needed.
//   • All U2..U_n use a 32-byte message — this fixes the padding.
//   • Fixed padding for a 32-byte message in a 64-byte SHA256 block:
//       words 8..14 = 0x00000000, word 15 = 0x00000300  (len = 768 bits)
//   • Precompute HMAC midstates as uint32[8] (not full ctx = 108 bytes).
//   • Inner loop copies only 2×32 = 64 bytes instead of 216 bytes.
//
// Outputs dk[32] in uint32[8] format (big-endian words).
//
// Implementation note: the hot loop (U2..U_n) exclusively uses the by-value
// compress (sha256_compress_bv) so that H_inner, H_outer, u_w, dk_w live in
// named registers instead of local memory. The only local array that
// remains is blk[16] (64 bytes), which is the single 64-byte message buffer
// shared by inner and outer halves of each iteration.
__device__ __forceinline__
void pbkdf2_sha256_rar5_u32(
    const uint8_t *pw, uint32_t pwlen,
    const uint8_t *salt, uint32_t saltlen,
    uint32_t iters,
    uint32_t dk_w[8])          // output: 32 bytes as uint32[8] BE words
{
    // ── 1. Build HMAC key (pad/hash to 64 bytes) ─────────────
    uint8_t k[64] = {0};
    if (pwlen > 64) {
        sha256_ctx_t tmp; sha256_init(&tmp);
        sha256_update(&tmp, pw, pwlen);
        sha256_final(&tmp, k);
    } else {
        for (uint32_t i = 0; i < pwlen; i++) k[i] = pw[i];
    }

    // ── 2. Compute ipad/opad midstates as sha256_h8_t (in registers) ──
    // ipad = k XOR 0x36, opad = k XOR 0x5c
    sha256_blk_t iblk, oblk;
    {
        uint32_t k0  = load_be32(k +  0), k1  = load_be32(k +  4);
        uint32_t k2  = load_be32(k +  8), k3  = load_be32(k + 12);
        uint32_t k4  = load_be32(k + 16), k5  = load_be32(k + 20);
        uint32_t k6  = load_be32(k + 24), k7  = load_be32(k + 28);
        uint32_t k8  = load_be32(k + 32), k9  = load_be32(k + 36);
        uint32_t k10 = load_be32(k + 40), k11 = load_be32(k + 44);
        uint32_t k12 = load_be32(k + 48), k13 = load_be32(k + 52);
        uint32_t k14 = load_be32(k + 56), k15 = load_be32(k + 60);
        const uint32_t IPAD = 0x36363636u, OPAD = 0x5c5c5c5cu;
        iblk = {k0^IPAD,  k1^IPAD,  k2^IPAD,  k3^IPAD,
                k4^IPAD,  k5^IPAD,  k6^IPAD,  k7^IPAD,
                k8^IPAD,  k9^IPAD,  k10^IPAD, k11^IPAD,
                k12^IPAD, k13^IPAD, k14^IPAD, k15^IPAD};
        oblk = {k0^OPAD,  k1^OPAD,  k2^OPAD,  k3^OPAD,
                k4^OPAD,  k5^OPAD,  k6^OPAD,  k7^OPAD,
                k8^OPAD,  k9^OPAD,  k10^OPAD, k11^OPAD,
                k12^OPAD, k13^OPAD, k14^OPAD, k15^OPAD};
    }

    sha256_h8_t H_inner = sha256_compress_bv(sha256_iv_bv(), iblk);
    sha256_h8_t H_outer = sha256_compress_bv(sha256_iv_bv(), oblk);

    // ── 3. U1 = HMAC(pw, salt || 0x00000001) ────────────────
    // Inner message: salt[16] || 0x00000001[4] = 20 bytes
    // Block: 20 bytes data + 36 bytes padding = 64 bytes
    sha256_h8_t u;
    {
        uint8_t msg[64] = {0};
        for (uint32_t i = 0; i < saltlen; i++) msg[i] = salt[i];
        msg[saltlen]   = 0; msg[saltlen+1] = 0;
        msg[saltlen+2] = 0; msg[saltlen+3] = 1; // INT(1) big-endian
        uint32_t msglen = saltlen + 4;
        msg[msglen] = 0x80;
        uint64_t bits = (uint64_t)(64 + msglen) * 8;
        msg[56]=(bits>>56)&0xff; msg[57]=(bits>>48)&0xff;
        msg[58]=(bits>>40)&0xff; msg[59]=(bits>>32)&0xff;
        msg[60]=(bits>>24)&0xff; msg[61]=(bits>>16)&0xff;
        msg[62]=(bits>> 8)&0xff; msg[63]= bits     &0xff;

        sha256_blk_t blk_u1 = {
            load_be32(msg+ 0), load_be32(msg+ 4), load_be32(msg+ 8), load_be32(msg+12),
            load_be32(msg+16), load_be32(msg+20), load_be32(msg+24), load_be32(msg+28),
            load_be32(msg+32), load_be32(msg+36), load_be32(msg+40), load_be32(msg+44),
            load_be32(msg+48), load_be32(msg+52), load_be32(msg+56), load_be32(msg+60),
        };

        // inner result
        sha256_h8_t hi = sha256_compress_bv(H_inner, blk_u1);

        // outer block: hi words || padding for 32-byte message (len=(64+32)*8=768=0x300)
        sha256_blk_t oblk2 = {
            hi.h0, hi.h1, hi.h2, hi.h3,
            hi.h4, hi.h5, hi.h6, hi.h7,
            0x80000000u, 0u, 0u, 0u,
            0u, 0u, 0u, 0x00000300u,
        };

        u = sha256_compress_bv(H_outer, oblk2);
    }

    // T starts as U1 (XOR accumulator below updates it)
    sha256_h8_t t = u;

    // ── 4. U2..U_n: all use 32-byte messages ─────────────────
    // blk is built fresh per call — words 8..15 are the invariant padding
    // for a 32-byte message (|message|+|block| = 96 bytes = 768 bits = 0x300).
    for (uint32_t iter = 1; iter < iters; iter++) {
        // ── inner: SHA256(H_inner_state || u || pad) ────────
        sha256_blk_t blk_in = {
            u.h0, u.h1, u.h2, u.h3,
            u.h4, u.h5, u.h6, u.h7,
            0x80000000u, 0u, 0u, 0u,
            0u, 0u, 0u, 0x00000300u,
        };
        sha256_h8_t hs = sha256_compress_bv(H_inner, blk_in);

        // ── outer: SHA256(H_outer_state || hs || pad) ───────
        sha256_blk_t blk_out = {
            hs.h0, hs.h1, hs.h2, hs.h3,
            hs.h4, hs.h5, hs.h6, hs.h7,
            0x80000000u, 0u, 0u, 0u,
            0u, 0u, 0u, 0x00000300u,
        };
        u = sha256_compress_bv(H_outer, blk_out);

        // T ^= U_{iter+1}
        t.h0 ^= u.h0; t.h1 ^= u.h1; t.h2 ^= u.h2; t.h3 ^= u.h3;
        t.h4 ^= u.h4; t.h5 ^= u.h5; t.h6 ^= u.h6; t.h7 ^= u.h7;
    }

    dk_w[0]=t.h0; dk_w[1]=t.h1; dk_w[2]=t.h2; dk_w[3]=t.h3;
    dk_w[4]=t.h4; dk_w[5]=t.h5; dk_w[6]=t.h6; dk_w[7]=t.h7;
}

// ── PBKDF2-HMAC-SHA256 pair (ILP=2 across 2 candidates) ───────
//
// Each call processes TWO INDEPENDENT PBKDF2-SHA256 derivations in lockstep.
//
// Tested on RTX 4060 Ti SM_89 (2026-04-14): NO measurable speedup vs the
// single-candidate path. Kept as documentation for future experiments — the
// kernel is compute-bound at ~67 KH/s on iter_count=15. With 32 warps/SM
// × 4 scheduler units, the pipeline is already saturated; doubling the
// independent work per thread doesn't help when there's no stall to fill.
//
// Retain for reference / potential use on architectures with fewer warps
// per SM. Current kernel (rar5_kdf.cu) uses the single-candidate path.
__device__ __forceinline__
void pbkdf2_sha256_rar5_u32_pair(
    const uint8_t *pw_a, uint32_t pwlen_a,
    const uint8_t *pw_b, uint32_t pwlen_b,
    const uint8_t *salt, uint32_t saltlen,
    uint32_t iters,
    uint32_t dk_a[8], uint32_t dk_b[8])
{
    // ── Init for A and B (sequential — small cost vs 32 768-iter hot loop) ──
    uint8_t ka[64] = {0}, kb[64] = {0};
    if (pwlen_a > 64) {
        sha256_ctx_t tmp; sha256_init(&tmp);
        sha256_update(&tmp, pw_a, pwlen_a); sha256_final(&tmp, ka);
    } else {
        for (uint32_t i = 0; i < pwlen_a; i++) ka[i] = pw_a[i];
    }
    if (pwlen_b > 64) {
        sha256_ctx_t tmp; sha256_init(&tmp);
        sha256_update(&tmp, pw_b, pwlen_b); sha256_final(&tmp, kb);
    } else {
        for (uint32_t i = 0; i < pwlen_b; i++) kb[i] = pw_b[i];
    }

    // Build ipad/opad blocks for both candidates.
    const uint32_t IPAD = 0x36363636u, OPAD = 0x5c5c5c5cu;
    sha256_blk_t iblk_a, oblk_a, iblk_b, oblk_b;
    {
        #define BUILD_BLK(PREFIX, K) do { \
            uint32_t w0=load_be32(K+0),  w1=load_be32(K+4),  w2=load_be32(K+8),  w3=load_be32(K+12); \
            uint32_t w4=load_be32(K+16), w5=load_be32(K+20), w6=load_be32(K+24), w7=load_be32(K+28); \
            uint32_t w8=load_be32(K+32), w9=load_be32(K+36), wA=load_be32(K+40), wB=load_be32(K+44); \
            uint32_t wC=load_be32(K+48), wD=load_be32(K+52), wE=load_be32(K+56), wF=load_be32(K+60); \
            iblk_##PREFIX = {w0^IPAD,w1^IPAD,w2^IPAD,w3^IPAD,w4^IPAD,w5^IPAD,w6^IPAD,w7^IPAD, \
                             w8^IPAD,w9^IPAD,wA^IPAD,wB^IPAD,wC^IPAD,wD^IPAD,wE^IPAD,wF^IPAD}; \
            oblk_##PREFIX = {w0^OPAD,w1^OPAD,w2^OPAD,w3^OPAD,w4^OPAD,w5^OPAD,w6^OPAD,w7^OPAD, \
                             w8^OPAD,w9^OPAD,wA^OPAD,wB^OPAD,wC^OPAD,wD^OPAD,wE^OPAD,wF^OPAD}; \
        } while (0)
        BUILD_BLK(a, ka);
        BUILD_BLK(b, kb);
        #undef BUILD_BLK
    }

    sha256_h8_t H_inner_a = sha256_compress_bv(sha256_iv_bv(), iblk_a);
    sha256_h8_t H_inner_b = sha256_compress_bv(sha256_iv_bv(), iblk_b);
    sha256_h8_t H_outer_a = sha256_compress_bv(sha256_iv_bv(), oblk_a);
    sha256_h8_t H_outer_b = sha256_compress_bv(sha256_iv_bv(), oblk_b);

    // U1 for A and B: salt || INT(1) → 20 bytes, padded.
    // Both use the SAME salt → build the block once, use twice.
    sha256_blk_t blk_u1;
    {
        uint8_t msg[64] = {0};
        for (uint32_t i = 0; i < saltlen; i++) msg[i] = salt[i];
        msg[saltlen+3] = 1; // INT(1) big-endian
        uint32_t msglen = saltlen + 4;
        msg[msglen] = 0x80;
        uint64_t bits = (uint64_t)(64 + msglen) * 8;
        msg[56]=(bits>>56)&0xff; msg[57]=(bits>>48)&0xff;
        msg[58]=(bits>>40)&0xff; msg[59]=(bits>>32)&0xff;
        msg[60]=(bits>>24)&0xff; msg[61]=(bits>>16)&0xff;
        msg[62]=(bits>> 8)&0xff; msg[63]= bits     &0xff;
        blk_u1 = {
            load_be32(msg+ 0), load_be32(msg+ 4), load_be32(msg+ 8), load_be32(msg+12),
            load_be32(msg+16), load_be32(msg+20), load_be32(msg+24), load_be32(msg+28),
            load_be32(msg+32), load_be32(msg+36), load_be32(msg+40), load_be32(msg+44),
            load_be32(msg+48), load_be32(msg+52), load_be32(msg+56), load_be32(msg+60),
        };
    }
    sha256_h8_t hi_a = sha256_compress_bv(H_inner_a, blk_u1);
    sha256_h8_t hi_b = sha256_compress_bv(H_inner_b, blk_u1);

    #define HI_TO_OUTERBLK(HI) sha256_blk_t{ \
        (HI).h0, (HI).h1, (HI).h2, (HI).h3, \
        (HI).h4, (HI).h5, (HI).h6, (HI).h7, \
        0x80000000u, 0u, 0u, 0u, 0u, 0u, 0u, 0x00000300u, \
    }
    sha256_h8_t u_a = sha256_compress_bv(H_outer_a, HI_TO_OUTERBLK(hi_a));
    sha256_h8_t u_b = sha256_compress_bv(H_outer_b, HI_TO_OUTERBLK(hi_b));

    sha256_h8_t t_a = u_a;
    sha256_h8_t t_b = u_b;

    // ── Hot loop: A and B interleaved ────────────────────────────
    // Writing the two streams side-by-side lets the compiler / scheduler
    // overlap dep chains — A's wait cycles fill with B's independent work.
    for (uint32_t iter = 1; iter < iters; iter++) {
        sha256_blk_t blk_in_a = HI_TO_OUTERBLK(u_a);
        sha256_blk_t blk_in_b = HI_TO_OUTERBLK(u_b);

        sha256_h8_t hs_a = sha256_compress_bv(H_inner_a, blk_in_a);
        sha256_h8_t hs_b = sha256_compress_bv(H_inner_b, blk_in_b);

        sha256_blk_t blk_ou_a = HI_TO_OUTERBLK(hs_a);
        sha256_blk_t blk_ou_b = HI_TO_OUTERBLK(hs_b);

        u_a = sha256_compress_bv(H_outer_a, blk_ou_a);
        u_b = sha256_compress_bv(H_outer_b, blk_ou_b);

        t_a.h0 ^= u_a.h0; t_a.h1 ^= u_a.h1; t_a.h2 ^= u_a.h2; t_a.h3 ^= u_a.h3;
        t_a.h4 ^= u_a.h4; t_a.h5 ^= u_a.h5; t_a.h6 ^= u_a.h6; t_a.h7 ^= u_a.h7;
        t_b.h0 ^= u_b.h0; t_b.h1 ^= u_b.h1; t_b.h2 ^= u_b.h2; t_b.h3 ^= u_b.h3;
        t_b.h4 ^= u_b.h4; t_b.h5 ^= u_b.h5; t_b.h6 ^= u_b.h6; t_b.h7 ^= u_b.h7;
    }
    #undef HI_TO_OUTERBLK

    dk_a[0]=t_a.h0; dk_a[1]=t_a.h1; dk_a[2]=t_a.h2; dk_a[3]=t_a.h3;
    dk_a[4]=t_a.h4; dk_a[5]=t_a.h5; dk_a[6]=t_a.h6; dk_a[7]=t_a.h7;
    dk_b[0]=t_b.h0; dk_b[1]=t_b.h1; dk_b[2]=t_b.h2; dk_b[3]=t_b.h3;
    dk_b[4]=t_b.h4; dk_b[5]=t_b.h5; dk_b[6]=t_b.h6; dk_b[7]=t_b.h7;
}

// ── PBKDF2-HMAC-SHA256 (generic, byte-output) ────────────────
// Wrapper around the optimised word-domain function.
// dklen must be exactly 32 for RAR5.
__device__ __forceinline__
void pbkdf2_sha256(const uint8_t *pw, uint32_t pwlen,
                   const uint8_t *salt, uint32_t saltlen,
                   uint32_t iters,
                   uint8_t *dk, uint32_t dklen)
{
    uint32_t dk_w[8];
    pbkdf2_sha256_rar5_u32(pw, pwlen, salt, saltlen, iters, dk_w);
    // Convert uint32[8] big-endian → bytes
    for (int i = 0; i < 8; i++) store_be32(dk + i*4, dk_w[i]);
}
