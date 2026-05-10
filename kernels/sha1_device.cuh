#pragma once
#include "common.cuh"

// ────────────────────────────────────────────────────────────
//  SHA-1 device implementation (for RAR3 KDF)
//
//  Key optimisations vs. naïve implementation:
//    • ctx.blk stored as uint32[16] big-endian (was uint8[64]).
//      Word-level writes in sha1_update cut local-memory ops by ~3×
//      for aligned data (password bytes are stride-256 aligned, salt is
//      short, counter is 3 bytes—handled by byte fallback).
//    • sha1_compress takes uint32[16] directly → no load_be32 on entry.
//    • sha1_compress is __noinline__: own register frame (~25 regs)
//      keeps the caller frame small → higher SM occupancy → better
//      latency hiding across the 262 144-iteration hot loop.
//    • w[16] circular buffer + #pragma unroll 80 → all modular indices
//      resolved at compile time → w[] stays in registers, not local mem.
// ────────────────────────────────────────────────────────────

typedef struct {
    uint32_t h[5];    // running hash state
    uint64_t total;   // bits processed so far
    uint32_t blk[16]; // current partial block as big-endian uint32[16]
    uint32_t boff;    // byte offset within blk (0–63)
} sha1_ctx_t;

// ── SHA-1 compression ────────────────────────────────────────
// Takes the block already in big-endian uint32 format (no conversion needed).
// SHA-1 schedule: w[i] = ROTL32(w[i-3]^w[i-8]^w[i-14]^w[i-16], 1)
// Circular-buffer w[16]: w[i-3]→(i+13)&15, w[i-8]→(i+8)&15,
//                         w[i-14]→(i+2)&15, w[i-16]→i&15.
// Full #pragma unroll 80 makes every modular index a compile-time constant
// so w[16] is allocated in registers, not local memory.
__device__ __noinline__
void sha1_compress(uint32_t h[5], const uint32_t blk[16]) {
    uint32_t w[16];
    #pragma unroll 16
    for (int i = 0; i < 16; i++) w[i] = blk[i];

    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];

    #pragma unroll 80
    for (int i = 0; i < 80; i++) {
        if (i >= 16)
            w[i&15] = ROTL32(w[(i+13)&15] ^ w[(i+8)&15] ^ w[(i+2)&15] ^ w[i&15], 1);
        uint32_t wi = w[i & 15];
        uint32_t f, k;
        if      (i < 20) { f = (b & c) | (~b & d);           k = 0x5A827999u; }
        else if (i < 40) { f = b ^ c ^ d;                     k = 0x6ED9EBA1u; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d);  k = 0x8F1BBCDCu; }
        else             { f = b ^ c ^ d;                     k = 0xCA62C1D6u; }
        uint32_t tmp = ROTL32(a, 5) + f + e + k + wi;
        e = d; d = c; c = ROTL32(b, 30); b = a; a = tmp;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
}

// ── Helpers: write into big-endian uint32 block ───────────────

// Write one byte at byte-offset boff (boff & 3 == 0 → MSB of its word).
__device__ __forceinline__
void sha1_put_byte(uint32_t blk[16], uint32_t boff, uint8_t b) {
    uint32_t wi    = boff >> 2;
    uint32_t shift = 24 - (boff & 3) * 8;  // 0→24, 1→16, 2→8, 3→0
    blk[wi] = (blk[wi] & ~(0xFFu << shift)) | ((uint32_t)b << shift);
}

// Write 4 bytes as one big-endian word (requires boff % 4 == 0).
__device__ __forceinline__
void sha1_put_word(uint32_t blk[16], uint32_t boff, const uint8_t *data) {
    blk[boff >> 2] = ((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16)
                   | ((uint32_t)data[2] <<  8) |  (uint32_t)data[3];
}

// Helper: reset block to all-zero after a compress
__device__ __forceinline__
void sha1_blk_zero(uint32_t blk[16]) {
    #pragma unroll 16
    for (int i = 0; i < 16; i++) blk[i] = 0;
}

// ── Init ─────────────────────────────────────────────────────

__device__ __forceinline__
void sha1_init(sha1_ctx_t *ctx) {
    ctx->h[0] = 0x67452301u; ctx->h[1] = 0xEFCDAB89u;
    ctx->h[2] = 0x98BADCFEu; ctx->h[3] = 0x10325476u;
    ctx->h[4] = 0xC3D2E1F0u;
    ctx->total = 0;
    ctx->boff  = 0;
    sha1_blk_zero(ctx->blk);
}

// ── Update ───────────────────────────────────────────────────
// Uses 32-bit (word) writes when byte-offset is 4-aligned and ≥4 bytes
// remain: ~3× fewer local-memory store operations for aligned data.

__device__ __forceinline__
void sha1_update(sha1_ctx_t *ctx, const uint8_t *data, uint32_t len) {
    ctx->total += (uint64_t)len * 8;
    uint32_t boff = ctx->boff;

    while (len > 0) {
        // Word writes while 4-byte aligned and ≥4 bytes left
        while ((boff & 3) == 0 && len >= 4) {
            sha1_put_word(ctx->blk, boff, data);
            data += 4; len -= 4; boff += 4;
            if (boff == 64) {
                sha1_compress(ctx->h, ctx->blk);
                sha1_blk_zero(ctx->blk);
                boff = 0;
            }
        }
        // Byte fallback for alignment head or short tail
        if (len > 0) {
            sha1_put_byte(ctx->blk, boff, *data++);
            len--;
            if (++boff == 64) {
                sha1_compress(ctx->h, ctx->blk);
                sha1_blk_zero(ctx->blk);
                boff = 0;
            }
        }
    }
    ctx->boff = boff;
}

// ── Final ────────────────────────────────────────────────────

__device__ __forceinline__
void sha1_final(sha1_ctx_t *ctx, uint8_t digest[20]) {
    uint64_t bits = ctx->total;
    uint32_t boff = ctx->boff;

    // Append 0x80 padding byte
    sha1_put_byte(ctx->blk, boff, 0x80);
    if (++boff == 64) {
        sha1_compress(ctx->h, ctx->blk);
        sha1_blk_zero(ctx->blk);
        boff = 0;
    }

    // If no room for the 8-byte length field (need boff ≤ 56)
    if (boff > 56) {
        // blk[boff..63] are already zero from last reset
        sha1_compress(ctx->h, ctx->blk);
        sha1_blk_zero(ctx->blk);
        boff = 0;
    }
    // blk[boff..55] are already zero; write big-endian bit count at [56..63]
    ctx->blk[14] = (uint32_t)(bits >> 32);
    ctx->blk[15] = (uint32_t)(bits & 0xFFFFFFFFu);
    sha1_compress(ctx->h, ctx->blk);

    store_be32(digest +  0, ctx->h[0]);
    store_be32(digest +  4, ctx->h[1]);
    store_be32(digest +  8, ctx->h[2]);
    store_be32(digest + 12, ctx->h[3]);
    store_be32(digest + 16, ctx->h[4]);
}

// Convenience: hash a single flat buffer
__device__ __forceinline__
void sha1_hash(const uint8_t *data, uint32_t len, uint8_t digest[20]) {
    sha1_ctx_t ctx;
    sha1_init(&ctx);
    sha1_update(&ctx, data, len);
    sha1_final(&ctx, digest);
}

// ── By-value interface (keeps h in registers) ─────────────────
//
// sha1_h5_t uses named fields — never stored behind a pointer in the
// caller, so NVCC allocates h0..h4 in registers.
//
// sha1_compress_bv is __noinline__: its w[16]+a-e live in their own
// register frame; the caller sees just 5 uint32 in / 5 uint32 out.

typedef struct { uint32_t h0,h1,h2,h3,h4; } sha1_h5_t;

__device__ __noinline__
sha1_h5_t sha1_compress_bv(sha1_h5_t hin, const uint32_t blk[16])
{
    uint32_t w[16];
    #pragma unroll 16
    for (int i = 0; i < 16; i++) w[i] = blk[i];

    uint32_t a=hin.h0, b=hin.h1, c=hin.h2, d=hin.h3, e=hin.h4;

    #pragma unroll 80
    for (int i = 0; i < 80; i++) {
        if (i >= 16)
            w[i&15] = ROTL32(w[(i+13)&15] ^ w[(i+8)&15] ^ w[(i+2)&15] ^ w[i&15], 1);
        uint32_t wi=w[i&15], f, k;
        if      (i<20){f=(b&c)|(~b&d); k=0x5A827999u;}
        else if (i<40){f=b^c^d;        k=0x6ED9EBA1u;}
        else if (i<60){f=(b&c)|(b&d)|(c&d); k=0x8F1BBCDCu;}
        else          {f=b^c^d;        k=0xCA62C1D6u;}
        uint32_t tmp=ROTL32(a,5)+f+e+k+wi;
        e=d; d=c; c=ROTL32(b,30); b=a; a=tmp;
    }

    sha1_h5_t hout = {hin.h0+a, hin.h1+b, hin.h2+c, hin.h3+d, hin.h4+e};
    return hout;
}
