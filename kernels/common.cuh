#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

// Maximum password size in bytes.
// RAR3 stores UTF-16LE (max 127 chars × 2 = 254 bytes rounded to 256).
// RAR5 stores UTF-8 (max 127 chars = 127 bytes, rounded to 128).
// We use 256 to support both safely.
#define MAX_PW_BYTES 256

// No match sentinel
#define NO_MATCH (-1)

// Rotate-right 32-bit.
// Use the __funnelshift_r intrinsic so nvcc emits a single SHF.R.WRAP.B32
// on every arch SM_35+ regardless of pattern-recognition heuristics.
#define ROTR32(x, n)  __funnelshift_r((uint32_t)(x), (uint32_t)(x), (n))
// Rotate-left 32-bit (used by SHA-1). Kept as the bit-pattern form — nvcc
// reliably recognises it. Switching could affect RAR3 codegen which we
// don't want to perturb here.
#define ROTL32(x, n)  (((uint32_t)(x) << (n)) | ((uint32_t)(x) >> (32 - (n))))

// Byte-swap 32-bit big-endian <-> host
__device__ __forceinline__ uint32_t bswap32(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}

// Load 4 bytes as big-endian uint32
__device__ __forceinline__ uint32_t load_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16)
         | ((uint32_t)p[2] << 8)  |  (uint32_t)p[3];
}

// Store uint32 as big-endian bytes
__device__ __forceinline__ void store_be32(uint8_t *p, uint32_t v) {
    p[0] = (v >> 24) & 0xff;
    p[1] = (v >> 16) & 0xff;
    p[2] = (v >>  8) & 0xff;
    p[3] =  v        & 0xff;
}

// Constant-time memory compare (returns 1 if equal)
__device__ __forceinline__ int memcmp_ct(const uint8_t *a, const uint8_t *b, int len) {
    uint32_t diff = 0;
    for (int i = 0; i < len; i++) diff |= a[i] ^ b[i];
    return (diff == 0) ? 1 : 0;
}
