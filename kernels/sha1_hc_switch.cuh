// switch_buffer_by_offset_{be,carry_be}_S — shift-merge helpers.
// Used by sha1_update_64_hc to shift-merge a
// 64-byte aligned source buffer into the current SHA-1 block at an
// arbitrary byte offset, with carry into the next block if needed.
//
// Every write target is a compile-time array index — nvcc keeps w0..w3
// and c0..c3 in registers as long as the caller doesn't take their address.

#pragma once
#include "sha1_hc.cuh"

// `offset` must be in [0, 63]. `offset_switch` = offset / 4.
// Shifts w0..w3 (64-byte source, 16 u32 BE words) right by `offset` bytes.
// Positions [0..offset) of the output are zero. Source is consumed entirely.
__device__ __forceinline__
void switch_buf_be(uint32_t w0[4], uint32_t w1[4], uint32_t w2[4], uint32_t w3[4], uint32_t offset)
{
    const int sw = (int)(offset >> 2);
    switch (sw) {
    case 0:
        w3[3] = hc_bytealign_be_S(w3[2], w3[3], offset);
        w3[2] = hc_bytealign_be_S(w3[1], w3[2], offset);
        w3[1] = hc_bytealign_be_S(w3[0], w3[1], offset);
        w3[0] = hc_bytealign_be_S(w2[3], w3[0], offset);
        w2[3] = hc_bytealign_be_S(w2[2], w2[3], offset);
        w2[2] = hc_bytealign_be_S(w2[1], w2[2], offset);
        w2[1] = hc_bytealign_be_S(w2[0], w2[1], offset);
        w2[0] = hc_bytealign_be_S(w1[3], w2[0], offset);
        w1[3] = hc_bytealign_be_S(w1[2], w1[3], offset);
        w1[2] = hc_bytealign_be_S(w1[1], w1[2], offset);
        w1[1] = hc_bytealign_be_S(w1[0], w1[1], offset);
        w1[0] = hc_bytealign_be_S(w0[3], w1[0], offset);
        w0[3] = hc_bytealign_be_S(w0[2], w0[3], offset);
        w0[2] = hc_bytealign_be_S(w0[1], w0[2], offset);
        w0[1] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w0[0] = hc_bytealign_be_S(0u,    w0[0], offset);
        break;
    case 1:
        w3[3] = hc_bytealign_be_S(w3[1], w3[2], offset);
        w3[2] = hc_bytealign_be_S(w3[0], w3[1], offset);
        w3[1] = hc_bytealign_be_S(w2[3], w3[0], offset);
        w3[0] = hc_bytealign_be_S(w2[2], w2[3], offset);
        w2[3] = hc_bytealign_be_S(w2[1], w2[2], offset);
        w2[2] = hc_bytealign_be_S(w2[0], w2[1], offset);
        w2[1] = hc_bytealign_be_S(w1[3], w2[0], offset);
        w2[0] = hc_bytealign_be_S(w1[2], w1[3], offset);
        w1[3] = hc_bytealign_be_S(w1[1], w1[2], offset);
        w1[2] = hc_bytealign_be_S(w1[0], w1[1], offset);
        w1[1] = hc_bytealign_be_S(w0[3], w1[0], offset);
        w1[0] = hc_bytealign_be_S(w0[2], w0[3], offset);
        w0[3] = hc_bytealign_be_S(w0[1], w0[2], offset);
        w0[2] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w0[1] = hc_bytealign_be_S(0u,    w0[0], offset);
        w0[0] = 0;
        break;
    case 2:
        w3[3] = hc_bytealign_be_S(w3[0], w3[1], offset);
        w3[2] = hc_bytealign_be_S(w2[3], w3[0], offset);
        w3[1] = hc_bytealign_be_S(w2[2], w2[3], offset);
        w3[0] = hc_bytealign_be_S(w2[1], w2[2], offset);
        w2[3] = hc_bytealign_be_S(w2[0], w2[1], offset);
        w2[2] = hc_bytealign_be_S(w1[3], w2[0], offset);
        w2[1] = hc_bytealign_be_S(w1[2], w1[3], offset);
        w2[0] = hc_bytealign_be_S(w1[1], w1[2], offset);
        w1[3] = hc_bytealign_be_S(w1[0], w1[1], offset);
        w1[2] = hc_bytealign_be_S(w0[3], w1[0], offset);
        w1[1] = hc_bytealign_be_S(w0[2], w0[3], offset);
        w1[0] = hc_bytealign_be_S(w0[1], w0[2], offset);
        w0[3] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w0[2] = hc_bytealign_be_S(0u,    w0[0], offset);
        w0[1] = 0; w0[0] = 0;
        break;
    case 3:
        w3[3] = hc_bytealign_be_S(w2[3], w3[0], offset);
        w3[2] = hc_bytealign_be_S(w2[2], w2[3], offset);
        w3[1] = hc_bytealign_be_S(w2[1], w2[2], offset);
        w3[0] = hc_bytealign_be_S(w2[0], w2[1], offset);
        w2[3] = hc_bytealign_be_S(w1[3], w2[0], offset);
        w2[2] = hc_bytealign_be_S(w1[2], w1[3], offset);
        w2[1] = hc_bytealign_be_S(w1[1], w1[2], offset);
        w2[0] = hc_bytealign_be_S(w1[0], w1[1], offset);
        w1[3] = hc_bytealign_be_S(w0[3], w1[0], offset);
        w1[2] = hc_bytealign_be_S(w0[2], w0[3], offset);
        w1[1] = hc_bytealign_be_S(w0[1], w0[2], offset);
        w1[0] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w0[3] = hc_bytealign_be_S(0u,    w0[0], offset);
        w0[2] = 0; w0[1] = 0; w0[0] = 0;
        break;
    case 4:
        w3[3] = hc_bytealign_be_S(w2[2], w2[3], offset);
        w3[2] = hc_bytealign_be_S(w2[1], w2[2], offset);
        w3[1] = hc_bytealign_be_S(w2[0], w2[1], offset);
        w3[0] = hc_bytealign_be_S(w1[3], w2[0], offset);
        w2[3] = hc_bytealign_be_S(w1[2], w1[3], offset);
        w2[2] = hc_bytealign_be_S(w1[1], w1[2], offset);
        w2[1] = hc_bytealign_be_S(w1[0], w1[1], offset);
        w2[0] = hc_bytealign_be_S(w0[3], w1[0], offset);
        w1[3] = hc_bytealign_be_S(w0[2], w0[3], offset);
        w1[2] = hc_bytealign_be_S(w0[1], w0[2], offset);
        w1[1] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w1[0] = hc_bytealign_be_S(0u,    w0[0], offset);
        w0[3] = 0; w0[2] = 0; w0[1] = 0; w0[0] = 0;
        break;
    case 5:
        w3[3] = hc_bytealign_be_S(w2[1], w2[2], offset);
        w3[2] = hc_bytealign_be_S(w2[0], w2[1], offset);
        w3[1] = hc_bytealign_be_S(w1[3], w2[0], offset);
        w3[0] = hc_bytealign_be_S(w1[2], w1[3], offset);
        w2[3] = hc_bytealign_be_S(w1[1], w1[2], offset);
        w2[2] = hc_bytealign_be_S(w1[0], w1[1], offset);
        w2[1] = hc_bytealign_be_S(w0[3], w1[0], offset);
        w2[0] = hc_bytealign_be_S(w0[2], w0[3], offset);
        w1[3] = hc_bytealign_be_S(w0[1], w0[2], offset);
        w1[2] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w1[1] = hc_bytealign_be_S(0u,    w0[0], offset);
        w1[0] = 0; w0[3] = 0; w0[2] = 0; w0[1] = 0; w0[0] = 0;
        break;
    case 6:
        w3[3] = hc_bytealign_be_S(w2[0], w2[1], offset);
        w3[2] = hc_bytealign_be_S(w1[3], w2[0], offset);
        w3[1] = hc_bytealign_be_S(w1[2], w1[3], offset);
        w3[0] = hc_bytealign_be_S(w1[1], w1[2], offset);
        w2[3] = hc_bytealign_be_S(w1[0], w1[1], offset);
        w2[2] = hc_bytealign_be_S(w0[3], w1[0], offset);
        w2[1] = hc_bytealign_be_S(w0[2], w0[3], offset);
        w2[0] = hc_bytealign_be_S(w0[1], w0[2], offset);
        w1[3] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w1[2] = hc_bytealign_be_S(0u,    w0[0], offset);
        w1[1] = 0; w1[0] = 0; w0[3] = 0; w0[2] = 0; w0[1] = 0; w0[0] = 0;
        break;
    case 7:
        w3[3] = hc_bytealign_be_S(w1[3], w2[0], offset);
        w3[2] = hc_bytealign_be_S(w1[2], w1[3], offset);
        w3[1] = hc_bytealign_be_S(w1[1], w1[2], offset);
        w3[0] = hc_bytealign_be_S(w1[0], w1[1], offset);
        w2[3] = hc_bytealign_be_S(w0[3], w1[0], offset);
        w2[2] = hc_bytealign_be_S(w0[2], w0[3], offset);
        w2[1] = hc_bytealign_be_S(w0[1], w0[2], offset);
        w2[0] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w1[3] = hc_bytealign_be_S(0u,    w0[0], offset);
        w1[2] = 0; w1[1] = 0; w1[0] = 0; w0[3] = 0; w0[2] = 0; w0[1] = 0; w0[0] = 0;
        break;
    case 8:
        w3[3] = hc_bytealign_be_S(w1[2], w1[3], offset);
        w3[2] = hc_bytealign_be_S(w1[1], w1[2], offset);
        w3[1] = hc_bytealign_be_S(w1[0], w1[1], offset);
        w3[0] = hc_bytealign_be_S(w0[3], w1[0], offset);
        w2[3] = hc_bytealign_be_S(w0[2], w0[3], offset);
        w2[2] = hc_bytealign_be_S(w0[1], w0[2], offset);
        w2[1] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w2[0] = hc_bytealign_be_S(0u,    w0[0], offset);
        w1[3]=0; w1[2]=0; w1[1]=0; w1[0]=0; w0[3]=0; w0[2]=0; w0[1]=0; w0[0]=0;
        break;
    case 9:
        w3[3] = hc_bytealign_be_S(w1[1], w1[2], offset);
        w3[2] = hc_bytealign_be_S(w1[0], w1[1], offset);
        w3[1] = hc_bytealign_be_S(w0[3], w1[0], offset);
        w3[0] = hc_bytealign_be_S(w0[2], w0[3], offset);
        w2[3] = hc_bytealign_be_S(w0[1], w0[2], offset);
        w2[2] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w2[1] = hc_bytealign_be_S(0u,    w0[0], offset);
        w2[0]=0; w1[3]=0; w1[2]=0; w1[1]=0; w1[0]=0; w0[3]=0; w0[2]=0; w0[1]=0; w0[0]=0;
        break;
    case 10:
        w3[3] = hc_bytealign_be_S(w1[0], w1[1], offset);
        w3[2] = hc_bytealign_be_S(w0[3], w1[0], offset);
        w3[1] = hc_bytealign_be_S(w0[2], w0[3], offset);
        w3[0] = hc_bytealign_be_S(w0[1], w0[2], offset);
        w2[3] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w2[2] = hc_bytealign_be_S(0u,    w0[0], offset);
        w2[1]=0; w2[0]=0; w1[3]=0; w1[2]=0; w1[1]=0; w1[0]=0; w0[3]=0; w0[2]=0; w0[1]=0; w0[0]=0;
        break;
    case 11:
        w3[3] = hc_bytealign_be_S(w0[3], w1[0], offset);
        w3[2] = hc_bytealign_be_S(w0[2], w0[3], offset);
        w3[1] = hc_bytealign_be_S(w0[1], w0[2], offset);
        w3[0] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w2[3] = hc_bytealign_be_S(0u,    w0[0], offset);
        w2[2]=0; w2[1]=0; w2[0]=0; w1[3]=0; w1[2]=0; w1[1]=0; w1[0]=0; w0[3]=0; w0[2]=0; w0[1]=0; w0[0]=0;
        break;
    case 12:
        w3[3] = hc_bytealign_be_S(w0[2], w0[3], offset);
        w3[2] = hc_bytealign_be_S(w0[1], w0[2], offset);
        w3[1] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w3[0] = hc_bytealign_be_S(0u,    w0[0], offset);
        w2[3]=0; w2[2]=0; w2[1]=0; w2[0]=0; w1[3]=0; w1[2]=0; w1[1]=0; w1[0]=0; w0[3]=0; w0[2]=0; w0[1]=0; w0[0]=0;
        break;
    case 13:
        w3[3] = hc_bytealign_be_S(w0[1], w0[2], offset);
        w3[2] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w3[1] = hc_bytealign_be_S(0u,    w0[0], offset);
        w3[0]=0; w2[3]=0; w2[2]=0; w2[1]=0; w2[0]=0; w1[3]=0; w1[2]=0; w1[1]=0; w1[0]=0; w0[3]=0; w0[2]=0; w0[1]=0; w0[0]=0;
        break;
    case 14:
        w3[3] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w3[2] = hc_bytealign_be_S(0u,    w0[0], offset);
        w3[1]=0; w3[0]=0; w2[3]=0; w2[2]=0; w2[1]=0; w2[0]=0; w1[3]=0; w1[2]=0; w1[1]=0; w1[0]=0; w0[3]=0; w0[2]=0; w0[1]=0; w0[0]=0;
        break;
    default: // 15
        w3[3] = hc_bytealign_be_S(0u,    w0[0], offset);
        w3[2]=0; w3[1]=0; w3[0]=0; w2[3]=0; w2[2]=0; w2[1]=0; w2[0]=0; w1[3]=0; w1[2]=0; w1[1]=0; w1[0]=0; w0[3]=0; w0[2]=0; w0[1]=0; w0[0]=0;
        break;
    }
}

// Same but also emits a carry (c0..c3) — the 64 bytes that wrapped past the
// input buffer. All 16 offset-cases, verbatim (MIT).
#include "sha1_hc_carry.inc"
#define switch_buf_carry_be switch_buf_carry_be_full

// Unused — kept for historical reference
#if 0
__device__ __forceinline__
void switch_buf_carry_be_OLD(
    uint32_t w0[4], uint32_t w1[4], uint32_t w2[4], uint32_t w3[4],
    uint32_t c0[4], uint32_t c1[4], uint32_t c2[4], uint32_t c3[4],
    uint32_t offset)
{
    const int sw = (int)(offset >> 2);
    switch (sw) {
    case 0:
        c0[0] = hc_bytealign_be_S(w3[3], 0u,    offset);
        w3[3] = hc_bytealign_be_S(w3[2], w3[3], offset);
        w3[2] = hc_bytealign_be_S(w3[1], w3[2], offset);
        w3[1] = hc_bytealign_be_S(w3[0], w3[1], offset);
        w3[0] = hc_bytealign_be_S(w2[3], w3[0], offset);
        w2[3] = hc_bytealign_be_S(w2[2], w2[3], offset);
        w2[2] = hc_bytealign_be_S(w2[1], w2[2], offset);
        w2[1] = hc_bytealign_be_S(w2[0], w2[1], offset);
        w2[0] = hc_bytealign_be_S(w1[3], w2[0], offset);
        w1[3] = hc_bytealign_be_S(w1[2], w1[3], offset);
        w1[2] = hc_bytealign_be_S(w1[1], w1[2], offset);
        w1[1] = hc_bytealign_be_S(w1[0], w1[1], offset);
        w1[0] = hc_bytealign_be_S(w0[3], w1[0], offset);
        w0[3] = hc_bytealign_be_S(w0[2], w0[3], offset);
        w0[2] = hc_bytealign_be_S(w0[1], w0[2], offset);
        w0[1] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w0[0] = hc_bytealign_be_S(0u,    w0[0], offset);
        break;
    case 1:
        c0[1] = hc_bytealign_be_S(w3[3], 0u,    offset);
        c0[0] = hc_bytealign_be_S(w3[2], w3[3], offset);
        w3[3] = hc_bytealign_be_S(w3[1], w3[2], offset);
        w3[2] = hc_bytealign_be_S(w3[0], w3[1], offset);
        w3[1] = hc_bytealign_be_S(w2[3], w3[0], offset);
        w3[0] = hc_bytealign_be_S(w2[2], w2[3], offset);
        w2[3] = hc_bytealign_be_S(w2[1], w2[2], offset);
        w2[2] = hc_bytealign_be_S(w2[0], w2[1], offset);
        w2[1] = hc_bytealign_be_S(w1[3], w2[0], offset);
        w2[0] = hc_bytealign_be_S(w1[2], w1[3], offset);
        w1[3] = hc_bytealign_be_S(w1[1], w1[2], offset);
        w1[2] = hc_bytealign_be_S(w1[0], w1[1], offset);
        w1[1] = hc_bytealign_be_S(w0[3], w1[0], offset);
        w1[0] = hc_bytealign_be_S(w0[2], w0[3], offset);
        w0[3] = hc_bytealign_be_S(w0[1], w0[2], offset);
        w0[2] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w0[1] = hc_bytealign_be_S(0u,    w0[0], offset);
        w0[0] = 0;
        break;
    case 2:
        c0[2] = hc_bytealign_be_S(w3[3], 0u,    offset);
        c0[1] = hc_bytealign_be_S(w3[2], w3[3], offset);
        c0[0] = hc_bytealign_be_S(w3[1], w3[2], offset);
        w3[3] = hc_bytealign_be_S(w3[0], w3[1], offset);
        w3[2] = hc_bytealign_be_S(w2[3], w3[0], offset);
        w3[1] = hc_bytealign_be_S(w2[2], w2[3], offset);
        w3[0] = hc_bytealign_be_S(w2[1], w2[2], offset);
        w2[3] = hc_bytealign_be_S(w2[0], w2[1], offset);
        w2[2] = hc_bytealign_be_S(w1[3], w2[0], offset);
        w2[1] = hc_bytealign_be_S(w1[2], w1[3], offset);
        w2[0] = hc_bytealign_be_S(w1[1], w1[2], offset);
        w1[3] = hc_bytealign_be_S(w1[0], w1[1], offset);
        w1[2] = hc_bytealign_be_S(w0[3], w1[0], offset);
        w1[1] = hc_bytealign_be_S(w0[2], w0[3], offset);
        w1[0] = hc_bytealign_be_S(w0[1], w0[2], offset);
        w0[3] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w0[2] = hc_bytealign_be_S(0u,    w0[0], offset);
        w0[1] = 0; w0[0] = 0;
        break;
    case 3:
        c0[3] = hc_bytealign_be_S(w3[3], 0u,    offset);
        c0[2] = hc_bytealign_be_S(w3[2], w3[3], offset);
        c0[1] = hc_bytealign_be_S(w3[1], w3[2], offset);
        c0[0] = hc_bytealign_be_S(w3[0], w3[1], offset);
        w3[3] = hc_bytealign_be_S(w2[3], w3[0], offset);
        w3[2] = hc_bytealign_be_S(w2[2], w2[3], offset);
        w3[1] = hc_bytealign_be_S(w2[1], w2[2], offset);
        w3[0] = hc_bytealign_be_S(w2[0], w2[1], offset);
        w2[3] = hc_bytealign_be_S(w1[3], w2[0], offset);
        w2[2] = hc_bytealign_be_S(w1[2], w1[3], offset);
        w2[1] = hc_bytealign_be_S(w1[1], w1[2], offset);
        w2[0] = hc_bytealign_be_S(w1[0], w1[1], offset);
        w1[3] = hc_bytealign_be_S(w0[3], w1[0], offset);
        w1[2] = hc_bytealign_be_S(w0[2], w0[3], offset);
        w1[1] = hc_bytealign_be_S(w0[1], w0[2], offset);
        w1[0] = hc_bytealign_be_S(w0[0], w0[1], offset);
        w0[3] = hc_bytealign_be_S(0u,    w0[0], offset);
        w0[2] = 0; w0[1] = 0; w0[0] = 0;
        break;
    default:
        // For offsets 4..15, port later if needed. For RAR3 with pw_len <= 28,
        // pw_salt_len <= 36, plus 3 counter bytes → max pos reached is <= 63
        // but specific paths above cover the common cycles. If we hit here,
        // correctness is preserved by falling back to the non-carry path
        // (which zeros w0..w3) after the transform.
        // Fallback: emit c* = 0, call switch_buf_be on w*, so the transform
        // runs on the merged data without producing a carry.
        {
            c0[0] = 0; c0[1] = 0; c0[2] = 0; c0[3] = 0;
            c1[0] = 0; c1[1] = 0; c1[2] = 0; c1[3] = 0;
            c2[0] = 0; c2[1] = 0; c2[2] = 0; c2[3] = 0;
            c3[0] = 0; c3[1] = 0; c3[2] = 0; c3[3] = 0;
            switch_buf_be(w0, w1, w2, w3, offset);
        }
        break;
    }
    // c1/c2/c3 are zero for offsets 0..3 (3 bytes of overflow max when
    // pw_salt_len ≤ 64 and pos+len ≤ 64+12). Initialise them.
    if (sw <= 3) {
        c1[0]=0; c1[1]=0; c1[2]=0; c1[3]=0;
        c2[0]=0; c2[1]=0; c2[2]=0; c2[3]=0;
        c3[0]=0; c3[1]=0; c3[2]=0; c3[3]=0;
    }
}
#endif // old carry stub

// sha1_update_64_hc: merge up-to-64 aligned BE words (w0..w3) into the
// running ctx (ctx_w0..ctx_w3, h, ctx_len). `len` is the actual byte count
// (≤ 64) from the source.
__device__ __forceinline__
void sha1_update_64_hc(
    uint32_t h[5],
    uint32_t ctx_w0[4], uint32_t ctx_w1[4], uint32_t ctx_w2[4], uint32_t ctx_w3[4],
    uint32_t &ctx_len,
    uint32_t w0[4], uint32_t w1[4], uint32_t w2[4], uint32_t w3[4],
    uint32_t len)
{
    if (len == 0) return;
    const uint32_t pos = ctx_len & 63u;
    ctx_len += len;

    if (pos == 0) {
        ctx_w0[0] = w0[0]; ctx_w0[1] = w0[1]; ctx_w0[2] = w0[2]; ctx_w0[3] = w0[3];
        ctx_w1[0] = w1[0]; ctx_w1[1] = w1[1]; ctx_w1[2] = w1[2]; ctx_w1[3] = w1[3];
        ctx_w2[0] = w2[0]; ctx_w2[1] = w2[1]; ctx_w2[2] = w2[2]; ctx_w2[3] = w2[3];
        ctx_w3[0] = w3[0]; ctx_w3[1] = w3[1]; ctx_w3[2] = w3[2]; ctx_w3[3] = w3[3];
        if (len == 64) {
            sha1_transform_hc(ctx_w0, ctx_w1, ctx_w2, ctx_w3, h);
            ctx_w0[0]=0; ctx_w0[1]=0; ctx_w0[2]=0; ctx_w0[3]=0;
            ctx_w1[0]=0; ctx_w1[1]=0; ctx_w1[2]=0; ctx_w1[3]=0;
            ctx_w2[0]=0; ctx_w2[1]=0; ctx_w2[2]=0; ctx_w2[3]=0;
            ctx_w3[0]=0; ctx_w3[1]=0; ctx_w3[2]=0; ctx_w3[3]=0;
        }
    } else {
        if ((pos + len) < 64u) {
            switch_buf_be(w0, w1, w2, w3, pos);
            ctx_w0[0] |= w0[0]; ctx_w0[1] |= w0[1]; ctx_w0[2] |= w0[2]; ctx_w0[3] |= w0[3];
            ctx_w1[0] |= w1[0]; ctx_w1[1] |= w1[1]; ctx_w1[2] |= w1[2]; ctx_w1[3] |= w1[3];
            ctx_w2[0] |= w2[0]; ctx_w2[1] |= w2[1]; ctx_w2[2] |= w2[2]; ctx_w2[3] |= w2[3];
            ctx_w3[0] |= w3[0]; ctx_w3[1] |= w3[1]; ctx_w3[2] |= w3[2]; ctx_w3[3] |= w3[3];
        } else {
            uint32_t c0[4] = {0,0,0,0};
            uint32_t c1[4] = {0,0,0,0};
            uint32_t c2[4] = {0,0,0,0};
            uint32_t c3[4] = {0,0,0,0};
            switch_buf_carry_be(w0, w1, w2, w3, c0, c1, c2, c3, pos);
            ctx_w0[0] |= w0[0]; ctx_w0[1] |= w0[1]; ctx_w0[2] |= w0[2]; ctx_w0[3] |= w0[3];
            ctx_w1[0] |= w1[0]; ctx_w1[1] |= w1[1]; ctx_w1[2] |= w1[2]; ctx_w1[3] |= w1[3];
            ctx_w2[0] |= w2[0]; ctx_w2[1] |= w2[1]; ctx_w2[2] |= w2[2]; ctx_w2[3] |= w2[3];
            ctx_w3[0] |= w3[0]; ctx_w3[1] |= w3[1]; ctx_w3[2] |= w3[2]; ctx_w3[3] |= w3[3];
            sha1_transform_hc(ctx_w0, ctx_w1, ctx_w2, ctx_w3, h);
            ctx_w0[0] = c0[0]; ctx_w0[1] = c0[1]; ctx_w0[2] = c0[2]; ctx_w0[3] = c0[3];
            ctx_w1[0] = c1[0]; ctx_w1[1] = c1[1]; ctx_w1[2] = c1[2]; ctx_w1[3] = c1[3];
            ctx_w2[0] = c2[0]; ctx_w2[1] = c2[1]; ctx_w2[2] = c2[2]; ctx_w2[3] = c2[3];
            ctx_w3[0] = c3[0]; ctx_w3[1] = c3[1]; ctx_w3[2] = c3[2]; ctx_w3[3] = c3[3];
        }
    }
}
