// RAR 1.5 probabilistic filter kernel (Arq B M1).
//
// One thread per candidate password. Pipeline per thread:
//   1. SetKey15 + Crypt15 on first K bytes of packed data.
//   2. Unpack15 filter: same state machine as CPU reference, but:
//      - no window reads/writes (filter_mode);
//      - bail after N iterations;
//      - emit Survivor if iter_count == N && dest_consumed <= dest_max.
//
// Verdict is a u32 bitmap in d_survivors (ceil(n/32) words).
// Parity target: rar15_filter_cpu in src/rar/rar15.rs.

#include <cstdint>
#include "common.cuh"

#define MAX_K_BYTES 512
#define WIN_MASK    0xFFFFu

// ── Huffman decode tables (copied verbatim from unpack15.rs) ────────

__device__ __constant__ uint32_t c_dec_l1[11]  = {0x8000,0xa000,0xc000,0xd000,0xe000,0xea00,0xee00,0xf000,0xf200,0xf200,0xffff};
__device__ __constant__ uint32_t c_pos_l1[13]  = {0,0,0,2,3,5,7,11,16,20,24,32,32};
__device__ __constant__ uint32_t c_dec_l2[10]  = {0xa000,0xc000,0xd000,0xe000,0xea00,0xee00,0xf000,0xf200,0xf240,0xffff};
__device__ __constant__ uint32_t c_pos_l2[13]  = {0,0,0,0,5,7,9,13,18,22,26,34,36};
__device__ __constant__ uint32_t c_dec_hf0[9]  = {0x8000,0xc000,0xe000,0xf200,0xf200,0xf200,0xf200,0xf200,0xffff};
__device__ __constant__ uint32_t c_pos_hf0[13] = {0,0,0,0,0,8,16,24,33,33,33,33,33};
__device__ __constant__ uint32_t c_dec_hf1[8]  = {0x2000,0xc000,0xe000,0xf000,0xf200,0xf200,0xf7e0,0xffff};
__device__ __constant__ uint32_t c_pos_hf1[13] = {0,0,0,0,0,0,4,44,60,76,80,80,127};
__device__ __constant__ uint32_t c_dec_hf2[8]  = {0x1000,0x2400,0x8000,0xc000,0xfa00,0xffff,0xffff,0xffff};
__device__ __constant__ uint32_t c_pos_hf2[13] = {0,0,0,0,0,0,2,7,53,117,233,0,0};
__device__ __constant__ uint32_t c_dec_hf3[7]  = {0x800,0x2400,0xee00,0xfe80,0xffff,0xffff,0xffff};
__device__ __constant__ uint32_t c_pos_hf3[13] = {0,0,0,0,0,0,0,2,16,218,251,0,0};
__device__ __constant__ uint32_t c_dec_hf4[6]  = {0xff00,0xffff,0xffff,0xffff,0xffff,0xffff};
__device__ __constant__ uint32_t c_pos_hf4[13] = {0,0,0,0,0,0,0,0,0,255,0,0,0};

#define STARTL1  2u
#define STARTL2  3u
#define STARTHF0 4u
#define STARTHF1 5u
#define STARTHF2 5u
#define STARTHF3 6u
#define STARTHF4 8u

__device__ __constant__ uint32_t c_short_len1[16] = {1,3,4,4,5,6,7,8,8,4,4,5,6,6,4,0};
__device__ __constant__ uint32_t c_short_xor1[15] = {0,0xa0,0xd0,0xe0,0xf0,0xf8,0xfc,0xfe,0xff,0xc0,0x80,0x90,0x98,0x9c,0xb0};
__device__ __constant__ uint32_t c_short_len2[16] = {2,3,3,3,4,4,5,6,6,4,4,5,6,6,4,0};
__device__ __constant__ uint32_t c_short_xor2[15] = {0,0x40,0x60,0xa0,0xd0,0xe0,0xf0,0xf8,0xfc,0xc0,0x80,0x90,0x98,0x9c,0xb0};

// ── CRC32 table (poly 0xEDB88320) in shared memory ─────────────────
// Built cooperatively by first 256 threads of block.

__device__ __forceinline__ uint32_t crc32_word(uint32_t i) {
    uint32_t c = i;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        c = (c & 1u) ? ((c >> 1) ^ 0xEDB88320u) : (c >> 1);
    }
    return c;
}

// ── Crypt15: SetKey15 + stream decrypt ──────────────────────────────

struct Rar15Key {
    uint16_t k0, k1, k2, k3;
};

__device__ __forceinline__ Rar15Key setkey15(
    const uint8_t* pw, int len, const uint32_t* crc_tab)
{
    uint32_t psw_crc = 0xffffffffu;
    for (int i = 0; i < len; ++i) {
        psw_crc = crc_tab[(psw_crc ^ pw[i]) & 0xff] ^ (psw_crc >> 8);
    }
    Rar15Key k;
    k.k0 = (uint16_t)(psw_crc & 0xffff);
    k.k1 = (uint16_t)((psw_crc >> 16) & 0xffff);
    k.k2 = 0;
    k.k3 = 0;
    for (int i = 0; i < len; ++i) {
        uint32_t p = pw[i];
        uint32_t ctp = crc_tab[p];
        k.k2 ^= (uint16_t)(p ^ ctp);
        k.k3 = (uint16_t)(k.k3 + (uint16_t)(p + (ctp >> 16)));
    }
    return k;
}

__device__ __forceinline__ uint16_t rotr16(uint16_t v, int n) {
    return (uint16_t)((v >> n) | (v << (16 - n)));
}

__device__ __forceinline__ uint8_t crypt15_byte(
    Rar15Key& k, uint8_t in, const uint32_t* crc_tab)
{
    k.k0 = (uint16_t)(k.k0 + 0x1234);
    uint32_t idx = ((uint32_t)k.k0 & 0x1fe) >> 1;
    uint32_t ct  = crc_tab[idx];
    k.k1 ^= (uint16_t)ct;
    k.k2  = (uint16_t)(k.k2 - (uint16_t)(ct >> 16));
    k.k0 ^= k.k2;
    uint16_t r1 = rotr16(k.k3, 1);
    k.k3 = rotr16((uint16_t)(r1 ^ k.k1), 1);
    k.k0 ^= k.k3;
    return (uint8_t)(in ^ (uint8_t)(k.k0 >> 8));
}

// ── Unpack15 per-thread state ───────────────────────────────────────

struct U15 {
    // Bit stream
    uint32_t in_addr;
    uint32_t in_bit;

    // Main state
    int64_t  dest_size;
    uint32_t unp_ptr;
    uint32_t prev_ptr;
    uint8_t  first_win;
    uint32_t iter_count;

    // Flag parsing
    int32_t  flags_cnt;
    uint32_t flag_buf;

    // Decoder state
    uint32_t st_mode;
    uint32_t l_count;
    uint32_t num_huf;
    uint32_t buf60;

    // Distance history
    uint32_t old_dist[4];
    uint32_t old_dist_ptr;
    uint32_t last_dist;
    uint32_t last_length;

    // Adaptive averages
    uint32_t avr_plc;
    uint32_t avr_plc_b;
    uint32_t avr_ln1;
    uint32_t avr_ln2;
    uint32_t avr_ln3;
    uint32_t nhfb;
    uint32_t nlzb;
    uint32_t max_dist3;

    // Huffman tables (adaptive)
    uint16_t ch_set  [256];
    uint16_t ch_set_a[256];
    uint16_t ch_set_b[256];
    uint16_t ch_set_c[256];
    uint8_t  nto_pl  [256];
    uint8_t  nto_pl_b[256];
    uint8_t  nto_pl_c[256];

    // Decrypted packed prefix (also 8-byte pad)
    uint8_t  stream[MAX_K_BYTES + 8];
    uint32_t read_top;

    uint8_t  decode_error;
};

// ── Bit reader ──────────────────────────────────────────────────────

__device__ __forceinline__ uint32_t fgetbits(const U15& u) {
    uint32_t a = u.in_addr;
    uint32_t b0 = u.stream[a];
    uint32_t b1 = u.stream[a + 1];
    uint32_t b2 = u.stream[a + 2];
    uint32_t bit_field = (b0 << 16) | (b1 << 8) | b2;
    return (bit_field >> (8 - u.in_bit)) & 0xffffu;
}

__device__ __forceinline__ void faddbits(U15& u, uint32_t bits) {
    uint32_t total = bits + u.in_bit;
    u.in_addr += (total >> 3);
    u.in_bit   = total & 7u;
}

// ── CorrHuff: reset adaptive tables after overflow ──────────────────

__device__ void corr_huff(uint16_t* cs, uint8_t* nto) {
    int pos = 0;
    for (int i = 7; i >= 0; --i) {
        for (int j = 0; j < 32; ++j) {
            cs[pos] = (uint16_t)((cs[pos] & 0xff00) | (uint16_t)(i & 0xff));
            ++pos;
        }
    }
    for (int i = 0; i < 256; ++i) nto[i] = 0;
    for (int i = 6; i >= 0; --i) {
        nto[i] = (uint8_t)((7 - i) * 32);
    }
}

__device__ void init_huff(U15& u) {
    for (int i = 0; i < 256; ++i) {
        u.ch_set  [i] = (uint16_t)(i << 8);
        u.ch_set_b[i] = (uint16_t)(i << 8);
        u.ch_set_a[i] = (uint16_t)(i);
        u.ch_set_c[i] = (uint16_t)((((uint32_t)((~(uint32_t)i) + 1u)) & 0xffu) << 8);
        u.nto_pl  [i] = 0;
        u.nto_pl_b[i] = 0;
        u.nto_pl_c[i] = 0;
    }
    corr_huff(u.ch_set_b, u.nto_pl_b);
}

// ── Decode helpers ──────────────────────────────────────────────────

__device__ __forceinline__ uint32_t decode_num_ct(
    U15& u, uint32_t num_in, uint32_t start_pos,
    const uint32_t* dec_tab, int dec_len, const uint32_t* pos_tab)
{
    uint32_t num = num_in & 0xfff0u;
    int i = 0;
    uint32_t start = start_pos;
    while (i < dec_len && dec_tab[i] <= num) {
        ++start;
        ++i;
    }
    faddbits(u, start);
    uint32_t prev = (i == 0) ? 0u : dec_tab[i - 1];
    return ((num - prev) >> (16 - start)) + pos_tab[start];
}

__device__ bool unp_read_buf(const U15& u) {
    return u.in_addr < u.read_top;
}

__device__ void get_flags_buf(U15& u) {
    uint32_t bf = fgetbits(u);
    uint32_t fp = decode_num_ct(u, bf, STARTHF2, c_dec_hf2, 8, c_pos_hf2);
    if (fp >= 256u) { u.decode_error = 1; return; }

    for (int guard = 0; guard < 16; ++guard) {
        uint32_t flags = (uint32_t)u.ch_set_c[fp];
        u.flag_buf = flags >> 8;
        uint32_t idx = flags & 0xffu;
        uint32_t new_place = (uint32_t)u.nto_pl_c[idx];
        u.nto_pl_c[idx] = (uint8_t)(u.nto_pl_c[idx] + 1);
        uint32_t flags_plus1 = flags + 1u;
        if ((flags_plus1 & 0xffu) != 0u) {
            u.ch_set_c[fp] = u.ch_set_c[new_place];
            u.ch_set_c[new_place] = (uint16_t)flags_plus1;
            return;
        }
        corr_huff(u.ch_set_c, u.nto_pl_c);
    }
}

// ── copy_string15 (filter mode: only advance pointer + dest) ───────

__device__ __forceinline__ void copy_string15(U15& u, uint32_t /*distance*/, uint32_t length) {
    u.dest_size -= (int64_t)length;
    u.unp_ptr = (u.unp_ptr + length) & WIN_MASK;
}

// ── ShortLZ ─────────────────────────────────────────────────────────

__device__ void short_lz(U15& u) {
    u.num_huf = 0;
    uint32_t bit_field = fgetbits(u);
    if (u.l_count == 2) {
        faddbits(u, 1);
        if (bit_field >= 0x8000u) {
            copy_string15(u, u.last_dist, u.last_length);
            return;
        }
        bit_field = (bit_field << 1) & 0xffffu;
        u.l_count = 0;
    }
    bit_field >>= 8;

    bool use_tab1 = (u.avr_ln1 < 37u);
    const uint32_t* slen = use_tab1 ? c_short_len1 : c_short_len2;
    const uint32_t* sxor = use_tab1 ? c_short_xor1 : c_short_xor2;
    uint32_t buf60 = u.buf60;

    uint32_t length = 0;
    while (true) {
        uint32_t sl = slen[length];
        if (use_tab1 && length == 1) sl = buf60 + 3;
        else if (!use_tab1 && length == 3) sl = buf60 + 3;

        uint32_t mask = ~(0xffu >> sl);
        if (((bit_field ^ sxor[length]) & mask) == 0u) break;
        ++length;
        if (length >= 15u) { break; }
    }
    uint32_t sl_final = slen[length];
    if (use_tab1 && length == 1) sl_final = buf60 + 3;
    else if (!use_tab1 && length == 3) sl_final = buf60 + 3;
    faddbits(u, sl_final);

    if (length >= 9u) {
        if (length == 9u) {
            u.l_count += 1;
            copy_string15(u, u.last_dist, u.last_length);
            return;
        }
        if (length == 14u) {
            u.l_count = 0;
            uint32_t bf = fgetbits(u);
            uint32_t len2 = decode_num_ct(u, bf, STARTL2, c_dec_l2, 10, c_pos_l2) + 5u;
            uint32_t distance = (fgetbits(u) >> 1) | 0x8000u;
            faddbits(u, 15);
            u.last_length = len2;
            u.last_dist   = distance;
            copy_string15(u, distance, len2);
            return;
        }

        u.l_count = 0;
        uint32_t save_length = length;
        uint32_t idx = (u.old_dist_ptr - (length - 9u)) & 3u;
        uint32_t distance = u.old_dist[idx];
        uint32_t bf = fgetbits(u);
        uint32_t l3 = decode_num_ct(u, bf, STARTL1, c_dec_l1, 11, c_pos_l1) + 2u;
        if (l3 == 0x101u && save_length == 10u) {
            u.buf60 ^= 1u;
            return;
        }
        if (distance > 256u) ++l3;
        if (distance >= u.max_dist3) ++l3;
        u.old_dist[u.old_dist_ptr] = distance;
        u.old_dist_ptr = (u.old_dist_ptr + 1u) & 3u;
        u.last_length = l3;
        u.last_dist   = distance;
        copy_string15(u, distance, l3);
        return;
    }

    u.l_count = 0;
    u.avr_ln1 += length;
    u.avr_ln1 -= u.avr_ln1 >> 4;

    uint32_t bf = fgetbits(u);
    uint32_t distance_place = decode_num_ct(u, bf, STARTHF2, c_dec_hf2, 8, c_pos_hf2) & 0xffu;
    int32_t dp = (int32_t)distance_place;
    uint32_t distance = (uint32_t)u.ch_set_a[dp];
    --dp;
    if (dp != -1) {
        uint32_t last_distance = (uint32_t)u.ch_set_a[dp];
        u.ch_set_a[dp + 1] = (uint16_t)last_distance;
        u.ch_set_a[dp]     = (uint16_t)distance;
    }
    uint32_t len = length + 2u;
    distance += 1u;
    u.old_dist[u.old_dist_ptr] = distance;
    u.old_dist_ptr = (u.old_dist_ptr + 1u) & 3u;
    u.last_length = len;
    u.last_dist   = distance;
    copy_string15(u, distance, len);
}

// ── LongLZ ──────────────────────────────────────────────────────────

__device__ void long_lz(U15& u) {
    u.num_huf = 0;
    u.nlzb += 16u;
    if (u.nlzb > 0xffu) {
        u.nlzb = 0x90u;
        u.nhfb >>= 1;
    }
    uint32_t old_avr2 = u.avr_ln2;

    uint32_t bf = fgetbits(u);
    uint32_t length;
    if (u.avr_ln2 >= 122u) {
        length = decode_num_ct(u, bf, STARTL2, c_dec_l2, 10, c_pos_l2);
    } else if (u.avr_ln2 >= 64u) {
        length = decode_num_ct(u, bf, STARTL1, c_dec_l1, 11, c_pos_l1);
    } else if (bf < 0x100u) {
        length = bf;
        faddbits(u, 16);
    } else {
        uint32_t l = 0;
        while (((bf << l) & 0x8000u) == 0u) {
            ++l;
            if (l >= 16u) break;
        }
        length = l;
        faddbits(u, l + 1);
    }

    u.avr_ln2 += length;
    u.avr_ln2 -= u.avr_ln2 >> 5;

    bf = fgetbits(u);
    uint32_t distance_place;
    if (u.avr_plc_b > 0x28ffu) {
        distance_place = decode_num_ct(u, bf, STARTHF2, c_dec_hf2, 8, c_pos_hf2);
    } else if (u.avr_plc_b > 0x6ffu) {
        distance_place = decode_num_ct(u, bf, STARTHF1, c_dec_hf1, 8, c_pos_hf1);
    } else {
        distance_place = decode_num_ct(u, bf, STARTHF0, c_dec_hf0, 9, c_pos_hf0);
    }

    u.avr_plc_b += distance_place;
    u.avr_plc_b -= u.avr_plc_b >> 8;

    uint32_t distance;
    uint32_t new_distance_place;
    int guard = 0;
    while (true) {
        uint32_t dp_idx = distance_place & 0xffu;
        distance = (uint32_t)u.ch_set_b[dp_idx];
        uint32_t idx = distance & 0xffu;
        new_distance_place = (uint32_t)u.nto_pl_b[idx];
        u.nto_pl_b[idx] = (uint8_t)(u.nto_pl_b[idx] + 1);
        distance += 1u;
        if ((distance & 0xffu) == 0u) {
            corr_huff(u.ch_set_b, u.nto_pl_b);
            if (++guard > 16) { u.decode_error = 1; return; }
        } else {
            break;
        }
    }

    uint32_t dp_idx = distance_place & 0xffu;
    u.ch_set_b[dp_idx]            = u.ch_set_b[new_distance_place];
    u.ch_set_b[new_distance_place] = (uint16_t)distance;

    uint32_t dist = ((distance & 0xff00u) | (fgetbits(u) >> 8)) >> 1;
    faddbits(u, 7);

    uint32_t old_avr3 = u.avr_ln3;
    if (length != 1u && length != 4u) {
        if (length == 0u && dist <= u.max_dist3) {
            u.avr_ln3 += 1u;
            u.avr_ln3 -= u.avr_ln3 >> 8;
        } else if (u.avr_ln3 > 0u) {
            u.avr_ln3 -= 1u;
        }
    }
    length += 3u;
    if (dist >= u.max_dist3) ++length;
    if (dist <= 256u) length += 8u;
    if (old_avr3 > 0xb0u || (u.avr_plc >= 0x2a00u && old_avr2 < 0x40u)) {
        u.max_dist3 = 0x7f00u;
    } else {
        u.max_dist3 = 0x2001u;
    }
    u.old_dist[u.old_dist_ptr] = dist;
    u.old_dist_ptr = (u.old_dist_ptr + 1u) & 3u;
    u.last_length = length;
    u.last_dist   = dist;
    copy_string15(u, dist, length);
}

// ── HuffDecode ──────────────────────────────────────────────────────

__device__ void huff_decode(U15& u) {
    uint32_t bf = fgetbits(u);

    int32_t byte_place;
    if (u.avr_plc > 0x75ffu) {
        byte_place = (int32_t)decode_num_ct(u, bf, STARTHF4, c_dec_hf4, 6, c_pos_hf4);
    } else if (u.avr_plc > 0x5dffu) {
        byte_place = (int32_t)decode_num_ct(u, bf, STARTHF3, c_dec_hf3, 7, c_pos_hf3);
    } else if (u.avr_plc > 0x35ffu) {
        byte_place = (int32_t)decode_num_ct(u, bf, STARTHF2, c_dec_hf2, 8, c_pos_hf2);
    } else if (u.avr_plc > 0x0dffu) {
        byte_place = (int32_t)decode_num_ct(u, bf, STARTHF1, c_dec_hf1, 8, c_pos_hf1);
    } else {
        byte_place = (int32_t)decode_num_ct(u, bf, STARTHF0, c_dec_hf0, 9, c_pos_hf0);
    }
    byte_place &= 0xff;

    if (u.st_mode != 0u) {
        if (byte_place == 0 && bf > 0xfffu) {
            byte_place = 0x100;
        }
        byte_place -= 1;
        if (byte_place == -1) {
            uint32_t bf2 = fgetbits(u);
            faddbits(u, 1);
            if (bf2 & 0x8000u) {
                u.num_huf = 0;
                u.st_mode = 0;
                return;
            } else {
                uint32_t length = (bf2 & 0x4000u) ? 4u : 3u;
                faddbits(u, 1);
                uint32_t bf3 = fgetbits(u);
                uint32_t d = decode_num_ct(u, bf3, STARTHF2, c_dec_hf2, 8, c_pos_hf2);
                uint32_t distance = (d << 5) | (fgetbits(u) >> 11);
                faddbits(u, 5);
                copy_string15(u, distance, length);
                return;
            }
        }
    } else {
        uint32_t prev = u.num_huf;
        u.num_huf += 1u;
        if (prev >= 16u && u.flags_cnt == 0) {
            u.st_mode = 1u;
        }
    }

    u.avr_plc += (uint32_t)byte_place;
    u.avr_plc -= u.avr_plc >> 8;
    u.nhfb += 16u;
    if (u.nhfb > 0xffu) {
        u.nhfb = 0x90u;
        u.nlzb >>= 1;
    }

    uint32_t bp = (uint32_t)byte_place;
    // filter_mode: skip window write
    u.unp_ptr = (u.unp_ptr + 1u) & WIN_MASK;
    u.dest_size -= 1;

    uint32_t cur_byte;
    uint32_t new_byte_place;
    int guard = 0;
    while (true) {
        cur_byte = (uint32_t)u.ch_set[bp];
        uint32_t idx = cur_byte & 0xffu;
        new_byte_place = (uint32_t)u.nto_pl[idx];
        u.nto_pl[idx] = (uint8_t)(u.nto_pl[idx] + 1);
        cur_byte += 1u;
        if ((cur_byte & 0xffu) > 0xa1u) {
            corr_huff(u.ch_set, u.nto_pl);
            if (++guard > 16) { u.decode_error = 1; return; }
        } else {
            break;
        }
    }

    u.ch_set[bp]              = u.ch_set[new_byte_place];
    u.ch_set[new_byte_place]  = (uint16_t)cur_byte;
}

// ── Main filter driver ──────────────────────────────────────────────

__device__ bool run_filter(U15& u, uint32_t iter_limit) {
    init_huff(u);
    u.unp_ptr = 0;
    u.dest_size -= 1;
    if (u.dest_size >= 0) {
        get_flags_buf(u);
        u.flags_cnt = 8;
    }

    while (u.dest_size >= 0) {
        if (u.iter_count >= iter_limit) break;
        u.iter_count += 1u;

        u.unp_ptr &= WIN_MASK;
        if (u.prev_ptr > u.unp_ptr) u.first_win = 1;
        u.prev_ptr = u.unp_ptr;

        if (u.in_addr + 30u > u.read_top && !unp_read_buf(u)) break;

        if (u.st_mode != 0u) {
            huff_decode(u);
            if (u.decode_error) return false;
            continue;
        }

        u.flags_cnt -= 1;
        if (u.flags_cnt < 0) {
            get_flags_buf(u);
            u.flags_cnt = 7;
            if (u.decode_error) return false;
        }

        if (u.flag_buf & 0x80u) {
            u.flag_buf <<= 1;
            if (u.nlzb > u.nhfb) long_lz(u); else huff_decode(u);
        } else {
            u.flag_buf <<= 1;
            u.flags_cnt -= 1;
            if (u.flags_cnt < 0) {
                get_flags_buf(u);
                u.flags_cnt = 7;
                if (u.decode_error) return false;
            }
            if (u.flag_buf & 0x80u) {
                u.flag_buf <<= 1;
                if (u.nlzb > u.nhfb) huff_decode(u); else long_lz(u);
            } else {
                u.flag_buf <<= 1;
                short_lz(u);
            }
        }
        if (u.decode_error) return false;
    }

    return u.iter_count >= iter_limit;
}

// ── Kernel entry ────────────────────────────────────────────────────

extern "C" __global__ void rar15_filter(
    const uint8_t*  d_passwords,   // flat tid*MAX_PW_BYTES
    const int32_t*  d_pw_lengths,
    int32_t         n_passwords,
    const uint8_t*  d_packed_prefix, // first k_bytes of packed stream (global mem)
    int32_t         k_bytes,
    int32_t         n_iters,
    int64_t         dest_max,
    int64_t         unp_size,
    uint32_t*       d_survivors)   // bitmap, ceil(n/32) words
{
    // CRC32 table in shared memory (1 KB). Cooperative build across block.
    __shared__ uint32_t s_crc_tab[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        s_crc_tab[i] = crc32_word((uint32_t)i);
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_passwords) return;

    int pw_len = d_pw_lengths[tid];
    if (pw_len < 0) pw_len = 0;
    if (pw_len > MAX_PW_BYTES) pw_len = MAX_PW_BYTES;
    const uint8_t* pw = d_passwords + (size_t)tid * MAX_PW_BYTES;

    // 1. SetKey15
    Rar15Key key = setkey15(pw, pw_len, s_crc_tab);

    // 2. Decrypt first k_bytes into local stream buffer
    U15 u;
    int k = k_bytes;
    if (k > MAX_K_BYTES) k = MAX_K_BYTES;
    #pragma unroll 1
    for (int i = 0; i < k; ++i) {
        u.stream[i] = crypt15_byte(key, d_packed_prefix[i], s_crc_tab);
    }
    // 8-byte zero pad (mirrors BitInput::new)
    #pragma unroll
    for (int i = 0; i < 8; ++i) u.stream[k + i] = 0;
    u.read_top = (uint32_t)k;

    // 3. Init remaining state
    u.in_addr = 0; u.in_bit = 0;
    u.dest_size = (int64_t)unp_size;
    u.unp_ptr = 0; u.prev_ptr = 0; u.first_win = 0;
    u.iter_count = 0;
    u.flags_cnt = 0; u.flag_buf = 0;
    u.st_mode = 0; u.l_count = 0; u.num_huf = 0; u.buf60 = 0;
    u.old_dist[0] = 0xffffffffu; u.old_dist[1] = 0xffffffffu;
    u.old_dist[2] = 0xffffffffu; u.old_dist[3] = 0xffffffffu;
    u.old_dist_ptr = 0; u.last_dist = 0xffffffffu; u.last_length = 0;
    u.avr_plc = 0x3500u; u.avr_plc_b = 0;
    u.avr_ln1 = 0; u.avr_ln2 = 0; u.avr_ln3 = 0;
    u.nhfb = 0x80u; u.nlzb = 0x80u; u.max_dist3 = 0x2001u;
    u.decode_error = 0;

    // 4. Run filter
    bool survivor_raw = run_filter(u, (uint32_t)n_iters);

    // 5. Check dest_max threshold
    int64_t dest_consumed = (int64_t)unp_size - 1 - u.dest_size;
    bool survivor = survivor_raw && !u.decode_error && (dest_consumed <= dest_max);

    // 6. Write bit in bitmap
    if (survivor) {
        uint32_t word = (uint32_t)(tid >> 5);
        uint32_t bit  = 1u << (tid & 31);
        atomicOr(&d_survivors[word], bit);
    }
}
