//! RAR 1.5 LZH decompressor — port of UnRAR `unpack15.cpp`.
//!
//! Non-solid mode only (single-file archives). Output is a freshly-built `Vec<u8>`
//! of exactly `unp_size` bytes; caller compares `crc32_ieee` of that
//! buffer against FILE_CRC to validate a candidate password.
//!
//! ## Filter mode (Arq B preview)
//!
//! `Unpack15::filter(packed, unp_size, k_bytes, n_iters)` runs the same
//! decoder with two cheap modifications:
//!   * window reads/writes are skipped (we do not need the output bytes);
//!   * the main loop bails as Survivor after `n_iters` successful
//!     decisions, or as Reject if the bit stream exhausts earlier.
//!
//! This is a CPU reference implementation of the GPU filter for the
//! RAR 1.5 Arq-B cracker. It runs ~10-20× faster than `unpack()` on
//! wrong passwords because it never touches the 64 KB window and exits
//! early; the final CRC check is delegated to `unpack()` for survivors
//! on the host side.

use anyhow::{bail, Result};

use super::bitinput::BitInput;

// ── Huffman decode tables (from unpack15.cpp) ───────────────────────

const STARTL1: u32 = 2;
const DEC_L1:  [u32; 11] = [0x8000,0xa000,0xc000,0xd000,0xe000,0xea00,0xee00,0xf000,0xf200,0xf200,0xffff];
const POS_L1:  [u32; 13] = [0,0,0,2,3,5,7,11,16,20,24,32,32];

const STARTL2: u32 = 3;
const DEC_L2:  [u32; 10] = [0xa000,0xc000,0xd000,0xe000,0xea00,0xee00,0xf000,0xf200,0xf240,0xffff];
const POS_L2:  [u32; 13] = [0,0,0,0,5,7,9,13,18,22,26,34,36];

const STARTHF0: u32 = 4;
const DEC_HF0:  [u32; 9]  = [0x8000,0xc000,0xe000,0xf200,0xf200,0xf200,0xf200,0xf200,0xffff];
const POS_HF0:  [u32; 13] = [0,0,0,0,0,8,16,24,33,33,33,33,33];

const STARTHF1: u32 = 5;
const DEC_HF1:  [u32; 8]  = [0x2000,0xc000,0xe000,0xf000,0xf200,0xf200,0xf7e0,0xffff];
const POS_HF1:  [u32; 13] = [0,0,0,0,0,0,4,44,60,76,80,80,127];

const STARTHF2: u32 = 5;
const DEC_HF2:  [u32; 8]  = [0x1000,0x2400,0x8000,0xc000,0xfa00,0xffff,0xffff,0xffff];
const POS_HF2:  [u32; 13] = [0,0,0,0,0,0,2,7,53,117,233,0,0];

const STARTHF3: u32 = 6;
const DEC_HF3:  [u32; 7]  = [0x800,0x2400,0xee00,0xfe80,0xffff,0xffff,0xffff];
const POS_HF3:  [u32; 13] = [0,0,0,0,0,0,0,2,16,218,251,0,0];

const STARTHF4: u32 = 8;
const DEC_HF4:  [u32; 6]  = [0xff00,0xffff,0xffff,0xffff,0xffff,0xffff];
const POS_HF4:  [u32; 13] = [0,0,0,0,0,0,0,0,0,255,0,0,0];

// ── ShortLZ tables (function-local in unrar) ────────────────────────

const SHORT_LEN1: [u32; 16] = [1,3,4,4,5,6,7,8,8,4,4,5,6,6,4,0];
const SHORT_XOR1: [u32; 15] = [0,0xa0,0xd0,0xe0,0xf0,0xf8,0xfc,0xfe,0xff,0xc0,0x80,0x90,0x98,0x9c,0xb0];
const SHORT_LEN2: [u32; 16] = [2,3,3,3,4,4,5,6,6,4,4,5,6,6,4,0];
const SHORT_XOR2: [u32; 15] = [0,0x40,0x60,0xa0,0xd0,0xe0,0xf0,0xf8,0xfc,0xc0,0x80,0x90,0x98,0x9c,0xb0];

// ── Unpack15 state ──────────────────────────────────────────────────

const WIN_SIZE: usize = 0x10000;
const WIN_MASK: u32   = 0xffff;

pub struct Unpack15 {
    bit:         BitInput,
    window:      Box<[u8; WIN_SIZE]>,
    written:     Vec<u8>,
    dest_size:   i64,

    unp_ptr:     u32,
    wr_ptr:      u32,
    prev_ptr:    u32,
    first_win:   bool,

    old_dist:    [u32; 4],
    old_dist_ptr: usize,
    last_dist:   u32,
    last_length: u32,

    flags_cnt:   i32,
    flag_buf:    u32,
    st_mode:     u32,
    l_count:     u32,
    num_huf:     u32,
    buf60:       u32,
    avr_plc:     u32,
    avr_plc_b:   u32,
    avr_ln1:     u32,
    avr_ln2:     u32,
    avr_ln3:     u32,
    nhfb:        u32,
    nlzb:        u32,
    max_dist3:   u32,

    ch_set:      [u16; 256],
    ch_set_a:    [u16; 256],
    ch_set_b:    [u16; 256],
    ch_set_c:    [u16; 256],
    nto_pl:      [u8; 256],
    nto_pl_b:    [u8; 256],
    nto_pl_c:    [u8; 256],

    // ── Filter mode ─────────────────────────────────────────
    /// When true, skip all window reads/writes and bail after
    /// `iter_limit` main-loop iterations returning the current
    /// `FilterResult`. See module docs.
    filter_mode: bool,
    iter_limit:  usize,
    iter_count:  usize,
    /// Set to true when decoding hits an impossible value (e.g. distance_place
    /// out of range). Used as a Reject signal in filter mode.
    decode_error: bool,
}

/// Filter mode verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterResult {
    /// The bit stream decoded cleanly for `n_iters` iterations without
    /// exhausting the packed stream. Caller should run strict verify.
    Survivor,
    /// The bit stream exhausted too early, or a decoder guard triggered —
    /// the password is definitively wrong.
    Reject,
}

/// Debug stats for the M0 probe. Not used in production.
#[derive(Debug, Clone, Copy)]
pub struct FilterStats {
    pub iters_done:    usize,
    pub dest_consumed: i64,
    pub bits_consumed: u64,
}

impl Unpack15 {
    fn new(packed: Vec<u8>, unp_size: u64) -> Self {
        Self {
            bit:         BitInput::new(packed),
            window:      Box::new([0u8; WIN_SIZE]),
            written:     Vec::with_capacity(unp_size.min(1 << 20) as usize),
            dest_size:   unp_size as i64,

            unp_ptr:     0,
            wr_ptr:      0,
            prev_ptr:    0,
            first_win:   false,

            old_dist:    [u32::MAX; 4],
            old_dist_ptr: 0,
            last_dist:   u32::MAX,
            last_length: 0,

            flags_cnt:   0,
            flag_buf:    0,
            st_mode:     0,
            l_count:     0,
            num_huf:     0,
            buf60:       0,
            avr_plc:     0x3500,
            avr_plc_b:   0,
            avr_ln1:     0,
            avr_ln2:     0,
            avr_ln3:     0,
            nhfb:        0x80,
            nlzb:        0x80,
            max_dist3:   0x2001,

            ch_set:      [0; 256],
            ch_set_a:    [0; 256],
            ch_set_b:    [0; 256],
            ch_set_c:    [0; 256],
            nto_pl:      [0; 256],
            nto_pl_b:    [0; 256],
            nto_pl_c:    [0; 256],

            filter_mode:  false,
            iter_limit:   usize::MAX,
            iter_count:   0,
            decode_error: false,
        }
    }

    /// Top-level entry. Non-solid only.
    pub fn unpack(packed: Vec<u8>, unp_size: u64) -> Result<Vec<u8>> {
        let mut u = Self::new(packed, unp_size);
        u.run()?;
        Ok(u.written)
    }

    /// Filter-mode entry: see module docs. `k_bytes` truncates the packed
    /// stream (so only a small prefix is decoded), `n_iters` caps the main
    /// loop. Cheaper than `unpack` — intended for culling wrong passwords
    /// before running strict verify on survivors.
    pub fn filter(
        packed: &[u8],
        unp_size: u64,
        k_bytes: usize,
        n_iters: usize,
    ) -> FilterResult {
        Self::filter_stats(packed, unp_size, k_bytes, n_iters).0
    }

    /// Like `filter` but also returns the debug stats used during the M0
    /// probe (number of iterations completed, bytes of output "produced").
    pub fn filter_stats(
        packed: &[u8],
        unp_size: u64,
        k_bytes: usize,
        n_iters: usize,
    ) -> (FilterResult, FilterStats) {
        let slice = if k_bytes < packed.len() { &packed[..k_bytes] } else { packed };
        let mut u = Self::new(slice.to_vec(), unp_size);
        u.filter_mode = true;
        u.iter_limit  = n_iters;
        let verdict = u.run_filter();
        let stats = FilterStats {
            iters_done:     u.iter_count,
            dest_consumed:  (unp_size as i64) - 1 - u.dest_size, // how many output bytes
            bits_consumed:  (u.bit.in_addr as u64) * 8 + u.bit.in_bit as u64,
        };
        (verdict, stats)
    }

    fn run_filter(&mut self) -> FilterResult {
        // Same driver as `run()` but swallows errors and reports
        // Reject/Survivor. We only care about bit-stream consistency.
        let _ = self.run();
        if self.decode_error {
            FilterResult::Reject
        } else if self.iter_count >= self.iter_limit {
            FilterResult::Survivor
        } else {
            // Loop exited early without hitting the iteration cap →
            // bit-stream exhausted / truncated before n_iters. Reject.
            FilterResult::Reject
        }
    }

    // ── Bit-stream availability ─────────────────────────────────

    /// Mirror of UnRAR `UnpReadBuf` for a static buffer: returns false when
    /// we're about to read past the end of the packed stream.
    fn unp_read_buf(&self) -> bool {
        self.bit.in_addr < self.bit.read_top()
    }

    // ── Main driver ─────────────────────────────────────────────

    fn run(&mut self) -> Result<()> {
        self.init_huff();
        self.unp_ptr = 0;

        self.dest_size -= 1;
        if self.dest_size >= 0 {
            self.get_flags_buf();
            self.flags_cnt = 8;
        }

        while self.dest_size >= 0 {
            if self.filter_mode && self.iter_count >= self.iter_limit { break; }
            self.iter_count += 1;

            self.unp_ptr &= WIN_MASK;

            if self.prev_ptr > self.unp_ptr {
                self.first_win = true;
            }
            self.prev_ptr = self.unp_ptr;

            let read_top = self.bit.read_top();
            if self.bit.in_addr + 30 > read_top && !self.unp_read_buf() {
                break;
            }

            if !self.filter_mode
                && ((self.wr_ptr.wrapping_sub(self.unp_ptr)) & WIN_MASK) < 270
                && self.wr_ptr != self.unp_ptr
            {
                self.unp_write_buf20();
            }

            if self.st_mode != 0 {
                self.huff_decode();
                continue;
            }

            self.flags_cnt -= 1;
            if self.flags_cnt < 0 {
                self.get_flags_buf();
                self.flags_cnt = 7;
            }

            if self.flag_buf & 0x80 != 0 {
                self.flag_buf <<= 1;
                if self.nlzb > self.nhfb {
                    self.long_lz();
                } else {
                    self.huff_decode();
                }
            } else {
                self.flag_buf <<= 1;
                self.flags_cnt -= 1;
                if self.flags_cnt < 0 {
                    self.get_flags_buf();
                    self.flags_cnt = 7;
                }
                if self.flag_buf & 0x80 != 0 {
                    self.flag_buf <<= 1;
                    if self.nlzb > self.nhfb {
                        self.huff_decode();
                    } else {
                        self.long_lz();
                    }
                } else {
                    self.flag_buf <<= 1;
                    self.short_lz();
                }
            }
        }

        if !self.filter_mode {
            self.unp_write_buf20();
            if (self.written.len() as i64) < (self.dest_size + 1).max(0) {
                // Loop exited early (stream truncated) without producing the
                // promised number of bytes. Signal corruption.
                bail!("Unpack15: bitstream exhausted before unp_size reached");
            }
        }
        Ok(())
    }

    // ── Init ────────────────────────────────────────────────────

    fn init_huff(&mut self) {
        for i in 0..256u16 {
            self.ch_set[i as usize]   = i << 8;
            self.ch_set_b[i as usize] = i << 8;
            self.ch_set_a[i as usize] = i;
            self.ch_set_c[i as usize] = ((!i).wrapping_add(1) & 0xff) << 8;
        }
        self.nto_pl.fill(0);
        self.nto_pl_b.fill(0);
        self.nto_pl_c.fill(0);
        // CorrHuff on ChSetB / NToPlB
        let mut set = [0u16; 256];
        set.copy_from_slice(&self.ch_set_b);
        let mut nto = [0u8; 256];
        nto.copy_from_slice(&self.nto_pl_b);
        corr_huff(&mut set, &mut nto);
        self.ch_set_b.copy_from_slice(&set);
        self.nto_pl_b.copy_from_slice(&nto);
    }

    // ── Write/read helpers ──────────────────────────────────────

    fn unp_write_buf20(&mut self) {
        if self.unp_ptr < self.wr_ptr {
            // Wrap: tail of window, then head.
            let tail_len = (WIN_SIZE as u32 - self.wr_ptr) & WIN_MASK;
            if tail_len > 0 {
                self.written.extend_from_slice(
                    &self.window[self.wr_ptr as usize..(self.wr_ptr + tail_len) as usize],
                );
            }
            self.written.extend_from_slice(&self.window[..self.unp_ptr as usize]);
        } else {
            self.written.extend_from_slice(
                &self.window[self.wr_ptr as usize..self.unp_ptr as usize],
            );
        }
        self.wr_ptr = self.unp_ptr;
    }

    fn copy_string15(&mut self, distance: u32, length: u32) {
        self.dest_size -= length as i64;
        if self.filter_mode {
            // Only advance the cursor; window content is irrelevant for
            // bit-stream correctness tracking.
            self.unp_ptr = (self.unp_ptr + length) & WIN_MASK;
            return;
        }
        let corrupt = (!self.first_win && distance > self.unp_ptr)
            || distance as usize > WIN_SIZE
            || distance == 0;
        if corrupt {
            for _ in 0..length {
                self.window[self.unp_ptr as usize] = 0;
                self.unp_ptr = (self.unp_ptr + 1) & WIN_MASK;
            }
        } else {
            for _ in 0..length {
                let src = (self.unp_ptr.wrapping_sub(distance)) & WIN_MASK;
                self.window[self.unp_ptr as usize] = self.window[src as usize];
                self.unp_ptr = (self.unp_ptr + 1) & WIN_MASK;
            }
        }
    }

    // ── Decode helpers ──────────────────────────────────────────

    fn decode_num(&mut self, num_in: u32, start_pos: u32, dec_tab: &[u32], pos_tab: &[u32]) -> u32 {
        let num = num_in & 0xfff0;
        let mut i = 0usize;
        let mut start = start_pos;
        while i < dec_tab.len() && dec_tab[i] <= num {
            start += 1;
            i += 1;
        }
        self.bit.faddbits(start);
        let prev = if i == 0 { 0 } else { dec_tab[i - 1] };
        ((num - prev) >> (16 - start)) + pos_tab[start as usize]
    }

    fn get_flags_buf(&mut self) {
        let bf = self.bit.fgetbits();
        let flags_place = self.decode_num(bf, STARTHF2, &DEC_HF2, &POS_HF2);
        if (flags_place as usize) >= self.ch_set_c.len() {
            return;
        }
        let fp = flags_place as usize;
        // Bounded iteration — CorrHuff re-seeds the table to a known-good
        // shape whose low-byte values span 0..=7, so the loop terminates
        // in at most a handful of iterations; cap at 16 to avoid hanging
        // on corrupt streams.
        for _ in 0..16 {
            let flags = self.ch_set_c[fp] as u32;
            self.flag_buf = flags >> 8;
            let idx = (flags & 0xff) as usize;
            let new_place = self.nto_pl_c[idx] as u32;
            self.nto_pl_c[idx] = self.nto_pl_c[idx].wrapping_add(1);
            let flags_plus1 = flags.wrapping_add(1);
            if (flags_plus1 & 0xff) != 0 {
                self.ch_set_c[fp] = self.ch_set_c[new_place as usize];
                self.ch_set_c[new_place as usize] = flags_plus1 as u16;
                return;
            }
            corr_huff(&mut self.ch_set_c, &mut self.nto_pl_c);
        }
    }

    // ── ShortLZ ─────────────────────────────────────────────────

    fn short_lz(&mut self) {
        self.num_huf = 0;
        let mut bit_field = self.bit.fgetbits();
        if self.l_count == 2 {
            self.bit.faddbits(1);
            if bit_field >= 0x8000 {
                let d = self.last_dist;
                let l = self.last_length;
                self.copy_string15(d, l);
                return;
            }
            bit_field = (bit_field << 1) & 0xffff;
            self.l_count = 0;
        }
        bit_field >>= 8;

        let (short_len, short_xor, use_tab1): (&[u32], &[u32], bool) =
            if self.avr_ln1 < 37 {
                (&SHORT_LEN1, &SHORT_XOR1, true)
            } else {
                (&SHORT_LEN2, &SHORT_XOR2, false)
            };
        let buf60 = self.buf60;
        let get_short_len = |pos: usize| -> u32 {
            if use_tab1 {
                if pos == 1 { buf60 + 3 } else { short_len[pos] }
            } else {
                if pos == 3 { buf60 + 3 } else { short_len[pos] }
            }
        };

        let mut length: u32 = 0;
        loop {
            let sl = get_short_len(length as usize);
            let mask = !(0xffu32 >> sl);
            if ((bit_field ^ short_xor[length as usize]) & mask) == 0 {
                break;
            }
            length += 1;
            if length as usize >= short_xor.len() {
                // Corrupt stream — avoid infinite loop.
                break;
            }
        }
        self.bit.faddbits(get_short_len(length as usize));

        if length >= 9 {
            if length == 9 {
                self.l_count += 1;
                let d = self.last_dist;
                let l = self.last_length;
                self.copy_string15(d, l);
                return;
            }
            if length == 14 {
                self.l_count = 0;
                let bf = self.bit.fgetbits();
                let length = self.decode_num(bf, STARTL2, &DEC_L2, &POS_L2) + 5;
                let distance = (self.bit.fgetbits() >> 1) | 0x8000;
                self.bit.faddbits(15);
                self.last_length = length;
                self.last_dist   = distance;
                self.copy_string15(distance, length);
                return;
            }

            self.l_count = 0;
            let save_length = length;
            let idx = (self.old_dist_ptr as u32).wrapping_sub(length - 9) & 3;
            let distance = self.old_dist[idx as usize];
            let bf = self.bit.fgetbits();
            let mut length = self.decode_num(bf, STARTL1, &DEC_L1, &POS_L1) + 2;
            if length == 0x101 && save_length == 10 {
                self.buf60 ^= 1;
                return;
            }
            if distance > 256 { length += 1; }
            if distance >= self.max_dist3 { length += 1; }

            self.old_dist[self.old_dist_ptr] = distance;
            self.old_dist_ptr = (self.old_dist_ptr + 1) & 3;
            self.last_length = length;
            self.last_dist   = distance;
            self.copy_string15(distance, length);
            return;
        }

        self.l_count = 0;
        self.avr_ln1 += length;
        self.avr_ln1 -= self.avr_ln1 >> 4;

        let bf = self.bit.fgetbits();
        let distance_place = self.decode_num(bf, STARTHF2, &DEC_HF2, &POS_HF2) & 0xff;
        let mut dp = distance_place as i32;
        let mut distance = self.ch_set_a[dp as usize] as u32;
        dp -= 1;
        if dp != -1 {
            let last_distance = self.ch_set_a[dp as usize] as u32;
            self.ch_set_a[(dp + 1) as usize] = last_distance as u16;
            self.ch_set_a[dp as usize]       = distance as u16;
        }
        let length = length + 2;
        distance += 1;
        self.old_dist[self.old_dist_ptr] = distance;
        self.old_dist_ptr = (self.old_dist_ptr + 1) & 3;
        self.last_length = length;
        self.last_dist   = distance;
        self.copy_string15(distance, length);
    }

    // ── LongLZ ──────────────────────────────────────────────────

    fn long_lz(&mut self) {
        self.num_huf = 0;
        self.nlzb += 16;
        if self.nlzb > 0xff {
            self.nlzb = 0x90;
            self.nhfb >>= 1;
        }
        let old_avr2 = self.avr_ln2;

        let bf = self.bit.fgetbits();
        let length;
        if self.avr_ln2 >= 122 {
            length = self.decode_num(bf, STARTL2, &DEC_L2, &POS_L2);
        } else if self.avr_ln2 >= 64 {
            length = self.decode_num(bf, STARTL1, &DEC_L1, &POS_L1);
        } else if bf < 0x100 {
            length = bf;
            self.bit.faddbits(16);
        } else {
            let mut l = 0u32;
            while ((bf << l) & 0x8000) == 0 {
                l += 1;
                if l >= 16 { break; }
            }
            length = l;
            self.bit.faddbits(l + 1);
        }

        self.avr_ln2 += length;
        self.avr_ln2 -= self.avr_ln2 >> 5;

        let bf = self.bit.fgetbits();
        let distance_place = if self.avr_plc_b > 0x28ff {
            self.decode_num(bf, STARTHF2, &DEC_HF2, &POS_HF2)
        } else if self.avr_plc_b > 0x6ff {
            self.decode_num(bf, STARTHF1, &DEC_HF1, &POS_HF1)
        } else {
            self.decode_num(bf, STARTHF0, &DEC_HF0, &POS_HF0)
        };

        self.avr_plc_b += distance_place;
        self.avr_plc_b -= self.avr_plc_b >> 8;

        let mut distance;
        let mut new_distance_place;
        loop {
            distance = self.ch_set_b[(distance_place & 0xff) as usize] as u32;
            let idx = (distance & 0xff) as usize;
            new_distance_place = self.nto_pl_b[idx] as u32;
            self.nto_pl_b[idx] = self.nto_pl_b[idx].wrapping_add(1);
            distance = distance.wrapping_add(1);
            if (distance & 0xff) == 0 {
                corr_huff(&mut self.ch_set_b, &mut self.nto_pl_b);
            } else {
                break;
            }
        }

        let dp_idx = (distance_place & 0xff) as usize;
        self.ch_set_b[dp_idx] = self.ch_set_b[new_distance_place as usize];
        self.ch_set_b[new_distance_place as usize] = distance as u16;

        let distance = ((distance & 0xff00) | (self.bit.fgetbits() >> 8)) >> 1;
        self.bit.faddbits(7);

        let old_avr3 = self.avr_ln3;
        let mut length = length;
        if length != 1 && length != 4 {
            if length == 0 && distance <= self.max_dist3 {
                self.avr_ln3 += 1;
                self.avr_ln3 -= self.avr_ln3 >> 8;
            } else if self.avr_ln3 > 0 {
                self.avr_ln3 -= 1;
            }
        }
        length += 3;
        if distance >= self.max_dist3 { length += 1; }
        if distance <= 256 { length += 8; }
        if old_avr3 > 0xb0 || (self.avr_plc >= 0x2a00 && old_avr2 < 0x40) {
            self.max_dist3 = 0x7f00;
        } else {
            self.max_dist3 = 0x2001;
        }
        self.old_dist[self.old_dist_ptr] = distance;
        self.old_dist_ptr = (self.old_dist_ptr + 1) & 3;
        self.last_length = length;
        self.last_dist   = distance;
        self.copy_string15(distance, length);
    }

    // ── HuffDecode ──────────────────────────────────────────────

    fn huff_decode(&mut self) {
        let bf = self.bit.fgetbits();

        let mut byte_place = if self.avr_plc > 0x75ff {
            self.decode_num(bf, STARTHF4, &DEC_HF4, &POS_HF4) as i32
        } else if self.avr_plc > 0x5dff {
            self.decode_num(bf, STARTHF3, &DEC_HF3, &POS_HF3) as i32
        } else if self.avr_plc > 0x35ff {
            self.decode_num(bf, STARTHF2, &DEC_HF2, &POS_HF2) as i32
        } else if self.avr_plc > 0x0dff {
            self.decode_num(bf, STARTHF1, &DEC_HF1, &POS_HF1) as i32
        } else {
            self.decode_num(bf, STARTHF0, &DEC_HF0, &POS_HF0) as i32
        };
        byte_place &= 0xff;

        if self.st_mode != 0 {
            if byte_place == 0 && bf > 0xfff {
                byte_place = 0x100;
            }
            byte_place -= 1;
            if byte_place == -1 {
                let bf = self.bit.fgetbits();
                self.bit.faddbits(1);
                if bf & 0x8000 != 0 {
                    self.num_huf = 0;
                    self.st_mode = 0;
                    return;
                } else {
                    let length = if bf & 0x4000 != 0 { 4 } else { 3 };
                    self.bit.faddbits(1);
                    let bf = self.bit.fgetbits();
                    let d = self.decode_num(bf, STARTHF2, &DEC_HF2, &POS_HF2);
                    let distance = (d << 5) | (self.bit.fgetbits() >> 11);
                    self.bit.faddbits(5);
                    self.copy_string15(distance, length);
                    return;
                }
            }
        } else {
            // C does `NumHuf++ >= 16` — compare pre-increment value.
            let prev = self.num_huf;
            self.num_huf += 1;
            if prev >= 16 && self.flags_cnt == 0 {
                self.st_mode = 1;
            }
        }

        self.avr_plc += byte_place as u32;
        self.avr_plc -= self.avr_plc >> 8;
        self.nhfb += 16;
        if self.nhfb > 0xff {
            self.nhfb = 0x90;
            self.nlzb >>= 1;
        }

        let bp = byte_place as usize;
        if !self.filter_mode {
            self.window[self.unp_ptr as usize] = (self.ch_set[bp] >> 8) as u8;
        }
        self.unp_ptr = (self.unp_ptr + 1) & WIN_MASK;
        self.dest_size -= 1;

        let mut cur_byte: u32;
        let mut new_byte_place: u32;
        loop {
            cur_byte = self.ch_set[bp] as u32;
            let idx = (cur_byte & 0xff) as usize;
            new_byte_place = self.nto_pl[idx] as u32;
            self.nto_pl[idx] = self.nto_pl[idx].wrapping_add(1);
            cur_byte = cur_byte.wrapping_add(1);
            if (cur_byte & 0xff) > 0xa1 {
                corr_huff(&mut self.ch_set, &mut self.nto_pl);
            } else {
                break;
            }
        }

        self.ch_set[bp] = self.ch_set[new_byte_place as usize];
        self.ch_set[new_byte_place as usize] = cur_byte as u16;
    }
}

/// Reset `char_set` / `num_to_place` to the canonical starting state used
/// after overflow of the adaptive counters. Port of `Unpack::CorrHuff`.
fn corr_huff(char_set: &mut [u16; 256], num_to_place: &mut [u8; 256]) {
    // for I=7..=0: 32 entries per bucket, low byte = I, high byte untouched.
    let mut pos = 0usize;
    for i in (0..=7i32).rev() {
        for _ in 0..32 {
            char_set[pos] = (char_set[pos] & 0xff00) | (i as u16 & 0xff);
            pos += 1;
        }
    }
    num_to_place.fill(0);
    for i in (0..=6i32).rev() {
        num_to_place[i as usize] = ((7 - i) * 32) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corr_huff_shape() {
        let mut cs = [0u16; 256];
        let mut nto = [0u8; 256];
        corr_huff(&mut cs, &mut nto);
        // First 32 entries → low byte = 7, next 32 → 6, ...
        assert_eq!(cs[0]   & 0xff, 7);
        assert_eq!(cs[31]  & 0xff, 7);
        assert_eq!(cs[32]  & 0xff, 6);
        assert_eq!(cs[255] & 0xff, 0);
        assert_eq!(nto[6], 32);
        assert_eq!(nto[0], 7 * 32);
    }

    #[test]
    fn init_huff_first_entries() {
        let mut u = Unpack15::new(vec![0u8; 64], 0);
        u.init_huff();
        assert_eq!(u.ch_set[0],   0x0000);
        assert_eq!(u.ch_set[1],   0x0100);
        assert_eq!(u.ch_set_a[1], 0x0001);
    }

    #[test]
    fn filter_rejects_obviously_truncated_stream() {
        // Only 4 bytes of packed data, way less than the 30-byte headroom
        // the main loop requires → filter must bail immediately.
        let verdict = Unpack15::filter(&[0u8; 4], 1024, 4, 32);
        assert_eq!(verdict, FilterResult::Reject);
    }
}
