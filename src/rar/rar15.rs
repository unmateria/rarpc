//! RAR 1.5 legacy encryption primitive (Crypt15 stream cipher).
//!
//! Source: UnRAR source (crypt.cpp `SetKey15` / `Crypt15`). The cipher is a
//! tiny 8-byte-state stream cipher keyed by the CRC32 of the password and
//! updated byte-wise.
//!
//! Combined with the LZH decompressor in `rar::unpack15`, this gives full
//! end-to-end password verification for RAR 1.5 archives: decrypt the
//! packed stream → decompress → compare CRC32 against FILE_CRC.

use anyhow::Result;

use super::unpack15::Unpack15;

// ── CRC32 table (polynomial 0xEDB88320) ─────────────────────────────

pub fn crc32_table() -> [u32; 256] {
    let mut tab = [0u32; 256];
    for i in 0..256u32 {
        let mut c = i;
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0xEDB88320 } else { c >> 1 };
        }
        tab[i as usize] = c;
    }
    tab
}

/// CRC32 running (no final XOR) — matches UnRAR `CRC32(StartCRC, data, size)`.
pub fn crc32_update(mut crc: u32, data: &[u8], tab: &[u32; 256]) -> u32 {
    for &b in data {
        crc = tab[((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    crc
}

pub fn crc32_ieee(data: &[u8]) -> u32 {
    let tab = crc32_table();
    !crc32_update(0xffff_ffff, data, &tab)
}

// ── Crypt15 stream cipher ───────────────────────────────────────────

#[derive(Clone)]
pub struct Rar15Cipher {
    pub key: [u16; 4],
    pub crc_tab: [u32; 256],
}

impl Rar15Cipher {
    /// `SetKey15` in unrar. Password is the OEM-encoded byte sequence
    /// (Windows-1252 on Spanish machines — ASCII passes through unchanged).
    pub fn new(password: &[u8]) -> Self {
        let crc_tab = crc32_table();
        let psw_crc = crc32_update(0xffff_ffff, password, &crc_tab);

        let mut key = [0u16; 4];
        key[0] = (psw_crc & 0xffff) as u16;
        key[1] = ((psw_crc >> 16) & 0xffff) as u16;
        key[2] = 0;
        key[3] = 0;
        for &p in password {
            let p32 = p as u32;
            let ctp = crc_tab[p as usize];
            key[2] ^= (p32 ^ ctp) as u16; // truncated to u16
            key[3] = key[3].wrapping_add((p32.wrapping_add(ctp >> 16)) as u16);
        }
        Self { key, crc_tab }
    }

    /// `Crypt15` in unrar. Decrypt/encrypt `data` in place (stream cipher is
    /// involutive — same operation for both directions).
    pub fn crypt(&mut self, data: &mut [u8]) {
        for b in data.iter_mut() {
            self.key[0] = self.key[0].wrapping_add(0x1234);
            let idx = ((self.key[0] as u32) & 0x1fe) >> 1;
            let ct  = self.crc_tab[idx as usize];
            self.key[1] ^= ct as u16;
            self.key[2] = self.key[2].wrapping_sub((ct >> 16) as u16);
            self.key[0] ^= self.key[2];
            // Two consecutive rotate-right-by-1 on 16-bit value, XOR key[1]
            // between them (this is exactly what unrar does).
            let r1 = self.key[3].rotate_right(1);
            self.key[3] = r1 ^ self.key[1];
            self.key[3] = self.key[3].rotate_right(1);
            self.key[0] ^= self.key[3];
            let k = (self.key[0] >> 8) as u8;
            *b ^= k;
        }
    }
}

// ── Filter parameters (Arq B) ───────────────────────────────────────

/// Tuning params for the probabilistic filter (Arq B).
/// Defaults: K=512, N=64, d_max=80 → pass-rate ~0.02% with 0 FN.
/// The same values are plumbed to both the CPU reference filter
/// (see `rar15_filter_cpu`) and the GPU kernel (`rar15_filter.cu`).
#[derive(Debug, Clone, Copy)]
pub struct Rar15FilterParams {
    pub k_bytes:  usize,
    pub n_iters:  usize,
    pub dest_max: i64,
}

impl Default for Rar15FilterParams {
    fn default() -> Self {
        Self { k_bytes: 512, n_iters: 64, dest_max: 80 }
    }
}

/// CPU reference filter — same verdict criterion the GPU kernel emits.
/// Returns `true` iff the password survives (i.e. requires strict verify).
pub fn rar15_filter_cpu(
    info: &Rar15Info,
    password: &[u8],
    params: Rar15FilterParams,
) -> bool {
    use crate::rar::unpack15::{FilterResult, Unpack15};

    let take = params.k_bytes.min(info.packed_data.len());
    let mut stream = info.packed_data[..take].to_vec();
    Rar15Cipher::new(password).crypt(&mut stream);
    let (verdict, stats) = Unpack15::filter_stats(
        &stream, info.unp_size as u64, stream.len(), params.n_iters,
    );
    matches!(verdict, FilterResult::Survivor) && stats.dest_consumed <= params.dest_max
}

// ── Archive metadata for RAR 1.5 ────────────────────────────────────

/// Everything needed to test a password against an encrypted RAR 1.5 file.
/// We keep the FULL packed data (typical 1997 archive: 30 KB) because the
/// verifier must decompress the whole thing to check `file_crc`.
#[derive(Debug, Clone)]
pub struct Rar15Info {
    pub packed_data: Vec<u8>, // the encrypted compressed stream
    pub unp_size:    u32,
    pub file_crc:    u32,
    pub unp_ver:     u8,      // 0x0f = RAR 1.5
    pub method:      u8,      // 0x30..0x35
}

impl Rar15Info {
    /// Full candidate verification: decrypt the packed stream with the
    /// candidate password, decompress (for method 0x31..0x35) or use the
    /// bytes as-is (0x30 = STORE), and compare CRC32 against `file_crc`.
    ///
    /// Called per-candidate on the CPU (no GPU path for Unpack15 today).
    pub fn verify_password(&self, password: &[u8]) -> Result<bool> {
        let mut stream = self.packed_data.clone();
        Rar15Cipher::new(password).crypt(&mut stream);

        if self.method == 0x30 {
            if stream.len() != self.unp_size as usize {
                return Ok(false);
            }
            return Ok(crc32_ieee(&stream) == self.file_crc);
        }

        // Compressed — run Unpack15. Any error (including corrupt output on
        // wrong key) is a NEGATIVE verdict, not a hard failure.
        let unpacked = match Unpack15::unpack(stream, self.unp_size as u64) {
            Ok(v)  => v,
            Err(_) => return Ok(false),
        };
        if unpacked.len() != self.unp_size as usize {
            return Ok(false);
        }
        Ok(crc32_ieee(&unpacked) == self.file_crc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_table_first_entries() {
        let tab = crc32_table();
        assert_eq!(tab[0], 0x00000000);
        assert_eq!(tab[1], 0x77073096);
        assert_eq!(tab[0xff], 0x2d02ef8d);
    }

    #[test]
    fn crc_of_hello() {
        // CRC32 of "hello" is 0x3610a686
        assert_eq!(crc32_ieee(b"hello"), 0x3610a686);
    }

    #[test]
    fn cipher_keys_deterministic() {
        let c1 = Rar15Cipher::new(b"test");
        let c2 = Rar15Cipher::new(b"test");
        assert_eq!(c1.key, c2.key);
        let c3 = Rar15Cipher::new(b"TEST");
        assert_ne!(c1.key, c3.key);
    }

    #[test]
    fn cipher_involutive() {
        // Crypt15 is XOR-based: encrypt + decrypt should round-trip.
        let original = b"Hello, RAR 1.5 world!".to_vec();
        let mut data = original.clone();

        Rar15Cipher::new(b"password").crypt(&mut data);
        assert_ne!(data, original, "cipher must alter data");

        Rar15Cipher::new(b"password").crypt(&mut data);
        assert_eq!(data, original, "same-key crypt twice == identity");
    }

}
