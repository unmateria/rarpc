use anyhow::Result;
use sha1::{Digest as _, Sha1};

/// How to verify a candidate password from the decrypted first block.
#[derive(Debug, Clone, PartialEq)]
pub enum Rar3CheckMode {
    /// Encrypted-headers archive (0x73 + ENCRYPTED_VER):
    /// block[2] must be a valid RAR3 HEAD_TYPE (typically 0x74).
    HeadType,
    /// File-level STORE archive, pack_size ≤ 16:
    /// CRC32 of block[0..pack_size] must equal file_crc.
    StoreCrc,
    /// Compressed or large STORE file: GPU passes candidates through,
    /// CPU double-checks with the Heuristic (block[0] in valid RAR3 types).
    /// Will have rare false positives filtered by the CPU double-check in engine.
    Heuristic,
}

/// Encryption metadata extracted from a RAR3 archive
#[derive(Debug, Clone)]
pub struct Rar3Info {
    pub salt:       [u8; 8],
    pub enc_block:  [u8; 16],   // first 16 bytes of encrypted file/header data
    pub check_mode: Rar3CheckMode,
    pub head_type:  u8,          // HeadType: expected value at block[2]
    pub file_crc:   u32,         // StoreCrc: CRC32 of unpacked file content
    pub pack_size:  u8,          // StoreCrc: encrypted bytes in first block (≤ 16)
    // Legacy 2-byte field kept so existing GPU dispatch compiles during transition
    pub auth_check: [u8; 2],
}

impl Rar3Info {
    /// RAR3 KDF: SHA-1 based, 262144 iterations
    /// Returns (key[16], iv[16])
    pub fn derive_key(password: &str, salt: &[u8; 8]) -> ([u8; 16], [u8; 16]) {
        // Encode password as UTF-16LE
        let utf16_bytes: Vec<u8> = password
            .encode_utf16()
            .flat_map(|c| c.to_le_bytes())
            .collect();

        let mut ctx = Sha1::new();
        let mut key = [0u8; 16];
        let mut iv  = [0u8; 16];
        let mut key_idx: usize = 0;
        let mut iv_idx:  usize = 0;

        const ITERS: u32 = 0x40000;
        const SAMPLE: u32 = 0x4000;

        for i in 0u32..ITERS {
            let i_bytes = [
                (i & 0xff) as u8,
                ((i >> 8) & 0xff) as u8,
                ((i >> 16) & 0xff) as u8,
            ];
            ctx.update(&utf16_bytes);
            ctx.update(salt);
            ctx.update(&i_bytes);

            // Sample for key (every SAMPLE-1 steps)
            if (i & (SAMPLE - 1)) == (SAMPLE - 1) {
                let digest = ctx.clone().finalize();
                if key_idx < 16 {
                    key[key_idx] = digest[0];
                    key_idx += 1;
                }
            }
            // Sample for IV (at offset SAMPLE/2 - 1)
            if (i & (SAMPLE - 1)) == (SAMPLE / 2 - 1) {
                let digest = ctx.clone().finalize();
                if iv_idx < 16 {
                    iv[iv_idx] = digest[0];
                    iv_idx += 1;
                }
            }
        }

        (key, iv)
    }

    /// CPU-side password verification
    pub fn verify_password(&self, password: &str) -> Result<bool> {
        use aes::Aes128;
        use cbc::Decryptor;
        use cipher::{BlockDecryptMut, KeyIvInit};

        let (key, iv) = Self::derive_key(password, &self.salt);

        type Aes128CbcDec = Decryptor<Aes128>;
        let mut block = self.enc_block;
        let decryptor = Aes128CbcDec::new(&key.into(), &iv.into());
        {
            use cipher::block_padding::NoPadding;
            decryptor
                .decrypt_padded_mut::<NoPadding>(&mut block)
                .map_err(|e| anyhow::anyhow!("AES decrypt error: {:?}", e))?;
        }

        Ok(match self.check_mode {
            Rar3CheckMode::HeadType => {
                // Encrypted-headers: byte[2] is HEAD_TYPE of the first file block.
                let valid = [0x72u8, 0x73, 0x74, 0x75, 0x76, 0x7a, 0x7b];
                valid.contains(&block[2])
            }
            Rar3CheckMode::StoreCrc => {
                // STORE file fitting in one block: CRC32 of raw content must match.
                let n = self.pack_size as usize;
                crc32_ieee(&block[..n]) == self.file_crc
            }
            Rar3CheckMode::Heuristic => {
                // Compressed / large file: check byte[0] is plausibly a RAR3 head
                // type or compressed-data magic. Engine does a CPU double-check so
                // false positives here are filtered before reporting to the user.
                let valid = [0x72u8, 0x73, 0x74, 0x75, 0x76, 0x7a, 0x7b];
                valid.contains(&block[0])
            }
        })
    }
}

// ── CRC32 (IEEE 802.3 polynomial) ───────────────────────────────
pub fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xEDB8_8320u32 & (0u32.wrapping_sub(crc & 1)));
        }
    }
    crc ^ 0xFFFF_FFFFu32
}
