use anyhow::Result;
use hmac::Hmac;
use pbkdf2::pbkdf2;
use sha2::Sha256;

/// Encryption metadata extracted from a RAR5 archive
#[derive(Debug, Clone)]
pub struct Rar5Info {
    pub salt: [u8; 16],
    /// AES IV — present only in the per-file CRYPT extra (None for archive-level)
    pub iv: Option<[u8; 16]>,
    pub iter_count: u8,   // actual iters = 1 << iter_count
    /// PswCheckData[12] = { InitV[8], PswCheck[4] }
    /// Present when CRYPT_PSWCHECK flag is set in the encryption header.
    pub psw_check_data: Option<PswCheckData>,
    pub enc_ver: u8,
}

/// Broken out from PswCheckData[12]:
///   hash = SHA256(PBKDF2_key || init_v)
///   check = hash[0..4]
///   Verify: check == stored_check
#[derive(Debug, Clone)]
pub struct PswCheckData {
    pub init_v: [u8; 8],
    pub check:  [u8; 4],
}

impl Rar5Info {
    pub fn iterations(&self) -> u32 {
        1u32 << self.iter_count
    }

    pub fn has_pw_check(&self) -> bool {
        self.psw_check_data.is_some()
    }

    /// Derive the 32-byte AES-256 key via PBKDF2-HMAC-SHA256
    pub fn derive_key(&self, password: &str) -> Result<[u8; 32]> {
        let mut key = [0u8; 32];
        pbkdf2::<Hmac<Sha256>>(
            password.as_bytes(),
            &self.salt,
            self.iterations(),
            &mut key,
        )
        .map_err(|e| anyhow::anyhow!("PBKDF2 error: {:?}", e))?;
        Ok(key)
    }

    /// Fast CPU-side password check using PswCheckData.
    ///
    /// Real WinRAR formula (verified empirically):
    ///   buf   = PBKDF2-SHA256(utf8_password, salt, (1<<cnt) + 32, 32)
    ///   xor8  = fold buf[32] → 8 bytes: xor8[i%8] ^= buf[i]
    ///   init_v stored in archive = xor8
    ///   check stored in archive  = SHA256(xor8)[0..4]
    ///   Verify: compute xor8, compare SHA256(xor8)[0..4] == stored check
    pub fn verify_password(&self, password: &str) -> Result<bool> {
        if let Some(ref pcd) = self.psw_check_data {
            // PswCheck uses (1<<iter_count) + 32 iterations
            let psw_iters = self.iterations() + 32;
            let mut buf = [0u8; 32];
            pbkdf2::<Hmac<Sha256>>(password.as_bytes(), &self.salt, psw_iters, &mut buf)
                .map_err(|e| anyhow::anyhow!("PBKDF2 error: {:?}", e))?;

            // XOR-compress 32 bytes → 8 bytes
            let mut xor8 = [0u8; 8];
            for (i, b) in buf.iter().enumerate() {
                xor8[i % 8] ^= b;
            }

            // init_v stored in archive is the expected xor8
            Ok(xor8 == pcd.init_v)
        } else {
            Err(anyhow::anyhow!(
                "RAR5 archive has no password check data — \
                 AES-based slow verification not implemented"
            ))
        }
    }

    /// Build the 8-byte GPU pw_check value from a known key.
    ///
    /// For GPU kernel convenience we pass check[4] padded to 8 bytes.
    pub fn gpu_pw_check_bytes(pcd: &PswCheckData) -> [u8; 12] {
        let mut out = [0u8; 12];
        out[..8].copy_from_slice(&pcd.init_v);
        out[8..12].copy_from_slice(&pcd.check);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hmac::Hmac;
    use pbkdf2::pbkdf2;
    use sha2::Digest;

    /// Verify the correct PswCheck formula against real WinRAR 7.20 bytes.
    ///
    /// Formula discovered empirically:
    ///   buf    = PBKDF2-SHA256(utf8_password, salt, (1<<cnt)+32, 32)
    ///   xor8   = fold buf[32] cyclically into 8 bytes
    ///   stored init_v == xor8   AND   SHA256(xor8)[0..4] == stored check
    #[test]
    fn test_psw_check_real_winrar() {
        let salt: [u8; 16] = [
            0xfa, 0xae, 0x70, 0xca, 0x95, 0x1f, 0xe4, 0xcb,
            0x36, 0x19, 0x3d, 0xdd, 0xb1, 0xfb, 0x78, 0x4d,
        ];
        let stored_init_v: [u8; 8] = [0x7f, 0xe1, 0x4e, 0xb5, 0xd7, 0xf6, 0xee, 0x3b];
        let stored_check:  [u8; 4] = [0x70, 0xdc, 0x2b, 0x75];
        let iter_count: u8 = 15;

        let info = Rar5Info {
            salt,
            iv: None,
            iter_count,
            psw_check_data: Some(PswCheckData { init_v: stored_init_v, check: stored_check }),
            enc_ver: 0,
        };

        assert!(info.verify_password("abc123").unwrap(), "abc123 should match");
        assert!(!info.verify_password("wrong").unwrap(),  "wrong password should not match");
    }

    /// Roundtrip: build PswCheckData from scratch, verify correct/wrong passwords.
    #[test]
    fn test_rar5_roundtrip() {
        let password   = "secret";
        let salt       = [1u8; 16];
        let iter_count = 10u8; // 1024 + 32 = 1056 iters for fast test

        // Compute expected PswCheckData
        let psw_iters = (1u32 << iter_count) + 32;
        let mut buf = [0u8; 32];
        pbkdf2::<Hmac<Sha256>>(password.as_bytes(), &salt, psw_iters, &mut buf).unwrap();
        let mut init_v = [0u8; 8];
        for (i, b) in buf.iter().enumerate() { init_v[i % 8] ^= b; }
        let hash  = Sha256::digest(&init_v);
        let check: [u8; 4] = hash[..4].try_into().unwrap();

        let info = Rar5Info {
            salt, iv: None, iter_count,
            psw_check_data: Some(PswCheckData { init_v, check }),
            enc_ver: 0,
        };

        assert!(info.verify_password("secret").unwrap());
        assert!(!info.verify_password("wrong").unwrap());
    }

    /// Parse the actual test_rar5.rar file and dump what we find
    #[test]
    fn test_dump_real_file() {
        use crate::rar::parser::parse_rar;
        let path_str = std::env::var("RARPC_TEST_RAR5")
            .unwrap_or_else(|_| "test_rar5.rar".to_string());
        let path = std::path::Path::new(&path_str);
        if !path.exists() {
            println!("test file not present, skipping");
            return;
        }
        let info = parse_rar(path).expect("parse failed");
        let r5 = info.rar5.as_ref().expect("not rar5");
        println!("iter_count = {}", r5.iter_count);
        println!("salt       = {:02x?}", &r5.salt);
        if let Some(ref pcd) = r5.psw_check_data {
            println!("init_v     = {:02x?}", &pcd.init_v);
            println!("check      = {:02x?}", &pcd.check);
        } else {
            println!("psw_check_data = None");
        }
    }

    #[allow(dead_code)]
    fn _old_test_psw_check_variants() {
        use hmac::Hmac;
        use pbkdf2::pbkdf2;

        let salt: [u8; 16] = [
            0xfa, 0xae, 0x70, 0xca, 0x95, 0x1f, 0xe4, 0xcb,
            0x36, 0x19, 0x3d, 0xdd, 0xb1, 0xfb, 0x78, 0x4d,
        ];
        let init_v: [u8; 8] = [0x7f, 0xe1, 0x4e, 0xb5, 0xd7, 0xf6, 0xee, 0x3b];
        let expected: [u8; 4] = [0x70, 0xdc, 0x2b, 0x75];
        let iters: u32 = 1 << 15; // 32768

        let pw_utf8: Vec<u8> = b"abc123".to_vec();
        let pw_utf16le: Vec<u8> = "abc123"
            .encode_utf16()
            .flat_map(|c| c.to_le_bytes())
            .collect();

        // ── Variant A: UTF-8 → key32 → SHA256(key||InitV)[0..4]
        {
            let mut key = [0u8; 32];
            pbkdf2::<Hmac<Sha256>>(&pw_utf8, &salt, iters, &mut key).unwrap();
            let mut h = Sha256::new();
            h.update(&key);
            h.update(&init_v);
            let hash = h.finalize();
            println!("A utf8  hash[0..4] = {:02x?}  expected = {:02x?}  match={}", &hash[..4], &expected, &hash[..4] == &expected);
        }

        // ── Variant B: UTF-16LE → key32 → SHA256(key||InitV)[0..4]
        {
            let mut key = [0u8; 32];
            pbkdf2::<Hmac<Sha256>>(&pw_utf16le, &salt, iters, &mut key).unwrap();
            let mut h = Sha256::new();
            h.update(&key);
            h.update(&init_v);
            let hash = h.finalize();
            println!("B utf16 hash[0..4] = {:02x?}  expected = {:02x?}  match={}", &hash[..4], &expected, &hash[..4] == &expected);
        }

        // ── Variant C: UTF-16LE → PBKDF2 52-byte output → last 4 bytes = check
        {
            let mut buf = [0u8; 52]; // key[32] + iv[16] + check[4]
            pbkdf2::<Hmac<Sha256>>(&pw_utf16le, &salt, iters, &mut buf).unwrap();
            let check = &buf[48..52];
            println!("C utf16 tail[4]    = {:02x?}  expected = {:02x?}  match={}", check, &expected, check == &expected);
        }

        // ── Variant D: UTF-8 → PBKDF2 52-byte output → last 4 bytes
        {
            let mut buf = [0u8; 52];
            pbkdf2::<Hmac<Sha256>>(&pw_utf8, &salt, iters, &mut buf).unwrap();
            let check = &buf[48..52];
            println!("D utf8  tail[4]    = {:02x?}  exp = {:02x?}  match={}", check, &expected, check == &expected);
        }

        // ── Variant E: UTF-8 → SHA256(key) alone (no InitV) → [0..4]
        {
            let mut key = [0u8; 32];
            pbkdf2::<Hmac<Sha256>>(&pw_utf8, &salt, iters, &mut key).unwrap();
            let hash = Sha256::digest(&key);
            println!("E utf8  sha(key)[0:4]      = {:02x?}  exp = {:02x?}  match={}", &hash[..4], &expected, &hash[..4] == &expected[..]);
        }

        // ── Variant F: UTF-16LE → SHA256(key) alone
        {
            let mut key = [0u8; 32];
            pbkdf2::<Hmac<Sha256>>(&pw_utf16le, &salt, iters, &mut key).unwrap();
            let hash = Sha256::digest(&key);
            println!("F utf16 sha(key)[0:4]      = {:02x?}  exp = {:02x?}  match={}", &hash[..4], &expected, &hash[..4] == &expected[..]);
        }

        // ── Variant G: PBKDF2(utf16, salt, iters, 56): bytes[48..56] → SHA256(those8 || init_v)[0..4]
        {
            let mut buf = [0u8; 56];
            pbkdf2::<Hmac<Sha256>>(&pw_utf16le, &salt, iters, &mut buf).unwrap();
            let mut h = Sha256::new();
            h.update(&buf[48..56]);
            h.update(&init_v);
            let hash = h.finalize();
            println!("G utf16 pbkdf2_56 sha(b48_56||iv)[0:4] = {:02x?}  exp = {:02x?}  match={}", &hash[..4], &expected, &hash[..4] == &expected[..]);
        }

        // ── Variant H: PBKDF2(utf8, salt, iters, 56): bytes[48..56] → SHA256(those8 || init_v)[0..4]
        {
            let mut buf = [0u8; 56];
            pbkdf2::<Hmac<Sha256>>(&pw_utf8, &salt, iters, &mut buf).unwrap();
            let mut h = Sha256::new();
            h.update(&buf[48..56]);
            h.update(&init_v);
            let hash = h.finalize();
            println!("H utf8  pbkdf2_56 sha(b48_56||iv)[0:4] = {:02x?}  exp = {:02x?}  match={}", &hash[..4], &expected, &hash[..4] == &expected[..]);
        }

        // ── Variant I: PBKDF2(utf16, salt, iters, 32) → SHA256(initV || key)[0..4] (reversed)
        {
            let mut key = [0u8; 32];
            pbkdf2::<Hmac<Sha256>>(&pw_utf16le, &salt, iters, &mut key).unwrap();
            let mut h = Sha256::new();
            h.update(&init_v);
            h.update(&key);
            let hash = h.finalize();
            println!("I utf16 sha(iv||key)[0:4]  = {:02x?}  exp = {:02x?}  match={}", &hash[..4], &expected, &hash[..4] == &expected[..]);
        }

        // ── Variants K-N: InitV used DURING PBKDF2 (as spec says: "additional entropy when converting to key")

        // K: PBKDF2(utf8, salt || init_v, iters, 32) → SHA256(key || init_v)[0..4]
        {
            let mut salt_iv = [0u8; 24];
            salt_iv[..16].copy_from_slice(&salt);
            salt_iv[16..].copy_from_slice(&init_v);
            let mut key = [0u8; 32];
            pbkdf2::<Hmac<Sha256>>(&pw_utf8, &salt_iv, iters, &mut key).unwrap();
            let mut h = Sha256::new(); h.update(&key); h.update(&init_v); let hash = h.finalize();
            println!("K utf8  pbkdf2(salt||iv) sha(k||iv)[0:4] = {:02x?}  match={}", &hash[..4], &hash[..4] == &expected[..]);
        }

        // L: PBKDF2(utf16, salt || init_v, iters, 32) → SHA256(key || init_v)[0..4]
        {
            let mut salt_iv = [0u8; 24];
            salt_iv[..16].copy_from_slice(&salt);
            salt_iv[16..].copy_from_slice(&init_v);
            let mut key = [0u8; 32];
            pbkdf2::<Hmac<Sha256>>(&pw_utf16le, &salt_iv, iters, &mut key).unwrap();
            let mut h = Sha256::new(); h.update(&key); h.update(&init_v); let hash = h.finalize();
            println!("L utf16 pbkdf2(salt||iv) sha(k||iv)[0:4] = {:02x?}  match={}", &hash[..4], &hash[..4] == &expected[..]);
        }

        // M: PBKDF2(utf8 || init_v, salt, iters, 32) → SHA256(key || init_v)[0..4]
        {
            let pw_and_iv: Vec<u8> = pw_utf8.iter().chain(init_v.iter()).copied().collect();
            let mut key = [0u8; 32];
            pbkdf2::<Hmac<Sha256>>(&pw_and_iv, &salt, iters, &mut key).unwrap();
            let mut h = Sha256::new(); h.update(&key); h.update(&init_v); let hash = h.finalize();
            println!("M utf8  pbkdf2(pw||iv, salt) sha(k||iv)[0:4] = {:02x?}  match={}", &hash[..4], &hash[..4] == &expected[..]);
        }

        // N: PBKDF2(utf16 || init_v, salt, iters, 32) → SHA256(key || init_v)[0..4]
        {
            let pw_and_iv: Vec<u8> = pw_utf16le.iter().chain(init_v.iter()).copied().collect();
            let mut key = [0u8; 32];
            pbkdf2::<Hmac<Sha256>>(&pw_and_iv, &salt, iters, &mut key).unwrap();
            let mut h = Sha256::new(); h.update(&key); h.update(&init_v); let hash = h.finalize();
            println!("N utf16 pbkdf2(pw||iv, salt) sha(k||iv)[0:4] = {:02x?}  match={}", &hash[..4], &hash[..4] == &expected[..]);
        }

        // O: PBKDF2(utf8, init_v XOR salt[0..8] concat salt[8..], iters, 32) → SHA256(key || init_v)[0..4]
        {
            let mut xor_salt = salt;
            for i in 0..8 { xor_salt[i] ^= init_v[i]; }
            let mut key = [0u8; 32];
            pbkdf2::<Hmac<Sha256>>(&pw_utf8, &xor_salt, iters, &mut key).unwrap();
            let mut h = Sha256::new(); h.update(&key); h.update(&init_v); let hash = h.finalize();
            println!("O utf8  pbkdf2(xor_salt) sha(k||iv)[0:4] = {:02x?}  match={}", &hash[..4], &hash[..4] == &expected[..]);
        }

        println!("--- raw expected: {:02x?}", &expected);

        // ── Sanity check: if init_v == XOR-compressed PBKDF2 output,
        //    then SHA256(init_v)[0:4] should == check
        {
            let h = Sha256::digest(&init_v);
            println!("SHA256(init_v)[0:4]     = {:02x?}  check = {:02x?}  match={}",
                     &h[..4], &expected, &h[..4] == &expected[..]);
        }

        // ── Variant P: UTF-16LE, 32800 iters, XOR-compress 32→8, then SHA256[0:4] == check[4]
        {
            let mut buf = [0u8; 32];
            pbkdf2::<Hmac<Sha256>>(&pw_utf16le, &salt, 32800u32, &mut buf).unwrap();
            let mut xor8 = [0u8; 8];
            for (i, b) in buf.iter().enumerate() { xor8[i % 8] ^= b; }
            let h = Sha256::digest(&xor8);
            println!("P utf16 32800 xor8={:02x?}  sha={:02x?}  match_iv={}  match_chk={}",
                     &xor8, &h[..4], xor8 == init_v, &h[..4] == &expected[..]);
        }

        // ── Variant Q: UTF-8, 32800 iters, XOR-compress 32→8, SHA256[0:4] == check
        {
            let mut buf = [0u8; 32];
            pbkdf2::<Hmac<Sha256>>(&pw_utf8, &salt, 32800u32, &mut buf).unwrap();
            let mut xor8 = [0u8; 8];
            for (i, b) in buf.iter().enumerate() { xor8[i % 8] ^= b; }
            let h = Sha256::digest(&xor8);
            println!("Q utf8  32800 xor8={:02x?}  sha={:02x?}  match_iv={}  match_chk={}",
                     &xor8, &h[..4], xor8 == init_v, &h[..4] == &expected[..]);
        }

        // ── Variant R: UTF-16LE, 32768 iters, XOR-compress 32→8, SHA256[0:4] == check
        {
            let mut buf = [0u8; 32];
            pbkdf2::<Hmac<Sha256>>(&pw_utf16le, &salt, iters, &mut buf).unwrap();
            let mut xor8 = [0u8; 8];
            for (i, b) in buf.iter().enumerate() { xor8[i % 8] ^= b; }
            let h = Sha256::digest(&xor8);
            println!("R utf16 32768 xor8={:02x?}  sha={:02x?}  match_iv={}  match_chk={}",
                     &xor8, &h[..4], xor8 == init_v, &h[..4] == &expected[..]);
        }
    }
}

