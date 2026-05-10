//! CPU fallback implementations — Rayon-parallel verification.

use rayon::prelude::*;

use crate::rar::rar3::Rar3Info;
use crate::rar::rar5::Rar5Info;
use crate::rar::rar15::Rar15Info;

// ── RAR5 ─────────────────────────────────────────────────────

/// Parallel CPU verification for RAR5 candidates.
/// Returns the index of the matching password, or None.
pub fn rar5_verify_cpu(passwords: &[Vec<u8>], info: &Rar5Info) -> Option<usize> {
    passwords.par_iter().position_any(|pw| {
        let s = String::from_utf8_lossy(pw);
        info.verify_password(&s).unwrap_or(false)
    })
}

// ── RAR3 ─────────────────────────────────────────────────────

/// Parallel CPU verification for RAR3 candidates.
pub fn rar3_verify_cpu(passwords: &[Vec<u8>], info: &Rar3Info) -> Option<usize> {
    passwords.par_iter().position_any(|pw| {
        let s = String::from_utf8_lossy(pw);
        info.verify_password(&s).unwrap_or(false)
    })
}

// ── RAR 1.5 ───────────────────────────────────────────────────

/// Parallel CPU verification for RAR 1.5 candidates. Each thread decrypts
/// a fresh copy of `packed_data` and feeds it through `Unpack15` before
/// matching CRC32 — GPU path is deferred (divergent LZH state is a poor
/// fit for warp execution).
pub fn rar15_verify_cpu(passwords: &[Vec<u8>], info: &Rar15Info) -> Option<usize> {
    passwords.par_iter().position_any(|pw| info.verify_password(pw).unwrap_or(false))
}

// ── Test vectors ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use hmac::Hmac;
    use pbkdf2::pbkdf2;

    #[test]
    fn test_rar5_pbkdf2_non_trivial() {
        let pw = b"test";
        let salt = [0u8; 16];
        let mut key = [0u8; 32];
        pbkdf2::<Hmac<sha2::Sha256>>(pw, &salt, 1, &mut key).unwrap();
        assert_ne!(key, [0u8; 32]);
    }

    #[test]
    fn test_rar5_roundtrip() {
        // Build a synthetic Rar5Info with a known password, verify it passes.
        // Uses the correct WinRAR formula:
        //   buf  = PBKDF2(utf8_pw, salt, (1<<cnt)+32, 32)
        //   xor8 = fold buf cyclically into 8 bytes
        //   init_v = xor8,  check = SHA256(xor8)[0..4]
        use crate::rar::rar5::PswCheckData;
        use sha2::Digest;

        let password   = "secret";
        let salt       = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let iter_count = 10u8; // 1024+32 iters — fast for tests

        let psw_iters = (1u32 << iter_count) + 32;
        let mut buf = [0u8; 32];
        pbkdf2::<Hmac<sha2::Sha256>>(password.as_bytes(), &salt, psw_iters, &mut buf).unwrap();
        let mut init_v = [0u8; 8];
        for (i, b) in buf.iter().enumerate() { init_v[i % 8] ^= b; }
        let hash  = sha2::Sha256::digest(&init_v);
        let check: [u8; 4] = hash[..4].try_into().unwrap();

        let info = Rar5Info {
            salt, iv: None, iter_count,
            psw_check_data: Some(PswCheckData { init_v, check }),
            enc_ver: 0,
        };

        assert!(info.verify_password("secret").unwrap());
        assert!(!info.verify_password("wrong").unwrap());
    }

    #[test]
    fn test_rar3_kdf_basic() {
        let (key, iv) = Rar3Info::derive_key("test", &[0u8; 8]);
        assert_eq!(key.len(), 16);
        assert_eq!(iv.len(), 16);
        assert_ne!(key, [0u8; 16]);
    }

    #[test]
    fn test_rar5_cpu_verify() {
        use crate::rar::rar5::PswCheckData;
        use sha2::Digest;

        let salt       = [7u8; 16];
        let iter_count = 10u8; // 1024+32 iters — fast for tests

        // Build PswCheckData for "hello" using the correct formula
        let psw_iters = (1u32 << iter_count) + 32;
        let mut buf = [0u8; 32];
        pbkdf2::<Hmac<sha2::Sha256>>(b"hello", &salt, psw_iters, &mut buf).unwrap();
        let mut init_v = [0u8; 8];
        for (i, b) in buf.iter().enumerate() { init_v[i % 8] ^= b; }
        let hash  = sha2::Sha256::digest(&init_v);
        let check: [u8; 4] = hash[..4].try_into().unwrap();

        let info = Rar5Info {
            salt, iv: None, iter_count,
            psw_check_data: Some(PswCheckData { init_v, check }),
            enc_ver: 0,
        };

        let candidates: Vec<Vec<u8>> = vec![
            b"bad1".to_vec(), b"hello".to_vec(), b"bad2".to_vec(),
        ];
        let result = rar5_verify_cpu(&candidates, &info);
        assert_eq!(result, Some(1));
    }
}
