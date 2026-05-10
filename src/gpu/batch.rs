//! Batch packing utilities shared by rar3_gpu and rar5_gpu.
//!
//! Passwords are stored in a flat, fixed-stride layout:
//!   passwords[tid * MAX_PW_BYTES .. tid * MAX_PW_BYTES + len[tid]]
//! so that CUDA threads can access them with aligned, coalesced reads.

pub const MAX_PW_BYTES: usize = 256;

/// Pack a slice of variable-length byte strings into a flat fixed-stride buffer.
/// Returns (flat_buffer, lengths) ready for H→D copy.
pub fn pack_passwords(passwords: &[Vec<u8>]) -> (Vec<u8>, Vec<i32>) {
    let n = passwords.len();
    let mut flat = vec![0u8; n * MAX_PW_BYTES];
    let mut lengths = vec![0i32; n];

    for (i, pw) in passwords.iter().enumerate() {
        let len = pw.len().min(MAX_PW_BYTES);
        flat[i * MAX_PW_BYTES .. i * MAX_PW_BYTES + len].copy_from_slice(&pw[..len]);
        lengths[i] = len as i32;
    }

    (flat, lengths)
}
