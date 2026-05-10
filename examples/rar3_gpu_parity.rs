//! RAR3 GPU KDF ↔ CPU KDF parity test.
//!
//! Exercises the `rar3_kdf_dump` debug kernel against `Rar3Info::derive_key`
//! over a batch of random short-ASCII passwords. Gate: 100% byte-equal
//! key[16]+iv[16] pairs.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use rarpc::gpu::batch::pack_passwords;
use rarpc::gpu::context::GpuContext;
use rarpc::rar::rar3::Rar3Info;

fn main() {
    let ctx = match GpuContext::new(0) {
        Ok(c) => c,
        Err(e) => { eprintln!("no GPU: {}, skipping", e); return; }
    };
    let module = ctx.ctx.load_module(Ptx::from_src(rarpc::RAR3_PTX)).unwrap();
    let func = module.load_function("rar3_kdf_dump").unwrap();

    let salt: [u8; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];

    // Generate passwords: start with some known cases, then random 4-10 char ASCII.
    let mut pws_ascii: Vec<String> = vec![
        "a".into(), "ab".into(), "test".into(), "hello".into(),
        "test123".into(), "abcdef".into(), "12345678".into(), "pass".into(),
    ];
    let mut s: u64 = 0xcafebabedeadbeef;
    for _ in 0..92 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let len = 4 + ((s >> 56) as usize % 7);
        let mut pw = String::with_capacity(len);
        for _ in 0..len {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            pw.push((b'a' + ((s >> 56) as u8 % 26)) as char);
        }
        pws_ascii.push(pw);
    }

    // Encode each password as UTF-16LE bytes (matches host packing convention).
    let pws_utf16: Vec<Vec<u8>> = pws_ascii.iter()
        .map(|s| s.encode_utf16().flat_map(|c| c.to_le_bytes()).collect::<Vec<u8>>())
        .collect();

    let n = pws_utf16.len() as i32;
    let (flat, lengths) = pack_passwords(&pws_utf16);

    let d_pw  = ctx.stream.clone_htod(&flat).unwrap();
    let d_len = ctx.stream.clone_htod(&lengths).unwrap();
    let d_salt = ctx.stream.clone_htod(salt.as_slice()).unwrap();
    let host_out = vec![0u8; pws_utf16.len() * 32];
    let mut d_out = ctx.stream.clone_htod(&host_out).unwrap();

    const BLOCK: u32 = 64;
    let grid = ((n as u32) + BLOCK - 1) / BLOCK;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut args = ctx.stream.launch_builder(&func);
    args.arg(&d_pw);
    args.arg(&d_len);
    args.arg(&n);
    args.arg(&d_salt);
    args.arg(&mut d_out);
    unsafe { args.launch(cfg) }.unwrap();
    ctx.stream.synchronize().unwrap();

    let gpu_out: Vec<u8> = ctx.stream.clone_dtoh(&d_out).unwrap();

    let mut mismatches = 0usize;
    let mut first_bad = None;
    for (i, pw) in pws_ascii.iter().enumerate() {
        let (cpu_key, cpu_iv) = Rar3Info::derive_key(pw, &salt);
        let gpu_slice = &gpu_out[i * 32 .. i * 32 + 32];
        let gpu_key = &gpu_slice[0..16];
        let gpu_iv  = &gpu_slice[16..32];
        if gpu_key != cpu_key || gpu_iv != cpu_iv {
            mismatches += 1;
            if first_bad.is_none() {
                first_bad = Some((i, pw.clone(), cpu_key, cpu_iv, gpu_key.to_vec(), gpu_iv.to_vec()));
            }
        }
    }

    if let Some((i, pw, ck, civ, gk, gi)) = first_bad {
        println!("FIRST MISMATCH at index {} pw='{}'", i, pw);
        println!("  CPU key: {:02x?}", ck);
        println!("  GPU key: {:02x?}", gk);
        println!("  CPU iv:  {:02x?}", civ);
        println!("  GPU iv:  {:02x?}", gi);
    }
    println!("Total: {} mismatches out of {}", mismatches, pws_ascii.len());
    if mismatches > 0 {
        std::process::exit(1);
    }
    println!("PARITY OK ✓");
}
