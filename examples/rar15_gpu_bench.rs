//! RAR 1.5 GPU filter throughput bench.

use std::time::Instant;
use rarpc::gpu::context::GpuContext;
use rarpc::gpu::rar15_gpu::Rar15Gpu;
use rarpc::rar::parser::parse_rar;
use rarpc::rar::rar15::Rar15FilterParams;

fn main() {
    let path = std::env::args().nth(1)
        .or_else(|| std::env::var("RARPC_TEST_RAR15").ok())
        .expect("pass RAR 1.5 path as arg or set RARPC_TEST_RAR15");
    let info = parse_rar(std::path::Path::new(&path)).unwrap();
    let info15 = info.rar15.as_ref().unwrap();
    let ctx = GpuContext::new(0).unwrap();
    let gpu = Rar15Gpu::new_for_archive(&ctx, info15, Rar15FilterParams::default()).unwrap();

    // 100k random passwords
    let n = 1_000_000usize;
    let mut pws = Vec::with_capacity(n);
    let mut s: u64 = 0x12345678_9abcdef0;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let len = 8 + ((s >> 56) as usize % 3);
        let mut pw = Vec::with_capacity(len);
        for _ in 0..len {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            pw.push(b'a' + ((s >> 56) as u8 % 26));
        }
        pws.push(pw);
    }

    // Warmup
    let _ = gpu.filter_batch(&pws[..1000]).unwrap();

    // Bench
    let t0 = Instant::now();
    let mut total_survivors = 0usize;
    let iters = 5;
    for _ in 0..iters {
        let s = gpu.filter_batch(&pws).unwrap();
        total_survivors += s.len();
    }
    let dt = t0.elapsed();
    let total = (n * iters) as f64;
    let kh = total / dt.as_secs_f64() / 1000.0;
    println!("{} cand in {:.2}s → {:.1} KH/s", n * iters, dt.as_secs_f64(), kh);
    println!("survivors: {} / {} ({:.3}%)",
        total_survivors, n * iters, 100.0 * total_survivors as f64 / total);
}
