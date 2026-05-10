//! RAR 1.5 GPU filter ↔ CPU reference filter parity (M1 gate).
//!
//! Runs N random passwords + the correct one through both filters and asserts
//! 100% agreement. Gate: 0 disagreements over 1000 candidates.
//!
//! Requires env vars:
//!   RARPC_TEST_RAR15  — path to a RAR 1.5 archive
//!   RARPC_TEST_PW     — the correct password for that archive

use rarpc::gpu::context::GpuContext;
use rarpc::gpu::rar15_gpu::Rar15Gpu;
use rarpc::rar::parser::parse_rar;
use rarpc::rar::rar15::{rar15_filter_cpu, Rar15FilterParams};

fn random_pws(n: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(n);
    let mut s: u64 = 0xdead_beef_cafe_f00d;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let len = 8 + ((s >> 56) as usize % 3); // 8..10
        let mut pw = Vec::with_capacity(len);
        for _ in 0..len {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let c = b'a' + ((s >> 56) as u8 % 26);
            pw.push(c);
        }
        out.push(pw);
    }
    out
}

#[test]
fn gpu_cpu_filter_parity() {
    let rar_path = match std::env::var("RARPC_TEST_RAR15") {
        Ok(p) if std::path::Path::new(&p).exists() => p,
        _ => {
            eprintln!("RARPC_TEST_RAR15 not set or file missing, skipping parity test");
            return;
        }
    };
    let correct_pw = match std::env::var("RARPC_TEST_PW") {
        Ok(pw) => pw,
        _ => {
            eprintln!("RARPC_TEST_PW not set, skipping parity test");
            return;
        }
    };

    let info = parse_rar(std::path::Path::new(&rar_path)).expect("parse RAR 1.5");
    let info15 = info.rar15.as_ref().expect("RAR 1.5");

    let ctx = match GpuContext::new(0) {
        Ok(c) => c,
        Err(e) => { eprintln!("no GPU: {}, skipping", e); return; }
    };
    let params = Rar15FilterParams::default();
    let gpu = Rar15Gpu::new_for_archive(&ctx, info15, params).expect("Rar15Gpu");

    let mut pws = random_pws(1000);
    pws.push(correct_pw.into_bytes());

    let gpu_survivors: std::collections::HashSet<usize> =
        gpu.filter_batch(&pws).expect("filter_batch").into_iter().collect();

    let mut cpu_survivors: std::collections::HashSet<usize> =
        std::collections::HashSet::new();
    for (i, pw) in pws.iter().enumerate() {
        if rar15_filter_cpu(info15, pw, params) { cpu_survivors.insert(i); }
    }

    let only_gpu: Vec<_> = gpu_survivors.difference(&cpu_survivors).collect();
    let only_cpu: Vec<_> = cpu_survivors.difference(&gpu_survivors).collect();

    eprintln!("GPU survivors: {}  CPU survivors: {}", gpu_survivors.len(), cpu_survivors.len());
    eprintln!("only_gpu: {}  only_cpu: {}", only_gpu.len(), only_cpu.len());

    let correct_idx = pws.len() - 1;
    assert!(cpu_survivors.contains(&correct_idx), "correct pw not a CPU survivor");
    assert!(gpu_survivors.contains(&correct_idx), "correct pw not a GPU survivor");

    assert_eq!(only_gpu.len(), 0, "GPU-only survivors: {:?}", only_gpu);
    assert_eq!(only_cpu.len(), 0, "CPU-only survivors: {:?}", only_cpu);
}
