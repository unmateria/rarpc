//! M0 probe for RAR 1.5 Arq B — offline correctness + pass-rate sweep.
//!
//! Not part of the release. Throwaway tool that decides whether the
//! probabilistic filter approach (GPU filter → CPU strict verify) is
//! viable for a given RAR 1.5 archive.
//!
//! Gate:
//!   * **0 falsos negativos** sobre el corpus positivo (la clave correcta).
//!   * **pass-rate ≤ 10 %** sobre 100 K passwords aleatorios.
//!   * **≤ 20 μs/candidato** en CPU, si no el filter es demasiado caro.
//!
//! Run with:
//!     cargo run --release --example rar15_filter_probe -- <path-to-rar>
//!
//! Pass the correct password as the second argument, or set RARPC_TEST_PW.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use rarpc::rar::parser::parse_rar;
use rarpc::rar::rar15::Rar15Info;
use rarpc::rar::unpack15::{FilterResult, FilterStats, Unpack15};
use rarpc::rar::rar15::Rar15Cipher;

fn main() -> anyhow::Result<()> {
    let mut args = env::args().skip(1);
    let rar_path: PathBuf = match args.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: rar15_filter_probe <RAR 1.5 path> [correct_password]");
            std::process::exit(2);
        }
    };
    let correct_pw: Vec<u8> = args
        .next()
        .or_else(|| std::env::var("RARPC_TEST_PW").ok())
        .expect("pass the correct password as 2nd arg or set RARPC_TEST_PW")
        .into_bytes();

    let info_full = parse_rar(&rar_path)?;
    let info: &Rar15Info = info_full.rar15.as_ref()
        .ok_or_else(|| anyhow::anyhow!("not a RAR 1.5 archive"))?;

    println!("Archive: {}", rar_path.display());
    println!("  packed:   {} bytes", info.packed_data.len());
    println!("  unp_size: {} bytes", info.unp_size);
    println!("  method:   0x{:02x}", info.method);
    println!("  file_crc: 0x{:08x}", info.file_crc);
    println!("  correct pw: {:?}", String::from_utf8_lossy(&correct_pw));
    println!();

    // Sanity: full strict verify on the correct password must pass.
    let strict_ok = info.verify_password(&correct_pw)?;
    if !strict_ok {
        eprintln!("ERROR: strict verify_password failed for the provided correct password.");
        std::process::exit(1);
    }
    println!("✓ strict verify_password on correct pw → true");
    println!();

    // Generate 100 K random 8-10 char passwords from [a-z0-9].
    const N_WRONG: usize = 100_000;
    let wrongs = generate_wrongs(N_WRONG);
    println!("Generated {} wrong passwords (8-10 chars, [a-z0-9])", wrongs.len());
    println!();

    // First pass — distribution diagnostic: plot dest_consumed after N iters
    // for the correct pw vs a sample of wrong ones. If the distributions
    // overlap heavily we have no signal.
    println!("── Diagnostic: dest_consumed distribution after N=64 iters, K=4096 ──");
    let (_, correct_stats) = run_filter_stats(info, &correct_pw, 4096, 64);
    println!("correct pw: iters={}, dest_consumed={}, bits_consumed={}",
             correct_stats.iters_done, correct_stats.dest_consumed, correct_stats.bits_consumed);
    let mut dest_samples: Vec<i64> = Vec::with_capacity(1000);
    let mut bits_samples: Vec<u64> = Vec::with_capacity(1000);
    let mut iter_samples: Vec<usize> = Vec::with_capacity(1000);
    for pw in wrongs.iter().take(1000) {
        let (_, s) = run_filter_stats(info, pw, 4096, 64);
        dest_samples.push(s.dest_consumed);
        bits_samples.push(s.bits_consumed);
        iter_samples.push(s.iters_done);
    }
    print_histogram("wrong dest_consumed", &dest_samples, correct_stats.dest_consumed);
    print_histogram_u64("wrong bits_consumed", &bits_samples, correct_stats.bits_consumed);
    print_histogram_usize("wrong iters_done", &iter_samples, correct_stats.iters_done);
    println!();

    // Sweep (K, N, dest_max). We reject on dest_consumed > dest_max after N
    // iters (empirically the correct pw emits far fewer output bytes per
    // iteration than random streams — see diagnostic above).
    let sweep: &[(usize, usize, i64)] = &[
        // (K, N, dest_max)
        (256,  32, 80),
        (512,  32, 80),
        (512,  64, 80),
        (1024, 64, 90),
        (1024, 64, 100),
        (2048, 64, 100),
        (1024, 128, 150),
        (2048, 128, 150),
        (4096, 128, 150),
    ];

    println!("{:>6} {:>5} {:>6}  {:>12}  {:>10}  {:>10}  {:>10}",
             "K", "N", "d_max", "true_pos", "pass %", "μs/cand", "gate");
    println!("{}", "─".repeat(74));

    let mut any_gate_ok = false;
    for &(k, n, dmax) in sweep {
        let (r_correct, s_correct) = run_filter_stats(info, &correct_pw, k, n);
        let true_pos = matches!(r_correct, FilterResult::Survivor)
            && s_correct.dest_consumed <= dmax;

        let t0 = Instant::now();
        let mut survivors = 0usize;
        for pw in wrongs.iter() {
            let (v, s) = run_filter_stats(info, pw, k, n);
            if matches!(v, FilterResult::Survivor) && s.dest_consumed <= dmax {
                survivors += 1;
            }
        }
        let dt = t0.elapsed();
        let pass_rate = survivors as f64 / wrongs.len() as f64;
        let per_cand_us = dt.as_micros() as f64 / wrongs.len() as f64;

        let gate = true_pos && pass_rate <= 0.10 && per_cand_us <= 20.0;
        any_gate_ok |= gate;

        println!("{:>6} {:>5} {:>6}  {:>12}  {:>9.3}%  {:>9.2}   {:>10}",
                 k, n, dmax,
                 if true_pos { "SURVIVOR ✓" } else { "REJECT ✗" },
                 pass_rate * 100.0,
                 per_cand_us,
                 if gate { "PASS ✓" } else { "—" });
    }

    println!();
    if any_gate_ok {
        println!("✓ M0 gate met — Arq B is viable. Pick the (K, N) with the lowest pass-rate that satisfies the gate.");
    } else {
        println!("✗ M0 gate FAILED for every (K, N). Arq B is not viable. Fallback: Arq A (full cipher on GPU, strict unpack on CPU).");
    }

    Ok(())
}

fn run_filter(info: &Rar15Info, pw: &[u8], k_bytes: usize, n_iters: usize) -> FilterResult {
    run_filter_stats(info, pw, k_bytes, n_iters).0
}

fn run_filter_stats(
    info: &Rar15Info, pw: &[u8], k_bytes: usize, n_iters: usize,
) -> (FilterResult, FilterStats) {
    let take = k_bytes.min(info.packed_data.len());
    let mut stream = info.packed_data[..take].to_vec();
    Rar15Cipher::new(pw).crypt(&mut stream);
    Unpack15::filter_stats(&stream, info.unp_size as u64, stream.len(), n_iters)
}

fn print_histogram(label: &str, samples: &[i64], correct: i64) {
    let mut v = samples.to_vec();
    v.sort();
    let min = v.first().copied().unwrap_or(0);
    let max = v.last().copied().unwrap_or(0);
    let median = v[v.len() / 2];
    let mean: f64 = samples.iter().map(|&x| x as f64).sum::<f64>() / samples.len() as f64;
    let p01 = v[v.len() / 100];
    let p99 = v[v.len() * 99 / 100];
    println!(
        "  {:>22}: min={:>6} p1={:>6} median={:>6} mean={:>7.1} p99={:>6} max={:>6} | correct={}",
        label, min, p01, median, mean, p99, max, correct,
    );
}

fn print_histogram_u64(label: &str, samples: &[u64], correct: u64) {
    let signed: Vec<i64> = samples.iter().map(|&x| x as i64).collect();
    print_histogram(label, &signed, correct as i64);
}

fn print_histogram_usize(label: &str, samples: &[usize], correct: usize) {
    let signed: Vec<i64> = samples.iter().map(|&x| x as i64).collect();
    print_histogram(label, &signed, correct as i64);
}

fn generate_wrongs(n: usize) -> Vec<Vec<u8>> {
    // Deterministic xorshift so runs are reproducible.
    let mut state: u64 = 0x0123_4567_89ab_cdef;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    const ALPHA: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let len = 8 + (next() % 3) as usize; // 8..=10
        let mut pw = Vec::with_capacity(len);
        for _ in 0..len {
            let r = (next() % ALPHA.len() as u64) as usize;
            pw.push(ALPHA[r]);
        }
        out.push(pw);
    }
    out
}
