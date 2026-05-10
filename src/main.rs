use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::time::Instant;

use rarpc::attack::{AttackConfig, AttackMode};
use rarpc::attack::engine::Engine;
use rarpc::gpu::context::GpuContext;
use rarpc::rar::parser::parse_rar;
use rarpc::session::Session;

// ── CLI definition ───────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "rarpc",
    version = "0.1.0",
    about = "RAR password cracker — CUDA GPU acceleration",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Crack a RAR archive password
    Crack(CrackArgs),
    /// Benchmark GPU hash rate
    Bench(BenchArgs),
    /// List available GPUs
    Gpus,
    /// Train a character trigram Markov model from a password corpus (rockyou, etc).
    TrainMarkov(TrainMarkovArgs),
    /// Measure candidate-count curve for a generator against a target list
    /// (time-to-crack in units of candidates, not seconds).
    BenchTtc(BenchTtcArgs),
}

#[derive(clap::Args)]
struct CrackArgs {
    /// Path to the RAR file
    #[arg(value_name = "FILE")]
    file: PathBuf,

    /// Wordlist file (dictionary attack)
    #[arg(short, long, value_name = "DICT")]
    wordlist: Option<PathBuf>,

    /// Brute-force attack
    #[arg(short, long)]
    brute: bool,

    /// Mask attack pattern (e.g. ?l?l?d?d)
    #[arg(short, long, value_name = "MASK")]
    mask: Option<String>,

    /// Path to a rulefile (used together with --wordlist).
    #[arg(long, value_name = "RULES")]
    rules: Option<PathBuf>,

    /// Path to a Markov model (trained via `rarpc train-markov`). Mutually
    /// exclusive with --wordlist / --brute / --mask.
    #[arg(long, value_name = "MODEL")]
    markov: Option<PathBuf>,

    /// Character set for brute force (default: lowercase+digits)
    #[arg(short, long, default_value = "abcdefghijklmnopqrstuvwxyz0123456789")]
    charset: String,

    /// Minimum password length
    #[arg(long, default_value_t = 1)]
    min_len: usize,

    /// Maximum password length
    #[arg(long, default_value_t = 8)]
    max_len: usize,

    /// GPU device index (default: 0)
    #[arg(short = 'g', long, default_value_t = 0)]
    gpu: usize,

    /// Number of GPUs to use (0 = all available)
    #[arg(long, default_value_t = 1)]
    num_gpus: usize,

    /// Batch size (passwords per GPU launch, power of 2 recommended)
    #[arg(long, default_value_t = 1 << 20)]
    batch: u32,

    /// Save/restore session file
    #[arg(short, long, value_name = "SESSION")]
    session: Option<PathBuf>,

    /// CPU-only fallback (no CUDA required)
    #[arg(long)]
    cpu_only: bool,

    /// Disable the 2-stream RAR5 pipeline (overlap CPU↔GPU work).
    /// Equivalent to `RARPC_PIPELINE=0`. Useful for A/B measurement.
    #[arg(long)]
    no_pipeline: bool,
}

#[derive(clap::Args)]
struct BenchArgs {
    /// GPU device index
    #[arg(short = 'g', long, default_value_t = 0)]
    gpu: usize,

    /// RAR version to benchmark (3 or 5)
    #[arg(long, default_value_t = 5)]
    rar_version: u8,

    /// Duration in seconds
    #[arg(long, default_value_t = 10)]
    duration: u64,

    /// Profiling mode: run a single fixed-size batch (no timing loop).
    /// Provides Nsight Compute with a deterministic kernel region.
    #[arg(long)]
    profile: bool,

    /// Batch size in profile mode. Default 16K keeps each launch under ~200ms
    /// so Windows WDDM TDR doesn't kill it and Nsight Compute can replay.
    #[arg(long, default_value_t = 1 << 14)]
    profile_batch: u32,

    /// RAR5 only: run two timed benches back-to-back (sequential then pipelined)
    /// and print the KH/s ratio. Used to validate Experiment 3 (CPU↔GPU overlap).
    #[arg(long)]
    pipelined_compare: bool,
}

#[derive(clap::Args)]
struct TrainMarkovArgs {
    /// Corpus file — one password per line (UTF-8).
    #[arg(value_name = "CORPUS")]
    corpus: PathBuf,

    /// Output model file (bincode).
    #[arg(value_name = "OUT")]
    output: PathBuf,
}

#[derive(clap::Args)]
struct BenchTtcArgs {
    /// File with one target password per line.
    #[arg(long)]
    targets: PathBuf,

    /// Upper bound on candidates to try per target (safety guard).
    #[arg(long, default_value_t = 10_000_000)]
    max_candidates: u64,

    /// Generator mode: "brute" | "rules" | "markov".
    #[arg(long)]
    mode: String,

    /// `--mode brute` / `rules`: charset (brute) or wordlist (rules).
    #[arg(long)]
    wordlist: Option<PathBuf>,

    /// `--mode rules`: ruleset file.
    #[arg(long)]
    rules: Option<PathBuf>,

    /// `--mode markov`: model file.
    #[arg(long)]
    markov: Option<PathBuf>,

    /// `--mode brute`: charset.
    #[arg(long, default_value = "abcdefghijklmnopqrstuvwxyz0123456789")]
    charset: String,

    #[arg(long, default_value_t = 1)]
    min_len: usize,

    #[arg(long, default_value_t = 8)]
    max_len: usize,
}

// ── Main ─────────────────────────────────────────────────────

fn main() -> Result<()> {
    // Launch GUI when invoked with no arguments
    if std::env::args().len() == 1 {
        return rarpc::gui::run_gui();
    }

    let cli = Cli::parse();

    match cli.command {
        Commands::Gpus => cmd_list_gpus(),
        Commands::Bench(args) => cmd_bench(args),
        Commands::Crack(args) => cmd_crack(args),
        Commands::TrainMarkov(args) => cmd_train_markov(args),
        Commands::BenchTtc(args) => cmd_bench_ttc(args),
    }
}

/// Bench time-to-crack in *candidate count* units. Reads a target list, runs
/// the requested generator, and reports how many candidates each target takes
/// to find. Output: median + mean + coverage at max_candidates cap.
///
/// This is purely a generator-quality metric — no GPU, no real .rar files.
/// Verification is byte-equality against the target string (faster to iterate
/// than building synthetic PswCheckData per target).
fn cmd_bench_ttc(args: BenchTtcArgs) -> Result<()> {
    use rarpc::attack::bruteforce::BruteForce;
    use rarpc::attack::markov::{MarkovModel, MarkovSource};
    use rarpc::attack::rules::{load_rules, RulesSource};
    use rarpc::attack::source::BatchSource;
    use rarpc::attack::wordlist::Wordlist;
    use std::collections::HashSet;
    use std::io::BufRead;

    let targets: HashSet<Vec<u8>> = {
        let f = std::fs::File::open(&args.targets)?;
        std::io::BufReader::new(f).lines()
            .filter_map(|l| l.ok())
            .map(|s| s.trim_end_matches(['\r','\n']).as_bytes().to_vec())
            .filter(|v| !v.is_empty())
            .collect()
    };
    if targets.is_empty() {
        bail!("no targets in {:?}", args.targets);
    }
    println!("Targets: {}", targets.len());

    let mut src: Box<dyn BatchSource> = match args.mode.as_str() {
        "brute" => Box::new(BruteForce::new(&args.charset, args.min_len, args.max_len)),
        "rules" => {
            let wl = args.wordlist.as_ref()
                .context("--rules mode requires --wordlist")?;
            let rp = args.rules.as_ref()
                .context("--rules mode requires --rules")?;
            Box::new(RulesSource::new(Wordlist::open(wl)?, load_rules(rp)?))
        }
        "markov" => {
            let m = args.markov.as_ref()
                .context("--markov mode requires --markov")?;
            Box::new(MarkovSource::new(MarkovModel::load(m)?, args.min_len, args.max_len))
        }
        other => bail!("unknown --mode {:?}", other),
    };

    let mut remaining: HashSet<Vec<u8>> = targets.clone();
    let mut hit_positions: Vec<u64> = Vec::new();
    let mut batch: Vec<Vec<u8>> = Vec::new();
    let mut pos: u64 = 0;

    'outer: while pos < args.max_candidates && !remaining.is_empty() {
        let got = src.next_batch(&mut batch, 8192)?;
        if got == 0 { break; }
        for pw in &batch {
            pos += 1;
            if remaining.remove(pw) {
                hit_positions.push(pos);
                if remaining.is_empty() { break 'outer; }
            }
            if pos >= args.max_candidates { break 'outer; }
        }
        if src.is_exhausted() { break; }
    }

    hit_positions.sort_unstable();
    let found  = hit_positions.len();
    let total  = targets.len();
    let median = hit_positions.get(found / 2).copied().unwrap_or(0);
    let mean: f64 = if found == 0 { 0.0 } else {
        hit_positions.iter().sum::<u64>() as f64 / found as f64
    };

    println!("mode           = {}", args.mode);
    println!("candidates     = {}", pos);
    println!("coverage       = {}/{}  ({:.1}%)", found, total, found as f64 * 100.0 / total as f64);
    println!("median tries   = {}", median);
    println!("mean tries     = {:.0}", mean);
    Ok(())
}

fn cmd_train_markov(args: TrainMarkovArgs) -> Result<()> {
    use rarpc::attack::markov::MarkovModel;
    println!("Training char trigram model from {:?}", args.corpus);
    let model = MarkovModel::train(&args.corpus)?;
    println!(
        "  alphabet = {} chars, total params = {}",
        model.alphabet.len(),
        model.logp_start.len() + model.logp_after1.len() + model.logp_full.len(),
    );
    model.save(&args.output)?;
    println!("Saved model → {:?}", args.output);
    Ok(())
}

fn cmd_list_gpus() -> Result<()> {
    let gpus = GpuContext::enumerate()?;
    if gpus.is_empty() {
        println!("No CUDA-capable GPUs found.");
        return Ok(());
    }
    println!("{:>4}  {:40}  {:>6}", "Idx", "Name", "VRAM");
    println!("{}", "-".repeat(56));
    for g in &gpus {
        println!("{:>4}  {:40}  {:>5}M", g.index, g.name, g.vram_mb);
    }
    Ok(())
}

fn cmd_bench(args: BenchArgs) -> Result<()> {
    use std::time::Duration;

    let ctx = GpuContext::new(args.gpu)
        .with_context(|| format!("Cannot open GPU {}", args.gpu))?;

    // ── Profile mode: single deterministic launch for Nsight Compute ─
    if args.profile && args.rar_version == 3 {
        use rarpc::gpu::rar3_gpu::Rar3Gpu;
        let batch = args.profile_batch as usize;
        println!("Profile mode: one RAR3 launch of {} candidates on GPU {}", batch, args.gpu);
        let fake_passwords: Vec<Vec<u8>> = (0..batch)
            .map(|i| format!("{:08x}", i).into_bytes())
            .collect();
        let gpu3 = Rar3Gpu::new(&ctx)?;
        let salt = [0u8; 8];
        let enc = [0u8; 16];
        let _ = gpu3.crack_batch(&fake_passwords, &salt, &enc, 2, 0, 0, 0)?;
        let start = Instant::now();
        let _ = gpu3.crack_batch(&fake_passwords, &salt, &enc, 2, 0, 0, 0)?;
        let elapsed = start.elapsed().as_secs_f64();
        println!("Measured: {} cand in {:.4}s ({:.0} H/s)", batch, elapsed, batch as f64 / elapsed);
        return Ok(());
    }
    if args.profile {
        if args.rar_version != 5 {
            bail!("--profile is currently only supported for RAR3 and RAR5");
        }
        use rarpc::gpu::rar5_gpu::Rar5Gpu;
        use rarpc::rar::rar5::{PswCheckData, Rar5Info};

        let batch = args.profile_batch as usize;
        println!(
            "Profile mode: one RAR5 launch of {} candidates on GPU {}",
            batch, args.gpu
        );

        let fake_passwords: Vec<Vec<u8>> = (0..batch)
            .map(|i| format!("{:08x}", i).into_bytes())
            .collect();

        // Synthetic Rar5Info. init_v is all-ones so real passwords never
        // match — the kernel runs the full PBKDF2 chain for every candidate.
        let info = Rar5Info {
            salt: [0u8; 16],
            iv: None,
            iter_count: 15, // 32768 AES iters, +32 for PswCheck
            psw_check_data: Some(PswCheckData {
                init_v: [0xffu8; 8],
                check: [0xffu8; 4],
            }),
            enc_ver: 0,
        };

        let gpu5 = Rar5Gpu::new(&ctx)?;

        // Warm-up launch — makes sure JIT, allocators, etc. are settled
        // before the measured launch.
        let _ = gpu5.crack_batch(&fake_passwords, &info)?;

        let start = Instant::now();
        let _ = gpu5.crack_batch(&fake_passwords, &info)?;
        let elapsed = start.elapsed().as_secs_f64();
        let rate = batch as f64 / elapsed;
        println!(
            "Measured launch: {} candidates in {:.4}s ({:.0} H/s)",
            batch, elapsed, rate
        );
        return Ok(());
    }

    println!("Benchmarking GPU {} for RAR{}...", args.gpu, args.rar_version);

    let batch = 1u32 << 20;
    let duration = Duration::from_secs(args.duration);

    // Generate dummy passwords for benchmark
    let fake_passwords: Vec<Vec<u8>> = (0..batch as usize)
        .map(|i| format!("{:08x}", i).into_bytes())
        .collect();

    // GPU object creation (PTX load) is hoisted before the timing window
    // so the benchmark measures only kernel dispatch throughput.
    if args.rar_version == 5 {
        use rarpc::gpu::rar5_gpu::Rar5Gpu;
        use rarpc::rar::rar5::{PswCheckData, Rar5Info};
        let gpu5 = Rar5Gpu::new(&ctx)?;
        let info = Rar5Info {
            salt: [0u8; 16],
            iv: None,
            iter_count: 15,
            psw_check_data: Some(PswCheckData {
                init_v: [0u8; 8],
                check: [0xffu8; 4],
            }),
            enc_ver: 0,
        };

        if args.pipelined_compare {
            let seq  = bench_seq_rar5(&gpu5, &info, &fake_passwords, duration)?;
            let pipe = bench_pipelined_rar5(&gpu5, &ctx, &info, &fake_passwords, duration)?;
            let ratio = pipe / seq;
            println!("RAR5 sequential : {:.2} KH/s", seq / 1000.0);
            println!("RAR5 pipelined  : {:.2} KH/s", pipe / 1000.0);
            println!("ratio (pipe/seq): {:.3}x  ({:+.1}%)", ratio, (ratio - 1.0) * 100.0);
            return Ok(());
        }

        let start = Instant::now();
        let mut total = 0u64;
        while start.elapsed() < duration {
            let _ = gpu5.crack_batch(&fake_passwords, &info)?;
            total += batch as u64;
        }
        let elapsed = start.elapsed().as_secs_f64();
        let rate = total as f64 / elapsed;
        println!("RAR5: {:.0} H/s  ({:.2} KH/s)", rate, rate / 1000.0);
    } else if args.rar_version == 3 {
        use rarpc::gpu::rar3_gpu::Rar3Gpu;
        let gpu3 = Rar3Gpu::new(&ctx)?;
        let salt = [0u8; 8];
        let enc = [0u8; 16];
        let start = Instant::now();
        let mut total = 0u64;
        while start.elapsed() < duration {
            let _ = gpu3.crack_batch(&fake_passwords, &salt, &enc, 2, 0, 0, 0)?;
            total += batch as u64;
        }
        let elapsed = start.elapsed().as_secs_f64();
        let rate = total as f64 / elapsed;
        println!("RAR3: {:.0} H/s  ({:.2} KH/s)", rate, rate / 1000.0);
    } else if args.rar_version == 15 {
        use rarpc::gpu::rar15_gpu::Rar15Gpu;
        use rarpc::rar::rar15::{Rar15FilterParams, Rar15Info};
        let info15 = Rar15Info {
            packed_data: vec![0u8; 512],
            unp_size: 28843,
            file_crc: 0xFFFFFFFF,
            unp_ver: 0x0f,
            method: 0x33,
        };
        let gpu15 = Rar15Gpu::new_for_archive(&ctx, &info15, Rar15FilterParams::default())?;
        let start = Instant::now();
        let mut total = 0u64;
        while start.elapsed() < duration {
            let _ = gpu15.filter_batch(&fake_passwords)?;
            total += batch as u64;
        }
        let elapsed = start.elapsed().as_secs_f64();
        let rate = total as f64 / elapsed;
        println!("RAR1.5: {:.0} H/s  ({:.2} MH/s)", rate, rate / 1_000_000.0);
    } else {
        bail!("Unsupported --rar-version {}. Use 3, 5, or 15.", args.rar_version);
    }

    Ok(())
}

fn cmd_crack(args: CrackArgs) -> Result<()> {
    if args.no_pipeline {
        rarpc::attack::engine::force_disable_pipeline();
    }

    // Parse the RAR file
    let rar_info = parse_rar(&args.file)
        .with_context(|| format!("Failed to parse {:?}", args.file))?;

    println!("RAR version : {:?}", rar_info.version);
    println!("Encryption  : {}", rar_info.encryption_name());
    println!("File        : {:?}", args.file);

    // Determine attack mode
    let mode = if let Some(model) = &args.markov {
        AttackMode::Markov {
            model: model.clone(),
            min_len: args.min_len,
            max_len: args.max_len,
        }
    } else if let (Some(wl), Some(rules)) = (&args.wordlist, &args.rules) {
        AttackMode::Rules { wordlist: wl.clone(), rules: rules.clone() }
    } else if let Some(wl) = &args.wordlist {
        AttackMode::Wordlist(wl.clone())
    } else if let Some(mask) = &args.mask {
        AttackMode::Mask(mask.clone())
    } else if args.brute {
        AttackMode::BruteForce {
            charset: args.charset.clone(),
            min_len: args.min_len,
            max_len: args.max_len,
        }
    } else {
        bail!("Specify an attack mode: --wordlist, --brute, --mask, --markov, or --wordlist + --rules");
    };

    let config = AttackConfig {
        mode,
        batch_size: args.batch,
        gpu_index: args.gpu,
        num_gpus: args.num_gpus,
        cpu_only: args.cpu_only,
    };

    // Load or create session
    let mut session = if let Some(ref sp) = args.session {
        if sp.exists() {
            Session::load(sp).with_context(|| "Failed to load session")?
        } else {
            Session::new(&args.file, &config)
        }
    } else {
        Session::new(&args.file, &config)
    };

    // Progress bar
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.cyan} [{elapsed_precise}] {pos} tried | {per_sec} | {msg}",
        )
        .unwrap()
        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ "),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let start = Instant::now();

    // Build and run the engine
    let mut engine = Engine::new(rar_info, config, session.clone())?;

    let session_path = args.session.clone();
    let batch_size = args.batch as u64;
    let mut last_save = 0u64;

    match engine.run(|tried, candidate| {
        pb.set_position(tried);
        pb.set_message(format!("testing: {}", candidate));
        if let Some(ref sp) = session_path {
            if tried - last_save >= batch_size * 10 {
                session.position = tried;
                let _ = session.save(sp);
                last_save = tried;
            }
        }
        true // keep going
    })? {
        Some(password) => {
            pb.finish_and_clear();
            let elapsed = start.elapsed();
            println!("\n✓ Password found: {}", password);
            println!(
                "  Tried {} candidates in {:.1}s ({:.0} H/s)",
                engine.tried(),
                elapsed.as_secs_f64(),
                engine.tried() as f64 / elapsed.as_secs_f64()
            );
            session.found_password = Some(password);
        }
        None => {
            pb.finish_and_clear();
            println!("\n✗ Password not found in the given search space.");
        }
    }

    // Save session if requested
    if let Some(ref sp) = args.session {
        session.position = engine.tried();
        session.save(sp)?;
    }

    Ok(())
}

// ── Bench helpers (Exp 3) ────────────────────────────────────

/// Sequential RAR5 bench: single-stream crack_batch loop. Returns H/s.
fn bench_seq_rar5(
    gpu5: &rarpc::gpu::rar5_gpu::Rar5Gpu,
    info: &rarpc::rar::rar5::Rar5Info,
    passwords: &[Vec<u8>],
    duration: std::time::Duration,
) -> Result<f64> {
    let batch = passwords.len() as u64;
    // Warm-up
    let _ = gpu5.crack_batch(passwords, info)?;
    let start = Instant::now();
    let mut total = 0u64;
    while start.elapsed() < duration {
        let _ = gpu5.crack_batch(passwords, info)?;
        total += batch;
    }
    let elapsed = start.elapsed().as_secs_f64();
    Ok(total as f64 / elapsed)
}

/// Pipelined RAR5 bench: two streams, upload+launch of batch N+1 while kernel N
/// is still running. Each iteration clones the host passwords (owned by
/// InFlight) so CPU work parity matches the real pipelined engine.
/// Returns H/s.
fn bench_pipelined_rar5(
    gpu5: &rarpc::gpu::rar5_gpu::Rar5Gpu,
    ctx: &rarpc::gpu::context::GpuContext,
    info: &rarpc::rar::rar5::Rar5Info,
    passwords: &[Vec<u8>],
    duration: std::time::Duration,
) -> Result<f64> {
    let batch = passwords.len() as u64;
    let stream_a = gpu5.stream().clone();
    let stream_b = ctx.new_stream()?;
    let streams = [stream_a, stream_b];
    let mut s = 0usize;

    // Warm-up on stream 0
    {
        let _ = gpu5.crack_batch(passwords, info)?;
    }

    let start = Instant::now();
    let mut total = 0u64;

    // Seed
    let seed_pws = passwords.to_vec();
    let mut current = {
        let mut ifl = gpu5.upload_async(seed_pws, info, streams[s].clone())?
            .expect("non-empty batch with pw check data");
        gpu5.launch_async(&mut ifl)?;
        s ^= 1;
        Some(ifl)
    };

    while let Some(running) = current.take() {
        let now = start.elapsed();
        // Launch the next batch while `running` is still on the GPU.
        let next = if now < duration {
            let pws = passwords.to_vec();
            let mut ifl = gpu5.upload_async(pws, info, streams[s].clone())?
                .expect("non-empty batch");
            gpu5.launch_async(&mut ifl)?;
            s ^= 1;
            Some(ifl)
        } else {
            None
        };

        // Sync + fetch `running`.
        let _ = gpu5.fetch_result(running)?;
        total += batch;

        current = next;
        if current.is_none() { break; }
    }

    let elapsed = start.elapsed().as_secs_f64();
    Ok(total as f64 / elapsed)
}
