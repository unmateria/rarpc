use anyhow::Result;
use std::sync::Arc;

use cudarc::driver::CudaStream;

use crate::attack::{AttackConfig, AttackMode};
use crate::attack::bruteforce::BruteForce;
use crate::attack::combinator::CombinatorSource;
use crate::attack::markov::{MarkovModel, MarkovSource};
use crate::attack::mask::MaskAttack;
use crate::attack::rules::{load_rules, RulesSource};
use crate::attack::source::BatchSource;
use crate::attack::wordlist::Wordlist;
use crate::cpu::fallback;
use crate::gpu::context::GpuContext;
use crate::gpu::rar5_gpu::Rar5Gpu;
use crate::gpu::rar3_gpu::Rar3Gpu;
use crate::gpu::rar15_gpu::Rar15Gpu;
use crate::rar::rar3::Rar3CheckMode;
use crate::rar::rar15::Rar15FilterParams;
use crate::rar::{RarInfo, RarVersion};
use crate::session::Session;

pub struct Engine {
    rar_info: RarInfo,
    config:   AttackConfig,
    tried:    u64,
    gpu5:     Option<Rar5Gpu>,
    gpu3:     Option<Rar3Gpu>,
    gpu15:    Option<Rar15Gpu>,
    /// Second stream, allocated on-demand when the RAR5 pipelined path runs.
    stream_b: Option<Arc<CudaStream>>,
    gpu_ctx:  Option<Arc<GpuContext>>,
    pipelined: bool,
}

impl Engine {
    pub fn new(rar_info: RarInfo, config: AttackConfig, session: Session) -> Result<Self> {
        let pipelined = pipeline_enabled();
        let (gpu5, gpu3, gpu15, gpu_ctx) = if config.cpu_only {
            (None, None, None, None)
        } else {
            match GpuContext::new(config.gpu_index) {
                Ok(ctx) => {
                    println!("GPU {}: {}", ctx.info.index, ctx.info.name);
                    let g5  = Rar5Gpu::new(&ctx).ok();
                    let g3  = Rar3Gpu::new(&ctx).ok();
                    let g15 = match (&rar_info.version, rar_info.rar15.as_ref()) {
                        (RarVersion::Rar15, Some(info15)) => {
                            Rar15Gpu::new_for_archive(&ctx, info15, Rar15FilterParams::default()).ok()
                        }
                        _ => None,
                    };
                    (g5, g3, g15, Some(Arc::new(ctx)))
                }
                Err(e) => {
                    eprintln!("Warning: CUDA unavailable ({}), falling back to CPU.", e);
                    (None, None, None, None)
                }
            }
        };
        Ok(Self {
            rar_info,
            config,
            tried: session.position,
            gpu5,
            gpu3,
            gpu15,
            stream_b: None,
            gpu_ctx,
            pipelined,
        })
    }

    pub fn tried(&self) -> u64 { self.tried }

    /// Run the attack.
    ///
    /// `progress_cb(tried, current_pw)` is called after each batch.
    /// Return `false` from the callback to cancel the run early.
    pub fn run<F>(&mut self, mut progress_cb: F) -> Result<Option<String>>
    where
        F: FnMut(u64, &str) -> bool,
    {
        match self.config.mode.clone() {
            AttackMode::Wordlist(path) => self.run_wordlist(path, &mut progress_cb),
            AttackMode::BruteForce { charset, min_len, max_len } => {
                self.run_bruteforce(charset, min_len, max_len, &mut progress_cb)
            }
            AttackMode::Mask(mask) => self.run_mask(mask, &mut progress_cb),
            AttackMode::Rules { wordlist, rules } => {
                self.run_rules(wordlist, rules, &mut progress_cb)
            }
            AttackMode::Combinator { wordlist, rules, triple } => {
                self.run_combinator(wordlist, rules, triple, &mut progress_cb)
            }
            AttackMode::Markov { model, min_len, max_len } => {
                self.run_markov(model, min_len, max_len, &mut progress_cb)
            }
        }
    }

    // ── Dispatch to GPU or CPU ────────────────────────────────

    fn dispatch_batch(&self, passwords: &[Vec<u8>]) -> Result<Option<String>> {
        match self.rar_info.version {
            RarVersion::Rar5  => self.dispatch_rar5(passwords),
            RarVersion::Rar3  => self.dispatch_rar3(passwords),
            RarVersion::Rar15 => self.dispatch_rar15(passwords),
        }
    }

    fn dispatch_rar15(&self, passwords: &[Vec<u8>]) -> Result<Option<String>> {
        let info = self.rar_info.rar15.as_ref().unwrap();

        // GPU filter → CPU strict verify on survivors.
        if let Some(gpu15) = self.gpu15.as_ref() {
            let survivors = gpu15.filter_batch(passwords)?;
            if survivors.is_empty() { return Ok(None); }
            let subset: Vec<Vec<u8>> = survivors.iter().map(|&i| passwords[i].clone()).collect();
            if let Some(local) = fallback::rar15_verify_cpu(&subset, info) {
                let orig = survivors[local];
                return Ok(Some(String::from_utf8_lossy(&passwords[orig]).into_owned()));
            }
            return Ok(None);
        }

        // CPU-only fallback.
        let idx  = fallback::rar15_verify_cpu(passwords, info);
        Ok(idx.map(|i| String::from_utf8_lossy(&passwords[i]).into_owned()))
    }

    fn dispatch_rar5(&self, passwords: &[Vec<u8>]) -> Result<Option<String>> {
        let info = self.rar_info.rar5.as_ref().unwrap();

        if !info.has_pw_check() {
            anyhow::bail!(
                "This RAR5 archive does not contain a password check value \
                 (CRYPT_PSWCHECK flag not set).\n\
                 Slow verification via AES decryption is not yet implemented.\n\
                 Most WinRAR-created archives include this check — if yours does not, \
                 it may have been created with a third-party tool."
            );
        }

        let match_idx = if let Some(ref gpu) = self.gpu5 {
            gpu.crack_batch(passwords, info)?
        } else {
            fallback::rar5_verify_cpu(passwords, info)
        };

        Ok(match_idx.map(|i| String::from_utf8_lossy(&passwords[i]).into_owned()))
    }

    fn dispatch_rar3(&self, passwords: &[Vec<u8>]) -> Result<Option<String>> {
        let info = self.rar_info.rar3.as_ref().unwrap();

        let utf16_pws: Vec<Vec<u8>> = passwords
            .iter()
            .map(|pw| {
                String::from_utf8_lossy(pw)
                    .encode_utf16()
                    .flat_map(|c| c.to_le_bytes())
                    .collect()
            })
            .collect();

        let (check_mode, head_type, file_crc, pack_size) = match info.check_mode {
            Rar3CheckMode::HeadType  => (0i32, info.head_type as i32, 0u32, 0i32),
            Rar3CheckMode::StoreCrc  => (1i32, 0i32, info.file_crc, info.pack_size as i32),
            Rar3CheckMode::Heuristic => (2i32, 0i32, 0u32, 0i32),
        };

        let match_idx = if let Some(ref gpu) = self.gpu3 {
            gpu.crack_batch(
                &utf16_pws, &info.salt, &info.enc_block,
                check_mode, head_type, file_crc, pack_size,
            )?
        } else {
            fallback::rar3_verify_cpu(passwords, info)
        };

        // CPU double-check eliminates GPU false positives before declaring victory.
        if let Some(idx) = match_idx {
            let pw = String::from_utf8_lossy(&passwords[idx]);
            if info.verify_password(&pw).unwrap_or(false) {
                return Ok(Some(pw.into_owned()));
            }
            // GPU false positive — treat this batch as no-match and continue.
        }
        Ok(None)
    }

    // ── Pipelined loop (RAR5 only) ────────────────────────────

    /// True if we should use the 2-stream pipelined path for this run.
    fn pipelined_rar5_available(&self) -> bool {
        self.pipelined
            && matches!(self.rar_info.version, RarVersion::Rar5)
            && self.gpu5.is_some()
            && self.gpu_ctx.is_some()
            && self.rar_info.rar5.as_ref().map_or(false, |r| r.has_pw_check())
    }

    /// Generic pipelined loop: two streams, overlap CPU batch-generation with
    /// GPU kernel execution. Drains slot_a before generating next; at any
    /// moment only one InFlight is "live" for hit-reporting.
    fn run_pipelined_rar5<G, F>(&mut self, gen: &mut G, cb: &mut F) -> Result<Option<String>>
    where
        G: BatchSource,
        F: FnMut(u64, &str) -> bool,
    {
        let gpu = self.gpu5.as_ref().expect("pipelined path guards gpu5");
        let info = self.rar_info.rar5.as_ref().expect("rar5 info").clone();

        // Lazily allocate stream B the first time we enter the pipelined path.
        if self.stream_b.is_none() {
            let ctx = self.gpu_ctx.as_ref().expect("gpu ctx present");
            self.stream_b = Some(ctx.new_stream()?);
        }
        let stream_a = gpu.stream().clone();
        let stream_b = self.stream_b.as_ref().unwrap().clone();

        let streams = [stream_a, stream_b];
        let mut s = 0usize; // cursor into `streams`

        let batch_size = self.config.batch_size as usize;
        let mut scratch: Vec<Vec<u8>> = Vec::with_capacity(batch_size);

        // Seed: produce and launch the first batch.
        let mut current = {
            let n = gen.next_batch(&mut scratch, batch_size)?;
            if n == 0 { return Ok(None); }
            let pws = std::mem::take(&mut scratch);
            scratch = Vec::with_capacity(batch_size);
            let mut ifl = match gpu.upload_async(pws, &info, streams[s].clone())? {
                Some(i) => i,
                None => return Ok(None),
            };
            gpu.launch_async(&mut ifl)?;
            s ^= 1;
            Some(ifl)
        };

        // Main loop: while one batch runs, prepare the next on the other stream.
        while let Some(running) = current.take() {
            let running_count = running.len() as u64;
            let running_sample = running.first_sample();

            // Produce and launch the NEXT batch on the idle stream. This is
            // the overlap window: CPU generation + H→D uploads run while the
            // kernel for `running` is still on the GPU.
            let mut next: Option<_> = None;
            if !gen.is_exhausted() {
                let n = gen.next_batch(&mut scratch, batch_size)?;
                if n > 0 {
                    let pws = std::mem::take(&mut scratch);
                    scratch = Vec::with_capacity(batch_size);
                    if let Some(mut ifl) = gpu.upload_async(pws, &info, streams[s].clone())? {
                        gpu.launch_async(&mut ifl)?;
                        s ^= 1;
                        next = Some(ifl);
                    }
                }
            }

            // Drain `running`: sync its stream, fetch result. If hit → return.
            // Earliest-batch-wins tiebreak is automatic because we fetch in
            // launch order before considering the next slot.
            if let Some(pw) = gpu.fetch_result(running)? {
                return Ok(Some(pw));
            }

            self.tried += running_count;
            if !cb(self.tried, &running_sample) { break; }

            current = next;
        }

        Ok(None)
    }

    /// Sequential (legacy) loop — used for RAR3, CPU-only, or when pipelining
    /// is disabled via flag/env. The generator is consumed through the
    /// `BatchSource` trait exactly as in the pipelined path so semantics match.
    fn run_sequential<G, F>(&mut self, gen: &mut G, cb: &mut F) -> Result<Option<String>>
    where
        G: BatchSource,
        F: FnMut(u64, &str) -> bool,
    {
        let effective = match self.rar_info.version {
            RarVersion::Rar3  => (self.config.batch_size as usize).min(8192),
            RarVersion::Rar15 => (self.config.batch_size as usize).min(16384),
            RarVersion::Rar5  => self.config.batch_size as usize,
        };
        let batch_size = effective;
        let mut batch: Vec<Vec<u8>> = Vec::with_capacity(batch_size);
        loop {
            if gen.is_exhausted() { break; }
            let n = gen.next_batch(&mut batch, batch_size)?;
            if n == 0 { break; }
            if let Some(pw) = self.dispatch_batch(&batch)? { return Ok(Some(pw)); }
            self.tried += n as u64;
            let sample = String::from_utf8_lossy(&batch[0]).into_owned();
            if !cb(self.tried, &sample) { break; }
        }
        Ok(None)
    }

    /// Single entry point for every attack mode — picks pipelined or
    /// sequential based on archive version + GPU availability + env flags.
    fn drive<G, F>(&mut self, gen: &mut G, cb: &mut F) -> Result<Option<String>>
    where
        G: BatchSource,
        F: FnMut(u64, &str) -> bool,
    {
        if self.pipelined_rar5_available() {
            self.run_pipelined_rar5(gen, cb)
        } else {
            self.run_sequential(gen, cb)
        }
    }

    // ── Attack loops ──────────────────────────────────────────

    fn run_wordlist<F>(&mut self, path: std::path::PathBuf, cb: &mut F) -> Result<Option<String>>
    where F: FnMut(u64, &str) -> bool
    {
        let mut wl = Wordlist::open(&path)?;
        if self.tried > 0 { wl.seek(self.tried)?; }
        self.drive(&mut wl, cb)
    }

    fn run_bruteforce<F>(
        &mut self, charset: String, min_len: usize, max_len: usize, cb: &mut F,
    ) -> Result<Option<String>>
    where F: FnMut(u64, &str) -> bool
    {
        let mut bf = BruteForce::new(&charset, min_len, max_len);
        if self.tried > 0 { bf.seek(self.tried); }
        self.drive(&mut bf, cb)
    }

    fn run_mask<F>(&mut self, mask: String, cb: &mut F) -> Result<Option<String>>
    where F: FnMut(u64, &str) -> bool
    {
        let mut ma = MaskAttack::parse(&mask)?;
        if self.tried > 0 { ma.seek(self.tried); }
        self.drive(&mut ma, cb)
    }

    fn run_rules<F>(
        &mut self,
        wordlist: std::path::PathBuf,
        rules_path: std::path::PathBuf,
        cb: &mut F,
    ) -> Result<Option<String>>
    where F: FnMut(u64, &str) -> bool
    {
        let wl = Wordlist::open(&wordlist)?;
        let rules = load_rules(&rules_path)?;
        let mut src = RulesSource::new(wl, rules);
        if self.tried > 0 { src.seek(self.tried)?; }
        self.drive(&mut src, cb)
    }

    fn run_combinator<F>(
        &mut self,
        wordlist: std::path::PathBuf,
        rules_path: Option<std::path::PathBuf>,
        triple: bool,
        cb: &mut F,
    ) -> Result<Option<String>>
    where F: FnMut(u64, &str) -> bool
    {
        let comb = CombinatorSource::from_wordlist(&wordlist, triple)?;
        match rules_path {
            Some(rp) => {
                let rules = load_rules(&rp)?;
                let mut src = RulesSource::new(comb, rules);
                if self.tried > 0 { src.seek(self.tried)?; }
                self.drive(&mut src, cb)
            }
            None => {
                let mut src = comb;
                if self.tried > 0 { src.seek(self.tried)?; }
                self.drive(&mut src, cb)
            }
        }
    }

    fn run_markov<F>(
        &mut self,
        model_path: std::path::PathBuf,
        min_len: usize,
        max_len: usize,
        cb: &mut F,
    ) -> Result<Option<String>>
    where F: FnMut(u64, &str) -> bool
    {
        let model = MarkovModel::load(&model_path)?;
        let mut src = MarkovSource::new(model, min_len, max_len);
        if self.tried > 0 { src.seek(self.tried)?; }
        self.drive(&mut src, cb)
    }
}

/// Check env `RARPC_PIPELINE`:
///   "0" / "off" / "false" → disabled (legacy sequential path).
///   anything else or unset → enabled.
fn pipeline_enabled() -> bool {
    match std::env::var("RARPC_PIPELINE") {
        Ok(v) => {
            let v = v.to_ascii_lowercase();
            !(v == "0" || v == "off" || v == "false" || v == "no")
        }
        Err(_) => true,
    }
}

/// CLI helper: main uses this to honour `--no-pipeline`.
pub fn force_disable_pipeline() {
    std::env::set_var("RARPC_PIPELINE", "0");
}
