use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use eframe::egui;
use rfd::FileDialog;

use crate::attack::{AttackConfig, AttackMode};
use crate::attack::engine::Engine;
use crate::rar::parser::parse_rar;
use crate::session::Session;

// ── Character group constants ─────────────────────────────────

const CHARS_LOWER:   &str = "abcdefghijklmnopqrstuvwxyz";
const CHARS_UPPER:   &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const CHARS_DIGITS:  &str = "0123456789";
const CHARS_SYMBOLS: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
const CHARS_SPACE:   &str = " ";
const CHARS_HEX:     &str = "0123456789abcdef";

fn charset_add(charset: &mut String, group: &str) {
    for c in group.chars() {
        if !charset.contains(c) {
            charset.push(c);
        }
    }
}

fn charset_remove(charset: &mut String, group: &str) {
    charset.retain(|c| !group.contains(c));
}

fn charset_has(charset: &str, group: &str) -> bool {
    group.chars().all(|c| charset.contains(c))
}

// ── Variant ruleset synthesis ─────────────────────────────────

fn variant_rule_count(case: bool, digits: bool, leet: bool, symbols: bool) -> usize {
    build_variant_rules(case, digits, leet, symbols).lines().count()
}

/// Build a ruleset from the GUI checkboxes. One rule per line.
/// Always includes `:` (noop) so the bare word is tried first.
fn build_variant_rules(case: bool, digits: bool, leet: bool, symbols: bool) -> String {
    let mut out = String::from(":\n");
    if case {
        out.push_str("l\nu\nc\nC\nt\n");
    }
    if digits {
        for d in 0..=9u8 {
            out.push_str(&format!("${}\n", d));
        }
        for d1 in 0..=9u8 {
            for d2 in 0..=9u8 {
                out.push_str(&format!("${}${}\n", d1, d2));
            }
        }
        // Year-like suffixes 1960-2025 via rules $1$9$6$0 etc.
        for y in 1960u32..=2025 {
            let digits: Vec<char> = y.to_string().chars().collect();
            out.push_str(&format!(
                "${}${}${}${}\n",
                digits[0], digits[1], digits[2], digits[3]
            ));
        }
    }
    if leet {
        out.push_str("sa@\nse3\nsi1\nso0\nss$\n");
        // Combined leet (single pass)
        out.push_str("sa@se3si1so0ss$\n");
    }
    if symbols {
        for s in ['!', '?', '.', '_', '*'] {
            out.push_str(&format!("${}\n", s));
        }
    }
    out
}

/// Write the synthesised ruleset to a temp file and return its path.
fn write_variant_rules(
    case: bool, digits: bool, leet: bool, symbols: bool,
) -> std::io::Result<PathBuf> {
    use std::io::Write;
    let text = build_variant_rules(case, digits, leet, symbols);
    let path = std::env::temp_dir().join("rarpc_variants.rule");
    let mut f = std::fs::File::create(&path)?;
    f.write_all(text.as_bytes())?;
    Ok(path)
}

// ── Worker messages (crack) ───────────────────────────────────

enum CrackMsg {
    Progress { tried: u64, rate: f64, current: String },
    Done { result: Option<String>, tried: u64 },
    Error(String),
}

// ── Worker messages (benchmark) ───────────────────────────────

enum BenchMsg {
    Status(String),
    Result { ver: u8, hs: f64 },
    Error(String),
    Done,
}

// ── Tabs ──────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Wordlist,
    BruteForce,
    Mask,
    Benchmark,
    Help,
}

// ── App state ─────────────────────────────────────────────────

pub struct RarpcApp {
    // ── Common
    tab: Tab,

    // ── RAR file (shared by all crack modes)
    rar_file: String,
    /// Format detected immediately on file selection (e.g. "RAR5", "RAR3")
    rar_format_hint: Option<String>,

    // ── Wordlist
    wordlist_path: String,
    /// Combinator: emit word_i and word_i || word_j for every pair.
    /// Multiplies the search space by (1 + n_words).
    combinator_enabled: bool,
    /// Also emit triples w_i||w_j||w_k (n^3 additional candidates).
    combinator_triple: bool,
    /// Variant expansion — when true, feed the wordlist (or combinator output)
    /// through a ruleset synthesised from the checkboxes below.
    variants_enabled: bool,
    variants_case:    bool, // l, u, c, t
    variants_digits:  bool, // $0..$9, $0$0..$9$9, common years
    variants_leet:    bool, // sa@, se3, si1, so0, ss$
    variants_symbols: bool, // $!, $?, $., $_

    // ── Brute force
    charset: String,
    min_len: usize,
    max_len: usize,

    // ── Mask
    mask: String,

    // ── GPU / performance
    gpu_index: usize,
    batch: u32,
    cpu_only: bool,

    // ── Crack worker
    cracking: bool,
    crack_stop: Arc<AtomicBool>,
    crack_rx: Option<mpsc::Receiver<CrackMsg>>,

    // Crack stats
    tried: u64,
    crack_rate: f64,
    current: String,
    crack_elapsed: f64,
    crack_start: Option<Instant>,
    crack_result: Option<String>,
    crack_error: Option<String>,
    exhausted: bool,

    // ── Benchmark worker
    benching: bool,
    bench_gpu: usize,
    bench_duration: u64,
    bench_rx: Option<mpsc::Receiver<BenchMsg>>,
    bench_status: String,
    bench_rar5_hs: Option<f64>,
    bench_rar3_hs: Option<f64>,
    bench_rar15_hs: Option<f64>,
    bench_error: Option<String>,

    // ── Log (shared)
    log: Vec<String>,
}

impl Default for RarpcApp {
    fn default() -> Self {
        Self {
            tab: Tab::Wordlist,
            rar_file: String::new(),
            rar_format_hint: None,
            wordlist_path: String::new(),
            combinator_enabled: false,
            combinator_triple: false,
            variants_enabled: true,
            variants_case:    true,
            variants_digits:  true,
            variants_leet:    false,
            variants_symbols: false,
            charset: format!("{}{}", CHARS_LOWER, CHARS_DIGITS),
            min_len: 1,
            max_len: 8,
            mask: "?l?l?l?d?d".to_string(),
            gpu_index: 0,
            batch: 1 << 20,
            cpu_only: false,
            cracking: false,
            crack_stop: Arc::new(AtomicBool::new(false)),
            crack_rx: None,
            tried: 0,
            crack_rate: 0.0,
            current: String::new(),
            crack_elapsed: 0.0,
            crack_start: None,
            crack_result: None,
            crack_error: None,
            exhausted: false,
            benching: false,
            bench_gpu: 0,
            bench_duration: 10,
            bench_rx: None,
            bench_status: String::new(),
            bench_rar5_hs: None,
            bench_rar3_hs: None,
            bench_rar15_hs: None,
            bench_error: None,
            log: Vec::new(),
        }
    }
}

// ── Crack logic ───────────────────────────────────────────────

impl RarpcApp {
    fn start_crack(&mut self) {
        let rar_path = PathBuf::from(&self.rar_file);
        let rar_info = match parse_rar(&rar_path) {
            Ok(i) => i,
            Err(e) => {
                let msg = format!("Error al parsear RAR: {}", e);
                self.crack_error = Some(msg.clone());
                self.log.push(msg);
                return;
            }
        };

        let mode = match self.tab {
            Tab::Wordlist => {
                let wl = PathBuf::from(&self.wordlist_path);
                let any_variant = self.variants_enabled && (
                    self.variants_case || self.variants_digits
                    || self.variants_leet || self.variants_symbols
                );
                let rules_path = if any_variant {
                    match write_variant_rules(
                        self.variants_case, self.variants_digits,
                        self.variants_leet, self.variants_symbols,
                    ) {
                        Ok(p) => Some(p),
                        Err(e) => {
                            let msg = format!("Error al escribir ruleset temporal: {}", e);
                            self.crack_error = Some(msg.clone());
                            self.log.push(msg);
                            return;
                        }
                    }
                } else { None };

                match (self.combinator_enabled, rules_path) {
                    (true,  rp) => AttackMode::Combinator {
                        wordlist: wl, rules: rp, triple: self.combinator_triple,
                    },
                    (false, Some(rp)) => AttackMode::Rules { wordlist: wl, rules: rp },
                    (false, None)     => AttackMode::Wordlist(wl),
                }
            }
            Tab::BruteForce => AttackMode::BruteForce {
                charset: self.charset.clone(),
                min_len: self.min_len,
                max_len: self.max_len,
            },
            Tab::Mask       => AttackMode::Mask(self.mask.clone()),
            Tab::Benchmark | Tab::Help => return,
        };

        let config = AttackConfig {
            mode,
            batch_size: self.batch,
            gpu_index: self.gpu_index,
            num_gpus: 1,
            cpu_only: self.cpu_only,
        };

        let session = Session::new(&rar_path, &config);

        // Reset
        self.tried       = 0;
        self.crack_rate  = 0.0;
        self.current     = String::new();
        self.crack_elapsed = 0.0;
        self.crack_result  = None;
        self.crack_error   = None;
        self.exhausted     = false;
        self.cracking      = true;
        self.crack_start   = Some(Instant::now());

        self.crack_stop.store(false, Ordering::SeqCst);
        let stop = self.crack_stop.clone();

        let (tx, rx) = mpsc::channel::<CrackMsg>();
        self.crack_rx = Some(rx);

        let label = match self.tab {
            Tab::Wordlist   => "wordlist",
            Tab::BruteForce => "brute-force",
            Tab::Mask       => "mask",
            Tab::Benchmark | Tab::Help => unreachable!(),
        };
        self.log.push(format!(
            "[crack] inicio {} — {}",
            label,
            rar_path.file_name().unwrap_or_default().to_string_lossy()
        ));

        std::thread::spawn(move || {
            let mut engine = match Engine::new(rar_info, config, session) {
                Ok(e) => e,
                Err(e) => { let _ = tx.send(CrackMsg::Error(e.to_string())); return; }
            };

            let mut last_sent  = Instant::now();
            let mut last_tried = 0u64;
            let mut last_rate  = 0.0f64;

            let result = engine.run(|tried, candidate| {
                let now = Instant::now();
                let dt  = now.duration_since(last_sent).as_secs_f64();
                if dt >= 0.25 {
                    let rate   = (tried - last_tried) as f64 / dt;
                    last_tried = tried;
                    last_sent  = now;
                    last_rate  = rate;
                    let _ = tx.send(CrackMsg::Progress {
                        tried, rate, current: candidate.to_string(),
                    });
                } else if dt >= 0.08 {
                    let _ = tx.send(CrackMsg::Progress {
                        tried, rate: last_rate, current: candidate.to_string(),
                    });
                }
                !stop.load(Ordering::Relaxed)
            });

            let final_tried = engine.tried();
            match result {
                Ok(Some(pw)) => { let _ = tx.send(CrackMsg::Done { result: Some(pw), tried: final_tried }); }
                Ok(None)     => { let _ = tx.send(CrackMsg::Done { result: None, tried: final_tried }); }
                Err(e)       => { let _ = tx.send(CrackMsg::Error(e.to_string())); }
            }
        });
    }

    fn stop_crack(&mut self) {
        self.crack_stop.store(true, Ordering::SeqCst);
        self.log.push("[crack] stop solicitado".to_string());
    }

    fn poll_crack(&mut self, ctx: &egui::Context) {
        if let Some(t) = self.crack_start {
            if self.cracking { self.crack_elapsed = t.elapsed().as_secs_f64(); }
        }

        let mut done = false;
        if let Some(ref rx) = self.crack_rx {
            loop {
                match rx.try_recv() {
                    Ok(CrackMsg::Progress { tried, rate, current }) => {
                        self.tried = tried;
                        if rate > 0.0 { self.crack_rate = rate; }
                        self.current = current;
                        ctx.request_repaint();
                    }
                    Ok(CrackMsg::Done { result: pw, tried }) => {
                        self.cracking     = false;
                        self.crack_result = pw.clone();
                        self.exhausted    = pw.is_none();
                        self.tried        = tried;
                        self.crack_elapsed = self.crack_start
                            .map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
                        if let Some(ref p) = pw {
                            self.log.push(format!(
                                "[crack] ✓ Contraseña: {}  ({} intentos, {:.1}s)",
                                p, self.tried, self.crack_elapsed
                            ));
                        } else {
                            self.log.push(format!(
                                "[crack] ✗ No encontrada ({} intentos, {:.1}s)",
                                self.tried, self.crack_elapsed
                            ));
                        }
                        done = true;
                        ctx.request_repaint();
                        break;
                    }
                    Ok(CrackMsg::Error(e)) => {
                        self.cracking    = false;
                        self.crack_error = Some(e.clone());
                        self.log.push(format!("[crack] error: {}", e));
                        done = true;
                        ctx.request_repaint();
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty)        => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        self.cracking = false;
                        done = true;
                        break;
                    }
                }
            }
        }
        if done { self.crack_rx = None; }
    }
}

// ── Benchmark logic ───────────────────────────────────────────

impl RarpcApp {
    fn start_bench(&mut self) {
        self.bench_status   = "Iniciando…".to_string();
        self.bench_rar5_hs  = None;
        self.bench_rar3_hs  = None;
        self.bench_rar15_hs = None;
        self.bench_error    = None;
        self.benching      = true;

        let gpu    = self.bench_gpu;
        let dur    = Duration::from_secs(self.bench_duration);
        let (tx, rx) = mpsc::channel::<BenchMsg>();
        self.bench_rx = Some(rx);

        self.log.push(format!("[bench] inicio GPU:{} dur:{}s", gpu, self.bench_duration));

        std::thread::spawn(move || {
            // ── RAR5 ─────────────────────────────────────
            let _ = tx.send(BenchMsg::Status("Benchmark RAR5 (PBKDF2-SHA256)…".into()));

            let ctx = match crate::gpu::context::GpuContext::new(gpu) {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(BenchMsg::Error(format!("GPU {}: {}", gpu, e)));
                    let _ = tx.send(BenchMsg::Done);
                    return;
                }
            };

            let batch = 1u32 << 20;
            let fake: Vec<Vec<u8>> = (0..batch as usize)
                .map(|i| format!("{:08x}", i).into_bytes())
                .collect();

            // RAR5 bench
            {
                use crate::gpu::rar5_gpu::Rar5Gpu;
                use crate::rar::rar5::{PswCheckData, Rar5Info};
                match Rar5Gpu::new(&ctx) {
                    Ok(gpu5) => {
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
                        let start = Instant::now();
                        let mut total = 0u64;
                        while start.elapsed() < dur {
                            if let Err(e) = gpu5.crack_batch(&fake, &info) {
                                let _ = tx.send(BenchMsg::Error(format!("RAR5: {}", e)));
                                let _ = tx.send(BenchMsg::Done);
                                return;
                            }
                            total += batch as u64;
                        }
                        let hs = total as f64 / start.elapsed().as_secs_f64();
                        let _ = tx.send(BenchMsg::Result { ver: 5, hs });
                    }
                    Err(e) => {
                        let _ = tx.send(BenchMsg::Error(format!("RAR5 GPU: {}", e)));
                    }
                }
            }

            // RAR3 bench
            let _ = tx.send(BenchMsg::Status("Benchmark RAR3 (SHA-1 KDF)…".into()));
            {
                use crate::gpu::rar3_gpu::Rar3Gpu;
                match Rar3Gpu::new(&ctx) {
                    Ok(gpu3) => {
                        let salt = [0u8; 8];
                        let enc  = [0u8; 16];
                        let start = Instant::now();
                        let mut total = 0u64;
                        while start.elapsed() < dur {
                            if let Err(e) = gpu3.crack_batch(&fake, &salt, &enc, 2, 0, 0, 0) {
                                let _ = tx.send(BenchMsg::Error(format!("RAR3: {}", e)));
                                let _ = tx.send(BenchMsg::Done);
                                return;
                            }
                            total += batch as u64;
                        }
                        let hs = total as f64 / start.elapsed().as_secs_f64();
                        let _ = tx.send(BenchMsg::Result { ver: 3, hs });
                    }
                    Err(e) => {
                        let _ = tx.send(BenchMsg::Error(format!("RAR3 GPU: {}", e)));
                    }
                }
            }

            // RAR 1.5 bench (GPU filter)
            let _ = tx.send(BenchMsg::Status("Benchmark RAR 1.5 (Crypt15 filter)…".into()));
            {
                use crate::gpu::rar15_gpu::Rar15Gpu;
                use crate::rar::rar15::{Rar15FilterParams, Rar15Info};
                let info15 = Rar15Info {
                    packed_data: vec![0u8; 512],
                    unp_size: 28843,
                    file_crc: 0xFFFFFFFF,
                    unp_ver: 0x0f,
                    method: 0x33,
                };
                let params = Rar15FilterParams::default();
                match Rar15Gpu::new_for_archive(&ctx, &info15, params) {
                    Ok(gpu15) => {
                        let start = Instant::now();
                        let mut total = 0u64;
                        while start.elapsed() < dur {
                            if let Err(e) = gpu15.filter_batch(&fake) {
                                let _ = tx.send(BenchMsg::Error(format!("RAR1.5: {}", e)));
                                let _ = tx.send(BenchMsg::Done);
                                return;
                            }
                            total += batch as u64;
                        }
                        let hs = total as f64 / start.elapsed().as_secs_f64();
                        let _ = tx.send(BenchMsg::Result { ver: 15, hs });
                    }
                    Err(e) => {
                        let _ = tx.send(BenchMsg::Error(format!("RAR1.5 GPU: {}", e)));
                    }
                }
            }

            let _ = tx.send(BenchMsg::Done);
        });
    }

    fn poll_bench(&mut self, ctx: &egui::Context) {
        let mut done = false;
        if let Some(ref rx) = self.bench_rx {
            loop {
                match rx.try_recv() {
                    Ok(BenchMsg::Status(s)) => {
                        self.bench_status = s.clone();
                        self.log.push(format!("[bench] {}", s));
                        ctx.request_repaint();
                    }
                    Ok(BenchMsg::Result { ver, hs }) => {
                        let label = fmt_hs(hs);
                        let ver_str = if ver == 15 { "1.5".to_string() } else { ver.to_string() };
                        self.log.push(format!("[bench] RAR{}: {}", ver_str, label));
                        match ver {
                            5  => { self.bench_rar5_hs = Some(hs); }
                            15 => { self.bench_rar15_hs = Some(hs); }
                            _  => { self.bench_rar3_hs = Some(hs); }
                        }
                        ctx.request_repaint();
                    }
                    Ok(BenchMsg::Error(e)) => {
                        self.bench_error = Some(e.clone());
                        self.log.push(format!("[bench] error: {}", e));
                        ctx.request_repaint();
                    }
                    Ok(BenchMsg::Done) => {
                        self.benching     = false;
                        self.bench_status = "Completado".to_string();
                        done = true;
                        ctx.request_repaint();
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty)        => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        self.benching = false;
                        done = true;
                        break;
                    }
                }
            }
        }
        if done { self.bench_rx = None; }
    }
}

// ── Format helpers ────────────────────────────────────────────

fn fmt_hs(hs: f64) -> String {
    if hs >= 1_000_000.0 { format!("{:.2} MH/s", hs / 1_000_000.0) }
    else if hs >= 1_000.0 { format!("{:.1} KH/s", hs / 1_000.0) }
    else { format!("{:.0} H/s", hs) }
}

fn fmt_elapsed(secs: f64) -> String {
    let s = secs as u64;
    if s < 60 { format!("{}s", s) }
    else if s < 3600 { format!("{}m {}s", s / 60, s % 60) }
    else { format!("{}h {}m", s / 3600, (s % 3600) / 60) }
}

// ── eframe App ────────────────────────────────────────────────

impl eframe::App for RarpcApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_crack(ctx);
        self.poll_bench(ctx);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("rarpc — RAR Password Cracker");
            ui.separator();

            // ── Tabs ──────────────────────────────────────────
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Wordlist,   "📖 Wordlist");
                ui.selectable_value(&mut self.tab, Tab::BruteForce, "🔑 Brute Force");
                ui.selectable_value(&mut self.tab, Tab::Mask,       "🎭 Mask");
                ui.selectable_value(&mut self.tab, Tab::Benchmark,  "⏱ Benchmark");
                ui.selectable_value(&mut self.tab, Tab::Help,       "? Help");
            });
            ui.separator();

            // ── Tab content ───────────────────────────────────
            match self.tab {
                Tab::Benchmark => self.ui_benchmark(ui),
                Tab::Help      => self.ui_help(ui),
                _              => self.ui_crack(ui),
            }
        });

        if self.cracking || self.benching {
            ctx.request_repaint_after(Duration::from_millis(200));
        }
    }
}

// ── Crack panel ───────────────────────────────────────────────

impl RarpcApp {
    fn ui_crack(&mut self, ui: &mut egui::Ui) {
        // RAR file selector
        if self.rar_file.is_empty() {
            // Big prominent button when no file chosen yet
            ui.add_space(8.0);
            ui.vertical_centered(|ui| {
                let btn = egui::Button::new(
                    egui::RichText::new("📂  Seleccionar archivo (.rar)")
                        .size(18.0)
                        .color(egui::Color32::WHITE),
                )
                .fill(egui::Color32::from_rgb(50, 100, 200))
                .min_size(egui::vec2(310.0, 52.0));
                if ui.add(btn).clicked() {
                    if let Some(p) = FileDialog::new()
                        .add_filter("Archivos RAR", &["rar"])
                        .pick_file()
                    {
                        let path_str = p.to_string_lossy().into_owned();
                        // Quick format detection for display
                        let hint = parse_rar(&p).ok()
                            .map(|info| info.encryption_name().to_string());
                        self.rar_format_hint = hint;
                        self.rar_file = path_str;
                    }
                }
            });
            ui.add_space(8.0);
        } else {
            // Compact row once a file is loaded
            ui.horizontal(|ui| {
                let fname = PathBuf::from(&self.rar_file)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| self.rar_file.clone());
                ui.label(egui::RichText::new(format!("📄 {}", fname)).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("✕ cambiar").clicked() {
                        self.rar_file.clear();
                        self.rar_format_hint = None;
                    }
                });
            });
            // Show detected format
            if let Some(ref hint) = self.rar_format_hint {
                ui.label(
                    egui::RichText::new(format!("🔒 {}", hint))
                        .small()
                        .color(egui::Color32::from_rgb(150, 200, 255)),
                );
            }
        }

        ui.add_space(6.0);

        // Mode-specific settings
        match self.tab {
            Tab::Wordlist   => self.ui_wordlist(ui),
            Tab::BruteForce => self.ui_bruteforce(ui),
            Tab::Mask       => self.ui_mask(ui),
            Tab::Benchmark | Tab::Help => unreachable!(),
        }

        ui.add_space(6.0);

        // GPU / performance (collapsible)
        egui::CollapsingHeader::new("⚙  GPU / Rendimiento")
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("gpu_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("GPU:");
                        ui.add(egui::DragValue::new(&mut self.gpu_index).range(0..=7));
                        ui.end_row();

                        ui.label("Batch size:");
                        ui.add(
                            egui::DragValue::new(&mut self.batch)
                                .range(1024..=(1 << 24))
                                .speed(4096.0),
                        );
                        ui.end_row();

                        ui.label("Solo CPU:");
                        ui.checkbox(&mut self.cpu_only, "");
                        ui.end_row();
                    });
            });

        ui.add_space(6.0);

        // Start / Stop
        ui.horizontal(|ui| {
            let can_start = !self.rar_file.is_empty()
                && match self.tab {
                    Tab::Wordlist   => !self.wordlist_path.is_empty(),
                    Tab::BruteForce => !self.charset.is_empty() && self.min_len <= self.max_len,
                    Tab::Mask       => !self.mask.is_empty(),
                    Tab::Benchmark | Tab::Help => false,
                };

            if !self.cracking {
                let btn = egui::Button::new(
                    egui::RichText::new("▶  Iniciar").color(egui::Color32::WHITE),
                ).fill(egui::Color32::from_rgb(34, 139, 34));
                if ui.add_enabled(can_start, btn).clicked() {
                    self.start_crack();
                }
            } else {
                let btn = egui::Button::new(
                    egui::RichText::new("■  Detener").color(egui::Color32::WHITE),
                ).fill(egui::Color32::from_rgb(178, 34, 34));
                if ui.add(btn).clicked() {
                    self.stop_crack();
                }
                ui.add_space(6.0);
                ui.spinner();
            }
        });

        ui.separator();

        // Stats
        egui::Grid::new("stats_grid")
            .num_columns(2)
            .spacing([16.0, 4.0])
            .show(ui, |ui| {
                ui.label("Estado:");
                if self.cracking {
                    ui.label(egui::RichText::new("Ejecutando…").color(egui::Color32::from_rgb(80, 200, 255)));
                } else if let Some(ref pw) = self.crack_result.clone() {
                    ui.label(
                        egui::RichText::new(format!("✓  {}", pw))
                            .color(egui::Color32::GREEN).strong(),
                    );
                } else if self.exhausted {
                    ui.label(
                        egui::RichText::new("✗  No encontrada — espacio agotado")
                            .color(egui::Color32::from_rgb(255, 120, 60)),
                    );
                } else if let Some(ref e) = self.crack_error.clone() {
                    ui.label(egui::RichText::new(format!("⚠  {}", e)).color(egui::Color32::RED));
                } else {
                    ui.label(egui::RichText::new("Inactivo").weak());
                }
                ui.end_row();

                ui.label("Intentos:");
                ui.label(format!("{}", self.tried));
                ui.end_row();

                ui.label("Velocidad:");
                ui.label(if self.crack_rate > 0.0 { fmt_hs(self.crack_rate) } else { "—".into() });
                ui.end_row();

                ui.label("Tiempo:");
                ui.label(if self.cracking || self.crack_elapsed > 0.0 {
                    fmt_elapsed(self.crack_elapsed)
                } else { "—".into() });
                ui.end_row();

                ui.label("Actual:");
                ui.label(egui::RichText::new(&self.current).monospace());
                ui.end_row();
            });

        // Log
        if !self.log.is_empty() {
            ui.add_space(4.0);
            ui.separator();
            ui.label(egui::RichText::new("Log").small().weak());
            egui::ScrollArea::vertical()
                .id_source("crack_log")
                .max_height(90.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for line in &self.log {
                        ui.label(egui::RichText::new(line).small().monospace());
                    }
                });
        }
    }

    // ── Wordlist panel ────────────────────────────────────────

    fn ui_wordlist(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Diccionario:");
            let te = egui::TextEdit::singleline(&mut self.wordlist_path)
                .hint_text("ruta/al/diccionario.txt")
                .desired_width(f32::INFINITY);
            ui.add(te);
            if ui.button("📂 Abrir…").clicked() {
                if let Some(p) = FileDialog::new()
                    .add_filter("Ficheros de texto", &["txt", "lst", "dic", "wordlist"])
                    .pick_file()
                {
                    self.wordlist_path = p.to_string_lossy().into_owned();
                }
            }
        });

        ui.add_space(6.0);
        ui.checkbox(&mut self.combinator_enabled,
            "Combinar palabras  (w_i y w_i+w_j para todo i,j)");
        if self.combinator_enabled {
            ui.indent("combinator_opts", |ui| {
                ui.checkbox(&mut self.combinator_triple,
                    "Triple combinacion  (+ w_i||w_j||w_k, n^3 candidatos extra)");
            });
        }

        ui.add_space(6.0);
        ui.checkbox(&mut self.variants_enabled, "Probar variantes por cada palabra");
        if self.variants_enabled {
            ui.indent("variants", |ui| {
                ui.checkbox(&mut self.variants_case,
                    "Capitalización  (minúsculas, MAYÚSCULAS, Capital, tOGGLE)");
                ui.checkbox(&mut self.variants_digits,
                    "Añadir dígitos  (word0 … word9, word00 … word99, +años 1960-2025)");
                ui.checkbox(&mut self.variants_leet,
                    "Leet  (a→@, e→3, i→1, o→0, s→$)");
                ui.checkbox(&mut self.variants_symbols,
                    "Añadir símbolos  (word! word? word. word_)");
                let n = variant_rule_count(
                    self.variants_case, self.variants_digits,
                    self.variants_leet, self.variants_symbols,
                );
                ui.label(format!(
                    "→ {} reglas por palabra (multiplicador del diccionario)",
                    n,
                ));
            });
        }
    }

    // ── Brute force panel ─────────────────────────────────────

    fn ui_bruteforce(&mut self, ui: &mut egui::Ui) {
        // Length row
        ui.horizontal(|ui| {
            ui.label("Longitud mín:");
            ui.add(egui::DragValue::new(&mut self.min_len).range(1..=64));
            ui.add_space(12.0);
            ui.label("máx:");
            ui.add(egui::DragValue::new(&mut self.max_len).range(1..=64));
        });

        ui.add_space(6.0);
        ui.label("Charset — añadir grupo:");

        // Character group toggle buttons
        ui.horizontal_wrapped(|ui| {
            // Each button adds/removes a group from the charset
            Self::group_toggle(ui, &mut self.charset, CHARS_LOWER,   "a–z");
            Self::group_toggle(ui, &mut self.charset, CHARS_UPPER,   "A–Z");
            Self::group_toggle(ui, &mut self.charset, CHARS_DIGITS,  "0–9");
            Self::group_toggle(ui, &mut self.charset, CHARS_SYMBOLS, "Símbolos");
            Self::group_toggle(ui, &mut self.charset, CHARS_SPACE,   "Espacio");
            Self::group_toggle(ui, &mut self.charset, CHARS_HEX,     "Hex (0-f)");

            ui.add_space(8.0);
            if ui.button("🗑 Limpiar").clicked() {
                self.charset.clear();
            }
        });

        // Big multiline charset display + edit
        ui.add_space(4.0);
        let char_count = self.charset.chars().count();
        ui.label(format!("Charset actual ({} caracteres):", char_count));

        let te = egui::TextEdit::multiline(&mut self.charset)
            .font(egui::TextStyle::Monospace)
            .desired_rows(4)
            .desired_width(f32::INFINITY)
            .hint_text("escribe o usa los botones de arriba");
        ui.add(te);
    }

    /// Renders a toggle button for a character group. Active = highlighted.
    fn group_toggle(ui: &mut egui::Ui, charset: &mut String, group: &str, label: &str) {
        let active = charset_has(charset, group);
        let color = if active {
            egui::Color32::from_rgb(60, 120, 200)
        } else {
            ui.visuals().widgets.inactive.bg_fill
        };
        let text_color = if active {
            egui::Color32::WHITE
        } else {
            ui.visuals().text_color()
        };
        let btn = egui::Button::new(egui::RichText::new(label).color(text_color))
            .fill(color);
        if ui.add(btn).clicked() {
            if active {
                charset_remove(charset, group);
            } else {
                charset_add(charset, group);
            }
        }
    }

    // ── Mask panel ────────────────────────────────────────────

    fn ui_mask(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Máscara:");
            ui.add(
                egui::TextEdit::singleline(&mut self.mask)
                    .hint_text("ej. ?u?l?l?d?d")
                    .desired_width(300.0),
            );
        });

        ui.add_space(4.0);

        // Quick mask builder
        ui.label("Insertar posición:");
        ui.horizontal_wrapped(|ui| {
            for (token, tip) in [
                ("?l", "a–z"),
                ("?u", "A–Z"),
                ("?d", "0–9"),
                ("?s", "símbolos"),
                ("?a", "todo"),
                ("?h", "hex"),
            ] {
                if ui.small_button(format!("{} ({})", token, tip)).clicked() {
                    self.mask.push_str(token);
                }
            }
            if ui.small_button("⌫ borrar último").clicked() {
                // Remove last ?x token
                if self.mask.len() >= 2 {
                    let new_len = self.mask.len() - 2;
                    self.mask.truncate(new_len);
                }
            }
        });

        ui.add_space(4.0);
        let estimated = self.estimate_mask_space();
        ui.label(
            egui::RichText::new(format!("Espacio de búsqueda: ~{}", estimated))
                .weak().small(),
        );
    }

    fn estimate_mask_space(&self) -> String {
        // Count variable slots and their sizes
        let mut space: u128 = 1;
        let mut i = 0;
        let chars: Vec<char> = self.mask.chars().collect();
        while i < chars.len() {
            if chars[i] == '?' && i + 1 < chars.len() {
                let size: u128 = match chars[i + 1] {
                    'l' => 26,
                    'u' => 26,
                    'd' => 10,
                    's' => 32,
                    'a' => 95,
                    'h' => 16,
                    _   => 1,
                };
                space = space.saturating_mul(size);
                i += 2;
            } else {
                i += 1;
            }
        }
        if space >= 1_000_000_000_000 {
            format!("{:.2}T", space as f64 / 1e12)
        } else if space >= 1_000_000_000 {
            format!("{:.2}G", space as f64 / 1e9)
        } else if space >= 1_000_000 {
            format!("{:.2}M", space as f64 / 1e6)
        } else if space >= 1_000 {
            format!("{:.1}K", space as f64 / 1e3)
        } else {
            format!("{}", space)
        }
    }
}

// ── Benchmark panel ───────────────────────────────────────────

impl RarpcApp {
    fn ui_benchmark(&mut self, ui: &mut egui::Ui) {
        ui.label("Mide la velocidad de la GPU en H/s para RAR5 y RAR3.");
        ui.add_space(8.0);

        egui::Grid::new("bench_cfg")
            .num_columns(2)
            .spacing([8.0, 6.0])
            .show(ui, |ui| {
                ui.label("GPU:");
                ui.add(egui::DragValue::new(&mut self.bench_gpu).range(0..=7));
                ui.end_row();

                ui.label("Duración:");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut self.bench_duration).range(3..=120));
                    ui.label("segundos");
                });
                ui.end_row();
            });

        ui.add_space(8.0);

        if !self.benching {
            let btn = egui::Button::new(
                egui::RichText::new("▶  Iniciar benchmark").color(egui::Color32::WHITE),
            ).fill(egui::Color32::from_rgb(34, 100, 180));
            if ui.add(btn).clicked() {
                self.start_bench();
            }
        } else {
            ui.horizontal(|ui| {
                let btn = egui::Button::new(
                    egui::RichText::new("Ejecutando…").color(egui::Color32::WHITE),
                ).fill(egui::Color32::from_rgb(120, 120, 120));
                ui.add_enabled(false, btn);
                ui.spinner();
                ui.label(egui::RichText::new(&self.bench_status).weak());
            });
        }

        ui.add_space(12.0);
        ui.separator();

        // Results table
        egui::Grid::new("bench_results")
            .num_columns(2)
            .spacing([24.0, 8.0])
            .show(ui, |ui| {
                ui.label(egui::RichText::new("RAR5 (PBKDF2-SHA256)").strong());
                if let Some(hs) = self.bench_rar5_hs {
                    ui.label(
                        egui::RichText::new(fmt_hs(hs))
                            .color(egui::Color32::from_rgb(80, 200, 120))
                            .strong()
                            .size(20.0),
                    );
                } else if self.benching {
                    ui.label(egui::RichText::new("…").weak());
                } else {
                    ui.label(egui::RichText::new("—").weak());
                }
                ui.end_row();

                ui.label(egui::RichText::new("RAR3 (SHA-1 KDF)").strong());
                if let Some(hs) = self.bench_rar3_hs {
                    ui.label(
                        egui::RichText::new(fmt_hs(hs))
                            .color(egui::Color32::from_rgb(80, 200, 120))
                            .strong()
                            .size(20.0),
                    );
                } else if self.benching {
                    ui.label(egui::RichText::new("…").weak());
                } else {
                    ui.label(egui::RichText::new("—").weak());
                }
                ui.end_row();

                ui.label(egui::RichText::new("RAR 1.5 (Crypt15 filter)").strong());
                if let Some(hs) = self.bench_rar15_hs {
                    ui.label(
                        egui::RichText::new(fmt_hs(hs))
                            .color(egui::Color32::from_rgb(80, 200, 120))
                            .strong()
                            .size(20.0),
                    );
                } else if self.benching {
                    ui.label(egui::RichText::new("…").weak());
                } else {
                    ui.label(egui::RichText::new("—").weak());
                }
                ui.end_row();
            });

        if let Some(ref e) = self.bench_error.clone() {
            ui.add_space(4.0);
            ui.label(egui::RichText::new(format!("⚠ {}", e)).color(egui::Color32::RED));
        }

        // Log (bench section)
        let bench_lines: Vec<&String> = self.log.iter()
            .filter(|l| l.starts_with("[bench]"))
            .collect();
        if !bench_lines.is_empty() {
            ui.add_space(8.0);
            ui.separator();
            ui.label(egui::RichText::new("Log").small().weak());
            egui::ScrollArea::vertical()
                .id_source("bench_log")
                .max_height(120.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for line in bench_lines {
                        ui.label(egui::RichText::new(line.as_str()).small().monospace());
                    }
                });
        }
    }
}

// ── Help panel ───────────────────────────────────────────

impl RarpcApp {
    fn ui_help(&self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .id_source("help_scroll")
            .show(ui, |ui| {
                ui.heading("Formatos soportados");
                ui.label("- RAR5 (AES-256 + PBKDF2-SHA256) — GPU + CPU fallback");
                ui.label("- RAR3 (AES-128 + SHA-1 KDF) — GPU + CPU fallback");
                ui.label("- RAR 1.5 (Crypt15 stream cipher) — GPU filter + CPU verify");
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Limitaciones:").strong());
                ui.label("  - RAR 2.0-2.8 no soportado");
                ui.label("  - RAR 1.5 solid archives no soportados");
                ui.label("  - RAR5 sin PswCheck requiere AES verify (no implementado)");

                ui.add_space(12.0);
                ui.heading("Modos de ataque");
                ui.label(egui::RichText::new("Wordlist").strong());
                ui.label("  Diccionario de passwords, uno por linea.");
                ui.label("  Opciones: combinar palabras (w_i||w_j), variantes (reglas).");
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Brute Force").strong());
                ui.label("  Genera todas las combinaciones de un charset dado.");
                ui.label("  Ejemplo: charset=abc, len=3 -> aaa, aab, aac, ..., ccc");
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Mask").strong());
                ui.label("  Patron con placeholders: ?l=a-z, ?u=A-Z, ?d=0-9,");
                ui.label("  ?s=simbolos, ?a=todos (95 chars), ?h=hex (0-f).");
                ui.label("  Ejemplo: ?u?l?l?l?d?d -> Abcd12");
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Rules").strong());
                ui.label("  Operadores: : l u c C t r d $X ^X sXY TN { } [ ] DN iNX @X");
                ui.label("  Se aplican sobre cada palabra del diccionario.");
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Combinator").strong());
                ui.label("  Emite w_i, w_i||w_j, y opcionalmente w_i||w_j||w_k (triple).");
                ui.label("  Total: n + n^2 (o n + n^2 + n^3 con triple).");
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Markov").strong());
                ui.label("  Modelo de trigramas entrenado con `rarpc train-markov`.");
                ui.label("  Genera candidatos ordenados por probabilidad.");

                ui.add_space(12.0);
                ui.heading("Uso CLI");
                ui.label(egui::RichText::new("rarpc crack <FILE> [OPTIONS]").monospace());
                ui.label("  --wordlist <DICT>    Diccionario");
                ui.label("  --brute              Fuerza bruta");
                ui.label("  --mask <MASK>        Ataque mascara");
                ui.label("  --rules <RULES>      Fichero de reglas");
                ui.label("  --markov <MODEL>     Modelo Markov");
                ui.label("  --charset <CHARS>    Charset para brute (default: a-z0-9)");
                ui.label("  --min-len N          Longitud minima (default: 1)");
                ui.label("  --max-len N          Longitud maxima (default: 8)");
                ui.label("  -g, --gpu N          Indice GPU (default: 0)");
                ui.label("  --batch N            Batch size (default: 1M)");
                ui.label("  --session <FILE>     Guardar/restaurar sesion");
                ui.label("  --cpu-only           Solo CPU (sin CUDA)");
                ui.label("  --no-pipeline        Desactivar pipeline 2-streams");
                ui.add_space(4.0);
                ui.label(egui::RichText::new("rarpc bench").monospace());
                ui.label("  Benchmark GPU (RAR5 y RAR3).");
                ui.add_space(4.0);
                ui.label(egui::RichText::new("rarpc gpus").monospace());
                ui.label("  Listar GPUs disponibles.");
                ui.add_space(4.0);
                ui.label(egui::RichText::new("rarpc train-markov <CORPUS> -o <MODEL>").monospace());
                ui.label("  Entrenar modelo Markov desde un corpus de passwords.");

                ui.add_space(12.0);
                ui.heading("Requisitos GPU");
                ui.label("- CUDA Toolkit 12.x");
                ui.label("- Visual Studio Build Tools 2022");
                ui.label("- GPU Ada SM_89 (RTX 4060 Ti, 4070, 4080, 4090)");
                ui.label("- Otras arquitecturas: cambiar -arch en build.rs");

                ui.add_space(12.0);
                ui.heading("Tips");
                ui.label("- Para archivos RAR5, la verificacion es rapida (~68 KH/s GPU).");
                ui.label("- RAR3 es mas lento (~76 KH/s) por su KDF de 262144 iteraciones.");
                ui.label("- RAR 1.5 usa GPU filter (6.7 MH/s) + CPU verify.");
                ui.label("- Combinator + Rules multiplica el espacio de busqueda.");
                ui.label("- Usa --session para poder interrumpir y continuar.");
            });
    }
}

// ── Entry point ───────────────────────────────────────────────

pub fn run_gui() -> anyhow::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("rarpc — RAR Password Cracker")
            .with_inner_size([600.0, 620.0])
            .with_min_inner_size([480.0, 500.0]),
        ..Default::default()
    };

    eframe::run_native(
        "rarpc",
        native_options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(RarpcApp::default()))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{:?}", e))
}
