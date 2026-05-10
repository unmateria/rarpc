//! Order-2 (trigram) character Markov candidate generator.
//!
//! The model stores log P(c | c-2, c-1) over a bounded byte alphabet. During
//! generation we expand partial prefixes via a min-heap keyed by their
//! cumulative -log-probability and emit complete candidates (length in
//! [min_len, max_len]) in best-first order across lengths.
//!
//! Trade-offs chosen for simplicity (vs. a fully featured Markov attack):
//!   * No length prior — a length-3 candidate with the same char probs as a
//!     length-8 candidate will always appear earlier because fewer -log p
//!     terms accumulate. Mitigation: set `--max-len == --min-len` if you want
//!     strictly-length-L output in probability order.
//!   * Session resume is coarse: `seek(n)` replays the heap from scratch and
//!     discards the first `n` candidates. Fine-grained resume would require
//!     serializing the heap state, which is more effort than justified for
//!     Exp 4 validation.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

use crate::attack::source::BatchSource;

/// Token reserved for the "before-password" context. We pick a class index
/// one past the real alphabet so it never collides with a real byte.
const START_CLASS: u16 = 0xFFFF;

/// Trigram character model. Stored in a compact binary form:
///
///   magic u32  "RMK1"
///   n     u32  alphabet size
///   alphabet [u8; n]
///   logp_start [f32; n]             — P(c | START, START)
///   logp_after1 [f32; n * n]        — P(c | START, c1)  indexed c1*n + c
///   logp_full   [f32; n * n * n]    — P(c | c2, c1)     indexed c2*n*n + c1*n + c
///
/// `logp_*` values are **natural log of probability** (i.e. −log p is the
/// heap weight). Probabilities are Laplace-smoothed so every slot is finite.
#[derive(Debug, Serialize, Deserialize)]
pub struct MarkovModel {
    pub alphabet: Vec<u8>,
    pub logp_start: Vec<f32>,
    pub logp_after1: Vec<f32>,
    pub logp_full: Vec<f32>,
}

impl MarkovModel {
    pub fn n(&self) -> usize { self.alphabet.len() }

    /// Reverse lookup: byte → class index, or None if not in alphabet.
    pub fn class_of(&self, byte: u8) -> Option<u16> {
        self.alphabet.iter().position(|&b| b == byte).map(|i| i as u16)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let bytes = bincode::serialize(self)?;
        let mut f = File::create(path)?;
        f.write_all(&bytes)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut f = File::open(path)
            .with_context(|| format!("opening Markov model {}", path.display()))?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        Ok(bincode::deserialize(&buf)?)
    }

    /// Train from a UTF-8 corpus file (one password per line). Characters
    /// outside the printable-ASCII range 0x20..=0x7E are dropped.
    pub fn train(corpus: &Path) -> Result<Self> {
        let f = File::open(corpus)
            .with_context(|| format!("opening corpus {}", corpus.display()))?;
        let reader = BufReader::new(f);

        // Collect alphabet from corpus (printable ASCII only).
        let mut present = [false; 128];
        let mut samples: Vec<Vec<u8>> = Vec::new();
        for line in reader.lines() {
            let line = line?;
            let bytes: Vec<u8> = line.bytes()
                .filter(|&b| (0x20..=0x7E).contains(&b))
                .collect();
            if bytes.is_empty() { continue; }
            for &b in &bytes { present[b as usize] = true; }
            samples.push(bytes);
        }
        if samples.is_empty() {
            bail!("corpus {} yielded zero usable passwords", corpus.display());
        }
        let alphabet: Vec<u8> = (0x20u8..=0x7E)
            .filter(|&b| present[b as usize])
            .collect();
        let n = alphabet.len();
        if n < 2 {
            bail!("corpus alphabet too small ({n} chars)");
        }

        // Byte → class lookup.
        let mut byte2class = [u16::MAX; 128];
        for (i, &b) in alphabet.iter().enumerate() {
            byte2class[b as usize] = i as u16;
        }

        // Laplace-smoothed counts.
        let mut start_counts   = vec![1.0f64; n];
        let mut after1_counts  = vec![1.0f64; n * n];
        let mut full_counts    = vec![1.0f64; n * n * n];
        let start_total_row    = n as f64;
        let mut after1_totals  = vec![n as f64; n];
        let mut full_totals    = vec![n as f64; n * n];

        for pw in &samples {
            let mut prev2 = START_CLASS;
            let mut prev1 = START_CLASS;
            for &b in pw {
                let c = byte2class[b as usize];
                if c == u16::MAX { continue; }
                let c = c as usize;
                match (prev2, prev1) {
                    (s2, s1) if s2 == START_CLASS && s1 == START_CLASS => {
                        start_counts[c] += 1.0;
                    }
                    (s2, s1) if s2 == START_CLASS => {
                        let p1 = s1 as usize;
                        after1_counts[p1 * n + c] += 1.0;
                        after1_totals[p1] += 1.0;
                    }
                    (p2, p1) => {
                        let p2 = p2 as usize; let p1 = p1 as usize;
                        full_counts[p2 * n * n + p1 * n + c] += 1.0;
                        full_totals[p2 * n + p1] += 1.0;
                    }
                }
                prev2 = prev1;
                prev1 = c as u16;
            }
        }

        // Convert to log-probabilities.
        let mut logp_start = vec![0.0f32; n];
        for c in 0..n {
            logp_start[c] = (start_counts[c] / start_total_row).ln() as f32;
        }
        let mut logp_after1 = vec![0.0f32; n * n];
        for p1 in 0..n {
            for c in 0..n {
                logp_after1[p1 * n + c] =
                    (after1_counts[p1 * n + c] / after1_totals[p1]).ln() as f32;
            }
        }
        let mut logp_full = vec![0.0f32; n * n * n];
        for p2 in 0..n {
            for p1 in 0..n {
                for c in 0..n {
                    logp_full[p2 * n * n + p1 * n + c] =
                        (full_counts[p2 * n * n + p1 * n + c]
                            / full_totals[p2 * n + p1]).ln() as f32;
                }
            }
        }

        Ok(Self { alphabet, logp_start, logp_after1, logp_full })
    }

    /// log P(next | prev2, prev1). Accepts `START_CLASS` in the history slots
    /// for the beginning of a password.
    fn logp(&self, prev2: u16, prev1: u16, next: u16) -> f32 {
        let n = self.n();
        let c = next as usize;
        match (prev2, prev1) {
            (s2, s1) if s2 == START_CLASS && s1 == START_CLASS => self.logp_start[c],
            (s2, p1) if s2 == START_CLASS => {
                self.logp_after1[p1 as usize * n + c]
            }
            (p2, p1) => {
                self.logp_full[p2 as usize * n * n + p1 as usize * n + c]
            }
        }
    }
}

// ── Best-first generator ──────────────────────────────────────────────

#[derive(Clone)]
struct Node {
    neg_logp: f32,   // -log p, the heap key (min-heap via Reverse below)
    chars:    Vec<u16>,
    prev2:    u16,
    prev1:    u16,
}

impl Node {
    fn bytes(&self, model: &MarkovModel) -> Vec<u8> {
        self.chars.iter().map(|&c| model.alphabet[c as usize]).collect()
    }
}

/// Min-heap ordering by neg_logp ascending (lowest cost = most probable).
impl PartialEq for Node { fn eq(&self, o: &Self) -> bool { self.neg_logp == o.neg_logp } }
impl Eq for Node {}
impl PartialOrd for Node { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so BinaryHeap (max-heap) becomes a min-heap over neg_logp.
        other.neg_logp.total_cmp(&self.neg_logp)
    }
}

pub struct MarkovSource {
    model:    MarkovModel,
    min_len:  usize,
    max_len:  usize,
    heap:     BinaryHeap<Node>,
    position: u64,
    exhausted: bool,
}

impl MarkovSource {
    pub fn new(model: MarkovModel, min_len: usize, max_len: usize) -> Self {
        assert!(min_len >= 1 && max_len >= min_len);
        let mut heap = BinaryHeap::new();
        heap.push(Node {
            neg_logp: 0.0,
            chars: Vec::new(),
            prev2: START_CLASS,
            prev1: START_CLASS,
        });
        Self { model, min_len, max_len, heap, position: 0, exhausted: false }
    }

    /// Emit the next most-probable candidate. `None` → generator exhausted.
    fn next_candidate(&mut self) -> Option<Vec<u8>> {
        let n = self.model.n() as u16;
        while let Some(node) = self.heap.pop() {
            let len = node.chars.len();

            // Emit if this node is itself a complete candidate.
            if len >= self.min_len && len <= self.max_len {
                let bytes = node.bytes(&self.model);
                // Continue expanding this node's children before returning so
                // neighbours of similar probability queue up correctly.
                if len < self.max_len {
                    self.expand_into_heap(&node, n);
                }
                self.position += 1;
                return Some(bytes);
            }

            // Otherwise only expand.
            if len < self.max_len {
                self.expand_into_heap(&node, n);
            }
        }
        self.exhausted = true;
        None
    }

    fn expand_into_heap(&mut self, node: &Node, n: u16) {
        for c in 0..n {
            let lp = self.model.logp(node.prev2, node.prev1, c);
            let mut chars = node.chars.clone();
            chars.push(c);
            self.heap.push(Node {
                neg_logp: node.neg_logp - lp,
                chars,
                prev2: node.prev1,
                prev1: c,
            });
        }
    }
}

impl BatchSource for MarkovSource {
    fn next_batch(&mut self, out: &mut Vec<Vec<u8>>, limit: usize) -> Result<usize> {
        out.clear();
        while out.len() < limit {
            match self.next_candidate() {
                Some(pw) => out.push(pw),
                None => break,
            }
        }
        Ok(out.len())
    }

    fn is_exhausted(&self) -> bool { self.exhausted }

    /// Coarse resume: replays the heap from scratch and discards `n` outputs.
    /// Cheap enough for typical session positions; documented caveat.
    fn seek(&mut self, n: u64) -> Result<()> {
        // Rebuild from scratch.
        self.heap.clear();
        self.heap.push(Node {
            neg_logp: 0.0,
            chars: Vec::new(),
            prev2: START_CLASS,
            prev1: START_CLASS,
        });
        self.position = 0;
        self.exhausted = false;
        for _ in 0..n {
            if self.next_candidate().is_none() { break; }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_corpus(lines: &[&str]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join("rarpc_markov_corpus.txt");
        let mut f = File::create(&p).unwrap();
        for l in lines { writeln!(f, "{}", l).unwrap(); }
        p
    }

    #[test]
    fn test_unigram_order_at_min1_max1() {
        // Corpus with strong letter skew: "aaaaab" → a should dominate.
        let path = write_corpus(&["aaaaab", "aaaaab", "aaaaab", "b"]);
        let model = MarkovModel::train(&path).unwrap();
        let mut src = MarkovSource::new(model, 1, 1);
        let mut out = Vec::new();
        src.next_batch(&mut out, 2).unwrap();
        // The most frequent start char is 'a'.
        assert_eq!(out[0], b"a");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_serialize_roundtrip() {
        let path = write_corpus(&["hello", "world", "helloworld"]);
        let model = MarkovModel::train(&path).unwrap();
        let tmp = std::env::temp_dir().join("rarpc_markov_model.bin");
        model.save(&tmp).unwrap();
        let m2 = MarkovModel::load(&tmp).unwrap();
        assert_eq!(model.alphabet, m2.alphabet);
        assert_eq!(model.logp_start.len(), m2.logp_start.len());
        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_candidates_all_within_length_bounds() {
        let path = write_corpus(&["abc", "abcd", "xyz"]);
        let model = MarkovModel::train(&path).unwrap();
        let mut src = MarkovSource::new(model, 2, 3);
        let mut out = Vec::new();
        src.next_batch(&mut out, 50).unwrap();
        assert!(!out.is_empty());
        for pw in &out {
            assert!(pw.len() >= 2 && pw.len() <= 3, "length out of bounds: {:?}", pw);
        }
        let _ = std::fs::remove_file(&path);
    }
}
