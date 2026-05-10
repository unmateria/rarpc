//! Combinator attack — emits single words, word-pair, and optionally triple
//! concatenations.
//!
//! Given a wordlist W = [w0, w1, …, w(n-1)] the candidate stream is:
//!
//!   w0, w0w0, w0w1, …, w0w(n-1),        [optionally: w0w0w0, w0w0w1, …]
//!   w1, w1w0, w1w1, …, w1w(n-1),        [optionally: w1w0w0, w1w0w1, …]
//!   …
//!
//! Without triple: total = n + n². With triple: total = n + n² + n³.
//!
//! Layered with `RulesSource` (see `attack::rules`), the combinator output
//! also gets suffixed/prefixed/leet-substituted, so users can tick
//! "combinar" + "añadir dígitos" and get candidates like `amigopedro99`.

use anyhow::Result;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::attack::source::BatchSource;

pub struct CombinatorSource {
    words: Vec<Vec<u8>>,
    triple: bool,
    /// `i` in range `0..=n`. `stage` says what to emit next:
    ///   0 → bare word[i]; switches to 1 after emission.
    ///   1 → word[i] || word[j]; iterate j=0..n, then stage=2 (if triple) or i+=1.
    ///   2 → word[i] || word[j2] || word[k]; iterate (j2,k) pairs, then i+=1.
    stage:    u8,
    i:        usize,
    j:        usize,
    j2:       usize,
    k:        usize,
    position: u64,
}

impl CombinatorSource {
    pub fn from_wordlist(path: &Path, triple: bool) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut words = Vec::new();
        for line in reader.lines() {
            let line = line?;
            let t = line.trim_end_matches(['\r', '\n']);
            if !t.is_empty() {
                words.push(t.as_bytes().to_vec());
            }
        }
        Ok(Self { words, triple, stage: 0, i: 0, j: 0, j2: 0, k: 0, position: 0 })
    }

    pub fn word_count(&self) -> usize { self.words.len() }

    pub fn total(&self) -> u64 {
        let n = self.words.len() as u64;
        if self.triple {
            n + n * n + n * n * n
        } else {
            n + n * n
        }
    }

    fn per_outer(&self) -> u64 {
        let n = self.words.len() as u64;
        if self.triple { 1 + n + n * n } else { 1 + n }
    }
}

impl BatchSource for CombinatorSource {
    fn next_batch(&mut self, out: &mut Vec<Vec<u8>>, limit: usize) -> Result<usize> {
        out.clear();
        let n = self.words.len();
        if n == 0 { return Ok(0); }
        while out.len() < limit && self.i < n {
            if self.stage == 0 {
                out.push(self.words[self.i].clone());
                self.stage = 1;
                self.j = 0;
                self.position += 1;
                continue;
            }
            if self.stage == 1 {
                let mut buf = Vec::with_capacity(
                    self.words[self.i].len() + self.words[self.j].len(),
                );
                buf.extend_from_slice(&self.words[self.i]);
                buf.extend_from_slice(&self.words[self.j]);
                out.push(buf);
                self.position += 1;
                self.j += 1;
                if self.j >= n {
                    if self.triple {
                        self.stage = 2;
                        self.j2 = 0;
                        self.k = 0;
                    } else {
                        self.i += 1;
                        self.stage = 0;
                    }
                }
                continue;
            }
            // stage == 2: triples word[i] || word[j2] || word[k]
            let mut buf = Vec::with_capacity(
                self.words[self.i].len() + self.words[self.j2].len() + self.words[self.k].len(),
            );
            buf.extend_from_slice(&self.words[self.i]);
            buf.extend_from_slice(&self.words[self.j2]);
            buf.extend_from_slice(&self.words[self.k]);
            out.push(buf);
            self.position += 1;
            self.k += 1;
            if self.k >= n {
                self.k = 0;
                self.j2 += 1;
                if self.j2 >= n {
                    self.i += 1;
                    self.stage = 0;
                }
            }
        }
        Ok(out.len())
    }

    fn is_exhausted(&self) -> bool {
        self.i >= self.words.len()
    }

    fn seek(&mut self, n_skip: u64) -> Result<()> {
        let n = self.words.len() as u64;
        if n == 0 { return Ok(()); }
        let per_outer = self.per_outer();
        let outer = n_skip / per_outer;
        let rem   = n_skip % per_outer;
        self.i = outer as usize;
        if rem == 0 {
            self.stage = 0;
            self.j = 0;
            self.j2 = 0;
            self.k = 0;
        } else if rem <= n {
            // Within pairs (rem=1 means first pair j=0)
            self.stage = 1;
            self.j = (rem - 1) as usize;
            self.j2 = 0;
            self.k = 0;
        } else {
            // Within triples: rem - 1 - n positions into the n^2 block
            let triple_pos = rem - 1 - n;
            self.stage = 2;
            self.j = 0;
            self.j2 = (triple_pos / n) as usize;
            self.k = (triple_pos % n) as usize;
        }
        self.position = n_skip;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_wl(tag: &str, lines: &[&str]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "rarpc_comb_test_{}_{}.txt",
            std::process::id(),
            tag,
        ));
        let mut f = File::create(&p).unwrap();
        for l in lines { writeln!(f, "{}", l).unwrap(); }
        p
    }

    #[test]
    fn emits_singles_and_pairs() {
        let p = write_wl("singles", &["a", "b"]);
        let mut src = CombinatorSource::from_wordlist(&p, false).unwrap();
        let mut out = Vec::new();
        src.next_batch(&mut out, 100).unwrap();
        let got: Vec<String> = out.iter()
            .map(|b| String::from_utf8(b.clone()).unwrap()).collect();
        assert_eq!(got, vec!["a", "aa", "ab", "b", "ba", "bb"]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn total_counts_n_plus_n_squared() {
        let p = write_wl("total", &["x", "y", "z"]);
        let src = CombinatorSource::from_wordlist(&p, false).unwrap();
        assert_eq!(src.total(), 3 + 9);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn seek_roundtrip() {
        let p = write_wl("seek", &["a", "b", "c"]);
        let mut full = CombinatorSource::from_wordlist(&p, false).unwrap();
        let mut all = Vec::new();
        full.next_batch(&mut all, 100).unwrap();
        assert_eq!(all.len(), 3 + 9);

        let mut partial = CombinatorSource::from_wordlist(&p, false).unwrap();
        partial.seek(5).unwrap();
        let mut rest = Vec::new();
        partial.next_batch(&mut rest, 100).unwrap();
        assert_eq!(rest, all[5..].to_vec());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn triple_emits_all() {
        let p = write_wl("triple", &["a", "b"]);
        let mut src = CombinatorSource::from_wordlist(&p, true).unwrap();
        let mut out = Vec::new();
        src.next_batch(&mut out, 200).unwrap();
        let got: Vec<String> = out.iter()
            .map(|b| String::from_utf8(b.clone()).unwrap()).collect();
        // n=2: singles=2, pairs=4, triples=8 → total=14
        assert_eq!(got.len(), 2 + 4 + 8);
        assert_eq!(src.total(), 14);
        // Order: a, aa, ab, aaa, aab, aba, abb, b, ba, bb, baa, bab, bba, bbb
        assert_eq!(got[0], "a");
        assert_eq!(got[1], "aa");
        assert_eq!(got[2], "ab");
        assert_eq!(got[3], "aaa");
        assert_eq!(got[6], "abb");
        assert_eq!(got[7], "b");
        assert_eq!(got[13], "bbb");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn triple_seek_roundtrip() {
        let p = write_wl("triple_seek", &["a", "b"]);
        let mut full = CombinatorSource::from_wordlist(&p, true).unwrap();
        let mut all = Vec::new();
        full.next_batch(&mut all, 200).unwrap();
        assert_eq!(all.len(), 14);

        for skip in [0, 1, 3, 7, 10, 13] {
            let mut partial = CombinatorSource::from_wordlist(&p, true).unwrap();
            partial.seek(skip).unwrap();
            let mut rest = Vec::new();
            partial.next_batch(&mut rest, 200).unwrap();
            assert_eq!(rest, all[skip as usize..].to_vec(),
                "seek({}) mismatch", skip);
        }
        let _ = std::fs::remove_file(&p);
    }
}
