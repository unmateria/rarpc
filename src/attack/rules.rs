//! Rules engine — standard transformation subset.
//!
//! A rule is a sequence of single-character operators applied to an input
//! password. We expose a small but practical subset so popular rulesets
//! (`best64.rule`, `leetspeak.rule`, …) can be ingested without modification:
//!
//!   `:`    noop (keep the word unchanged)
//!   `l`    lowercase all
//!   `u`    uppercase all
//!   `c`    capitalize (first upper, rest lower)
//!   `C`    invert-capitalize (first lower, rest upper)
//!   `t`    toggle case on every char
//!   `r`    reverse
//!   `d`    duplicate (pw + pw)
//!   `$X`   append literal byte X
//!   `^X`   prepend literal byte X
//!   `sXY`  substitute every X with Y
//!   `TN`   toggle case of char at position N (N in 0-9A-Z hex-like)
//!
//! `RulesSource` wraps another `BatchSource` and emits, for every input word,
//! every rule in the ruleset. Total candidate count = words × rules.

use anyhow::{anyhow, bail, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::attack::source::BatchSource;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    Noop,
    Lower,
    Upper,
    Capitalize,
    InvCapitalize,
    ToggleAll,
    Reverse,
    Duplicate,
    Append(u8),
    Prepend(u8),
    Substitute(u8, u8),
    TogglePos(u8),
    RotateLeft,
    RotateRight,
    DeleteFirst,
    DeleteLast,
    DeletePos(u8),
    InsertPos(u8, u8),
    Purge(u8),
}

/// A rule is a sequence of ops applied left-to-right.
#[derive(Debug, Clone, Default)]
pub struct Rule {
    pub ops: Vec<Op>,
}

impl Rule {
    /// Apply the rule in-place to `pw`, clearing `pw` first and filling with
    /// the transformed bytes derived from `src`.
    pub fn apply(&self, src: &[u8], out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(src);
        for op in &self.ops {
            apply_op(op, out);
        }
    }
}

fn apply_op(op: &Op, buf: &mut Vec<u8>) {
    match op {
        Op::Noop => {}
        Op::Lower => {
            for b in buf.iter_mut() { if b.is_ascii_uppercase() { *b += 32; } }
        }
        Op::Upper => {
            for b in buf.iter_mut() { if b.is_ascii_lowercase() { *b -= 32; } }
        }
        Op::Capitalize => {
            for (i, b) in buf.iter_mut().enumerate() {
                if i == 0 {
                    if b.is_ascii_lowercase() { *b -= 32; }
                } else if b.is_ascii_uppercase() {
                    *b += 32;
                }
            }
        }
        Op::InvCapitalize => {
            for (i, b) in buf.iter_mut().enumerate() {
                if i == 0 {
                    if b.is_ascii_uppercase() { *b += 32; }
                } else if b.is_ascii_lowercase() {
                    *b -= 32;
                }
            }
        }
        Op::ToggleAll => {
            for b in buf.iter_mut() {
                if b.is_ascii_lowercase()      { *b -= 32; }
                else if b.is_ascii_uppercase() { *b += 32; }
            }
        }
        Op::Reverse => buf.reverse(),
        Op::Duplicate => {
            let n = buf.len();
            buf.reserve(n);
            for i in 0..n { buf.push(buf[i]); }
        }
        Op::Append(c) => buf.push(*c),
        Op::Prepend(c) => buf.insert(0, *c),
        Op::Substitute(x, y) => {
            for b in buf.iter_mut() { if *b == *x { *b = *y; } }
        }
        Op::TogglePos(n) => {
            let i = *n as usize;
            if i < buf.len() {
                let b = &mut buf[i];
                if b.is_ascii_lowercase()      { *b -= 32; }
                else if b.is_ascii_uppercase() { *b += 32; }
            }
        }
        Op::RotateLeft => {
            if buf.len() > 1 {
                let first = buf[0];
                buf.remove(0);
                buf.push(first);
            }
        }
        Op::RotateRight => {
            if buf.len() > 1 {
                let last = buf[buf.len() - 1];
                buf.pop();
                buf.insert(0, last);
            }
        }
        Op::DeleteFirst => { if !buf.is_empty() { buf.remove(0); } }
        Op::DeleteLast  => { buf.pop(); }
        Op::DeletePos(n) => {
            let i = *n as usize;
            if i < buf.len() { buf.remove(i); }
        }
        Op::InsertPos(n, c) => {
            let i = (*n as usize).min(buf.len());
            buf.insert(i, *c);
        }
        Op::Purge(c) => { buf.retain(|b| *b != *c); }
    }
}

/// Parse an `N` position character: digits map 0-9, uppercase
/// letters map A→10, B→11, … Z→35. Used by operators like `T0`, `TA`, etc.
fn parse_pos(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'A'..=b'Z' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// Parse a single rule line. Whitespace between ops is allowed
/// (a plain `$1` is just one op).
pub fn parse_rule(line: &str) -> Result<Rule> {
    let bytes = line.as_bytes();
    let mut ops = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b' ' || b == b'\t' { i += 1; continue; }
        match b {
            b':' => { ops.push(Op::Noop); i += 1; }
            b'l' => { ops.push(Op::Lower); i += 1; }
            b'u' => { ops.push(Op::Upper); i += 1; }
            b'c' => { ops.push(Op::Capitalize); i += 1; }
            b'C' => { ops.push(Op::InvCapitalize); i += 1; }
            b't' => { ops.push(Op::ToggleAll); i += 1; }
            b'r' => { ops.push(Op::Reverse); i += 1; }
            b'd' => { ops.push(Op::Duplicate); i += 1; }
            b'$' => {
                let c = *bytes.get(i + 1)
                    .ok_or_else(|| anyhow!("rule '$' missing argument: {:?}", line))?;
                ops.push(Op::Append(c)); i += 2;
            }
            b'^' => {
                let c = *bytes.get(i + 1)
                    .ok_or_else(|| anyhow!("rule '^' missing argument: {:?}", line))?;
                ops.push(Op::Prepend(c)); i += 2;
            }
            b's' => {
                let x = *bytes.get(i + 1)
                    .ok_or_else(|| anyhow!("rule 's' missing source: {:?}", line))?;
                let y = *bytes.get(i + 2)
                    .ok_or_else(|| anyhow!("rule 's' missing dest: {:?}", line))?;
                ops.push(Op::Substitute(x, y)); i += 3;
            }
            b'T' => {
                let p = *bytes.get(i + 1)
                    .ok_or_else(|| anyhow!("rule 'T' missing position: {:?}", line))?;
                let pos = parse_pos(p)
                    .ok_or_else(|| anyhow!("rule 'T' bad position {:?}", p as char))?;
                ops.push(Op::TogglePos(pos)); i += 2;
            }
            b'{' => { ops.push(Op::RotateLeft);  i += 1; }
            b'}' => { ops.push(Op::RotateRight); i += 1; }
            b'[' => { ops.push(Op::DeleteFirst); i += 1; }
            b']' => { ops.push(Op::DeleteLast);  i += 1; }
            b'D' => {
                let p = *bytes.get(i + 1)
                    .ok_or_else(|| anyhow!("rule 'D' missing position: {:?}", line))?;
                let pos = parse_pos(p)
                    .ok_or_else(|| anyhow!("rule 'D' bad position {:?}", p as char))?;
                ops.push(Op::DeletePos(pos)); i += 2;
            }
            b'i' => {
                let p = *bytes.get(i + 1)
                    .ok_or_else(|| anyhow!("rule 'i' missing position: {:?}", line))?;
                let pos = parse_pos(p)
                    .ok_or_else(|| anyhow!("rule 'i' bad position {:?}", p as char))?;
                let c = *bytes.get(i + 2)
                    .ok_or_else(|| anyhow!("rule 'i' missing char: {:?}", line))?;
                ops.push(Op::InsertPos(pos, c)); i += 3;
            }
            b'@' => {
                let c = *bytes.get(i + 1)
                    .ok_or_else(|| anyhow!("rule '@' missing char: {:?}", line))?;
                ops.push(Op::Purge(c)); i += 2;
            }
            _ => bail!("unsupported rule operator {:?} in {:?}", b as char, line),
        }
    }
    Ok(Rule { ops })
}

/// Parse a full rulebook — one rule per line, `#` and blank lines ignored.
/// Rules that contain ops outside the supported subset are skipped with a
/// warning printed to stderr (so `best64.rule` mostly works).
pub fn load_rules(path: &Path) -> Result<Vec<Rule>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut rules = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') { continue; }
        match parse_rule(trimmed) {
            Ok(r)  => rules.push(r),
            Err(e) => eprintln!(
                "warning: skipping rule at {}:{}: {}",
                path.display(), lineno + 1, e
            ),
        }
    }
    if rules.is_empty() {
        bail!("no usable rules parsed from {}", path.display());
    }
    Ok(rules)
}

/// `RulesSource` wraps another source (typically `Wordlist`) and emits every
/// `(word, rule)` pair as a candidate password.
///
/// Batch order for input words W0, W1 and rules R0, R1, R2 is:
///   R0(W0), R1(W0), R2(W0), R0(W1), R1(W1), R2(W1), …
/// i.e. rules vary fastest. `position` counts pairs consumed (rules*word_idx
/// + rule_idx) for session resume.
pub struct RulesSource<S: BatchSource> {
    inner:        S,
    rules:        Vec<Rule>,
    /// Buffered input word (mutated by `inner.next_batch` via word_scratch).
    current_word: Option<Vec<u8>>,
    /// Next rule index to apply to `current_word`.
    next_rule:    usize,
    /// Scratch buffer for rule output — reused to avoid per-apply allocation.
    out_scratch:  Vec<u8>,
    /// Pre-fetched word queue to reduce per-word source.next_batch overhead.
    word_queue:   Vec<Vec<u8>>,
    word_idx:     usize,
    position:     u64,
}

impl<S: BatchSource> RulesSource<S> {
    pub fn new(inner: S, rules: Vec<Rule>) -> Self {
        assert!(!rules.is_empty(), "RulesSource needs ≥ 1 rule");
        Self {
            inner, rules,
            current_word: None, next_rule: 0,
            out_scratch: Vec::new(),
            word_queue: Vec::new(), word_idx: 0,
            position: 0,
        }
    }

    pub fn position(&self) -> u64 { self.position }

    fn refill_queue(&mut self) -> Result<()> {
        const WORD_BATCH: usize = 4096;
        self.word_queue.clear();
        self.word_idx = 0;
        let _ = self.inner.next_batch(&mut self.word_queue, WORD_BATCH)?;
        Ok(())
    }
}

impl<S: BatchSource> BatchSource for RulesSource<S> {
    fn next_batch(&mut self, out: &mut Vec<Vec<u8>>, limit: usize) -> Result<usize> {
        out.clear();
        let n_rules = self.rules.len();
        while out.len() < limit {
            // Ensure we have a current word to apply rules to.
            if self.current_word.is_none() {
                if self.word_idx >= self.word_queue.len() {
                    self.refill_queue()?;
                    if self.word_queue.is_empty() { break; }
                }
                self.current_word = Some(std::mem::take(&mut self.word_queue[self.word_idx]));
                self.word_idx += 1;
                self.next_rule = 0;
            }
            let word = self.current_word.as_ref().unwrap();
            // Emit rules[self.next_rule..n_rules].
            while self.next_rule < n_rules && out.len() < limit {
                self.rules[self.next_rule].apply(word, &mut self.out_scratch);
                out.push(self.out_scratch.clone());
                self.next_rule += 1;
                self.position += 1;
            }
            if self.next_rule >= n_rules {
                self.current_word = None;
            }
        }
        Ok(out.len())
    }

    fn is_exhausted(&self) -> bool {
        self.current_word.is_none()
            && self.word_idx >= self.word_queue.len()
            && self.inner.is_exhausted()
    }

    fn seek(&mut self, n: u64) -> Result<()> {
        let n_rules = self.rules.len() as u64;
        let skip_words = n / n_rules;
        let skip_rules = (n % n_rules) as usize;
        self.inner.seek(skip_words)?;
        self.position = n;
        self.next_rule = skip_rules;
        // Prime current_word if we need to start mid-rule.
        if skip_rules > 0 {
            let mut tmp = Vec::new();
            self.inner.next_batch(&mut tmp, 1)?;
            if let Some(w) = tmp.into_iter().next() {
                self.current_word = Some(w);
            } else {
                self.next_rule = 0;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attack::wordlist::Wordlist;
    use std::io::Write;

    fn apply(rule_str: &str, input: &str) -> String {
        let r = parse_rule(rule_str).unwrap();
        let mut out = Vec::new();
        r.apply(input.as_bytes(), &mut out);
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn test_noop() { assert_eq!(apply(":", "Foo"), "Foo"); }

    #[test]
    fn test_case_ops() {
        assert_eq!(apply("l", "FooBAR"), "foobar");
        assert_eq!(apply("u", "FooBAR"), "FOOBAR");
        assert_eq!(apply("c", "fooBAR"), "Foobar");
        assert_eq!(apply("C", "fooBAR"), "fOOBAR");
        assert_eq!(apply("t", "FooBAR"), "fOObar");
    }

    #[test]
    fn test_reverse_dup() {
        assert_eq!(apply("r", "abc"),  "cba");
        assert_eq!(apply("d", "abc"),  "abcabc");
    }

    #[test]
    fn test_append_prepend() {
        assert_eq!(apply("$1", "cat"),  "cat1");
        assert_eq!(apply("^!", "cat"),  "!cat");
    }

    #[test]
    fn test_substitute() {
        assert_eq!(apply("sa@", "banana"), "b@n@n@");
    }

    #[test]
    fn test_toggle_pos() {
        assert_eq!(apply("T0", "cat"), "Cat");
        assert_eq!(apply("T2", "cat"), "caT");
    }

    #[test]
    fn test_combined() {
        assert_eq!(apply("c$1", "password"), "Password1");
        assert_eq!(apply("l$!", "HELLO"),    "hello!");
    }

    #[test]
    fn test_rotate() {
        assert_eq!(apply("{", "abcde"), "bcdea");
        assert_eq!(apply("}", "abcde"), "eabcd");
    }

    #[test]
    fn test_delete() {
        assert_eq!(apply("[", "hello"), "ello");
        assert_eq!(apply("]", "hello"), "hell");
        assert_eq!(apply("D2", "hello"), "helo");
    }

    #[test]
    fn test_insert_pos() {
        assert_eq!(apply("i0!", "cat"), "!cat");
        assert_eq!(apply("i2X", "cat"), "caXt");
    }

    #[test]
    fn test_purge() {
        assert_eq!(apply("@a", "banana"), "bnn");
    }

    #[test]
    fn test_rules_source_basic() {
        let dir = std::env::temp_dir();
        let wl_path = dir.join("rarpc_rules_test.txt");
        {
            let mut f = File::create(&wl_path).unwrap();
            writeln!(f, "cat").unwrap();
            writeln!(f, "dog").unwrap();
        }
        let wl = Wordlist::open(&wl_path).unwrap();
        let rules = vec![parse_rule(":").unwrap(), parse_rule("u").unwrap()];
        let mut src = RulesSource::new(wl, rules);
        let mut out = Vec::new();
        src.next_batch(&mut out, 10).unwrap();
        let words: Vec<_> = out.iter().map(|b| String::from_utf8(b.clone()).unwrap()).collect();
        assert_eq!(words, ["cat", "CAT", "dog", "DOG"]);
        let _ = std::fs::remove_file(&wl_path);
    }
}
