//! Unified generator interface — lets the engine iterate over any candidate
//! stream (wordlist / brute / mask / rules / markov) through a single trait.
//!
//! The trait is intentionally tiny: it mirrors the methods every existing
//! generator already has. Impls are one-liners that delegate to the inherent
//! methods, so there is no behaviour change for existing code paths.

use anyhow::Result;

use crate::attack::bruteforce::BruteForce;
use crate::attack::mask::MaskAttack;
use crate::attack::wordlist::Wordlist;

/// Produces batches of candidate passwords. Used by the pipelined engine.
pub trait BatchSource {
    /// Fill `out` with up to `limit` candidates. Returns the count written.
    fn next_batch(&mut self, out: &mut Vec<Vec<u8>>, limit: usize) -> Result<usize>;

    /// True once no more candidates will ever be produced.
    fn is_exhausted(&self) -> bool;

    /// Skip forward `n` candidates (session resume).
    fn seek(&mut self, n: u64) -> Result<()>;
}

impl BatchSource for Wordlist {
    fn next_batch(&mut self, out: &mut Vec<Vec<u8>>, limit: usize) -> Result<usize> {
        Wordlist::next_batch(self, out, limit)
    }
    fn is_exhausted(&self) -> bool { Wordlist::is_exhausted(self) }
    fn seek(&mut self, n: u64) -> Result<()> { Wordlist::seek(self, n) }
}

impl BatchSource for BruteForce {
    fn next_batch(&mut self, out: &mut Vec<Vec<u8>>, limit: usize) -> Result<usize> {
        Ok(BruteForce::next_batch(self, out, limit))
    }
    fn is_exhausted(&self) -> bool { BruteForce::is_exhausted(self) }
    fn seek(&mut self, n: u64) -> Result<()> { BruteForce::seek(self, n); Ok(()) }
}

impl BatchSource for MaskAttack {
    fn next_batch(&mut self, out: &mut Vec<Vec<u8>>, limit: usize) -> Result<usize> {
        Ok(MaskAttack::next_batch(self, out, limit))
    }
    fn is_exhausted(&self) -> bool { MaskAttack::is_exhausted(self) }
    fn seek(&mut self, n: u64) -> Result<()> { MaskAttack::seek(self, n); Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trait_over_bruteforce() {
        let mut bf = BruteForce::new("ab", 1, 1);
        let src: &mut dyn BatchSource = &mut bf;
        let mut out = Vec::new();
        let n = src.next_batch(&mut out, 10).unwrap();
        assert_eq!(n, 2);
        assert!(src.is_exhausted());
    }

    #[test]
    fn trait_over_mask() {
        let mut m = MaskAttack::parse("?d").unwrap();
        let src: &mut dyn BatchSource = &mut m;
        let mut out = Vec::new();
        src.next_batch(&mut out, 20).unwrap();
        assert_eq!(out.len(), 10);
    }
}
