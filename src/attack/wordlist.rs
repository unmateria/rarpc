use anyhow::Result;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Buffered wordlist reader
pub struct Wordlist {
    reader: BufReader<File>,
    position: u64,
    exhausted: bool,
}

impl Wordlist {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        Ok(Self {
            reader: BufReader::new(file),
            position: 0,
            exhausted: false,
        })
    }

    /// Skip `n` lines to resume from a session
    pub fn seek(&mut self, n: u64) -> Result<()> {
        let mut buf = String::new();
        for _ in 0..n {
            buf.clear();
            let read = self.reader.read_line(&mut buf)?;
            if read == 0 {
                self.exhausted = true;
                break;
            }
            self.position += 1;
        }
        Ok(())
    }

    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    pub fn position(&self) -> u64 {
        self.position
    }

    /// Fill `out` with up to `limit` trimmed non-empty lines.
    /// Returns the number of passwords read.
    pub fn next_batch(&mut self, out: &mut Vec<Vec<u8>>, limit: usize) -> Result<usize> {
        out.clear();
        let mut buf = String::new();
        while out.len() < limit {
            buf.clear();
            let read = self.reader.read_line(&mut buf)?;
            if read == 0 {
                self.exhausted = true;
                break;
            }
            self.position += 1;
            let trimmed = buf.trim_end_matches(['\r', '\n']);
            if !trimmed.is_empty() {
                out.push(trimmed.as_bytes().to_vec());
            }
        }
        Ok(out.len())
    }
}
