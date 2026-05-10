/// Lexicographic brute-force generator over a fixed character set.
///
/// Passwords are generated in the order:
///   (all passwords of min_len) → ... → (all passwords of max_len)
///
/// Resumable: `position` is the global index into the search space.
pub struct BruteForce {
    charset: Vec<u8>,
    min_len: usize,
    max_len: usize,
    current: Vec<usize>,    // index into charset for each position
    current_len: usize,
    exhausted: bool,
}

impl BruteForce {
    pub fn new(charset: &str, min_len: usize, max_len: usize) -> Self {
        let charset: Vec<u8> = charset.bytes().collect();
        let exhausted = charset.is_empty() || min_len > max_len;
        let current = vec![0usize; min_len];
        Self {
            charset,
            min_len,
            max_len,
            current,
            current_len: min_len,
            exhausted,
        }
    }

    /// Resume from an absolute position in the search space
    pub fn seek(&mut self, position: u64) {
        // Count passwords per length
        let base = self.charset.len() as u64;
        let mut remaining = position;

        self.current_len = self.min_len;
        while self.current_len <= self.max_len {
            let count = base.pow(self.current_len as u32);
            if remaining < count {
                break;
            }
            remaining -= count;
            self.current_len += 1;
        }

        if self.current_len > self.max_len {
            self.exhausted = true;
            return;
        }

        // Decode `remaining` as a mixed-radix number
        self.current = vec![0usize; self.current_len];
        for i in (0..self.current_len).rev() {
            self.current[i] = (remaining % base) as usize;
            remaining /= base;
        }
    }

    /// Total search space size
    pub fn total_space(&self) -> u64 {
        let base = self.charset.len() as u64;
        (self.min_len..=self.max_len)
            .map(|len| base.pow(len as u32))
            .sum()
    }

    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    fn advance(&mut self) {
        // Increment the rightmost counter, carrying left
        let mut carry = true;
        for i in (0..self.current_len).rev() {
            if carry {
                self.current[i] += 1;
                if self.current[i] >= self.charset.len() {
                    self.current[i] = 0;
                } else {
                    carry = false;
                    break;
                }
            }
        }
        if carry {
            // Overflow at current length → increment length
            self.current_len += 1;
            if self.current_len > self.max_len {
                self.exhausted = true;
            } else {
                self.current = vec![0usize; self.current_len];
            }
        }
    }

    /// Fill `out` with the next batch of passwords.
    /// Returns the number of passwords actually written.
    pub fn next_batch(&mut self, out: &mut Vec<Vec<u8>>, limit: usize) -> usize {
        let mut count = 0;
        out.clear();
        while count < limit && !self.exhausted {
            let pw: Vec<u8> = self.current.iter().map(|&i| self.charset[i]).collect();
            out.push(pw);
            count += 1;
            self.advance();
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bruteforce_order() {
        let mut bf = BruteForce::new("ab", 1, 2);
        let mut batch = Vec::new();
        bf.next_batch(&mut batch, 100);
        let words: Vec<_> = batch.iter().map(|p| String::from_utf8(p.clone()).unwrap()).collect();
        assert_eq!(words, ["a", "b", "aa", "ab", "ba", "bb"]);
    }

    #[test]
    fn test_bruteforce_seek() {
        let mut bf = BruteForce::new("ab", 1, 2);
        bf.seek(3); // "ab" is at index 3 (a=0, b=1, aa=2, ab=3)
        let mut batch = Vec::new();
        bf.next_batch(&mut batch, 1);
        assert_eq!(batch[0], b"ab");
    }

    #[test]
    fn test_total_space() {
        let bf = BruteForce::new("ab", 1, 2);
        // 2^1 + 2^2 = 6
        assert_eq!(bf.total_space(), 6);
    }
}
