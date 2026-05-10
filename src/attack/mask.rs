/// Mask-based password generator.
///
/// Placeholders:
///   ?l  — lowercase a-z
///   ?u  — uppercase A-Z
///   ?d  — digits 0-9
///   ?s  — printable specials  !@#$%^&*()-_=+[]{}|;:'",.<>?/`~
///   ?a  — all of ?l + ?u + ?d + ?s
///   ?h  — hex digits 0-9a-f
///   Any other character — treated as a literal fixed character.
///
/// Example: `?u?l?l?l?d?d` → one uppercase + three lowercase + two digits.

const LOWER:   &str = "abcdefghijklmnopqrstuvwxyz";
const UPPER:   &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const DIGITS:  &str = "0123456789";
const SPECIAL: &str = "!@#$%^&*()-_=+[]{}|;:'\",.<>?/`~";
const HEX:     &str = "0123456789abcdef";

fn lower()   -> Vec<u8> { LOWER.bytes().collect() }
fn upper()   -> Vec<u8> { UPPER.bytes().collect() }
fn digits()  -> Vec<u8> { DIGITS.bytes().collect() }
fn special() -> Vec<u8> { SPECIAL.bytes().collect() }
fn hex()     -> Vec<u8> { HEX.bytes().collect() }
fn all()     -> Vec<u8> {
    let mut v: Vec<u8> = Vec::with_capacity(95);
    v.extend_from_slice(LOWER.as_bytes());
    v.extend_from_slice(UPPER.as_bytes());
    v.extend_from_slice(DIGITS.as_bytes());
    v.extend_from_slice(SPECIAL.as_bytes());
    v
}

/// One position in the mask.
enum Slot {
    Charset(Vec<u8>),   // variable: cycle over these bytes
    Literal(u8),        // fixed: always this byte
}

pub struct MaskAttack {
    slots: Vec<Slot>,
    indices: Vec<usize>, // current position in each charset slot
    exhausted: bool,
}

impl MaskAttack {
    /// Parse a mask string and build the generator.
    pub fn parse(mask: &str) -> anyhow::Result<Self> {
        let chars: Vec<char> = mask.chars().collect();
        let mut slots: Vec<Slot> = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '?' && i + 1 < chars.len() {
                let slot = match chars[i + 1] {
                    'l' => Slot::Charset(lower()),
                    'u' => Slot::Charset(upper()),
                    'd' => Slot::Charset(digits()),
                    's' => Slot::Charset(special()),
                    'a' => Slot::Charset(all()),
                    'h' => Slot::Charset(hex()),
                    '?' => Slot::Literal(b'?'), // escaped ?
                    c   => return Err(anyhow::anyhow!("Unknown mask placeholder ?{}", c)),
                };
                slots.push(slot);
                i += 2;
            } else {
                slots.push(Slot::Literal(chars[i] as u8));
                i += 1;
            }
        }

        let exhausted = slots.is_empty();
        let indices = vec![0usize; slots.len()];
        Ok(Self { slots, indices, exhausted })
    }

    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    /// Total number of candidates
    pub fn total_space(&self) -> u64 {
        self.slots
            .iter()
            .map(|s| match s {
                Slot::Charset(cs) => cs.len() as u64,
                Slot::Literal(_)  => 1,
            })
            .product()
    }

    fn current_password(&self) -> Vec<u8> {
        self.slots
            .iter()
            .enumerate()
            .map(|(i, s)| match s {
                Slot::Charset(cs) => cs[self.indices[i]],
                Slot::Literal(b)  => *b,
            })
            .collect()
    }

    fn advance(&mut self) {
        // Increment from rightmost variable slot
        let mut carry = true;
        for i in (0..self.slots.len()).rev() {
            if !carry { break; }
            match &self.slots[i] {
                Slot::Literal(_) => continue,
                Slot::Charset(cs) => {
                    self.indices[i] += 1;
                    if self.indices[i] >= cs.len() {
                        self.indices[i] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
        }
        if carry {
            self.exhausted = true;
        }
    }

    /// Advance the generator to position `n` (skip the first n candidates).
    /// Uses mixed-radix decomposition — O(slots) not O(n).
    pub fn seek(&mut self, mut n: u64) {
        // Decompose n into per-slot indices, right-to-left (rightmost varies fastest).
        for i in (0..self.slots.len()).rev() {
            match &self.slots[i] {
                Slot::Literal(_) => {} // fixed position, contributes no carry
                Slot::Charset(cs) => {
                    let base = cs.len() as u64;
                    self.indices[i] = (n % base) as usize;
                    n /= base;
                }
            }
        }
        if n > 0 {
            // n exceeded total search space
            self.exhausted = true;
        }
    }

    pub fn next_batch(&mut self, out: &mut Vec<Vec<u8>>, limit: usize) -> usize {
        out.clear();
        let mut count = 0;
        while count < limit && !self.exhausted {
            out.push(self.current_password());
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
    fn test_mask_literal() {
        let mut m = MaskAttack::parse("abc").unwrap();
        assert_eq!(m.total_space(), 1);
        let mut batch = Vec::new();
        m.next_batch(&mut batch, 10);
        assert_eq!(batch, vec![b"abc".to_vec()]);
    }

    #[test]
    fn test_mask_digits() {
        let mut m = MaskAttack::parse("?d?d").unwrap();
        assert_eq!(m.total_space(), 100);
        let mut batch = Vec::new();
        m.next_batch(&mut batch, 100);
        assert_eq!(&batch[0], b"00");
        assert_eq!(&batch[99], b"99");
    }

    #[test]
    fn test_mask_mixed() {
        let mut m = MaskAttack::parse("A?dB").unwrap();
        assert_eq!(m.total_space(), 10);
        let mut batch = Vec::new();
        m.next_batch(&mut batch, 10);
        assert_eq!(&batch[0], b"A0B");
        assert_eq!(&batch[9], b"A9B");
    }

    #[test]
    fn test_mask_seek_basic() {
        // Seek to position 50 in "?d?d" (00..99) → should start at "50"
        let mut m = MaskAttack::parse("?d?d").unwrap();
        m.seek(50);
        let mut batch = Vec::new();
        m.next_batch(&mut batch, 3);
        assert_eq!(&batch[0], b"50");
        assert_eq!(&batch[1], b"51");
        assert_eq!(&batch[2], b"52");
    }

    #[test]
    fn test_mask_seek_with_literal() {
        // "A?d?d": total_space = 100, seek(10) → "A10"
        let mut m = MaskAttack::parse("A?d?d").unwrap();
        m.seek(10);
        let mut batch = Vec::new();
        m.next_batch(&mut batch, 2);
        assert_eq!(&batch[0], b"A10");
        assert_eq!(&batch[1], b"A11");
    }

    #[test]
    fn test_mask_seek_exhausted() {
        let mut m = MaskAttack::parse("?d").unwrap(); // 10 candidates
        m.seek(10); // past the end
        assert!(m.is_exhausted());
    }

    #[test]
    fn test_mask_seek_resume_matches_sequential() {
        // Verify seek(n) + next_batch gives the same results as sequential generation.
        let mut full = MaskAttack::parse("?l?d").unwrap(); // 26*10 = 260 candidates
        let mut all_full = Vec::new();
        full.next_batch(&mut all_full, 260);

        let mut seeked = MaskAttack::parse("?l?d").unwrap();
        seeked.seek(100);
        let mut from_100 = Vec::new();
        seeked.next_batch(&mut from_100, 160);

        assert_eq!(all_full[100..], from_100[..]);
    }
}
