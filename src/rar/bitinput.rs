//! Bit-level input reader — port of UnRAR `getbits.hpp`/`getbits.cpp`.
//!
//! Reads 16 bits at a time, MSB first, from a flat byte buffer. Pads 8
//! zero-bytes past the end so `getbits()`/`addbits()` can safely peek
//! ahead the fixed amount UnRAR expects.

pub struct BitInput {
    pub buf:     Vec<u8>,
    pub in_addr: usize,
    pub in_bit:  u32,
}

impl BitInput {
    /// Take ownership of `data` and append 8 trailing zero bytes so that
    /// `getbits` is safe even at the very end of the stream.
    pub fn new(mut data: Vec<u8>) -> Self {
        data.extend_from_slice(&[0u8; 8]);
        Self { buf: data, in_addr: 0, in_bit: 0 }
    }

    /// Fully-usable length of the original stream (excluding the 8-byte pad).
    pub fn read_top(&self) -> usize {
        self.buf.len().saturating_sub(8)
    }

    /// Return 16 bits starting at the current bit position, MSB first.
    #[inline]
    pub fn getbits(&self) -> u32 {
        let a = self.in_addr;
        let b0 = self.buf[a] as u32;
        let b1 = self.buf[a + 1] as u32;
        let b2 = self.buf[a + 2] as u32;
        let bit_field = (b0 << 16) | (b1 << 8) | b2;
        (bit_field >> (8 - self.in_bit)) & 0xffff
    }

    #[inline]
    pub fn addbits(&mut self, bits: u32) {
        let total = bits + self.in_bit;
        self.in_addr += (total >> 3) as usize;
        self.in_bit = total & 7;
    }

    #[inline]
    pub fn fgetbits(&self) -> u32 { self.getbits() }

    #[inline]
    pub fn faddbits(&mut self, bits: u32) { self.addbits(bits) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn getbits_msb_first() {
        let mut b = BitInput::new(vec![0xAB, 0xCD, 0xEF]);
        assert_eq!(b.getbits(), 0xABCD);
        b.addbits(4);
        assert_eq!(b.getbits(), 0xBCDE);
        b.addbits(4);
        assert_eq!(b.getbits(), 0xCDEF);
    }

    #[test]
    fn addbits_crosses_byte() {
        let mut b = BitInput::new(vec![0xFF; 4]);
        b.addbits(11);
        assert_eq!(b.in_addr, 1);
        assert_eq!(b.in_bit, 3);
    }
}
