/*
 * Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//! Low-level VVC (H.266) bitstream writing: an MSB-first bit writer with
//! Exp-Golomb coding, RBSP trailing bits, emulation-prevention, and NAL-unit
//! framing. Everything above this layer (SPS/PPS/picture header/slice) is just
//! a sequence of `u(n)`, `ue(v)`, and `se(v)` syntax elements written here.
//!
//! Some items here are consumed by later pipeline stages (parameter sets, slice
//! headers) that are not yet wired up; `dead_code` is allowed module-wide until
//! then.
#![allow(dead_code)]

/// VVC NAL unit types (H.266 Table 5). Only the subset needed to emit a still
/// intra picture is enumerated; the numeric value is `nuh_unit_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum NalUnitType {
    /// Coded slice of an IDR picture, with leading RADL pictures permitted.
    IdrWRadl = 7,
    /// Coded slice of an IDR picture with no leading pictures.
    IdrNLp = 8,
    /// Operating point information.
    Opi = 12,
    /// Decoding capability information.
    Dci = 13,
    /// Video parameter set.
    Vps = 14,
    /// Sequence parameter set.
    Sps = 15,
    /// Picture parameter set.
    Pps = 16,
    /// Adaptation parameter set (prefix).
    PrefixAps = 17,
    /// Picture header.
    Ph = 19,
    /// Access unit delimiter.
    Aud = 20,
    /// Prefix SEI.
    PrefixSei = 23,
}

impl NalUnitType {
    pub(crate) fn value(self) -> u8 {
        self as u8
    }
}

/// MSB-first bit writer producing a raw byte stream (RBSP payload, pre-emulation).
#[derive(Default)]
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    /// Partially filled current byte, MSB-first.
    cur: u8,
    /// Number of bits already filled in `cur` (0..8).
    nbits: u8,
}

impl BitWriter {
    pub(crate) fn new() -> Self {
        BitWriter::default()
    }

    /// True when the next write would start a fresh byte (i.e. byte-aligned).
    pub(crate) fn is_byte_aligned(&self) -> bool {
        self.nbits == 0
    }

    /// Total bits written so far.
    pub(crate) fn bit_len(&self) -> usize {
        self.bytes.len() * 8 + self.nbits as usize
    }

    /// Write a single bit (`u(1)`).
    #[inline]
    pub(crate) fn put_bit(&mut self, bit: u32) {
        self.cur = (self.cur << 1) | (bit as u8 & 1);
        self.nbits += 1;
        if self.nbits == 8 {
            self.bytes.push(self.cur);
            self.cur = 0;
            self.nbits = 0;
        }
    }

    /// Write the low `n` bits of `value`, MSB-first (`u(n)`). `n` may be 0..=32.
    pub(crate) fn put_bits(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        let mut i = n;
        while i > 0 {
            i -= 1;
            self.put_bit((value >> i) & 1);
        }
    }

    /// Unsigned Exp-Golomb (`ue(v)`), H.266 §9.2.
    pub(crate) fn put_ue(&mut self, value: u32) {
        // codeNum = value; write leadingZeroBits then the (leadingZeroBits+1)-bit
        // representation of value+1.
        let v = value as u64 + 1;
        let len = 64 - v.leading_zeros(); // bit length of (value+1)
        // (len-1) leading zero bits, then the value+1 in `len` bits.
        for _ in 0..(len - 1) {
            self.put_bit(0);
        }
        // value+1 fits in `len` bits; emit it.
        let mut i = len;
        while i > 0 {
            i -= 1;
            self.put_bit(((v >> i) & 1) as u32);
        }
    }

    /// Signed Exp-Golomb (`se(v)`), H.266 §9.2.1. Mapping: 0,1,-1,2,-2,... ->
    /// codeNum 0,1,2,3,4,...
    pub(crate) fn put_se(&mut self, value: i32) {
        let code = if value <= 0 {
            (-(value as i64) as u64) * 2
        } else {
            (value as u64) * 2 - 1
        };
        self.put_ue(code as u32);
    }

    /// Append `rbsp_stop_one_bit` followed by zero alignment bits (H.266 §7.3.1.1).
    pub(crate) fn rbsp_trailing_bits(&mut self) {
        self.put_bit(1);
        while self.nbits != 0 {
            self.put_bit(0);
        }
    }

    /// Finish and return the RBSP bytes. Asserts byte alignment (call
    /// [`rbsp_trailing_bits`](Self::rbsp_trailing_bits) or [`byte_align`](Self::byte_align)
    /// first).
    pub(crate) fn into_bytes(self) -> Vec<u8> {
        debug_assert!(self.nbits == 0, "BitWriter not byte-aligned at finish");
        self.bytes
    }

    /// Pad with zero bits to the next byte boundary (alignment without a stop bit).
    pub(crate) fn byte_align(&mut self) {
        while self.nbits != 0 {
            self.put_bit(0);
        }
    }
}

/// Insert RBSP emulation-prevention bytes (H.266 §7.4.2): any `00 00 00`,
/// `00 00 01`, `00 00 02`, or `00 00 03` in the RBSP has a `0x03` inserted after
/// the second zero, becoming `00 00 03 xx`.
pub(crate) fn rbsp_to_ebsp(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + rbsp.len() / 16 + 4);
    let mut zeros = 0u32;
    for &b in rbsp {
        if zeros >= 2 && b <= 0x03 {
            out.push(0x03);
            zeros = 0;
        }
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        out.push(b);
    }
    out
}

/// Inverse of [`rbsp_to_ebsp`]: strip emulation-prevention `0x03` bytes from a
/// NAL payload to recover the raw RBSP.
pub(crate) fn ebsp_to_rbsp(ebsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ebsp.len());
    let mut zeros = 0;
    let mut i = 0;
    while i < ebsp.len() {
        let b = ebsp[i];
        if zeros >= 2 && b == 0x03 && i + 1 < ebsp.len() && ebsp[i + 1] <= 0x03 {
            zeros = 0; // drop the emulation_prevention_three_byte
        } else {
            out.push(b);
            zeros = if b == 0 { zeros + 1 } else { 0 };
        }
        i += 1;
    }
    out
}

/// Wrap an RBSP payload into a complete NAL unit: 2-byte NAL header (H.266
/// §7.3.1.2) + emulation-prevented payload. `temporal_id_plus1` is normally 1.
pub(crate) fn write_nal(nut: NalUnitType, rbsp: &[u8], temporal_id_plus1: u8) -> Vec<u8> {
    // forbidden_zero_bit(1)=0, nuh_reserved_zero_bit(1)=0, nuh_layer_id(6)=0,
    // nuh_unit_type(5), nuh_temporal_id_plus1(3).
    let mut hdr = BitWriter::new();
    hdr.put_bit(0); // forbidden_zero_bit
    hdr.put_bit(0); // nuh_reserved_zero_bit
    hdr.put_bits(0, 6); // nuh_layer_id
    hdr.put_bits(nut.value() as u32, 5);
    hdr.put_bits(temporal_id_plus1 as u32, 3);
    let header = hdr.into_bytes();

    let mut nal = Vec::with_capacity(2 + rbsp.len() + 8);
    nal.extend_from_slice(&header);
    nal.extend_from_slice(&rbsp_to_ebsp(rbsp));
    nal
}

/// Prefix a NAL unit with the 4-byte Annex-B start code `00 00 00 01`.
pub(crate) fn annexb(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len() + 4);
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.extend_from_slice(nal);
    out
}

/// MSB-first bit reader, the inverse of [`BitWriter`]. Used to verify written
/// parameter sets and (later) to parse them.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    bitpos: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        BitReader { data, bitpos: 0 }
    }

    pub(crate) fn bit_pos(&self) -> usize {
        self.bitpos
    }

    #[inline]
    pub(crate) fn read_bit(&mut self) -> u32 {
        let byte = self.bitpos >> 3;
        let bit = if byte < self.data.len() {
            ((self.data[byte] >> (7 - (self.bitpos & 7))) & 1) as u32
        } else {
            0
        };
        self.bitpos += 1;
        bit
    }

    pub(crate) fn read_bits(&mut self, n: u32) -> u32 {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.read_bit();
        }
        v
    }

    /// Unsigned Exp-Golomb (`ue(v)`).
    pub(crate) fn read_ue(&mut self) -> u32 {
        let mut zeros = 0;
        while self.read_bit() == 0 && zeros < 32 {
            zeros += 1;
        }
        if zeros == 0 {
            return 0;
        }
        let suffix = self.read_bits(zeros);
        (1 << zeros) - 1 + suffix
    }

    /// Signed Exp-Golomb (`se(v)`).
    pub(crate) fn read_se(&mut self) -> i32 {
        let code = self.read_ue();
        let k = ((code + 1) >> 1) as i32;
        if code & 1 == 1 { k } else { -k }
    }

    /// Advance to the next byte boundary (consume alignment bits).
    pub(crate) fn byte_align(&mut self) {
        while self.bitpos & 7 != 0 {
            self.bitpos += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reader_inverts_writer() {
        let mut w = BitWriter::new();
        w.put_bits(0b10110, 5);
        w.put_ue(0);
        w.put_ue(42);
        w.put_se(-7);
        w.put_se(13);
        w.put_bits(0xABCD, 16);
        w.byte_align();
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(5), 0b10110);
        assert_eq!(r.read_ue(), 0);
        assert_eq!(r.read_ue(), 42);
        assert_eq!(r.read_se(), -7);
        assert_eq!(r.read_se(), 13);
        assert_eq!(r.read_bits(16), 0xABCD);
    }

    #[test]
    fn ue_known_values() {
        // codeNum -> bit pattern (H.266 Table 9-2 style):
        // 0 -> "1", 1 -> "010", 2 -> "011", 3 -> "00100", 4 -> "00101"
        let cases: &[(u32, &[u32])] = &[
            (0, &[1]),
            (1, &[0, 1, 0]),
            (2, &[0, 1, 1]),
            (3, &[0, 0, 1, 0, 0]),
            (4, &[0, 0, 1, 0, 1]),
        ];
        for (v, bits) in cases {
            let mut w = BitWriter::new();
            w.put_ue(*v);
            w.byte_align();
            // Reconstruct the leading bits and compare.
            let total: Vec<u32> = {
                let bytes = w.bytes.clone();
                let mut out = Vec::new();
                for byte in bytes {
                    for i in (0..8).rev() {
                        out.push(((byte >> i) & 1) as u32);
                    }
                }
                out
            };
            assert_eq!(&total[..bits.len()], *bits, "ue({v})");
        }
    }

    #[test]
    fn se_maps_to_expected_codenum() {
        // se: 0->0, 1->1, -1->2, 2->3, -2->4
        let map: &[(i32, u32)] = &[(0, 0), (1, 1), (-1, 2), (2, 3), (-2, 4)];
        for (val, code) in map {
            let mut a = BitWriter::new();
            a.put_se(*val);
            a.byte_align();
            let mut b = BitWriter::new();
            b.put_ue(*code);
            b.byte_align();
            assert_eq!(a.bytes, b.bytes, "se({val}) should equal ue({code})");
        }
    }

    #[test]
    fn emulation_prevention_inserts_03() {
        assert_eq!(rbsp_to_ebsp(&[0, 0, 0]), vec![0, 0, 3, 0]);
        assert_eq!(rbsp_to_ebsp(&[0, 0, 1]), vec![0, 0, 3, 1]);
        assert_eq!(rbsp_to_ebsp(&[0, 0, 2]), vec![0, 0, 3, 2]);
        assert_eq!(rbsp_to_ebsp(&[0, 0, 3]), vec![0, 0, 3, 3]);
        // No insertion when not preceded by two zeros.
        assert_eq!(rbsp_to_ebsp(&[0, 1, 2]), vec![0, 1, 2]);
        // Long zero run: 00 00 00 00 -> 00 00 03 00 00  (then one more 0)
        assert_eq!(rbsp_to_ebsp(&[0, 0, 0, 0]), vec![0, 0, 3, 0, 0]);
    }

    #[test]
    fn nal_header_two_bytes() {
        let nal = write_nal(NalUnitType::Sps, &[0xFF], 1);
        // header(2) + payload. SPS = 15 = 0b01111.
        assert_eq!(nal.len(), 3);
        // byte0: 0 0 000000 0 -> top 8 bits: forbidden0, reserved0, layer_id[5:0]=0,
        //        then unit_type high bits. layer_id=0 so byte0 = 0b0000_0000 | (15>>4)=0
        assert_eq!(nal[0], 0x00);
        // byte1: unit_type low 4 bits (1111) << 4 ... no: unit_type(5) then tid+1(3)
        // bits: u4..u0 = 01111 ; tid+1=001  => 0111_1001 = 0x79
        assert_eq!(nal[1], 0x79);
    }

    #[test]
    fn rbsp_trailing_bits_aligns() {
        let mut w = BitWriter::new();
        w.put_bits(0b101, 3);
        w.rbsp_trailing_bits();
        assert!(w.is_byte_aligned());
        // 101 + stop bit 1 + four zero pad = 1011_0000 = 0xB0
        assert_eq!(w.into_bytes(), vec![0xB0]);
    }
}
