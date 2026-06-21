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

//! VVC (H.266) CABAC arithmetic encoder.
//!
//! The arithmetic *engine* — the 9-bit `range`, the `low` register, and the
//! carry/outstanding-bit byte output — is the same H.264-derived coder used by
//! HEVC, and is implemented here with the classic bit-by-bit renormalisation
//! (proven carry handling). What VVC changes, and what this module implements
//! faithfully, is the **probability model**:
//!
//!   * a dual probability state (`state[0]` fast, `state[1]` slow) updated at
//!     two different exponential rates, then averaged (H.266 §9.3.4.3.2), and
//!   * a multiplication-based LPS-range derivation `getLPS` (no HEVC-style
//!     `RangeTabLps` lookup).
//!
//! Constants and the update/init formulas mirror the VVC reference software
//! `BinProbModel_Std`.
#![allow(dead_code)]

/// Nominal probability precision (bits). H.266 fixes this at 15.
const PROB_BITS: u32 = 15;
/// Precision of the fast estimate `state[0]`.
const PROB_BITS_0: u32 = 10;
/// Precision of the slow estimate `state[1]`.
const PROB_BITS_1: u32 = 14;
/// Mask keeping `state[0]` at `PROB_BITS_0` precision within a 15-bit field.
const MASK_0: u16 = ((!(!0u32 << PROB_BITS_0)) << (PROB_BITS - PROB_BITS_0)) as u16; // 0x7FE0
/// Mask keeping `state[1]` at `PROB_BITS_1` precision within a 15-bit field.
const MASK_1: u16 = ((!(!0u32 << PROB_BITS_1)) << (PROB_BITS - PROB_BITS_1)) as u16; // 0x7FFE
/// Default packed window sizes (rate0=0 high nibble, rate1=8 -> 0x08). VVC's
/// default `log2WindowSize` resolves to this for most contexts.
const DWS: u8 = 8;

/// A single VVC CABAC context model: two probability estimates plus the packed
/// adaptation rates.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CtxModel {
    state: [u16; 2],
    rate: u8,
}

impl CtxModel {
    /// Initialise from a context init value, the slice QP, and the per-context
    /// `log2_window_size` (H.266 §9.3.2.3, matching `BinProbModel_Std::init`
    /// followed by `setLog2WindowSize`). `init_value` and `log2_window_size` are
    /// the per-context bytes from the initialisation tables; `qp` is the slice
    /// luma QP (clamped to 0..=63).
    pub(crate) fn init(init_value: u8, qp: u8, log2_window_size: u8) -> Self {
        let init_id = init_value as i32;
        let qp = (qp as i32).clamp(0, 63);
        let slope = (init_id >> 3) - 4;
        let offset = ((init_id & 7) * 18) + 1;
        let inistate = ((slope * (qp - 16)) >> 1) + offset;
        let state_clip = inistate.clamp(1, 127);
        let p1 = (state_clip << 8) as u16;
        let mut c = CtxModel {
            state: [p1 & MASK_0, p1 & MASK_1],
            rate: DWS,
        };
        c.set_log2_window_size(log2_window_size);
        c
    }

    /// Apply the packed adaptation rates (`BinProbModel_Std::setLog2WindowSize`).
    pub(crate) fn set_log2_window_size(&mut self, log2_window_size: u8) {
        let lws = log2_window_size as u32;
        let rate0 = 2 + ((lws >> 2) & 3);
        let rate1 = 3 + rate0 + (lws & 3);
        self.rate = (16 * rate0 + rate1) as u8;
    }

    /// Combined 8-bit probability estimate (0..=255). 128 is equiprobable.
    #[inline]
    fn prob8(self) -> u16 {
        (self.state[0].wrapping_add(self.state[1])) >> 8
    }

    /// Most-probable symbol (0 or 1): the top bit of the combined estimate.
    #[inline]
    pub(crate) fn mps(self) -> u8 {
        (self.prob8() >> 7) as u8
    }

    /// LPS sub-range for the current 9-bit coder `range`
    /// (`BinProbModel_Std::getLPS`).
    #[inline]
    pub(crate) fn get_lps(self, range: u32) -> u32 {
        let mut q = self.prob8();
        if q & 0x80 != 0 {
            q ^= 0xff; // fold to the smaller probability (0..=127)
        }
        (((q as u32 >> 2) * (range >> 5)) >> 1) + 4
    }

    /// Dual-rate probability update after coding `bin` (`BinProbModel_Std::update`).
    #[inline]
    pub(crate) fn update(&mut self, bin: u8) {
        let rate0 = (self.rate >> 4) as u32;
        let rate1 = (self.rate & 15) as u32;
        // Work in u32 to avoid any intermediate overflow, then re-store as u16.
        let mut s0 = self.state[0] as u32;
        let mut s1 = self.state[1] as u32;
        s0 -= (s0 >> rate0) & MASK_0 as u32;
        s1 -= (s1 >> rate1) & MASK_1 as u32;
        if bin != 0 {
            s0 += (0x7fffu32 >> rate0) & MASK_0 as u32;
            s1 += (0x7fffu32 >> rate1) & MASK_1 as u32;
        }
        self.state[0] = s0 as u16;
        self.state[1] = s1 as u16;
    }

    /// Test-only constructor with explicit internal state.
    #[cfg(test)]
    pub(crate) fn from_raw(state: [u16; 2], rate: u8) -> Self {
        CtxModel { state, rate }
    }
}

/// VVC CABAC encoder.
///
/// 9-bit `low`/`range` arithmetic coder with bit-FIFO renormalisation and
/// outstanding-bit (carry) handling, exactly as in the H.264/HEVC/VVC reference
/// arithmetic coder. The probability adaptation and LPS derivation are VVC's
/// ([`CtxModel`]).
#[derive(Clone)]
pub(crate) struct CabacEncoder {
    low: u32,
    m_range: u32,
    bits_outstanding: u32,
    first_bit: bool,
    bit_buffer: u8,
    bit_count: u8,
    pub(crate) output: Vec<u8>,
}

impl CabacEncoder {
    pub(crate) fn new() -> Self {
        CabacEncoder {
            low: 0,
            m_range: 510,
            bits_outstanding: 0,
            first_bit: true,
            bit_buffer: 0,
            bit_count: 0,
            output: Vec::new(),
        }
    }

    /// Current coder range (debug/validation aid).
    pub(crate) fn range(&self) -> u32 {
        self.m_range
    }

    #[inline]
    fn emit_bit(&mut self, b: u32) {
        self.bit_buffer = (self.bit_buffer << 1) | (b as u8 & 1);
        self.bit_count += 1;
        if self.bit_count == 8 {
            self.output.push(self.bit_buffer);
            self.bit_buffer = 0;
            self.bit_count = 0;
        }
    }

    /// Output a resolved bit plus any outstanding (carry-deferred) opposite bits.
    #[inline]
    fn put_bit(&mut self, b: u32) {
        if self.first_bit {
            self.first_bit = false;
        } else {
            self.emit_bit(b);
        }
        while self.bits_outstanding > 0 {
            self.emit_bit(1 - b);
            self.bits_outstanding -= 1;
        }
    }

    /// Renormalise after a context-coded or terminate bin.
    #[inline]
    fn renorm(&mut self) {
        while self.m_range < 256 {
            if self.low < 256 {
                self.put_bit(0);
            } else if self.low >= 512 {
                self.low -= 512;
                self.put_bit(1);
            } else {
                self.low -= 256;
                self.bits_outstanding += 1;
            }
            self.m_range <<= 1;
            self.low <<= 1;
        }
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /// Context-adaptive binary encoding (VVC `encodeBin`).
    #[inline]
    pub(crate) fn encode_bin(&mut self, bin_val: u8, ctx: &mut CtxModel) {
        let lps = ctx.get_lps(self.m_range);
        self.m_range -= lps;
        if (bin_val & 1) != ctx.mps() {
            // LPS path.
            self.low += self.m_range;
            self.m_range = lps;
        }
        ctx.update(bin_val & 1);
        self.renorm();
    }

    /// Equal-probability bypass encoding (`encodeBinEP`). Identical to HEVC.
    #[inline]
    pub(crate) fn encode_bypass(&mut self, bin_val: u8) {
        self.low <<= 1;
        if bin_val != 0 {
            self.low += self.m_range;
        }
        if self.low >= 1024 {
            self.put_bit(1);
            self.low -= 1024;
        } else if self.low < 512 {
            self.put_bit(0);
        } else {
            self.low -= 512;
            self.bits_outstanding += 1;
        }
    }

    /// Encode `n` equal-probability bypass bins from the low `n` bits of `value`
    /// (MSB first).
    pub(crate) fn encode_bypass_bits(&mut self, value: u32, n: u32) {
        let mut i = n;
        while i > 0 {
            i -= 1;
            self.encode_bypass(((value >> i) & 1) as u8);
        }
    }

    /// Encode the terminate bin (`encodeBinTrm`). When `flag == 1` the coder is
    /// flushed. Identical procedure to HEVC §9.3.4.3.5.
    pub(crate) fn encode_terminate(&mut self, flag: u8) {
        self.m_range -= 2;
        if flag != 0 {
            self.low += self.m_range;
            self.flush();
        } else {
            self.renorm();
        }
    }

    /// EncodeFlush: range = 2; renorm; emit the trailing bits with the stop bit.
    fn flush(&mut self) {
        self.m_range = 2;
        self.renorm();
        self.put_bit((self.low >> 9) & 1);
        let two = ((self.low >> 7) & 3) | 1;
        self.emit_bit((two >> 1) & 1);
        self.emit_bit(two & 1);
    }

    /// Byte-align the partial buffer (zero padding) and return the coded bytes.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        if self.bit_count > 0 {
            self.bit_buffer <<= 8 - self.bit_count;
            self.output.push(self.bit_buffer);
            self.bit_buffer = 0;
            self.bit_count = 0;
        }
        self.output
    }

    /// Reset to the initial coder state, keeping the `output` buffer's capacity.
    /// Lets one encoder be reused for many rate-distortion bit-count trials
    /// without reallocating, which is a hot path during mode decision.
    pub(crate) fn reset(&mut self) {
        self.low = 0;
        self.m_range = 510;
        self.bits_outstanding = 0;
        self.first_bit = true;
        self.bit_buffer = 0;
        self.bit_count = 0;
        self.output.clear();
    }

    /// Byte length the stream would have after `finish()`, without consuming or
    /// mutating the encoder — used to score a trial's coded size.
    pub(crate) fn flushed_len(&self) -> usize {
        self.output.len() + usize::from(self.bit_count > 0)
    }
}

pub(crate) use decoder::CabacDecoder;

/// VVC CABAC arithmetic decoder: the inverse of [`CabacEncoder`]. Follows the
/// standard decode framing (9-bit init, offset/range, MSB-first bit input with
/// end-of-input 1-stuffing) and the shared VVC probability model ([`CtxModel`]).
mod decoder {
    use super::CtxModel;

    /// Independent VVC CABAC decoder used purely to verify the encoder. It
    /// follows the standard arithmetic-decode framing (9-bit init, offset/range,
    /// MSB-first bit input with end-of-input 1-stuffing) and the VVC probability
    /// model ([`CtxModel`]). If random bin sequences survive an encode -> decode
    /// round trip with identical context models on both sides, the encoder's
    /// range/renorm/carry handling and probability updates are self-consistent.
    pub(crate) struct CabacDecoder<'a> {
        range: u32,
        offset: u32,
        data: &'a [u8],
        bitpos: usize, // bit index into data, MSB-first
    }

    impl<'a> CabacDecoder<'a> {
        pub(crate) fn new(data: &'a [u8]) -> Self {
            let mut d = CabacDecoder {
                range: 510,
                offset: 0,
                data,
                bitpos: 0,
            };
            d.offset = d.read_bits(9);
            d
        }

        #[inline]
        fn next_bit(&mut self) -> u32 {
            let byte_idx = self.bitpos >> 3;
            let bit = if byte_idx < self.data.len() {
                let b = self.data[byte_idx];
                ((b >> (7 - (self.bitpos & 7))) & 1) as u32
            } else {
                1 // past end of input: stuffing
            };
            self.bitpos += 1;
            bit
        }

        #[inline]
        fn read_bits(&mut self, n: u32) -> u32 {
            let mut v = 0;
            for _ in 0..n {
                v = (v << 1) | self.next_bit();
            }
            v
        }

        #[inline]
        fn renorm(&mut self) {
            while self.range < 256 {
                self.range <<= 1;
                self.offset = (self.offset << 1) | self.next_bit();
            }
        }

        #[inline]
        pub(crate) fn decode_bin(&mut self, ctx: &mut CtxModel) -> u8 {
            let mps = ctx.mps();
            let lps = ctx.get_lps(self.range);
            self.range -= lps;
            let bin = if self.offset >= self.range {
                self.offset -= self.range;
                self.range = lps;
                1 - mps
            } else {
                mps
            };
            ctx.update(bin);
            self.renorm();
            bin
        }

        #[inline]
        pub(crate) fn decode_bypass(&mut self) -> u8 {
            self.offset = (self.offset << 1) | self.next_bit();
            if self.offset >= self.range {
                self.offset -= self.range;
                1
            } else {
                0
            }
        }

        #[inline]
        pub(crate) fn decode_terminate(&mut self) -> u8 {
            self.range -= 2;
            if self.offset >= self.range {
                1
            } else {
                self.renorm();
                0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CabacDecoder;
    use super::*;

    /// Tiny xorshift PRNG so tests are deterministic without external crates.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn bit(&mut self) -> u8 {
            (self.next() & 1) as u8
        }
    }

    #[test]
    fn mask_constants_match_spec() {
        assert_eq!(MASK_0, 0x7FE0);
        assert_eq!(MASK_1, 0x7FFE);
    }

    #[test]
    fn init_clamps_and_is_deterministic() {
        // Extreme init values still produce in-range states.
        for &iv in &[0u8, 1, 17, 95, 128, 200, 255] {
            for &qp in &[0u8, 16, 26, 37, 51, 63] {
                let c = CtxModel::init(iv, qp, 8);
                let s = c.state[0] as u32 + c.state[1] as u32;
                assert!(s > 0 && s < (1 << 16));
                assert!(c.mps() == 0 || c.mps() == 1);
            }
        }
    }

    /// One mixed sequence of context, bypass, and a final terminate bin.
    fn roundtrip_mixed(seed: u64, n: usize) {
        let mut rng = Rng(seed);
        // A handful of distinct contexts with assorted init values.
        let init_vals = [25u8, 60, 110, 154, 199];
        let qp = 32u8;
        let make_ctxs = || -> Vec<CtxModel> {
            init_vals
                .iter()
                .map(|&v| CtxModel::init(v, qp, 8))
                .collect()
        };

        // Plan the bins up front so encoder and decoder agree on the schedule.
        enum Op {
            Ctx(usize, u8),
            Byp(u8),
        }
        let mut plan = Vec::with_capacity(n);
        for _ in 0..n {
            if rng.bit() == 0 {
                let ci = (rng.next() as usize) % init_vals.len();
                plan.push(Op::Ctx(ci, rng.bit()));
            } else {
                plan.push(Op::Byp(rng.bit()));
            }
        }

        let mut enc = CabacEncoder::new();
        let mut ectx = make_ctxs();
        for op in &plan {
            match op {
                Op::Ctx(ci, b) => enc.encode_bin(*b, &mut ectx[*ci]),
                Op::Byp(b) => enc.encode_bypass(*b),
            }
        }
        enc.encode_terminate(1);
        let bytes = enc.finish();

        let mut dec = CabacDecoder::new(&bytes);
        let mut dctx = make_ctxs();
        for (i, op) in plan.iter().enumerate() {
            let got = match op {
                Op::Ctx(ci, _) => dec.decode_bin(&mut dctx[*ci]),
                Op::Byp(_) => dec.decode_bypass(),
            };
            let want = match op {
                Op::Ctx(_, b) => *b,
                Op::Byp(b) => *b,
            };
            assert_eq!(got, want, "bin {i} mismatch (seed {seed})");
        }
        assert_eq!(dec.decode_terminate(), 1, "terminate (seed {seed})");
    }

    #[test]
    fn roundtrip_context_and_bypass() {
        for seed in 1..200u64 {
            roundtrip_mixed(seed.wrapping_mul(0x9E3779B97F4A7C15), 64);
        }
    }

    #[test]
    fn roundtrip_long_sequences() {
        for seed in 1..20u64 {
            roundtrip_mixed(seed.wrapping_mul(0xD1B54A32D192ED03), 4000);
        }
    }

    #[test]
    fn roundtrip_all_context_bins() {
        // Stress the probability adaptation: long runs of a single context.
        let mut rng = Rng(0xABCDEF);
        let qp = 28u8;
        for &iv in &[12u8, 77, 140, 222] {
            let mut plan = Vec::new();
            for _ in 0..2000 {
                plan.push(rng.bit());
            }
            let mut enc = CabacEncoder::new();
            let mut ec = CtxModel::init(iv, qp, 8);
            for &b in &plan {
                enc.encode_bin(b, &mut ec);
            }
            enc.encode_terminate(1);
            let bytes = enc.finish();
            let mut dec = CabacDecoder::new(&bytes);
            let mut dc = CtxModel::init(iv, qp, 8);
            for (i, &b) in plan.iter().enumerate() {
                assert_eq!(dec.decode_bin(&mut dc), b, "iv {iv} bin {i}");
            }
            assert_eq!(dec.decode_terminate(), 1);
        }
    }

    #[test]
    fn probability_adapts_toward_input() {
        // A mid-range context (init value 35 @ qp 32) is not saturated, so
        // feeding 1s should raise the combined estimate toward the maximum.
        let mut c = CtxModel::init(35, 32, 8);
        let prob_before = c.state[0] as u32 + c.state[1] as u32;
        let lps_before = c.get_lps(384);
        for _ in 0..16 {
            c.update(1);
        }
        let prob_after = c.state[0] as u32 + c.state[1] as u32;
        let lps_after = c.get_lps(384);
        assert!(
            prob_after > prob_before,
            "estimate must rise ({prob_after} > {prob_before})"
        );
        assert!(
            lps_after <= lps_before,
            "LPS must not grow ({lps_after} <= {lps_before})"
        );
    }
}
