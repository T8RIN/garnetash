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

#![allow(dead_code)]

pub(crate) const LOSSLESS_QP: u8 = 4;
#[inline]
pub(crate) fn ts_dequant_lossless(level: i32) -> i32 {
    ((level * DEQUANT_SCALE_QP4) + (1 << (RIGHT_SHIFT_QP4 - 1))) >> RIGHT_SHIFT_QP4
}

const DEQUANT_SCALE_QP4: i32 = 64; // g_InvQuantScales[0][4]
const RIGHT_SHIFT_QP4: i32 = 6; //  IQUANT_SHIFT - QP_per(=0)

#[inline]
pub(crate) fn derive_mod_coeff(right: i32, below: i32, abs_coeff: i32) -> i32 {
    if abs_coeff == 0 {
        return 0;
    }
    let pred = below.abs().max(right.abs());
    if abs_coeff == pred {
        1
    } else if abs_coeff < pred {
        abs_coeff + 1
    } else {
        abs_coeff
    }
}

/// Decoder-side inverse of [`derive_mod_coeff`] (H.266 `decDeriveModCoeff`).
///
/// ```text
/// pred = max(|below|, |right|)
/// if   absMod == 1 && pred > 0 -> pred
/// else                         -> absMod - (absMod <= pred)
/// ```
#[inline]
pub(crate) fn dec_derive_mod_coeff(right: i32, below: i32, abs_mod: i32) -> i32 {
    if abs_mod == 0 {
        return 0;
    }
    let pred = below.abs().max(right.abs());
    if abs_mod == 1 && pred > 0 {
        pred
    } else {
        abs_mod - (abs_mod <= pred) as i32
    }
}

/// Significance context offset: number of significant left/above neighbours
/// (0, 1 or 2). `sig(pos)` reports whether the neighbour level is non-zero.
#[inline]
pub(crate) fn sig_ctx_offset_ts(left_nonzero: bool, above_nonzero: bool) -> usize {
    left_nonzero as usize + above_nonzero as usize
}

/// Greater-than-one (lrg1) context offset: same neighbour count, except BDPCM
/// blocks always use offset 3.
#[inline]
pub(crate) fn gt1_ctx_offset_ts(left_nonzero: bool, above_nonzero: bool, bdpcm: bool) -> usize {
    if bdpcm {
        3
    } else {
        left_nonzero as usize + above_nonzero as usize
    }
}

/// Sign context offset from the signed left/above neighbour levels.
///
/// 0 when both are zero or have opposing signs, 1 when both are non-negative,
/// 2 otherwise; BDPCM adds 3.
#[inline]
pub(crate) fn sign_ctx_offset_ts(right: i32, below: i32, bdpcm: bool) -> usize {
    let base = if (right == 0 && below == 0) || (right * below) < 0 {
        0
    } else if right >= 0 && below >= 0 {
        1
    } else {
        2
    };
    base + if bdpcm { 3 } else { 0 }
}

/// Go-Rice parameter for transform-skip remainders: always 1 in VVC
/// (`templateAbsSumTS` returns a constant).
#[inline]
pub(crate) fn rice_param_ts() -> u32 {
    1
}

/// Encode one transform-skip residual block (TSRC), mirroring VTM
/// `CABACWriter::residual_codingTS`. `coeff` is the `w x h` signed residual in
/// raster order. Range-extension Rice signalling is disabled, so the Go-Rice
/// parameter is the constant 1.
pub(crate) fn encode_residual_ts(
    enc: &mut crate::cabac::CabacEncoder,
    ctx: &mut crate::cabac::Contexts,
    coeff: &[i32],
    w: usize,
    h: usize,
    bdpcm: bool,
) {
    let wig = w / 4;
    let hig = h / 4;
    let cg_scan = crate::residual::diag_scan(wig, hig);
    let sub_scan = crate::residual::diag_scan(4, 4);
    let num_cg = cg_scan.len();
    let mut num_ctx_bins: i32 = (((w * h) * 7) >> 2) as i32;

    let getv = |x: i32, y: i32| -> i32 {
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            coeff[(y as usize) * w + (x as usize)]
        } else {
            0
        }
    };
    // deriveModCoeff for (x, y) using the original (unmapped) neighbour
    // residuals; under BDPCM the remap is bypassed and levels code verbatim.
    let mod_abs = |x: usize, y: usize| -> i32 {
        if bdpcm {
            coeff[y * w + x].abs()
        } else {
            derive_mod_coeff(
                getv(x as i32 - 1, y as i32),
                getv(x as i32, y as i32 - 1),
                coeff[y * w + x].abs(),
            )
        }
    };

    let mut sig_cg = vec![false; wig * hig];
    let mut any_prior_sig = false;

    for (sub_set_id, &(cgx, cgy)) in cg_scan.iter().enumerate() {
        let pos: Vec<(usize, usize)> = sub_scan
            .iter()
            .map(|&(sx, sy)| (cgx * 4 + sx, cgy * 4 + sy))
            .collect();
        let group_sig = pos.iter().any(|&(x, y)| coeff[y * w + x] != 0);

        let infer = sub_set_id == num_cg - 1 && !any_prior_sig;
        if !infer {
            let s_left = if cgx > 0 {
                sig_cg[cgy * wig + cgx - 1]
            } else {
                false
            };
            let s_above = if cgy > 0 {
                sig_cg[(cgy - 1) * wig + cgx]
            } else {
                false
            };
            enc.encode_bin(
                group_sig as u8,
                &mut ctx.ts_sig_group[s_left as usize + s_above as usize],
            );
            if !group_sig {
                continue;
            }
        }
        sig_cg[cgy * wig + cgx] = true;
        any_prior_sig = true;

        let n = pos.len();
        let infer_sig_pos = n - 1;
        let mut num_nonzero = 0;
        let mut last_pass1: i32 = -1;
        let mut last_pass2: i32 = -1;

        // ── pass 1: significance, sign, gt1, parity ──
        let mut k = 0;
        while k < n && num_ctx_bins >= 4 {
            let (x, y) = pos[k];
            let cval = coeff[y * w + x];
            let left_nz = getv(x as i32 - 1, y as i32) != 0;
            let above_nz = getv(x as i32, y as i32 - 1) != 0;
            if num_nonzero > 0 || k != infer_sig_pos {
                enc.encode_bin(
                    (cval != 0) as u8,
                    &mut ctx.ts_sig_flag[sig_ctx_offset_ts(left_nz, above_nz)],
                );
                num_ctx_bins -= 1;
            }
            if cval != 0 {
                let sofs = sign_ctx_offset_ts(
                    getv(x as i32 - 1, y as i32),
                    getv(x as i32, y as i32 - 1),
                    bdpcm,
                );
                enc.encode_bin((cval < 0) as u8, &mut ctx.ts_sign_flag[sofs]);
                num_ctx_bins -= 1;
                num_nonzero += 1;
                let mut rem = mod_abs(x, y) - 1;
                let gt1 = (rem != 0) as u8;
                enc.encode_bin(
                    gt1,
                    &mut ctx.ts_lrg1_flag[gt1_ctx_offset_ts(left_nz, above_nz, bdpcm)],
                );
                num_ctx_bins -= 1;
                if gt1 == 1 {
                    rem -= 1;
                    enc.encode_bin((rem & 1) as u8, &mut ctx.ts_par_flag[0]);
                    num_ctx_bins -= 1;
                }
            }
            last_pass1 = k as i32;
            k += 1;
        }

        // ── pass 2: up to four greater-than flags ──
        let mut k = 0;
        while k < n && num_ctx_bins >= 4 {
            let (x, y) = pos[k];
            let abs_level = mod_abs(x, y);
            let mut cutoff = 2i32;
            for _ in 0..4 {
                if abs_level >= cutoff {
                    enc.encode_bin(
                        (abs_level >= cutoff + 2) as u8,
                        &mut ctx.ts_gtx_flag[(cutoff >> 1) as usize],
                    );
                    num_ctx_bins -= 1;
                }
                cutoff += 2;
            }
            last_pass2 = k as i32;
            k += 1;
        }

        // ── pass 3: Go-Rice remainder + late signs (all bypass) ──
        for (k, &(x, y)) in pos.iter().enumerate() {
            let ki = k as i32;
            let cutoff = if ki <= last_pass2 {
                10
            } else if ki <= last_pass1 {
                2
            } else {
                0
            };
            let abs_level = if cutoff == 0 || bdpcm {
                coeff[y * w + x].abs()
            } else {
                mod_abs(x, y)
            };
            if abs_level >= cutoff {
                let rem = if ki <= last_pass1 {
                    (abs_level - cutoff) >> 1
                } else {
                    abs_level
                };
                encode_rem_abs_ep(enc, rem as u32, rice_param_ts());
                if abs_level != 0 && ki > last_pass1 {
                    enc.encode_bypass((coeff[y * w + x] < 0) as u8);
                }
            }
        }
    }
}

use crate::residual::encode_rem_abs_ep;

pub(crate) mod test_support {
    use super::*;
    use crate::cabac::Contexts;
    use crate::cabac::engine::CabacDecoder;
    use crate::residual::diag_scan;
    use crate::residual::test_support::decode_rem_abs_ep;

    pub(crate) fn decode_residual_ts(
        dec: &mut CabacDecoder,
        ctx: &mut Contexts,
        w: usize,
        h: usize,
        bdpcm: bool,
    ) -> Vec<i32> {
        let wig = w / 4;
        let hig = h / 4;
        let cg_scan = diag_scan(wig, hig);
        let sub_scan = diag_scan(4, 4);
        let num_cg = cg_scan.len();
        let mut num_ctx_bins: i32 = (((w * h) * 7) >> 2) as i32;
        let mut coeff = vec![0i32; w * h];
        let mut sig_cg = vec![false; wig * hig];
        let mut any_prior_sig = false;

        for (sub_set_id, &(cgx, cgy)) in cg_scan.iter().enumerate() {
            let pos: Vec<(usize, usize)> = sub_scan
                .iter()
                .map(|&(sx, sy)| (cgx * 4 + sx, cgy * 4 + sy))
                .collect();
            let infer = sub_set_id == num_cg - 1 && !any_prior_sig;
            let group_sig = if infer {
                true
            } else {
                let s_left = if cgx > 0 {
                    sig_cg[cgy * wig + cgx - 1]
                } else {
                    false
                };
                let s_above = if cgy > 0 {
                    sig_cg[(cgy - 1) * wig + cgx]
                } else {
                    false
                };
                dec.decode_bin(&mut ctx.ts_sig_group[s_left as usize + s_above as usize]) == 1
            };
            if !group_sig {
                continue;
            }
            sig_cg[cgy * wig + cgx] = true;
            any_prior_sig = true;

            let nz = |c: &[i32], x: i32, y: i32| -> bool {
                x >= 0
                    && y >= 0
                    && (x as usize) < w
                    && (y as usize) < h
                    && c[(y as usize) * w + x as usize] != 0
            };
            let gv = |c: &[i32], x: i32, y: i32| -> i32 {
                if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                    c[(y as usize) * w + x as usize]
                } else {
                    0
                }
            };

            let n = pos.len();
            let infer_sig_pos = n - 1;
            let mut num_nonzero = 0u32;
            let mut sign_pattern = 0u32;
            let mut sig_blk: Vec<usize> = Vec::new();
            let mut last_pass1: i32 = -1;
            let mut last_pass2: i32 = -1;

            // ── pass 1 ──
            let mut k = 0;
            while k < n && num_ctx_bins >= 4 {
                let (x, y) = pos[k];
                let bp = y * w + x;
                let mut sig = (num_nonzero == 0 && k == infer_sig_pos) as u8;
                if sig == 0 {
                    let ofs = nz(&coeff, x as i32 - 1, y as i32) as usize
                        + nz(&coeff, x as i32, y as i32 - 1) as usize;
                    sig = dec.decode_bin(&mut ctx.ts_sig_flag[ofs]);
                    num_ctx_bins -= 1;
                }
                if sig == 1 {
                    let sofs = sign_ctx_offset_ts(
                        gv(&coeff, x as i32 - 1, y as i32),
                        gv(&coeff, x as i32, y as i32 - 1),
                        bdpcm,
                    );
                    let sign = dec.decode_bin(&mut ctx.ts_sign_flag[sofs]);
                    num_ctx_bins -= 1;
                    sign_pattern += (sign as u32) << num_nonzero;
                    sig_blk.push(bp);
                    num_nonzero += 1;
                    let g1 = gt1_ctx_offset_ts(
                        nz(&coeff, x as i32 - 1, y as i32),
                        nz(&coeff, x as i32, y as i32 - 1),
                        bdpcm,
                    );
                    let gt1 = dec.decode_bin(&mut ctx.ts_lrg1_flag[g1]);
                    num_ctx_bins -= 1;
                    let mut par = 0u8;
                    if gt1 == 1 {
                        par = dec.decode_bin(&mut ctx.ts_par_flag[0]);
                        num_ctx_bins -= 1;
                    }
                    coeff[bp] = if sign == 1 { -1 } else { 1 } * (1 + par as i32 + gt1 as i32);
                }
                last_pass1 = k as i32;
                k += 1;
            }

            // ── pass 2 ──
            let mut k = 0;
            while k < n && num_ctx_bins >= 4 {
                let (x, y) = pos[k];
                let bp = y * w + x;
                let mut tc = coeff[bp].abs();
                let mut cutoff = 2i32;
                for _ in 0..4 {
                    if tc >= cutoff {
                        let gt = dec.decode_bin(&mut ctx.ts_gtx_flag[(cutoff >> 1) as usize]);
                        tc += (gt as i32) << 1;
                        num_ctx_bins -= 1;
                    }
                    cutoff += 2;
                }
                coeff[bp] = tc;
                last_pass2 = k as i32;
                k += 1;
            }

            // ── pass 3 ──
            for (k, &(x, y)) in pos.iter().enumerate() {
                let bp = y * w + x;
                let ki = k as i32;
                let cutoff = if ki <= last_pass2 {
                    10
                } else if ki <= last_pass1 {
                    2
                } else {
                    0
                };
                let mut tc = coeff[bp].abs();
                if tc >= cutoff {
                    let rem = decode_rem_abs_ep(dec, rice_param_ts()) as i32;
                    tc += if ki <= last_pass1 { rem << 1 } else { rem };
                    if tc != 0 && ki > last_pass1 {
                        let sign = dec.decode_bypass();
                        sign_pattern += (sign as u32) << num_nonzero;
                        sig_blk.push(bp);
                        num_nonzero += 1;
                    }
                }
                if cutoff != 0 && tc > 0 && !bdpcm {
                    tc = dec_derive_mod_coeff(
                        gv(&coeff, x as i32 - 1, y as i32),
                        gv(&coeff, x as i32, y as i32 - 1),
                        tc,
                    );
                }
                coeff[bp] = tc;
            }

            // ── apply accumulated signs ──
            let mut sp = sign_pattern;
            for &bp in &sig_blk {
                let a = coeff[bp];
                coeff[bp] = if sp & 1 == 1 { -a } else { a };
                sp >>= 1;
            }
        }
        coeff
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::engine::CabacDecoder;
    use crate::cabac::{CabacEncoder, Contexts};
    use test_support::decode_residual_ts;

    /// Decode one TSRC block, mirroring vvdec `residual_coding_subblockTS`.
    /// Returns the `w x h` reconstructed signed residual in raster order.

    fn roundtrip_b(block: &[i32], w: usize, h: usize, bdpcm: bool) {
        let mut enc = CabacEncoder::new();
        let mut ectx = Contexts::new_intra(LOSSLESS_QP);
        encode_residual_ts(&mut enc, &mut ectx, block, w, h, bdpcm);
        enc.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = CabacDecoder::new(&bytes);
        let mut dctx = Contexts::new_intra(LOSSLESS_QP);
        let out = decode_residual_ts(&mut dec, &mut dctx, w, h, bdpcm);
        assert_eq!(dec.decode_terminate(), 1, "terminate mismatch for {w}x{h}");
        assert_eq!(
            out, block,
            "TSRC round-trip mismatch for {w}x{h} bdpcm={bdpcm}"
        );
    }

    fn roundtrip(block: &[i32], w: usize, h: usize) {
        roundtrip_b(block, w, h, false);
        roundtrip_b(block, w, h, true);
    }

    #[test]
    fn tsrc_round_trip_diverse_blocks() {
        // A small LCG gives reproducible pseudo-random residuals.
        let mut state = 0x9e3779b9u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for &(w, h) in &[(4usize, 4usize), (8, 8), (16, 16), (32, 32)] {
            // All-zero except a forced significant coeff would be illegal (cbf=0);
            // every test block below has at least one non-zero value.
            for trial in 0..6 {
                let mut block = vec![0i32; w * h];
                for v in block.iter_mut() {
                    let r = next();
                    *v = match trial {
                        0 => (r % 511) as i32 - 255, // full 8-bit residual range
                        1 => (r % 7) as i32 - 3,     // small values
                        2 => ((r % 2) as i32) * ((r >> 4 & 1) as i32 * 2 - 1) * 255, // sparse extremes
                        3 => (r % 3) as i32 - 1, // mostly -1/0/1
                        4 => (r as i32 % 21) - 10,
                        _ => (r % 101) as i32 - 50,
                    };
                }
                if block.iter().all(|&v| v == 0) {
                    block[0] = 1; // ensure the block is significant
                }
                roundtrip(&block, w, h);
            }
        }
    }

    #[test]
    fn tsrc_round_trip_boundary_levels() {
        // Dense large magnitudes deliberately exhaust the context-bin budget,
        // forcing the pass-3 Go-Rice path and the late-discovered-sign branch.
        for &(w, h) in &[(4usize, 4usize), (8, 8)] {
            let mut block = vec![0i32; w * h];
            for (i, v) in block.iter_mut().enumerate() {
                *v = if i % 2 == 0 { 200 } else { -173 };
            }
            roundtrip(&block, w, h);
        }
    }

    #[test]
    fn ts_dequant_is_identity_at_lossless_qp() {
        for level in -512..=512 {
            assert_eq!(ts_dequant_lossless(level), level, "level {level}");
        }
    }

    #[test]
    fn mod_coeff_remap_is_exactly_invertible() {
        // For every neighbour magnitude and every absolute level, the
        // encoder remap followed by the decoder inverse must recover the level.
        for right in -8..=8 {
            for below in -8..=8 {
                for abs_coeff in 1..=20 {
                    let coded = derive_mod_coeff(right, below, abs_coeff);
                    assert!(coded >= 1, "coded level must stay significant");
                    let back = dec_derive_mod_coeff(right, below, coded);
                    assert_eq!(
                        back, abs_coeff,
                        "right={right} below={below} abs={abs_coeff}"
                    );
                }
            }
        }
        assert_eq!(derive_mod_coeff(0, 0, 0), 0);
        assert_eq!(dec_derive_mod_coeff(0, 0, 0), 0);
    }

    #[test]
    fn sig_and_gt1_context_offsets() {
        assert_eq!(sig_ctx_offset_ts(false, false), 0);
        assert_eq!(sig_ctx_offset_ts(true, false), 1);
        assert_eq!(sig_ctx_offset_ts(true, true), 2);
        assert_eq!(gt1_ctx_offset_ts(true, true, false), 2);
        assert_eq!(gt1_ctx_offset_ts(false, false, true), 3); // BDPCM
    }

    #[test]
    fn sign_context_offsets_match_spec() {
        assert_eq!(sign_ctx_offset_ts(0, 0, false), 0); // both zero
        assert_eq!(sign_ctx_offset_ts(3, -2, false), 0); // opposing signs
        assert_eq!(sign_ctx_offset_ts(3, 2, false), 1); // both non-negative
        assert_eq!(sign_ctx_offset_ts(-3, -2, false), 2); // both negative
        assert_eq!(sign_ctx_offset_ts(-3, -2, true), 5); // + BDPCM
    }

    #[test]
    fn lossless_pipeline_reconstructs_source_exactly() {
        // The lossless recipe: level = (src - pred), identity dequant, then
        // recon = pred + dequant(level). Over the full 8-bit prediction and
        // source range this must return the source with no error and no
        // overflow in the level domain.
        for pred in [0i32, 1, 64, 127, 200, 255] {
            for src in 0..=255i32 {
                let level = src - pred; // residual, coded verbatim under TSRC
                let recon = pred + ts_dequant_lossless(level);
                assert_eq!(recon, src, "pred={pred} src={src}");
            }
        }
    }
}
