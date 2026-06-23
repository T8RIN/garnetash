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

use crate::cabac::{CabacEncoder, Contexts};
use crate::residual::{Component, encode_residual};
use crate::residual_ts::encode_residual_ts;

/// One transform unit's coefficient blocks: luma plus optional chroma (Cb, Cr)
/// for 4:2:0, all as raster (`y*w + x`) quantized-level arrays.
/// Which components a coding unit codes. Single tree codes luma+chroma together;
/// an intra dual tree codes luma and chroma in separate passes (VVC
/// `treeType` DUAL_TREE_LUMA / DUAL_TREE_CHROMA).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TreeType {
    Single,
    Luma,
    Chroma,
}

pub(crate) struct TuCoeffs<'a> {
    pub(crate) luma: &'a [i32],
    pub(crate) lw: usize,
    pub(crate) lh: usize,
    /// `(cb, cr, cw, ch)` when chroma is present.
    pub(crate) chroma: Option<(&'a [i32], &'a [i32], usize, usize)>,
    /// Lossless mode: code `transform_skip_flag = 1` then the transform-skip
    /// residual coder (TSRC) for each component, instead of regular residual
    /// coding. The "levels" are then the raw signed residual (src - pred).
    pub(crate) lossless: bool,
    /// Luma uses BDPCM (horizontal/vertical DPCM). Under BDPCM the luma
    /// `transform_skip_flag` is inferred (not coded) and TSRC runs in its
    /// BDPCM variant.
    pub(crate) luma_bdpcm: bool,
    /// Chroma (both Cb and Cr) uses BDPCM. Same inference/TSRC handling as luma,
    /// and the chroma CBFs use the BDPCM context indices (Cb→1, Cr→2).
    pub(crate) chroma_bdpcm: bool,
    /// Lossy transform-skip for luma: `transform_skip_flag = 1` is coded and the
    /// luma "levels" are the TS-quantized spatial residual coded with TSRC.
    /// Ignored in lossless mode (which always uses TS) and under BDPCM.
    pub(crate) luma_ts: bool,
    /// Lossy transform-skip for chroma (applies to both Cb and Cr).
    pub(crate) chroma_ts: bool,
    /// When `Some(delta)`, this transform unit is the first coefficient-bearing
    /// TU of its quantization group under adaptive quant, so `cu_qp_delta` is
    /// coded here (after the CBF flags, before residuals). `delta` is the signed
    /// luma QP difference from the QG predictor; the caller guarantees this TU
    /// has a coded block flag set.
    pub(crate) code_dqp: Option<i32>,
    /// Dependent quantization is used for this TU's transformed (non-TS)
    /// components (slice-level `sh_dep_quant_used_flag`).
    pub(crate) dep_quant: bool,
    /// Which components this unit codes (single tree, or a dual-tree luma /
    /// chroma pass).
    pub(crate) tree: TreeType,
}

#[inline]
fn any_nonzero(c: &[i32]) -> bool {
    c.iter().any(|&v| v != 0)
}

/// Code one component's residual: in lossless mode emit `transform_skip_flag`
/// (always 1) followed by TSRC; otherwise the regular residual coder.
/// Code one component's residual. Transform-skip is enabled in the SPS, so for a
/// non-BDPCM block the `transform_skip_flag` is coded: 1 selects the transform-
/// skip residual coder (TSRC), 0 the regular residual coder. Lossless always
/// uses TS; BDPCM infers TS (no flag). `ts` is the lossy per-component decision.
#[allow(clippy::too_many_arguments)]
fn encode_component_residual(
    enc: &mut CabacEncoder,
    ctx: &mut Contexts,
    coeff: &[i32],
    w: usize,
    h: usize,
    comp: Component,
    lossless: bool,
    bdpcm: bool,
    ts: bool,
    dep_quant: bool,
) {
    // Transform-skip is only available for blocks no larger than the signalled
    // maximum (32). For a 64-wide/-tall transform the flag is neither coded nor
    // usable, so such a block always takes the DCT path with high-frequency
    // zero-out.
    let ts_allowed = w <= 32 && h <= 32;
    let use_ts = (lossless || ts) && ts_allowed;
    debug_assert!(
        !lossless || ts_allowed,
        "lossless block >32 cannot use transform-skip"
    );
    if !bdpcm && ts_allowed {
        // transform_skip_flag, coded with MTSIndex ctx 4 (luma) / 5 (chroma).
        let ctx_idx = if comp.is_luma() { 4 } else { 5 };
        enc.encode_bin(use_ts as u8, &mut ctx.mts_index[ctx_idx]);
    }
    if use_ts {
        encode_residual_ts(enc, ctx, coeff, w, h, bdpcm);
    } else {
        encode_residual(enc, ctx, coeff, w, h, comp, dep_quant);
    }
}

/// Code one transform unit: chroma CBFs, the luma CBF, then the residual of
/// each component whose CBF is set.
/// Encode `cu_qp_delta_abs` (truncated-unary prefix, cMax 5, ctx0 for the first
/// bin and ctx1 for the rest; EG0 bypass suffix when `|delta| >= 5`) followed by
/// `cu_qp_delta_sign_flag` (bypass) when non-zero. Mirrors VTM `cu_qp_delta`.
fn encode_cu_qp_delta(enc: &mut CabacEncoder, ctx: &mut Contexts, delta: i32) {
    let abs = delta.unsigned_abs();
    let prefix = abs.min(5);
    for i in 0..prefix {
        let c = if i == 0 { 0 } else { 1 };
        enc.encode_bin(1, &mut ctx.cu_qp_delta_abs[c]);
    }
    if prefix < 5 {
        let c = if prefix == 0 { 0 } else { 1 };
        enc.encode_bin(0, &mut ctx.cu_qp_delta_abs[c]);
    }
    if abs >= 5 {
        // Exp-Golomb (order CU_DQP_EG_k = 0) of (abs - 5), bypass. Matches VTM
        // CABACWriter::exp_golomb_eqprob: a unary-ones prefix terminated by 0,
        // then `count` suffix bits.
        let mut symbol = abs - 5;
        let mut count = 0u32;
        let mut bins = 0u32;
        let mut num_bins = 0u32;
        while symbol >= (1 << count) {
            bins = (bins << 1) | 1;
            num_bins += 1;
            symbol -= 1 << count;
            count += 1;
        }
        bins <<= 1;
        num_bins += 1;
        enc.encode_bypass_bits(bins, num_bins);
        enc.encode_bypass_bits(symbol, count);
    }
    if abs > 0 {
        enc.encode_bypass((delta < 0) as u8);
    }
}

pub(crate) fn encode_transform_unit(enc: &mut CabacEncoder, ctx: &mut Contexts, tu: &TuCoeffs) {
    let code_luma = tu.tree != TreeType::Chroma;
    let code_chroma = tu.tree != TreeType::Luma && tu.chroma.is_some();
    let cbf_luma = code_luma && any_nonzero(tu.luma);
    let (cbf_cb, cbf_cr) = match tu.chroma {
        Some((cb, cr, _, _)) if code_chroma => (any_nonzero(cb), any_nonzero(cr)),
        _ => (false, false),
    };

    // cbf_cb, then cbf_cr (whose context depends on cbf_cb), then cbf_luma.
    // BDPCM blocks use fixed CBF context indices (Cb→1, Cr→2).
    if code_chroma {
        let cb_ctx = if tu.chroma_bdpcm { 1 } else { 0 };
        enc.encode_bin(cbf_cb as u8, &mut ctx.qt_cbf_cb[cb_ctx]);
        let cr_ctx = if tu.chroma_bdpcm { 2 } else { cbf_cb as usize };
        enc.encode_bin(cbf_cr as u8, &mut ctx.qt_cbf_cr[cr_ctx]);
    }
    // Luma cbf context: BDPCM blocks use index 1 (H.266 9.3.4.2.1), else 0.
    if code_luma {
        let luma_cbf_ctx = if tu.luma_bdpcm { 1 } else { 0 };
        enc.encode_bin(cbf_luma as u8, &mut ctx.qt_cbf_luma[luma_cbf_ctx]);
    }

    // Adaptive quant: when this is the QG's first coefficient-bearing TU the
    // caller passes the signed QP delta. It is coded after the CBF flags and
    // before the residuals (H.266 transform_unit). A cbf is guaranteed here.
    if let Some(delta) = tu.code_dqp {
        debug_assert!(cbf_luma || cbf_cb || cbf_cr);
        encode_cu_qp_delta(enc, ctx, delta);
    }

    if cbf_luma {
        encode_component_residual(
            enc,
            ctx,
            tu.luma,
            tu.lw,
            tu.lh,
            Component::Luma,
            tu.lossless,
            tu.luma_bdpcm,
            tu.luma_ts,
            tu.dep_quant,
        );
    }
    if code_chroma && let Some((cb, cr, cw, ch)) = tu.chroma {
        if cbf_cb {
            encode_component_residual(
                enc,
                ctx,
                cb,
                cw,
                ch,
                Component::Cb,
                tu.lossless,
                tu.chroma_bdpcm,
                tu.chroma_ts,
                tu.dep_quant,
            );
        }
        if cbf_cr {
            encode_component_residual(
                enc,
                ctx,
                cr,
                cw,
                ch,
                Component::Cr,
                tu.lossless,
                tu.chroma_bdpcm,
                tu.chroma_ts,
                tu.dep_quant,
            );
        }
    }
}

/// Code the transform tree for a leaf coding unit. With CU == TU there is never
/// an implicit split, so this defers directly to the single transform unit.
pub(crate) fn encode_transform_tree(enc: &mut CabacEncoder, ctx: &mut Contexts, tu: &TuCoeffs) {
    encode_transform_unit(enc, ctx, tu);
}

/// Test-only transform-unit decoder.
pub(crate) mod test_support {
    use super::*;
    use crate::cabac::engine::CabacDecoder;
    use crate::residual::test_support::decode_residual;
    use crate::residual_ts::test_support::decode_residual_ts;

    pub(crate) struct DecodedTu {
        pub(crate) luma: Vec<i32>,
        pub(crate) cb: Vec<i32>,
        pub(crate) cr: Vec<i32>,
        pub(crate) luma_ts: bool,
        pub(crate) cb_ts: bool,
        pub(crate) cr_ts: bool,
        /// Signed cu_qp_delta read in this TU (dual-tree luma path), else None.
        pub(crate) dqp: Option<i32>,
    }

    /// Read a non-BDPCM `transform_skip_flag` (transform-skip is enabled in the
    /// SPS) for one component, using MTSIndex ctx 4 (luma) / 5 (chroma).
    fn read_ts_flag(dec: &mut CabacDecoder, ctx: &mut Contexts, luma: bool) -> bool {
        let i = if luma { 4 } else { 5 };
        dec.decode_bin(&mut ctx.mts_index[i]) == 1
    }

    fn decode_comp(
        dec: &mut CabacDecoder,
        ctx: &mut Contexts,
        w: usize,
        h: usize,
        comp: Component,
        dep_quant: bool,
    ) -> (Vec<i32>, bool) {
        // transform_skip_flag is only present for blocks <= 32; a 64-wide/-tall
        // transform is always DCT-coded with high-frequency zero-out.
        let ts = if w <= 32 && h <= 32 {
            read_ts_flag(dec, ctx, comp.is_luma())
        } else {
            false
        };
        let levels = if ts {
            decode_residual_ts(dec, ctx, w, h, false)
        } else {
            decode_residual(dec, ctx, w, h, comp, dep_quant)
        };
        (levels, ts)
    }

    #[cfg(test)]
    pub(crate) fn decode_transform_unit(
        dec: &mut CabacDecoder,
        ctx: &mut Contexts,
        lw: usize,
        lh: usize,
        chroma: Option<(usize, usize)>,
    ) -> DecodedTu {
        let (cbf_cb, cbf_cr) = if chroma.is_some() {
            let b = dec.decode_bin(&mut ctx.qt_cbf_cb[0]) == 1;
            let r = dec.decode_bin(&mut ctx.qt_cbf_cr[b as usize]) == 1;
            (b, r)
        } else {
            (false, false)
        };
        let cbf_luma = dec.decode_bin(&mut ctx.qt_cbf_luma[0]) == 1;
        let (luma, luma_ts) = if cbf_luma {
            decode_comp(dec, ctx, lw, lh, Component::Luma, false)
        } else {
            (vec![0; lw * lh], false)
        };
        let (mut cb, mut cr) = (Vec::new(), Vec::new());
        let (mut cb_ts, mut cr_ts) = (false, false);
        if let Some((cw, ch)) = chroma {
            if cbf_cb {
                let (l, t) = decode_comp(dec, ctx, cw, ch, Component::Cb, false);
                cb = l;
                cb_ts = t;
            } else {
                cb = vec![0; cw * ch];
            }
            if cbf_cr {
                let (l, t) = decode_comp(dec, ctx, cw, ch, Component::Cr, false);
                cr = l;
                cr_ts = t;
            } else {
                cr = vec![0; cw * ch];
            }
        }
        DecodedTu {
            luma,
            cb,
            cr,
            luma_ts,
            cb_ts,
            cr_ts,
            dqp: None,
        }
    }

    /// Tree-aware mirror of [`super::encode_transform_unit`]: codes only the
    /// luma component (`TreeType::Luma`), only chroma (`TreeType::Chroma`), or
    /// both (`TreeType::Single`).
    pub(crate) fn decode_transform_unit_tree(
        dec: &mut CabacDecoder,
        ctx: &mut Contexts,
        lw: usize,
        lh: usize,
        chroma: Option<(usize, usize)>,
        tree: TreeType,
        dep_quant: bool,
        read_dqp: bool,
    ) -> DecodedTu {
        let code_luma = tree != TreeType::Chroma;
        let code_chroma = tree != TreeType::Luma && chroma.is_some();
        let (cbf_cb, cbf_cr) = if code_chroma {
            let b = dec.decode_bin(&mut ctx.qt_cbf_cb[0]) == 1;
            let r = dec.decode_bin(&mut ctx.qt_cbf_cr[b as usize]) == 1;
            (b, r)
        } else {
            (false, false)
        };
        let cbf_luma = if code_luma {
            dec.decode_bin(&mut ctx.qt_cbf_luma[0]) == 1
        } else {
            false
        };
        // cu_qp_delta: present after all CBF flags, before residuals, at the QG's
        // first coefficient-bearing TU (dual tree => luma tree only).
        let dqp = if read_dqp && (cbf_luma || cbf_cb || cbf_cr) {
            Some(decode_cu_qp_delta(dec, ctx))
        } else {
            None
        };
        let (luma, luma_ts) = if cbf_luma {
            decode_comp(dec, ctx, lw, lh, Component::Luma, dep_quant)
        } else {
            (vec![0; lw * lh], false)
        };
        let (mut cb, mut cr) = (Vec::new(), Vec::new());
        let (mut cb_ts, mut cr_ts) = (false, false);
        if code_chroma && let Some((cw, ch)) = chroma {
            if cbf_cb {
                let (l, t) = decode_comp(dec, ctx, cw, ch, Component::Cb, dep_quant);
                cb = l;
                cb_ts = t;
            } else {
                cb = vec![0; cw * ch];
            }
            if cbf_cr {
                let (l, t) = decode_comp(dec, ctx, cw, ch, Component::Cr, dep_quant);
                cr = l;
                cr_ts = t;
            } else {
                cr = vec![0; cw * ch];
            }
        }
        DecodedTu {
            luma,
            cb,
            cr,
            luma_ts,
            cb_ts,
            cr_ts,
            dqp,
        }
    }

    /// BDPCM-aware transform-unit decoder, mirroring [`encode_transform_unit`].
    /// `luma_bdpcm` / `chroma_bdpcm` select the BDPCM CBF/residual contexts; for
    /// BDPCM blocks `transform_skip_flag` is inferred (not coded) and the
    /// transform-skip residual coder is used with the BDPCM sign/context rules.
    fn decode_comp_full(
        dec: &mut CabacDecoder,
        ctx: &mut Contexts,
        w: usize,
        h: usize,
        comp: Component,
        bdpcm: bool,
        dep_quant: bool,
    ) -> (Vec<i32>, bool) {
        let ts_allowed = w <= 32 && h <= 32;
        // BDPCM => transform-skip is inferred (flag not present). Otherwise the
        // flag is present for blocks <= 32; 64-wide/-tall transforms are DCT.
        let ts = if bdpcm {
            true
        } else if ts_allowed {
            read_ts_flag(dec, ctx, comp.is_luma())
        } else {
            false
        };
        let levels = if ts {
            decode_residual_ts(dec, ctx, w, h, bdpcm)
        } else {
            decode_residual(dec, ctx, w, h, comp, dep_quant)
        };
        (levels, ts)
    }

    /// Decode `cu_qp_delta_abs` + sign, returning the signed delta (mirrors
    /// [`super::encode_cu_qp_delta`] / VTM `cu_qp_delta`).
    fn decode_cu_qp_delta(dec: &mut CabacDecoder, ctx: &mut Contexts) -> i32 {
        let mut ones = 0u32;
        while ones < 5 {
            let c = if ones == 0 { 0 } else { 1 };
            if dec.decode_bin(&mut ctx.cu_qp_delta_abs[c]) == 1 {
                ones += 1;
            } else {
                break;
            }
        }
        let mut abs = ones;
        if ones >= 5 {
            // Exp-Golomb (order 0) suffix, mirroring VTM exp_golomb_eqprob.
            let mut symbol = 0u32;
            let mut count = 0u32;
            let mut bit = 1u32;
            while bit != 0 {
                bit = dec.decode_bypass() as u32;
                symbol += bit << count;
                count += 1;
            }
            count -= 1;
            if count != 0 {
                let mut suffix = 0u32;
                for _ in 0..count {
                    suffix = (suffix << 1) | dec.decode_bypass() as u32;
                }
                symbol += suffix;
            }
            abs += symbol;
        }
        if abs == 0 {
            return 0;
        }
        let neg = dec.decode_bypass() == 1;
        if neg { -(abs as i32) } else { abs as i32 }
    }

    pub(crate) fn decode_transform_unit_full(
        dec: &mut CabacDecoder,
        ctx: &mut Contexts,
        lw: usize,
        lh: usize,
        chroma: Option<(usize, usize)>,
        luma_bdpcm: bool,
        chroma_bdpcm: bool,
        dep_quant: bool,
        read_dqp: bool,
    ) -> (DecodedTu, Option<i32>) {
        let (cbf_cb, cbf_cr) = if chroma.is_some() {
            let cb_ctx = if chroma_bdpcm { 1 } else { 0 };
            let b = dec.decode_bin(&mut ctx.qt_cbf_cb[cb_ctx]) == 1;
            let cr_ctx = if chroma_bdpcm { 2 } else { b as usize };
            let r = dec.decode_bin(&mut ctx.qt_cbf_cr[cr_ctx]) == 1;
            (b, r)
        } else {
            (false, false)
        };
        let luma_cbf_ctx = if luma_bdpcm { 1 } else { 0 };
        let cbf_luma = dec.decode_bin(&mut ctx.qt_cbf_luma[luma_cbf_ctx]) == 1;
        // Adaptive quant: when the QG has not yet coded its delta and any CBF is
        // set, the signed `cu_qp_delta` is present here (after CBFs, before
        // residuals). Garnetash leaves are <= 64, so the >64 forced-coding case
        // never applies.
        let dqp = if read_dqp && (cbf_luma || cbf_cb || cbf_cr) {
            Some(decode_cu_qp_delta(dec, ctx))
        } else {
            None
        };
        let (luma, luma_ts) = if cbf_luma {
            decode_comp_full(dec, ctx, lw, lh, Component::Luma, luma_bdpcm, dep_quant)
        } else {
            (vec![0; lw * lh], false)
        };
        let (mut cb, mut cr) = (Vec::new(), Vec::new());
        let (mut cb_ts, mut cr_ts) = (false, false);
        if let Some((cw, ch)) = chroma {
            if cbf_cb {
                let (l, t) =
                    decode_comp_full(dec, ctx, cw, ch, Component::Cb, chroma_bdpcm, dep_quant);
                cb = l;
                cb_ts = t;
            } else {
                cb = vec![0; cw * ch];
            }
            if cbf_cr {
                let (l, t) =
                    decode_comp_full(dec, ctx, cw, ch, Component::Cr, chroma_bdpcm, dep_quant);
                cr = l;
                cr_ts = t;
            } else {
                cr = vec![0; cw * ch];
            }
        }
        (
            DecodedTu {
                luma,
                cb,
                cr,
                luma_ts,
                cb_ts,
                cr_ts,
                dqp: None,
            },
            dqp,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use crate::cabac::engine::CabacDecoder;

    #[test]
    fn luma_only_round_trips() {
        let mut c = vec![0i32; 64];
        c[0] = 12;
        c[1] = -3;
        c[9] = 1;
        let mut enc = CabacEncoder::new();
        let mut ectx = Contexts::new_intra(28);
        let tu = TuCoeffs {
            tree: TreeType::Single,
            luma: &c,
            lw: 8,
            lh: 8,
            chroma: None,
            lossless: false,
            luma_bdpcm: false,
            chroma_bdpcm: false,
            luma_ts: false,
            chroma_ts: false,
            code_dqp: None,
            dep_quant: false,
        };
        encode_transform_tree(&mut enc, &mut ectx, &tu);
        enc.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = CabacDecoder::new(&bytes);
        let mut dctx = Contexts::new_intra(28);
        let got = decode_transform_unit(&mut dec, &mut dctx, 8, 8, None);
        assert_eq!(dec.decode_terminate(), 1);
        assert_eq!(got.luma, c);
    }

    #[test]
    fn dual_tree_luma_then_chroma_round_trips() {
        // Dual tree codes a luma-only TU then a chroma-only TU into one CABAC
        // stream. Each pass must code only its own components (the luma pass
        // emits no chroma CBFs; the chroma pass emits no luma CBF), and both
        // must round-trip.
        let mut y = vec![0i32; 256];
        let mut cb = vec![0i32; 64];
        let mut cr = vec![0i32; 64];
        y[0] = 40;
        y[3] = -7;
        y[17] = 2;
        cb[0] = -5;
        cb[1] = 3;
        cr[0] = 9;
        cr[2] = -2;
        for &qp in &[12u8, 28, 45] {
            let mut enc = CabacEncoder::new();
            let mut ectx = Contexts::new_intra(qp);
            let luma_tu = TuCoeffs {
                tree: TreeType::Luma,
                luma: &y,
                lw: 16,
                lh: 16,
                chroma: None,
                lossless: false,
                luma_bdpcm: false,
                chroma_bdpcm: false,
                luma_ts: false,
                chroma_ts: false,
                code_dqp: None,
                dep_quant: false,
            };
            encode_transform_tree(&mut enc, &mut ectx, &luma_tu);
            let chroma_tu = TuCoeffs {
                tree: TreeType::Chroma,
                luma: &[],
                lw: 8,
                lh: 8,
                chroma: Some((&cb, &cr, 8, 8)),
                lossless: false,
                luma_bdpcm: false,
                chroma_bdpcm: false,
                luma_ts: false,
                chroma_ts: false,
                code_dqp: None,
                dep_quant: false,
            };
            encode_transform_tree(&mut enc, &mut ectx, &chroma_tu);
            enc.encode_terminate(1);
            let bytes = enc.finish();

            let mut dec = CabacDecoder::new(&bytes);
            let mut dctx = Contexts::new_intra(qp);
            let lg = decode_transform_unit_tree(
                &mut dec,
                &mut dctx,
                16,
                16,
                None,
                TreeType::Luma,
                false,
                false,
            );
            let cg = decode_transform_unit_tree(
                &mut dec,
                &mut dctx,
                8,
                8,
                Some((8, 8)),
                TreeType::Chroma,
                false,
                false,
            );
            assert_eq!(dec.decode_terminate(), 1, "terminate qp={qp}");
            assert_eq!(lg.luma, y, "luma qp={qp}");
            assert_eq!(cg.cb, cb, "cb qp={qp}");
            assert_eq!(cg.cr, cr, "cr qp={qp}");
        }
    }

    #[test]
    fn luma_plus_chroma_round_trips() {
        // 16x16 luma with 8x8 chroma (4:2:0).
        let mut y = vec![0i32; 256];
        let mut cb = vec![0i32; 64];
        let mut cr = vec![0i32; 64];
        y[0] = 40;
        y[3] = -7;
        y[17] = 2;
        cb[0] = -5;
        cb[1] = 3;
        cb[8] = 1;
        cr[0] = 9;
        cr[2] = -2;
        for &qp in &[12u8, 28, 45] {
            let mut enc = CabacEncoder::new();
            let mut ectx = Contexts::new_intra(qp);
            let tu = TuCoeffs {
                tree: TreeType::Single,
                luma: &y,
                lw: 16,
                lh: 16,
                chroma: Some((&cb, &cr, 8, 8)),
                lossless: false,
                luma_bdpcm: false,
                chroma_bdpcm: false,
                luma_ts: false,
                chroma_ts: false,
                code_dqp: None,
                dep_quant: false,
            };
            encode_transform_tree(&mut enc, &mut ectx, &tu);
            enc.encode_terminate(1);
            let bytes = enc.finish();
            let mut dec = CabacDecoder::new(&bytes);
            let mut dctx = Contexts::new_intra(qp);
            let got = decode_transform_unit(&mut dec, &mut dctx, 16, 16, Some((8, 8)));
            assert_eq!(dec.decode_terminate(), 1, "qp={qp}");
            assert_eq!(got.luma, y, "luma qp={qp}");
            assert_eq!(got.cb, cb, "cb qp={qp}");
            assert_eq!(got.cr, cr, "cr qp={qp}");
        }
    }

    #[test]
    fn empty_chroma_codes_only_cbfs() {
        // Zero chroma residual: only cbf_cb=0, cbf_cr=0 coded for chroma.
        let mut y = vec![0i32; 64];
        y[0] = 5;
        let cb = vec![0i32; 16];
        let cr = vec![0i32; 16];
        let mut enc = CabacEncoder::new();
        let mut ectx = Contexts::new_intra(32);
        let tu = TuCoeffs {
            tree: TreeType::Single,
            luma: &y,
            lw: 8,
            lh: 8,
            chroma: Some((&cb, &cr, 4, 4)),
            lossless: false,
            luma_bdpcm: false,
            chroma_bdpcm: false,
            luma_ts: false,
            chroma_ts: false,
            code_dqp: None,
            dep_quant: false,
        };
        encode_transform_tree(&mut enc, &mut ectx, &tu);
        enc.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = CabacDecoder::new(&bytes);
        let mut dctx = Contexts::new_intra(32);
        let got = decode_transform_unit(&mut dec, &mut dctx, 8, 8, Some((4, 4)));
        assert_eq!(dec.decode_terminate(), 1);
        assert_eq!(got.luma, y);
        assert!(got.cb.iter().all(|&v| v == 0));
        assert!(got.cr.iter().all(|&v| v == 0));
    }
}
