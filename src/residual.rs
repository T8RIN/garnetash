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

use crate::cabac::{CabacEncoder, Contexts};

/// color component selecting the context sets and template rules used by the
/// coefficient coder. Cb and Cr share the same chroma context sets.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Component {
    Luma,
    Cb,
    Cr,
}

impl Component {
    #[inline]
    pub(crate) fn is_luma(self) -> bool {
        matches!(self, Component::Luma)
    }
}

const COEF_REMAIN_BIN_REDUCTION: u32 = 5;
const MAX_LOG2_TR_DR: u32 = 15;
const CTX_BIN_RATIO: i32 = 28; // MAX_TU_LEVEL_CTX_CODED_BIN_CONSTRAINT (luma & chroma)

/// `g_groupIdx`: maps a position coordinate to its last-significant group index.
#[rustfmt::skip]
static GROUP_IDX: [u32; 32] = [
    0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
];
/// `g_minInGroup`: lowest coordinate value in each last-significant group.
static MIN_IN_GROUP: [u32; 14] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96];
/// `g_goRiceParsCoeff`: Golomb-Rice parameter from the clamped neighbour sum.
#[rustfmt::skip]
static GO_RICE_PARS: [u32; 32] = [
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
];
/// `prefixCtx`: per-log2-size last-prefix context offset (luma).
static PREFIX_CTX: [i32; 8] = [0, 0, 0, 3, 6, 10, 15, 21];

/// Up-right diagonal scan over a `w x h` grid (VTM `ScanGenerator::DIAG`),
/// returning positions as `(x, y)` in scan order (DC first).
pub(crate) fn diag_scan(w: usize, h: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(w * h);
    let (mut line, mut col) = (0usize, 0usize);
    for _ in 0..w * h {
        out.push((col, line));
        if col == w - 1 || line == 0 {
            line += col + 1;
            col = 0;
            if line >= h {
                col += line - (h - 1);
                line = h - 1;
            }
        } else {
            col += 1;
            line -= 1;
        }
    }
    out
}

struct ScanTables {
    scan: Vec<(usize, usize)>,
    cg_scan: Vec<(usize, usize)>,
}

pub(crate) fn scan_coords(w: usize, h: usize) -> &'static [(usize, usize)] {
    &scan_tables(w.min(32), h.min(32)).scan
}

/// VVC last-significant-coefficient group index for a coordinate `c` (0..=31).
pub(crate) fn last_group_idx(c: usize) -> u32 {
    GROUP_IDX[c.min(31)]
}

/// Return the cached scan tables for a clamped block size `cw × ch`
/// (`cw, ch ∈ {4, 8, 16, 32}`). The 16-entry cache is built once and is
/// immutable afterwards, so concurrent readers never contend.
fn scan_tables(cw: usize, ch: usize) -> &'static ScanTables {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Vec<ScanTables>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| {
        let mut v = Vec::with_capacity(16);
        for lw in 2..6usize {
            for lh in 2..6usize {
                let (cw, ch) = (1usize << lw, 1usize << lh);
                let cg_scan = diag_scan(cw / 4, ch / 4);
                let sub_scan = diag_scan(4, 4);
                let mut scan = Vec::with_capacity(cw * ch);
                for &(cgx, cgy) in &cg_scan {
                    for &(sx, sy) in &sub_scan {
                        scan.push((cgx * 4 + sx, cgy * 4 + sy));
                    }
                }
                v.push(ScanTables { scan, cg_scan });
            }
        }
        v
    });
    let idx = (cw.trailing_zeros() as usize - 2) * 4 + (ch.trailing_zeros() as usize - 2);
    &cache[idx]
}

/// Coefficient-coding context for one square luma transform block: the grouped
/// diagonal scan, the coefficient-group scan, and the last-position parameters.
struct CoeffCtx {
    w: usize,
    h: usize,
    scan: &'static [(usize, usize)], // per scan position, (x, y) in the block
    scan_cg: &'static [(usize, usize)], // per CG scan position, (cgx, cgy)
    width_in_groups: usize,
    height_in_groups: usize,
    last_offset_x: i32,
    last_offset_y: i32,
    last_shift_x: i32,
    last_shift_y: i32,
    max_last_x: u32,
    max_last_y: u32,
}

impl CoeffCtx {
    fn new(w: usize, h: usize, comp: Component) -> Self {
        // VVC high-frequency zero-out (JVET_C0024_ZERO_OUT_TH = 32): for a
        // transform dimension of 64 only the low-frequency 32 coefficients can
        // be non-zero, so the scan, coefficient groups and last-position range
        // are clamped to 32 — identical to a 32-wide block. The buffer stride
        // (`w`/`h`) and the last-position *context* offset/shift still use the
        // actual size, matching vvdec's CoeffCodingContext exactly.
        let cw = w.min(32);
        let ch = h.min(32);
        // All our square sizes use 4x4 coefficient groups.
        let wig = cw / 4;
        let hig = ch / 4;
        let st = scan_tables(cw, ch);
        let log2w = w.trailing_zeros() as usize;
        let log2h = h.trailing_zeros() as usize;
        let (lox, loy, lsx, lsy) = if comp.is_luma() {
            (
                PREFIX_CTX[log2w],
                PREFIX_CTX[log2h],
                ((log2w + 1) >> 2) as i32,
                ((log2h + 1) >> 2) as i32,
            )
        } else {
            // chroma: lastOffset = 0, lastShift = Clip3(0, 2, size >> 3)
            (
                0,
                0,
                ((w >> 3) as i32).clamp(0, 2),
                ((h >> 3) as i32).clamp(0, 2),
            )
        };
        CoeffCtx {
            w,
            h,
            scan: &st.scan,
            scan_cg: &st.cg_scan,
            width_in_groups: wig,
            height_in_groups: hig,
            last_offset_x: lox,
            last_offset_y: loy,
            last_shift_x: lsx,
            last_shift_y: lsy,
            max_last_x: GROUP_IDX[cw - 1],
            max_last_y: GROUP_IDX[ch - 1],
        }
    }

    #[inline]
    fn block_pos(&self, scan_pos: usize) -> usize {
        let (x, y) = self.scan[scan_pos];
        y * self.w + x
    }
    #[inline]
    fn last_x_ctx(&self, pos: u32) -> usize {
        (self.last_offset_x + (pos as i32 >> self.last_shift_x)) as usize
    }
    #[inline]
    fn last_y_ctx(&self, pos: u32) -> usize {
        (self.last_offset_y + (pos as i32 >> self.last_shift_y)) as usize
    }
}

/// Significance-flag context plus the template variables that the subsequent
/// greater-than / parity flags reuse (VTM `sigCtxIdAbs`). Returns
/// `(ctx_index, diag, sum1)`.
fn sig_ctx(
    coeff: &[i32],
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    is_luma: bool,
) -> (usize, i32, i32) {
    let p = y * w + x;
    let diag = (x + y) as i32;
    let mut sum_abs = 0i32;
    let mut num_pos = 0i32;
    let mut upd = |v: i32| {
        let a = v.abs();
        sum_abs += (4 + (a & 1)).min(a);
        num_pos += (a != 0) as i32;
    };
    if x < w - 1 {
        upd(coeff[p + 1]);
        if x < w - 2 {
            upd(coeff[p + 2]);
        }
        if y < h - 1 {
            upd(coeff[p + w + 1]);
        }
    }
    if y < h - 1 {
        upd(coeff[p + w]);
        if y < h - 2 {
            upd(coeff[p + 2 * w]);
        }
    }
    let mut ctx_ofs = ((sum_abs + 1) >> 1).min(3) + if diag < 2 { 4 } else { 0 };
    if is_luma {
        ctx_ofs += if diag < 5 { 4 } else { 0 };
    }
    (ctx_ofs as usize, diag, sum_abs - num_pos)
}

/// Context offset for the gt1 / parity / gt2 flags (VTM `ctxOffsetAbs`).
fn ctx_offset_abs(diag: i32, sum1: i32, is_luma: bool) -> usize {
    if diag == -1 {
        return 0;
    }
    let mut offset = sum1.min(4) + 1;
    offset += if is_luma {
        if diag == 0 {
            15
        } else if diag < 3 {
            10
        } else if diag < 10 {
            5
        } else {
            0
        }
    } else if diag == 0 {
        5
    } else {
        0
    };
    offset as usize
}

/// Clamped neighbour absolute-sum for Golomb-Rice parameter derivation
/// (VTM `templateAbsSum`, history value 0).
fn template_abs_sum(
    coeff: &[i32],
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    base_level: i32,
) -> usize {
    let p = y * w + x;
    let mut sum = 0i32;
    if x < w - 1 {
        sum += coeff[p + 1].abs();
        if x < w - 2 {
            sum += coeff[p + 2].abs();
        }
        if y < h - 1 {
            sum += coeff[p + w + 1].abs();
        }
    }
    if y < h - 1 {
        sum += coeff[p + w].abs();
        if y < h - 2 {
            sum += coeff[p + 2 * w].abs();
        }
    }
    (sum - 5 * base_level).clamp(0, 31) as usize
}

#[inline]
fn derive_rice(coeff: &[i32], x: usize, y: usize, w: usize, h: usize, base_level: i32) -> u32 {
    GO_RICE_PARS[template_abs_sum(coeff, x, y, w, h, base_level)]
}

/// Encode a Golomb-Rice + limited-Exp-Golomb remainder (VTM `encodeRemAbsEP`).
pub(crate) fn encode_rem_abs_ep(enc: &mut CabacEncoder, bins: u32, rice: u32) {
    let cutoff = COEF_REMAIN_BIN_REDUCTION;
    let threshold = cutoff << rice;
    if bins < threshold {
        let length = (bins >> rice) + 1;
        enc.encode_bypass_bits((1 << length) - 2, length);
        enc.encode_bypass_bits(bins & ((1 << rice) - 1), rice);
    } else {
        let max_prefix_len = 32 - cutoff - MAX_LOG2_TR_DR;
        let code_value = (bins >> rice) - cutoff;
        let mut prefix_len = 0u32;
        let suffix_len;
        if code_value >= (1 << max_prefix_len) - 1 {
            prefix_len = max_prefix_len;
            suffix_len = MAX_LOG2_TR_DR;
        } else {
            while code_value > (2 << prefix_len) - 2 {
                prefix_len += 1;
            }
            suffix_len = prefix_len + rice + 1;
        }
        let total_prefix_len = prefix_len + cutoff;
        let prefix = (1u32 << total_prefix_len) - 1;
        let suffix = ((code_value - ((1 << prefix_len) - 1)) << rice) | (bins & ((1 << rice) - 1));
        enc.encode_bypass_bits(prefix, total_prefix_len);
        enc.encode_bypass_bits(suffix, suffix_len);
    }
}

/// Encode the last-significant-coefficient position.
fn encode_last_sig(
    enc: &mut CabacEncoder,
    ctx: &mut Contexts,
    cc: &CoeffCtx,
    scan_pos_last: usize,
    luma: bool,
) {
    let blk = cc.block_pos(scan_pos_last);
    let mut pos_x = (blk % cc.w) as u32;
    let mut pos_y = (blk / cc.w) as u32;
    let gx = GROUP_IDX[pos_x as usize];
    let gy = GROUP_IDX[pos_y as usize];
    for c in 0..gx {
        let i = cc.last_x_ctx(c);
        enc.encode_bin(
            1,
            if luma {
                &mut ctx.last_x[i]
            } else {
                &mut ctx.last_x_c[i]
            },
        );
    }
    if gx < cc.max_last_x {
        let i = cc.last_x_ctx(gx);
        enc.encode_bin(
            0,
            if luma {
                &mut ctx.last_x[i]
            } else {
                &mut ctx.last_x_c[i]
            },
        );
    }
    for c in 0..gy {
        let i = cc.last_y_ctx(c);
        enc.encode_bin(
            1,
            if luma {
                &mut ctx.last_y[i]
            } else {
                &mut ctx.last_y_c[i]
            },
        );
    }
    if gy < cc.max_last_y {
        let i = cc.last_y_ctx(gy);
        enc.encode_bin(
            0,
            if luma {
                &mut ctx.last_y[i]
            } else {
                &mut ctx.last_y_c[i]
            },
        );
    }
    if gx > 3 {
        pos_x -= MIN_IN_GROUP[gx as usize];
        for i in (0..=(((gx - 2) >> 1) as i32 - 1)).rev() {
            enc.encode_bypass(((pos_x >> i) & 1) as u8);
        }
    }
    if gy > 3 {
        pos_y -= MIN_IN_GROUP[gy as usize];
        for i in (0..=(((gy - 2) >> 1) as i32 - 1)).rev() {
            enc.encode_bypass(((pos_y >> i) & 1) as u8);
        }
    }
}

pub(crate) fn mts_signallable(levels: &[i32], w: usize, h: usize) -> bool {
    if last_sig_scan_pos(levels, w, h, Component::Luma) < 1 {
        return false;
    }
    for y in 0..h {
        for x in 0..w {
            if (x >= 16 || y >= 16) && levels[y * w + x] != 0 {
                return false;
            }
        }
    }
    true
}

pub(crate) fn last_sig_scan_pos(levels: &[i32], w: usize, h: usize, comp: Component) -> i32 {
    let cc = CoeffCtx::new(w, h, comp);
    let mut last = -1i32;
    for scan_pos in 0..cc.scan.len() {
        if levels[cc.block_pos(scan_pos)] != 0 {
            last = scan_pos as i32;
        }
    }
    last
}

#[allow(clippy::type_complexity)]
pub(crate) fn lfnst_present(
    sps_lfnst: bool,
    luma_w: usize,
    luma_h: usize,
    luma: (&[i32], bool),
    chroma: Option<(&[i32], &[i32], usize, usize, bool)>,
) -> bool {
    if !sps_lfnst || luma_w > 64 || luma_h > 64 {
        return false;
    }
    let mut last_scan_pos = false;
    let mut violates = false;
    let mut tr_skip = false;
    let mut visit = |levels: &[i32], w: usize, h: usize, ts: bool| {
        if levels.is_empty() {
            return; // component not present (e.g. luma in a chroma-only CU)
        }
        let sp_last = last_sig_scan_pos(levels, w, h, Component::Luma);
        if sp_last < 0 {
            return; // cbf = 0: this component contributes nothing
        }
        if ts {
            tr_skip = true; // a coded transform-skip component disables LFNST
            return;
        }
        if w >= 4 && h >= 4 {
            let max_pos = if (w == 4 && h == 4) || (w == 8 && h == 8) {
                7
            } else {
                15
            };
            if sp_last > max_pos {
                violates = true;
            }
            if sp_last >= 1 {
                last_scan_pos = true;
            }
        }
    };
    visit(luma.0, luma_w, luma_h, luma.1);
    if let Some((cb, cr, cw, ch, cts)) = chroma {
        visit(cb, cw, ch, cts);
        visit(cr, cw, ch, cts);
    }
    last_scan_pos && !violates && !tr_skip
}

/// (`y*w + x`) array of quantized levels; it must contain at least one nonzero.
pub(crate) fn encode_residual(
    enc: &mut CabacEncoder,
    ctx: &mut Contexts,
    coeff: &[i32],
    w: usize,
    h: usize,
    comp: Component,
    dep_quant: bool,
) {
    let luma = comp.is_luma();
    let cc = CoeffCtx::new(w, h, comp);
    // Number of coded positions after high-frequency zero-out (= w·h except for
    // a 64-wide/-tall block, where it is clamped to the 32×32 low-frequency
    // region). `cc.scan` already spans exactly this region.
    let max_coeff = cc.scan.len();
    let log2_cg = 4usize;

    // Last position and significant-group flags (scan-indexed).
    let mut scan_pos_last = -1i32;
    let num_cg = (max_coeff >> log2_cg).max(1);
    // num_cg ≤ 64 (max coded block is 32×32 = 1024 coeffs, 16 per 4×4 group),
    // so the per-group significance flags live on the stack — avoiding a heap
    // allocation on every (frequently RD-trialled) residual-coding call.
    debug_assert!(num_cg <= 64);
    let mut sig_scan_arr = [false; 64];
    let sig_scan = &mut sig_scan_arr[..num_cg];
    for scan_pos in 0..max_coeff {
        if coeff[cc.block_pos(scan_pos)] != 0 {
            scan_pos_last = scan_pos as i32;
            sig_scan[scan_pos >> log2_cg] = true;
        }
    }
    debug_assert!(scan_pos_last >= 0, "empty TU");
    let scan_pos_last = scan_pos_last as usize;

    encode_last_sig(enc, ctx, &cc, scan_pos_last, luma);

    let mut reg_bin_limit: i32 = (max_coeff as i32 * CTX_BIN_RATIO) >> 4;
    let last_cg = scan_pos_last >> log2_cg;
    // VTM `m_tmplCpDiag`/`m_tmplCpSum1`: set by `sigCtxIdAbs` and read by
    // `ctxOffsetAbs`. They persist across subblocks and stay at the sentinel
    // (-1) until the first significance context is derived — which never
    // happens for the inferred `scanPosLast`, so its gtx offset is 0.
    let mut tmpl_diag = -1i32;
    let mut tmpl_sum1 = 0i32;
    // Raster-indexed group flags, for the sig-group-flag neighbour context.
    let mut sig_raster_arr = [false; 64];
    let sig_raster = &mut sig_raster_arr[..num_cg];

    // Dependent-quantization quantizer state, advanced per coefficient in scan
    // (reverse) order across the whole block. Pinned at 0 when dependent quant
    // is disabled, so context selection and the bypass zero-position are
    // identical to scalar coding.
    let mut dq_state = 0u8;

    for sub_set_id in (0..=last_cg).rev() {
        let min_sub = sub_set_id << log2_cg;
        let max_sub = min_sub + (1 << log2_cg) - 1;
        let is_last = sub_set_id == last_cg;
        let is_not_first = sub_set_id != 0;
        let (cgx, cgy) = cc.scan_cg[sub_set_id];
        let sub_set_pos = cgy * cc.width_in_groups + cgx;
        let significant = sig_scan[sub_set_id];

        // significant_coeffgroup_flag (inferred for the last and DC groups).
        if !is_last && is_not_first {
            let sig_right = cgx + 1 < cc.width_in_groups && sig_raster[sub_set_pos + 1];
            let sig_lower =
                cgy + 1 < cc.height_in_groups && sig_raster[sub_set_pos + cc.width_in_groups];
            let gctx = (sig_right | sig_lower) as usize;
            enc.encode_bin(
                significant as u8,
                if luma {
                    &mut ctx.sig_group[gctx]
                } else {
                    &mut ctx.sig_group_c[gctx]
                },
            );
            if !significant {
                continue;
            }
        }
        if significant {
            sig_raster[sub_set_pos] = true;
        }

        let first_sig_pos = if is_last { scan_pos_last } else { max_sub };
        let infer_sig_pos: i32 = if first_sig_pos != scan_pos_last {
            if is_not_first { min_sub as i32 } else { -1 }
        } else {
            first_sig_pos as i32
        };

        let mut ctx_off = [0usize; 16];
        let mut num_nonzero = 0i32;
        let mut sign_pattern = 0u32;
        let mut rem_reg_bins = reg_bin_limit;
        let mut first_pos_2nd = min_sub as i32 - 1;

        let mut next_sig_pos = first_sig_pos as i32;
        while next_sig_pos >= min_sub as i32 && rem_reg_bins >= 4 {
            let sp = next_sig_pos as usize;
            let (x, y) = cc.scan[sp];
            let level = coeff[y * w + x];
            let sig = (level != 0) as u8;
            // Derive the significance context (and update the gtx template) only
            // in the cases where VTM calls `sigCtxIdAbs`. For the inferred
            // `scanPosLast`, neither branch runs, so the template keeps its
            // sentinel and the gtx offset stays 0.
            if num_nonzero > 0 || next_sig_pos != infer_sig_pos {
                let (sctx, diag, sum1) = sig_ctx(coeff, x, y, w, h, luma);
                tmpl_diag = diag;
                tmpl_sum1 = sum1;
                enc.encode_bin(sig, ctx.sig_model(luma, dq_state, sctx));
                rem_reg_bins -= 1;
            } else if next_sig_pos != scan_pos_last as i32 {
                let (_, diag, sum1) = sig_ctx(coeff, x, y, w, h, luma);
                tmpl_diag = diag;
                tmpl_sum1 = sum1;
            }
            if sig != 0 {
                let off = ctx_offset_abs(tmpl_diag, tmpl_sum1, luma);
                ctx_off[(next_sig_pos - min_sub as i32) as usize] = off;
                num_nonzero += 1;
                let abs_level = level.unsigned_abs();
                if next_sig_pos != scan_pos_last as i32 {
                    sign_pattern <<= 1;
                }
                if level < 0 {
                    sign_pattern |= 1;
                }
                let gt1 = abs_level > 1;
                enc.encode_bin(
                    gt1 as u8,
                    if luma {
                        &mut ctx.gt1_flag[off]
                    } else {
                        &mut ctx.gt1_flag_c[off]
                    },
                );
                rem_reg_bins -= 1;
                if gt1 {
                    enc.encode_bin(
                        (abs_level & 1) as u8,
                        if luma {
                            &mut ctx.par_flag[off]
                        } else {
                            &mut ctx.par_flag_c[off]
                        },
                    );
                    rem_reg_bins -= 1;
                    let gt2 = abs_level > 3;
                    enc.encode_bin(
                        gt2 as u8,
                        if luma {
                            &mut ctx.gt2_flag[off]
                        } else {
                            &mut ctx.gt2_flag_c[off]
                        },
                    );
                    rem_reg_bins -= 1;
                    if gt2 {
                        first_pos_2nd = first_pos_2nd.max(next_sig_pos);
                    }
                }
            }
            if dep_quant {
                dq_state = crate::depquant::next_state(dq_state, level);
            }
            next_sig_pos -= 1;
        }
        let min_pos_2nd = next_sig_pos;
        reg_bin_limit = rem_reg_bins;

        // 2nd pass: Golomb-Rice remainders for context-coded coeffs with level > 3.
        let mut scan_pos = first_pos_2nd;
        while scan_pos > min_pos_2nd {
            let (x, y) = cc.scan[scan_pos as usize];
            let abs_level = coeff[y * w + x].unsigned_abs();
            if abs_level >= 4 {
                let rice = derive_rice(coeff, x, y, w, h, 4);
                encode_rem_abs_ep(enc, (abs_level - 4) >> 1, rice);
            }
            scan_pos -= 1;
        }

        // bypass pass: coeffs beyond the context-bin budget.
        let mut scan_pos = min_pos_2nd;
        while scan_pos >= min_sub as i32 {
            let (x, y) = cc.scan[scan_pos as usize];
            let val = coeff[y * w + x];
            let abs_level = val.unsigned_abs();
            let rice = derive_rice(coeff, x, y, w, h, 0);
            let pos0 = (if dq_state < 2 { 1u32 } else { 2 }) << rice; // g_goRicePosCoeff0(state, rice)
            let rem = if abs_level == 0 {
                pos0
            } else if abs_level <= pos0 {
                abs_level - 1
            } else {
                abs_level
            };
            encode_rem_abs_ep(enc, rem, rice);
            if abs_level != 0 {
                num_nonzero += 1;
                sign_pattern <<= 1;
                if val < 0 {
                    sign_pattern |= 1;
                }
            }
            if dep_quant {
                dq_state = crate::depquant::next_state(dq_state, val);
            }
            scan_pos -= 1;
        }

        // signs (no sign-data-hiding in v1).
        enc.encode_bypass_bits(sign_pattern, num_nonzero as u32);
    }
}

pub(crate) mod test_support {
    use super::*;
    use crate::cabac::engine::CabacDecoder;

    pub(crate) fn dec_bits(dec: &mut CabacDecoder, n: u32) -> u32 {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | dec.decode_bypass() as u32;
        }
        v
    }

    pub(crate) fn decode_rem_abs_ep(dec: &mut CabacDecoder, rice: u32) -> u32 {
        let cutoff = COEF_REMAIN_BIN_REDUCTION;
        let cap = cutoff + (32 - cutoff - MAX_LOG2_TR_DR); // cutoff + maxPrefixLen
        let mut ones = 0u32;
        while ones < cap && dec.decode_bypass() == 1 {
            ones += 1;
        }
        let mask = (1u32 << rice) - 1;
        if ones < cutoff {
            (ones << rice) + dec_bits(dec, rice)
        } else {
            assert!(
                ones < cap,
                "Golomb-Rice cap hit; test coefficient too large"
            );
            let pl = ones - cutoff;
            let suf = dec_bits(dec, pl + rice);
            let code_value = ((1u32 << pl) - 1) + (suf >> rice);
            ((code_value + cutoff) << rice) + (suf & mask)
        }
    }

    fn decode_last_sig(
        dec: &mut CabacDecoder,
        ctx: &mut Contexts,
        cc: &CoeffCtx,
        luma: bool,
    ) -> usize {
        let mut gx = 0u32;
        while gx < cc.max_last_x && {
            let i = cc.last_x_ctx(gx);
            dec.decode_bin(if luma {
                &mut ctx.last_x[i]
            } else {
                &mut ctx.last_x_c[i]
            }) == 1
        } {
            gx += 1;
        }
        let mut gy = 0u32;
        while gy < cc.max_last_y && {
            let i = cc.last_y_ctx(gy);
            dec.decode_bin(if luma {
                &mut ctx.last_y[i]
            } else {
                &mut ctx.last_y_c[i]
            }) == 1
        } {
            gy += 1;
        }
        let mut pos_x = if gx > 3 {
            let nbits = (gx - 2) >> 1;
            MIN_IN_GROUP[gx as usize] + dec_bits(dec, nbits)
        } else {
            gx
        };
        let mut pos_y = if gy > 3 {
            let nbits = (gy - 2) >> 1;
            MIN_IN_GROUP[gy as usize] + dec_bits(dec, nbits)
        } else {
            gy
        };
        // Find scan position matching (pos_x, pos_y). A corrupt stream can decode
        // an out-of-block position that is not in the scan; fall back to 0 (a
        // valid index) so the decoder produces wrong output rather than panicking.
        let target = (pos_x as usize, pos_y as usize);
        let sp = cc.scan.iter().position(|&p| p == target).unwrap_or(0);
        let _ = (&mut pos_x, &mut pos_y);
        sp
    }

    pub(crate) fn decode_residual(
        dec: &mut CabacDecoder,
        ctx: &mut Contexts,
        w: usize,
        h: usize,
        comp: Component,
        dep_quant: bool,
    ) -> Vec<i32> {
        let luma = comp.is_luma();
        let cc = CoeffCtx::new(w, h, comp);
        let mut coeff = vec![0i32; w * h];
        let log2_cg = 4usize;
        let max_coeff = cc.scan.len();
        let scan_pos_last = decode_last_sig(dec, ctx, &cc, luma);
        let last_cg = scan_pos_last >> log2_cg;
        let mut reg_bin_limit: i32 = (max_coeff as i32 * CTX_BIN_RATIO) >> 4;
        // Mirror VTM `m_tmplCpDiag`/`m_tmplCpSum1` (see encoder).
        let mut tmpl_diag = -1i32;
        let mut tmpl_sum1 = 0i32;
        // Raster-indexed group flags, mirroring the encoder's neighbour context.
        let mut sig_raster = vec![false; (max_coeff >> log2_cg).max(1)];

        // Dependent-quantization quantizer state (see encoder). Pinned at 0 when
        // dependent quant is off, so decoding is identical to scalar coding.
        let mut dq_state = 0u8;

        for sub_set_id in (0..=last_cg).rev() {
            let min_sub = sub_set_id << log2_cg;
            let max_sub = min_sub + (1 << log2_cg) - 1;
            let is_last = sub_set_id == last_cg;
            let is_not_first = sub_set_id != 0;
            let (cgx, cgy) = cc.scan_cg[sub_set_id];
            let sub_set_pos = cgy * cc.width_in_groups + cgx;

            let mut significant = is_last; // last group inferred significant
            if !is_last && is_not_first {
                let sig_right = cgx + 1 < cc.width_in_groups && sig_raster[sub_set_pos + 1];
                let sig_lower =
                    cgy + 1 < cc.height_in_groups && sig_raster[sub_set_pos + cc.width_in_groups];
                let gctx = (sig_right | sig_lower) as usize;
                if dec.decode_bin(if luma {
                    &mut ctx.sig_group[gctx]
                } else {
                    &mut ctx.sig_group_c[gctx]
                }) == 1
                {
                    significant = true;
                } else {
                    continue;
                }
            }
            if significant {
                sig_raster[sub_set_pos] = true;
            }

            let first_sig_pos = if is_last { scan_pos_last } else { max_sub };
            let infer_sig_pos: i32 = if first_sig_pos != scan_pos_last {
                if is_not_first { min_sub as i32 } else { -1 }
            } else {
                first_sig_pos as i32
            };

            let mut par = [0i32; 16];
            let mut gt2 = [false; 16];
            let mut off_arr = [0usize; 16];
            let mut order: Vec<usize> = Vec::new(); // nonzero scan positions, processing order
            let mut num_nonzero = 0i32;
            let mut rem_reg_bins = reg_bin_limit;

            let mut first_pos_2nd = min_sub as i32 - 1;
            let mut next_sig_pos = first_sig_pos as i32;
            while next_sig_pos >= min_sub as i32 && rem_reg_bins >= 4 {
                let sp = next_sig_pos as usize;
                let (x, y) = cc.scan[sp];
                let sig = if num_nonzero > 0 || next_sig_pos != infer_sig_pos {
                    let (sctx, diag, sum1) = sig_ctx(&coeff, x, y, w, h, luma);
                    tmpl_diag = diag;
                    tmpl_sum1 = sum1;
                    let b = dec.decode_bin(ctx.sig_model(luma, dq_state, sctx));
                    rem_reg_bins -= 1;
                    b
                } else if next_sig_pos != scan_pos_last as i32 {
                    let (_, diag, sum1) = sig_ctx(&coeff, x, y, w, h, luma);
                    tmpl_diag = diag;
                    tmpl_sum1 = sum1;
                    1 // inferred significant
                } else {
                    1 // inferred significant (scanPosLast)
                };
                if sig != 0 {
                    let off = ctx_offset_abs(tmpl_diag, tmpl_sum1, luma);
                    off_arr[sp - min_sub] = off;
                    num_nonzero += 1;
                    order.push(sp);
                    let gt1 = dec.decode_bin(if luma {
                        &mut ctx.gt1_flag[off]
                    } else {
                        &mut ctx.gt1_flag_c[off]
                    }) == 1;
                    rem_reg_bins -= 1;
                    let mut abs_level = 1i32;
                    if gt1 {
                        let p = dec.decode_bin(if luma {
                            &mut ctx.par_flag[off]
                        } else {
                            &mut ctx.par_flag_c[off]
                        }) as i32;
                        rem_reg_bins -= 1;
                        let g2 = dec.decode_bin(if luma {
                            &mut ctx.gt2_flag[off]
                        } else {
                            &mut ctx.gt2_flag_c[off]
                        }) == 1;
                        rem_reg_bins -= 1;
                        par[sp - min_sub] = p;
                        gt2[sp - min_sub] = g2;
                        abs_level = if g2 { 4 + p } else { 2 + p };
                        if g2 {
                            first_pos_2nd = first_pos_2nd.max(next_sig_pos);
                        }
                    }
                    coeff[y * w + x] = abs_level; // provisional magnitude
                }
                if dep_quant {
                    dq_state = crate::depquant::next_state(dq_state, coeff[y * w + x]);
                }
                next_sig_pos -= 1;
            }
            let min_pos_2nd = next_sig_pos;
            reg_bin_limit = rem_reg_bins;

            // 2nd pass remainders.
            let mut scan_pos = first_pos_2nd;
            while scan_pos > min_pos_2nd {
                let sp = scan_pos as usize;
                if gt2[sp - min_sub] {
                    let (x, y) = cc.scan[sp];
                    let rice = derive_rice(&coeff, x, y, w, h, 4);
                    let rem = decode_rem_abs_ep(dec, rice);
                    coeff[y * w + x] = 4 + 2 * rem as i32 + par[sp - min_sub];
                }
                scan_pos -= 1;
            }

            // bypass pass.
            let mut scan_pos = min_pos_2nd;
            while scan_pos >= min_sub as i32 {
                let sp = scan_pos as usize;
                let (x, y) = cc.scan[sp];
                let rice = derive_rice(&coeff, x, y, w, h, 0);
                let pos0 = (if dq_state < 2 { 1u32 } else { 2 }) << rice;
                let rem = decode_rem_abs_ep(dec, rice);
                let abs_level = if rem < pos0 {
                    rem + 1
                } else if rem == pos0 {
                    0
                } else {
                    rem
                };
                if abs_level != 0 {
                    coeff[y * w + x] = abs_level as i32;
                    num_nonzero += 1;
                    order.push(sp);
                }
                if dep_quant {
                    dq_state = crate::depquant::next_state(dq_state, coeff[y * w + x]);
                }
                scan_pos -= 1;
            }

            // signs: numNonzero bits, MSB = first-processed (highest scan) nonzero.
            let signs = dec_bits(dec, num_nonzero as u32);
            let k = order.len() as u32;
            for (i, &sp) in order.iter().enumerate() {
                let bit = (signs >> (k - 1 - i as u32)) & 1;
                if bit == 1 {
                    let (x, y) = cc.scan[sp];
                    coeff[y * w + x] = -coeff[y * w + x];
                }
            }
        }
        coeff
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use crate::cabac::engine::CabacDecoder;

    fn round_trip(coeff: &[i32], w: usize, h: usize, qp: u8) -> Vec<i32> {
        let mut enc = CabacEncoder::new();
        let mut ectx = Contexts::new_intra(qp);
        encode_residual(&mut enc, &mut ectx, coeff, w, h, Component::Luma, false);
        enc.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = CabacDecoder::new(&bytes);
        let mut dctx = Contexts::new_intra(qp);
        let got = decode_residual(&mut dec, &mut dctx, w, h, Component::Luma, false);
        assert_eq!(dec.decode_terminate(), 1, "terminate mismatch");
        got
    }

    #[test]
    fn dq_chain_round_trips() {
        // Validates the whole dependent-quant chain end-to-end without VTM:
        // trellis levels -> encode_residual(dq) -> decode_residual(dq) must
        // recover the identical levels, proving the state threading in residual
        // coding is self-consistent (encoder and decoder walk the same state
        // machine). The DQ dequant is also exercised on the decoded levels.
        use crate::transform::{dequantize_dq_wh, dq_trellis_wh};
        for &(w, h) in &[(4usize, 4usize), (8, 8), (16, 16), (4, 16)] {
            for &qp in &[22u8, 32, 42] {
                let mut coeff = vec![0i32; w * h];
                let mut s = 0x1234_5678_9abc_def0u64 ^ (w as u64) ^ ((qp as u64) << 8);
                for c in coeff.iter_mut() {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                    *c = ((s >> 40) as i32 % 600) - 300;
                }
                let lambda = 0.57 * 2f64.powf((qp as f64 - 12.0) / 3.0);
                let levels = dq_trellis_wh(&coeff, w, h, qp, 8, lambda);
                if levels.iter().all(|&l| l == 0) {
                    continue;
                }
                let mut enc = CabacEncoder::new();
                let mut ectx = Contexts::new_intra(qp);
                encode_residual(&mut enc, &mut ectx, &levels, w, h, Component::Luma, true);
                enc.encode_terminate(1);
                let bytes = enc.finish();
                let mut dec = CabacDecoder::new(&bytes);
                let mut dctx = Contexts::new_intra(qp);
                let got = decode_residual(&mut dec, &mut dctx, w, h, Component::Luma, true);
                assert_eq!(dec.decode_terminate(), 1, "terminate w{w}h{h}qp{qp}");
                assert_eq!(got, levels, "DQ level round-trip mismatch w{w}h{h}qp{qp}");
                let scan = scan_coords(w, h);
                let _ = dequantize_dq_wh(&got, scan, w, h, qp, 8);
            }
        }
    }

    #[test]
    fn diag_scan_4x4_is_canonical() {
        let s = diag_scan(4, 4);
        assert_eq!(s[0], (0, 0));
        assert_eq!(s[1], (0, 1)); // (x=0,y=1)
        assert_eq!(s[2], (1, 0));
        assert_eq!(s.len(), 16);
        // every position visited once
        let mut seen = vec![false; 16];
        for &(x, y) in &s {
            seen[y * 4 + x] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn single_dc_round_trips() {
        for &n in &[4usize, 8, 16, 32] {
            let mut c = vec![0i32; n * n];
            c[0] = 7;
            assert_eq!(round_trip(&c, n, n, 32), c, "n={n}");
        }
    }

    #[test]
    fn small_blocks_round_trip() {
        for &n in &[4usize, 8] {
            // a handful of nonzeros including a negative and a >3 level
            let mut c = vec![0i32; n * n];
            c[0] = 5;
            c[1] = -2;
            c[n] = 1;
            c[n + 1] = -9;
            c[2] = 3;
            assert_eq!(round_trip(&c, n, n, 30), c, "n={n}");
        }
    }

    #[test]
    fn rectangular_blocks_round_trip() {
        // Chroma transform blocks under 4:2:2 are half-width: (N/2)×N. The
        // residual coder is (w,h)-parameterized; verify it round-trips for the
        // rectangular sizes that 4:2:2 produces (incl. a 64-tall block, where
        // the coded region is clamped to 32 rows by high-frequency zero-out).
        let mut seed = 0x4221_d00du32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed
        };
        for &(w, h) in &[
            (4usize, 8usize),
            (8, 16),
            (16, 32),
            (32, 64),
            (8, 4),
            (16, 8),
        ] {
            let (cw, ch) = (w.min(32), h.min(32)); // coded (non-zeroed) region
            for &qp in &[16u8, 30, 45] {
                let mut c = vec![0i32; w * h];
                let mut any = false;
                for y in 0..ch {
                    for x in 0..cw {
                        if next() % 100 < 30 {
                            let mag = (next() % 80 + 1) as i32;
                            c[y * w + x] = if next() & 1 == 0 { mag } else { -mag };
                            any = true;
                        }
                    }
                }
                if !any {
                    c[0] = 1;
                }
                let got = round_trip(&c, w, h, qp);
                assert_eq!(got, c, "{w}x{h} qp{qp}");
            }
        }
    }

    #[test]
    fn zeroed_64x64_block_round_trips() {
        // A 64×64 transform block only carries coefficients in the top-left
        // 32×32 (VVC high-frequency zero-out). The residual coder must code
        // exactly that region — last-position, CG scan and bin budget clamped
        // to 32 — while indexing the full stride-64 buffer.
        let mut seed = 0x0bad_f00du32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed
        };
        for &qp in &[10u8, 32, 51] {
            for _ in 0..6 {
                let mut c = vec![0i32; 64 * 64];
                let mut any = false;
                for y in 0..32 {
                    for x in 0..32 {
                        if next() % 100 < 25 {
                            let mag = (next() % 90 + 1) as i32;
                            c[y * 64 + x] = if next() & 1 == 0 { mag } else { -mag };
                            any = true;
                        }
                    }
                }
                if !any {
                    c[0] = 1;
                }
                let got = round_trip(&c, 64, 64, qp);
                assert_eq!(got, c, "qp={qp}");
                // And confirm nothing leaked outside the 32×32 region.
                for y in 0..64 {
                    for x in 0..64 {
                        if x >= 32 || y >= 32 {
                            assert_eq!(got[y * 64 + x], 0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn pseudo_random_blocks_round_trip() {
        // Deterministic pseudo-random sparse coefficient blocks across sizes/QPs.
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed
        };
        for &n in &[4usize, 8, 16, 32] {
            for &qp in &[10u8, 32, 51] {
                for _ in 0..8 {
                    let mut c = vec![0i32; n * n];
                    let mut any = false;
                    for v in c.iter_mut() {
                        // ~35% nonzero, magnitudes up to ~120 (well below the Rice cap)
                        if next() % 100 < 35 {
                            let mag = (next() % 120 + 1) as i32;
                            *v = if next() & 1 == 0 { mag } else { -mag };
                            any = true;
                        }
                    }
                    if !any {
                        c[0] = 1;
                    }
                    let got = round_trip(&c, n, n, qp);
                    assert_eq!(got, c, "n={n} qp={qp}");
                }
            }
        }
    }

    #[test]
    fn dense_high_magnitude_block_round_trips() {
        // Stresses the context-bin budget (forcing the bypass fallback) and the
        // Golomb-Rice remainder pass.
        let n = 8;
        let mut c = vec![0i32; n * n];
        for (i, v) in c.iter_mut().enumerate() {
            *v = (((i * 37) % 90) as i32) - 45;
        }
        c[0] = 100;
        assert_eq!(round_trip(&c, n, n, 37), c);
    }
}
