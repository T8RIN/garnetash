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

//! Intra prediction sample generation: reference-sample derivation with
//! substitution, and planar / DC / angular predictors (H.266 §8.4.5).
//!
//! Ported from VTM `xPredIntraPlanar` / `xPredIntraDc` / `xPredIntraAng`. The
//! angular path uses the VVC angle and inverse-angle tables, builds the extended
//! main/side reference arrays (including the negative-angle left extension via
//! `absInvAngle`), copies integer slopes directly, and interpolates fractional
//! positions with the 4-tap cubic (DCT-IF chroma) or smoothing filter selected
//! exactly as VTM does. Square blocks only, so no wide-angle remapping applies.
//!
//! Deferred for a later increment (documented simplifications, so this is not yet
//! a fully conformant predictor): position-dependent prediction combination
//! (PDPC) and the reference-sample smoothing ([1 2 1]/4) filter. The directional
//! core below is otherwise spec-faithful.
#![allow(dead_code)]

use crate::intra::{DC_IDX, PLANAR_IDX, VDIA_IDX};

const VER_IDX: i32 = 50;
const HOR_IDX: i32 = 18;
const DIA_IDX: i32 = 34;

/// Tangent of the prediction angle in 1/32-sample units, indexed by the absolute
/// angular mode offset (VTM `angTable`).
#[rustfmt::skip]
static ANG_TABLE: [i32; 32] = [
    0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 29,
    32, 35, 39, 45, 51, 57, 64, 73, 86, 102, 128, 171, 256, 341, 512, 1024,
];

/// Inverse angle `(512*32)/angle`, indexed by absolute angular mode offset
/// (VTM `invAngTable`), used to project the side reference onto the main array.
#[rustfmt::skip]
static INV_ANG_TABLE: [i32; 32] = [
    0, 16384, 8192, 5461, 4096, 2731, 2048, 1638, 1365, 1170, 1024, 910, 819, 712, 630, 565,
    512, 468, 420, 364, 321, 287, 256, 224, 191, 161, 128, 96, 64, 48, 32, 16,
];

/// 4-tap cubic interpolation filter (DCT-IF chroma table), indexed by the 1/32
/// fractional position then tap.
#[rustfmt::skip]
static CUBIC_FILTER: [[i32; 4]; 32] = [
    [0,64,0,0],[-1,63,2,0],[-2,62,4,0],[-2,60,7,-1],[-2,58,10,-2],[-3,57,12,-2],
    [-4,56,14,-2],[-4,55,15,-2],[-4,54,16,-2],[-5,53,18,-2],[-6,52,20,-2],[-6,49,24,-3],
    [-6,46,28,-4],[-5,44,29,-4],[-4,42,30,-4],[-4,39,33,-4],[-4,36,36,-4],[-4,33,39,-4],
    [-4,30,42,-4],[-4,29,44,-5],[-4,28,46,-6],[-3,24,49,-6],[-2,20,52,-6],[-2,18,53,-5],
    [-2,16,54,-4],[-2,15,55,-4],[-2,14,56,-4],[-2,12,57,-3],[-2,10,58,-2],[-1,7,60,-2],
    [0,4,62,-2],[0,2,63,-1],
];

/// Per-size threshold (VTM `m_aucIntraFilter`) selecting reference vs
/// interpolation filtering; indexed by `(log2W + log2H) >> 1`.
static INTRA_FILTER: [i32; 8] = [24, 24, 24, 14, 2, 0, 0, 0];

#[inline]
fn integer_slope(abs_ang: i32) -> bool {
    abs_ang & 0x1F == 0
}

#[inline]
fn clip_pel(v: i32, bd: u8) -> i32 {
    v.clamp(0, (1 << bd) - 1)
}

/// Reference samples for one block: `top[0] == left[0]` is the top-left corner;
/// `top[1..]` run rightward along the row above, `left[1..]` downward along the
/// column to the left. Each side is `2*size` samples long (already substituted).
pub(crate) struct RefSamples {
    pub(crate) top: Vec<i32>,
    pub(crate) left: Vec<i32>,
}

impl RefSamples {
    /// Build references from optional neighbour pixel rows, applying H.266
    /// reference-sample substitution: unavailable samples are filled by
    /// propagating the nearest available one (scanning from the bottom-left up
    /// the left column, across the corner, then along the top row); if none are
    /// available, all are set to the mid value `1 << (bitDepth-1)`.
    ///
    /// `corner` is the top-left sample; `above`/`left_px` provide up to `2*w` /
    /// `2*h` samples (rightward / downward). `None` marks an unavailable sample.
    pub(crate) fn build(
        w: usize,
        h: usize,
        corner: Option<i32>,
        above: &[Option<i32>],
        left_px: &[Option<i32>],
        bit_depth: u8,
    ) -> RefSamples {
        let n_top = 2 * w;
        let n_left = 2 * h;
        // Unified scan order: left column bottom->top, corner, top row left->right.
        let mut seq: Vec<Option<i32>> = Vec::with_capacity(n_left + 1 + n_top);
        for i in (0..n_left).rev() {
            seq.push(left_px.get(i).copied().flatten());
        }
        seq.push(corner);
        for i in 0..n_top {
            seq.push(above.get(i).copied().flatten());
        }
        substitute(&mut seq, bit_depth);
        // Split back out.
        let mut left = vec![0i32; n_left + 1];
        let mut top = vec![0i32; n_top + 1];
        let corner_val = seq[n_left].unwrap();
        left[0] = corner_val;
        top[0] = corner_val;
        for i in 0..n_left {
            left[i + 1] = seq[n_left - 1 - i].unwrap();
        }
        for i in 0..n_top {
            top[i + 1] = seq[n_left + 1 + i].unwrap();
        }
        RefSamples { top, left }
    }
}

/// In-place reference-sample substitution over the unified scan sequence.
fn substitute(seq: &mut [Option<i32>], bit_depth: u8) {
    let n = seq.len();
    let first = seq.iter().position(|s| s.is_some());
    let Some(first) = first else {
        let mid = 1i32 << (bit_depth - 1);
        for s in seq.iter_mut() {
            *s = Some(mid);
        }
        return;
    };
    // Fill leading unavailable with the first available.
    let fv = seq[first].unwrap();
    for s in seq.iter_mut().take(first) {
        *s = Some(fv);
    }
    // Forward-propagate the last available value across gaps.
    let mut last = fv;
    for s in seq.iter_mut().take(n).skip(first) {
        match *s {
            Some(v) => last = v,
            None => *s = Some(last),
        }
    }
}

/// Predict an `w*h` block (row-major) for the given luma intra `mode`.
/// Derived per-block prediction parameters (H.266 `initPredIntraParams`),
/// for square single-reference-line luma blocks.
struct IntraParams {
    apply_pdpc: bool,
    ref_filter: bool,
    interp: bool,
    angular_scale: i32,
}

/// VVC wide-angle intra mode remapping (VTM `getWideAngle`). For a non-square
/// block, angular modes pointing past the block's long side are remapped to
/// wide angles (mode ± 65) so the projected reference stays available. Square
/// blocks and the non-angular modes (planar/DC) are returned unchanged, so this
/// is a no-op for every square block the encoder produces.
pub(crate) fn wide_angle(w: usize, h: usize, mode: i32) -> i32 {
    if mode > DC_IDX as i32 && mode <= VDIA_IDX as i32 {
        const MODE_SHIFT: [i32; 6] = [0, 6, 10, 12, 14, 15];
        let log2w = w.trailing_zeros() as i32;
        let log2h = h.trailing_zeros() as i32;
        let delta = (log2w - log2h).unsigned_abs() as usize;
        if w > h && mode < 2 + MODE_SHIFT[delta] {
            return mode + (VDIA_IDX as i32 - 1);
        } else if h > w && mode > VDIA_IDX as i32 - MODE_SHIFT[delta] {
            return mode - (VDIA_IDX as i32 - 1);
        }
    }
    mode
}

/// VVC wide-angle remap for **LFNST set selection** (VTM `PU::getWideAngle`).
/// Identical to the prediction remap [`wide_angle`] except the tall-block
/// (`h > w`) branch subtracts `VDIA_IDX + 1` (67) instead of `VDIA_IDX - 1`
/// (65) — VTM keeps these two as deliberately distinct functions.
pub(crate) fn lfnst_wide_angle(w: usize, h: usize, mode: i32) -> i32 {
    if mode > DC_IDX as i32 && mode <= VDIA_IDX as i32 {
        const MODE_SHIFT: [i32; 6] = [0, 6, 10, 12, 14, 15];
        let log2w = w.trailing_zeros() as i32;
        let log2h = h.trailing_zeros() as i32;
        let delta = (log2w - log2h).unsigned_abs() as usize;
        if w > h && mode < 2 + MODE_SHIFT[delta] {
            return mode + (VDIA_IDX as i32 - 1);
        } else if h > w && mode > VDIA_IDX as i32 - MODE_SHIFT[delta] {
            return mode - (VDIA_IDX as i32 + 1);
        }
    }
    mode
}

fn intra_params(orig_mode: i32, eff_mode: i32, w: usize, h: usize, is_luma: bool) -> IntraParams {
    let log2w = w.trailing_zeros() as i32;
    let log2h = h.trailing_zeros() as i32;
    let mut p = IntraParams {
        apply_pdpc: w >= 4 && h >= 4, // multiRefIdx == 0
        ref_filter: false,
        interp: false,
        angular_scale: 0,
    };
    // Planar/DC vs angular is decided by the *original* signalled mode. The
    // effective (wide-angle remapped) mode can land on the value 0 or 1 for a
    // non-square block, but that is still an angular direction — not planar/DC.
    if orig_mode == PLANAR_IDX as i32 {
        p.ref_filter = is_luma && w * h > 32;
    } else if orig_mode == DC_IDX as i32 {
        // DC: no reference filter; PDPC applies.
    } else {
        let is_mode_ver = eff_mode >= DIA_IDX;
        let ang_mode = if is_mode_ver {
            eff_mode - VER_IDX
        } else {
            -(eff_mode - HOR_IDX)
        };
        let abs_ang = ANG_TABLE[ang_mode.unsigned_abs() as usize];
        let abs_inv_angle = INV_ANG_TABLE[ang_mode.unsigned_abs() as usize];
        if ang_mode < 0 {
            p.apply_pdpc = false;
        } else if ang_mode > 0 {
            let side_size = if is_mode_ver { h } else { w } as i32;
            let scale = 2.min(
                (side_size.ilog2() as i32) - (((3 * abs_inv_angle - 2) as u32).ilog2() as i32 - 8),
            );
            p.angular_scale = scale;
            p.apply_pdpc &= scale >= 0;
        }
        if is_luma {
            let diff = (eff_mode - HOR_IDX).abs().min((eff_mode - VER_IDX).abs());
            let log2_size = ((log2w + log2h) >> 1) as usize;
            if diff > INTRA_FILTER[log2_size] {
                let is_ref = integer_slope(abs_ang);
                p.ref_filter = is_ref;
                p.interp = !is_ref;
            }
        }
    }
    p
}

/// Apply the `[1 2 1]/4` reference-sample smoothing filter (H.266
/// `xFilterReferenceSamples`): a shared filtered corner, smoothed interiors, and
/// unchanged endpoints.
pub(crate) fn filter_references(refs: &RefSamples, w: usize, h: usize) -> RefSamples {
    let n_top = 2 * w;
    let n_left = 2 * h;
    let corner = (2 * refs.top[0] + refs.top[1] + refs.left[1] + 2) >> 2;
    let mut top = refs.top.clone();
    let mut left = refs.left.clone();
    top[0] = corner;
    left[0] = corner;
    #[allow(clippy::needless_range_loop)]
    for i in 1..n_top {
        top[i] = (refs.top[i - 1] + 2 * refs.top[i] + refs.top[i + 1] + 2) >> 2;
    }
    top[n_top] = refs.top[n_top];
    #[allow(clippy::needless_range_loop)]
    for i in 1..n_left {
        left[i] = (refs.left[i - 1] + 2 * refs.left[i] + refs.left[i + 1] + 2) >> 2;
    }
    left[n_left] = refs.left[n_left];
    RefSamples { top, left }
}

/// Position-dependent prediction combination for planar / DC (H.266 8.4.5.2.14).
fn pdpc_planar_dc(pred: &mut [i32], w: usize, h: usize, refs: &RefSamples) {
    let log2w = w.trailing_zeros() as i32;
    let log2h = h.trailing_zeros() as i32;
    let scale = (log2w + log2h - 2) >> 2;
    // Per-column left weight depends only on x; precompute once instead of
    // recomputing inside the y·x loop. Slicing the top row up front lets the
    // inner loop zip without bounds checks.
    let mut wl_col = [0i32; 64];
    for (x, wl) in wl_col[..w].iter_mut().enumerate() {
        *wl = 32 >> 31.min((x << 1) as i32 >> scale);
    }
    let top = &refs.top[1..1 + w];
    for (y, row) in pred.chunks_exact_mut(w).enumerate().take(h) {
        let wt = 32 >> 31.min((y << 1) as i32 >> scale);
        let left = refs.left[y + 1];
        for ((val, &wl), &top_x) in row.iter_mut().zip(&wl_col[..w]).zip(top) {
            let v = *val;
            *val = v + ((wl * (left - v) + wt * (top_x - v) + 32) >> 6);
        }
    }
}

/// Predict an `w*h` block (row-major) for the given luma intra `mode`, including
/// reference-sample smoothing and PDPC as selected by the prediction parameters.
pub(crate) fn predict(
    mode: u8,
    w: usize,
    h: usize,
    refs: &RefSamples,
    bit_depth: u8,
    is_luma: bool,
) -> Vec<i32> {
    let mut out = vec![0i32; w * h];
    let mut scratch = Vec::new();
    predict_into(
        &mut out,
        &mut scratch,
        None,
        mode,
        w,
        h,
        refs,
        bit_depth,
        is_luma,
    );
    out
}

/// Predict into a caller-supplied `out` buffer (length `w*h`), reusing `scratch`
/// for the angular transpose temporary. Lets hot loops (the 67-mode SATD scan)
/// avoid one heap allocation per mode. Produces output identical to [`predict`].
///
/// `filtered` may carry the smoothed references for this block, precomputed
/// once by the caller; they are mode-independent, so a hot loop can build them
/// a single time instead of re-filtering inside every mode that needs them.
/// Pass `None` to filter lazily when required.
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_into(
    out: &mut [i32],
    scratch: &mut Vec<i32>,
    filtered: Option<&RefSamples>,
    mode: u8,
    w: usize,
    h: usize,
    refs: &RefSamples,
    bit_depth: u8,
    is_luma: bool,
) {
    let eff_mode = wide_angle(w, h, mode as i32);
    let p = intra_params(mode as i32, eff_mode, w, h, is_luma);
    let local_filtered;
    let r: &RefSamples = if p.ref_filter {
        match filtered {
            Some(f) => f,
            None => {
                local_filtered = filter_references(refs, w, h);
                &local_filtered
            }
        }
    } else {
        refs
    };
    match mode as i32 {
        x if x == PLANAR_IDX as i32 => {
            predict_planar_into(out, w, h, r);
            if p.apply_pdpc {
                pdpc_planar_dc(out, w, h, r);
            }
        }
        x if x == DC_IDX as i32 => {
            predict_dc_into(out, w, h, r);
            if p.apply_pdpc {
                pdpc_planar_dc(out, w, h, r);
            }
        }
        _ => predict_angular_into(out, scratch, eff_mode, w, h, r, bit_depth, &p, is_luma),
    }
}

/// Planar prediction (bilinear blend of the four extended edge samples).
fn predict_planar_into(out: &mut [i32], w: usize, h: usize, refs: &RefSamples) {
    let log2w = w.trailing_zeros();
    let log2h = h.trailing_zeros();
    let offset = 1i32 << (log2w + log2h);
    let final_shift = 1 + log2w + log2h;

    // w, h are powers of two ≤ 64, so fixed stack scratch avoids the six per-call
    // heap allocations this hot, common mode would otherwise make.
    let mut top_row = [0i32; 65];
    let mut left_col = [0i32; 65];
    top_row[..(w + 1)].copy_from_slice(&refs.top[1..(w + 1 + 1)]);
    left_col[..(h + 1)].copy_from_slice(&refs.left[1..(h + 1 + 1)]);
    let bottom_left = left_col[h];
    let top_right = top_row[w];

    let mut bottom_row = [0i32; 64];
    for k in 0..w {
        bottom_row[k] = bottom_left - top_row[k];
        top_row[k] <<= log2h;
    }
    let mut right_col = [0i32; 64];
    let mut left_shifted = [0i32; 64];
    for k in 0..h {
        right_col[k] = top_right - left_col[k];
        left_shifted[k] = left_col[k] << log2w;
    }

    let mut top_acc = [0i32; 64];
    top_acc[..w].copy_from_slice(&top_row[..w]);
    for (y, row) in out[..w * h].chunks_exact_mut(w).enumerate() {
        let mut hor = left_shifted[y];
        let rc = right_col[y];
        for ((o, ta), &br) in row
            .iter_mut()
            .zip(top_acc[..w].iter_mut())
            .zip(&bottom_row[..w])
        {
            hor += rc;
            *ta += br;
            *o = ((hor << log2h) + (*ta << log2w) + offset) >> final_shift;
        }
    }
}

/// DC prediction (mean of the adjacent top and left reference samples).
fn predict_dc_into(out: &mut [i32], w: usize, h: usize, refs: &RefSamples) {
    let denom = if w == h { w << 1 } else { w.max(h) };
    let div_shift = (denom as u32).trailing_zeros();
    let div_offset = (denom >> 1) as i32;
    let mut sum = 0i32;
    if w >= h {
        sum += refs.top[1..1 + w].iter().sum::<i32>();
    }
    if w <= h {
        sum += refs.left[1..1 + h].iter().sum::<i32>();
    }
    let dc = (sum + div_offset) >> div_shift;
    out[..w * h].fill(dc);
}

/// Angular prediction for a directional mode (2..66, excluding planar/DC).
#[allow(clippy::too_many_arguments)]
fn predict_angular_into(
    out: &mut [i32],
    scratch: &mut Vec<i32>,
    mode: i32,
    w: usize,
    h: usize,
    refs: &RefSamples,
    bit_depth: u8,
    params: &IntraParams,
    is_luma: bool,
) {
    let is_mode_ver = mode >= DIA_IDX;
    let ang_mode = if is_mode_ver {
        mode - VER_IDX
    } else {
        -(mode - HOR_IDX)
    };
    let abs_ang_mode = ang_mode.unsigned_abs() as usize;
    let sign = if ang_mode < 0 { -1 } else { 1 };
    let abs_ang = ANG_TABLE[abs_ang_mode];
    let intra_pred_angle = sign * abs_ang;
    let abs_inv_angle = INV_ANG_TABLE[abs_ang_mode];

    let use_cubic = !params.interp;

    // Predict on possibly-swapped dimensions; transpose afterwards for hor modes.
    let (pw, ph) = if is_mode_ver { (w, h) } else { (h, w) };

    // Main/side reference arrays as plain slices, both indexed [0] = corner.
    let (main_src, side_src) = if is_mode_ver {
        (&refs.top, &refs.left)
    } else {
        (&refs.left, &refs.top)
    };

    // Build an extended main reference with a base offset so negative indices
    // (negative-angle left extension) are representable.
    let side_size = ph as i32;
    let base; // index in `main` corresponding to refMain[0]
    // The main reference is at most 2·pw+3 (≥0 angle) or ph+pw+2 (<0 angle)
    // entries, and pw,ph ≤ 64, so it fits a fixed stack buffer — avoiding a heap
    // allocation on every one of the 67 per-block mode evaluations.
    let mut main = [0i32; 2 * 64 + 4];
    if intra_pred_angle < 0 {
        base = side_size as usize;
        main[base..(pw + 1 + base + 1)].copy_from_slice(&main_src[..(pw + 1 + 1)]);
        // Extend to the left using the side reference.
        for k in (-side_size)..0 {
            let idx = ((-k * abs_inv_angle + 256) >> 9).min(side_size) as usize;
            main[(base as i32 + k) as usize] = side_src[idx];
        }
    } else {
        base = 0;
        let ref_len = 2 * pw; // m_topRefLength for square main side
        let main_len = ref_len + 3;
        let avail = main_src.len();
        main[..main_len.min(avail)].copy_from_slice(&main_src[..main_len.min(avail)]);
        // Replicate the tail (right extension).
        let last = main_src[avail - 1];
        main[avail..main_len].fill(last);
    }
    let side = side_src; // refSide, [0]=corner
    let apply_pdpc = params.apply_pdpc;
    let log2pw = pw.trailing_zeros() as i32;
    let log2ph = ph.trailing_zeros() as i32;

    // Vertical modes write straight into `out`; horizontal modes fill `scratch`
    // (pw×ph) and transpose into `out` afterwards. Both avoid a fresh alloc.
    if !is_mode_ver {
        scratch.clear();
        scratch.resize(pw * ph, 0);
    }
    let buf: &mut [i32] = if is_mode_ver {
        &mut out[..]
    } else {
        &mut scratch[..]
    };
    if intra_pred_angle == 0 {
        let scale = (log2pw + log2ph - 2) >> 2;
        let top_left = main[base];
        for y in 0..ph {
            let row = &mut buf[y * pw..y * pw + pw];
            for (o, &m) in row.iter_mut().zip(&main[base + 1..base + 1 + pw]) {
                *o = m;
            }
            if apply_pdpc {
                let left = side[1 + y];
                let lim = ((3 << scale) as usize).min(pw);
                for (x, o) in row[..lim].iter_mut().enumerate() {
                    let wl = 32 >> ((2 * x as i32) >> scale);
                    let val = *o;
                    *o = clip_pel(val + ((wl * (left - top_left) + 32) >> 6), bit_depth);
                }
            }
        }
    } else {
        for y in 0..ph {
            let delta_pos = intra_pred_angle * (y as i32 + 1);
            let delta_int = delta_pos >> 5;
            let delta_fract = delta_pos & 31;
            let row = &mut buf[y * pw..y * pw + pw];
            if !integer_slope(abs_ang) {
                if is_luma {
                    let smooth = [
                        16 - (delta_fract >> 1),
                        32 - (delta_fract >> 1),
                        16 + (delta_fract >> 1),
                        delta_fract >> 1,
                    ];
                    let f = if use_cubic {
                        &CUBIC_FILTER[delta_fract as usize]
                    } else {
                        &smooth
                    };
                    // Slice the exact reference window once (single bounds check),
                    // then iterate 4-wide windows the compiler proves in-bounds.
                    let start = (base as i32 + delta_int) as usize;
                    let win = &main[start..start + pw + 3];
                    for (o, p) in row.iter_mut().zip(win.windows(4)) {
                        let val = (f[0] * p[0] + f[1] * p[1] + f[2] * p[2] + f[3] * p[3] + 32) >> 6;
                        *o = clip_pel(val, bit_depth);
                    }
                } else {
                    // Chroma: 2-tap linear interpolation over consecutive samples.
                    let start = (base as i32 + delta_int + 1) as usize;
                    let win = &main[start..start + pw + 1];
                    for (o, p) in row.iter_mut().zip(win.windows(2)) {
                        *o = p[0] + ((delta_fract * (p[1] - p[0]) + 16) >> 5);
                    }
                }
            } else {
                let start = (base as i32 + delta_int + 1) as usize;
                for (o, &m) in row.iter_mut().zip(&main[start..start + pw]) {
                    *o = m;
                }
            }
            if apply_pdpc {
                // intra_pred_angle > 0 here (negative angles disable PDPC).
                let scale = params.angular_scale;
                let mut inv_angle_sum = 256;
                let lim = ((3 << scale) as usize).min(pw);
                for (x, o) in row[..lim].iter_mut().enumerate() {
                    inv_angle_sum += abs_inv_angle;
                    let wl = 32 >> ((2 * x as i32) >> scale);
                    let idx = (y as i32 + (inv_angle_sum >> 9) + 1) as usize;
                    let left = side[idx.min(side.len() - 1)];
                    let val = *o;
                    *o = val + ((wl * (left - val) + 32) >> 6);
                }
            }
        }
    }

    if !is_mode_ver {
        // Transpose the pw×ph scratch buffer into the w×h output.
        for (y, srow) in scratch[..pw * ph].chunks_exact(pw).enumerate() {
            for (x, &s) in srow.iter().enumerate() {
                out[x * w + y] = s;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refs_from(w: usize, h: usize, top_v: i32, left_v: i32, corner: i32) -> RefSamples {
        let above = vec![Some(top_v); 2 * w];
        let left = vec![Some(left_v); 2 * h];
        RefSamples::build(w, h, Some(corner), &above, &left, 8)
    }

    #[test]
    fn dc_is_mean_of_references() {
        let refs = refs_from(8, 8, 100, 100, 100);
        let p = predict(DC_IDX, 8, 8, &refs, 8, true);
        assert!(p.iter().all(|&v| v == 100));
        // Asymmetric: top=80, left=120 -> mean 100.
        let refs2 = refs_from(8, 8, 80, 120, 100);
        let p2 = predict(DC_IDX, 8, 8, &refs2, 8, true);
        // PDPC reshapes the block edges; the DC base shows through where the
        // PDPC weights have decayed to zero (bottom-right corner).
        assert_eq!(p2[7 * 8 + 7], 100, "DC base at PDPC-free corner");
    }

    #[test]
    fn planar_constant_references_are_constant() {
        let refs = refs_from(16, 16, 50, 50, 50);
        let p = predict(PLANAR_IDX, 16, 16, &refs, 8, true);
        assert!(p.iter().all(|&v| v == 50), "min {:?}", p.iter().min());
    }

    #[test]
    fn vertical_mode_copies_top_row() {
        // Ramp along the top reference; pure-vertical (mode 50) copies it down.
        let above: Vec<Option<i32>> = (0..16).map(|i| Some(10 + i as i32 * 3)).collect();
        let left = vec![Some(0); 16];
        let refs = RefSamples::build(8, 8, Some(10), &above, &left, 8);
        let p = predict(50, 8, 8, &refs, 8, true);
        // Pure-vertical copies the top row; PDPC only touches the left columns,
        // so the right columns (x >= 6) match the reference exactly.
        for y in 0..8 {
            for x in 6..8 {
                assert_eq!(p[y * 8 + x], refs.top[x + 1], "({x},{y})");
            }
        }
    }

    #[test]
    fn horizontal_mode_copies_left_column() {
        let left: Vec<Option<i32>> = (0..16).map(|i| Some(20 + i as i32 * 2)).collect();
        let above = vec![Some(0); 16];
        let refs = RefSamples::build(8, 8, Some(20), &above, &left, 8);
        let p = predict(18, 8, 8, &refs, 8, true);
        // Pure-horizontal copies the left column; after the transpose PDPC only
        // touches the top rows, so rows y >= 6 match the reference exactly.
        for y in 6..8 {
            let expect = refs.left[y + 1];
            assert!(
                p[y * 8..y * 8 + 8].iter().all(|&v| v == expect),
                "row {y}={expect}"
            );
        }
    }

    #[test]
    fn substitution_all_unavailable_is_midvalue() {
        let above = vec![None; 16];
        let left = vec![None; 16];
        let refs = RefSamples::build(8, 8, None, &above, &left, 8);
        assert!(refs.top.iter().all(|&v| v == 128));
        assert!(refs.left.iter().all(|&v| v == 128));
    }

    #[test]
    fn substitution_propagates_into_gaps() {
        // Only one top sample available; it must fill the whole sequence.
        let mut above = vec![None; 8];
        above[3] = Some(77);
        let left = vec![None; 4];
        let refs = RefSamples::build(4, 4, None, &above, &left, 8);
        assert!(refs.left.iter().all(|&v| v == 77), "left {:?}", refs.left);
        assert_eq!(refs.top[0], 77); // corner filled from the available sample
        assert_eq!(refs.top[4], 77);
    }

    #[test]
    fn output_within_pixel_range() {
        // Every mode must produce in-range samples for arbitrary references.
        let above: Vec<Option<i32>> = (0..32).map(|i| Some((i * 251 % 256) as i32)).collect();
        let left: Vec<Option<i32>> = (0..32).map(|i| Some((i * 113 % 256) as i32)).collect();
        let refs = RefSamples::build(16, 16, Some(128), &above, &left, 8);
        for mode in 0u8..67 {
            let p = predict(mode, 16, 16, &refs, 8, true);
            assert_eq!(p.len(), 256);
            assert!(
                p.iter().all(|&v| (0..=255).contains(&v)),
                "mode {mode} out of range"
            );
        }
    }

    #[test]
    fn reference_filter_is_121_over_4() {
        // Impulse in the top row; the smoothing filter spreads it as [1 2 1]/4.
        let mut above = vec![Some(0i32); 16];
        above[5] = Some(40);
        let left = vec![Some(0i32); 16];
        let refs = RefSamples::build(8, 8, Some(0), &above, &left, 8);
        let f = filter_references(&refs, 8, 8);
        // top index i corresponds to above[i-1]; the impulse at above[5] is top[6].
        assert_eq!(f.top[5], (0 + 2 * 0 + 40 + 2) >> 2); // 10
        assert_eq!(f.top[6], (0 + 2 * 40 + 0 + 2) >> 2); // 20
        assert_eq!(f.top[7], (40 + 2 * 0 + 0 + 2) >> 2); // 10
        assert_eq!(f.top[16], refs.top[16]); // endpoint unchanged
    }

    #[test]
    fn pdpc_blends_dc_edges_but_not_interior() {
        // Strong asymmetry between top and left so PDPC visibly bends the edges.
        let refs = refs_from(8, 8, 200, 0, 100);
        let p = predict(DC_IDX, 8, 8, &refs, 8, true);
        // Interior (bottom-right) keeps the DC base.
        let dc = (8 * 200 + 8 * 0 + 8) >> 4;
        assert_eq!(p[7 * 8 + 7], dc);
        // The top edge is pulled toward the (larger) top reference, the left
        // edge toward the (smaller) left reference: top-row sample exceeds the
        // left-column sample at matching offset.
        assert!(
            p[0 * 8 + 3] > p[3 * 8 + 0],
            "PDPC should bend edges asymmetrically"
        );
        assert!(p.iter().all(|&v| (0..=255).contains(&v)));
    }

    #[test]
    fn angular_diagonal_shifts_reference() {
        // Mode 66 (top-right diagonal, integer slope 32) copies the top
        // reference shifted by one extra sample per row.
        let above: Vec<Option<i32>> = (0..32).map(|i| Some(i as i32)).collect();
        let left = vec![Some(0); 16];
        let refs = RefSamples::build(8, 8, Some(0), &above, &left, 8);
        let p = predict(66, 8, 8, &refs, 8, true);
        // angle=32 -> deltaInt = (32*(y+1))>>5 = y+1; sample = top[x + deltaInt + 1].
        // PDPC blends the left columns toward the side reference; the right
        // columns (x >= 6) keep the pure shifted copy.
        for y in 0..8 {
            for x in 6..8 {
                assert_eq!(p[y * 8 + x], refs.top[x + (y + 1) + 1], "({x},{y})");
            }
        }
    }
}
