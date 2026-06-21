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

/// The three CCLM modes (which neighbour template fits the model).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CclmMode {
    /// LM_CHROMA: top + left templates.
    Lt,
    /// MDLM_L: left (and below-left) template only.
    L,
    /// MDLM_T: top (and above-right) template only.
    T,
}

/// VVC division-significand table (`DivSigTable`, H.266 8.4.5.2.13): 4-bit
/// significands minus 8 (the MSB is implied).
static DIV_SIG_TABLE: [i32; 16] = [0, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 0];

#[inline]
fn floor_log2(x: i32) -> i32 {
    debug_assert!(x > 0);
    31 - (x as u32).leading_zeros() as i32
}

pub(crate) fn cclm_model(
    sel_luma: [i32; 4],
    sel_chroma: [i32; 4],
    have_neighbours: bool,
    bd: u8,
) -> (i32, i32, u32) {
    if !have_neighbours {
        return (0, 1 << (bd - 1), 0);
    }
    let mut min_grp = [0usize, 2];
    let mut max_grp = [1usize, 3];
    if sel_luma[min_grp[0]] > sel_luma[min_grp[1]] {
        min_grp.swap(0, 1);
    }
    if sel_luma[max_grp[0]] > sel_luma[max_grp[1]] {
        max_grp.swap(0, 1);
    }
    if sel_luma[min_grp[0]] > sel_luma[max_grp[1]] {
        std::mem::swap(&mut min_grp, &mut max_grp);
    }
    if sel_luma[min_grp[1]] > sel_luma[max_grp[0]] {
        std::mem::swap(&mut min_grp[1], &mut max_grp[0]);
    }
    let min_luma = (sel_luma[min_grp[0]] + sel_luma[min_grp[1]] + 1) >> 1;
    let min_chroma = (sel_chroma[min_grp[0]] + sel_chroma[min_grp[1]] + 1) >> 1;
    let max_luma = (sel_luma[max_grp[0]] + sel_luma[max_grp[1]] + 1) >> 1;
    let max_chroma = (sel_chroma[max_grp[0]] + sel_chroma[max_grp[1]] + 1) >> 1;

    let diff = max_luma - min_luma;
    if diff <= 0 {
        return (0, min_chroma, 0);
    }
    let diff_c = max_chroma - min_chroma;
    let mut x = floor_log2(diff);
    let norm_diff = ((diff << 4) >> x) & 15;
    let v = DIV_SIG_TABLE[norm_diff as usize] | 8;
    x += (norm_diff != 0) as i32;
    let y = floor_log2(diff_c.abs().max(1)) + 1;
    let add = 1 << y >> 1;
    let mut a = (diff_c * v + add) >> y;
    let mut shift = 3 + x - y;
    if shift < 1 {
        shift = 1;
        a = if a == 0 {
            0
        } else if a < 0 {
            -15
        } else {
            15
        };
    }
    let b = min_chroma - ((a * min_luma) >> shift);
    (a, b, shift as u32)
}

pub(crate) fn cclm_select(
    top_luma: &[i32],
    top_chroma: &[i32],
    left_luma: &[i32],
    left_chroma: &[i32],
    above_avail: bool,
    left_avail: bool,
) -> ([i32; 4], [i32; 4], bool) {
    let mut sel_l = [0i32; 4];
    let mut sel_c = [0i32; 4];
    if !above_avail && !left_avail {
        return (sel_l, sel_c, false);
    }
    let actual_top = if above_avail {
        top_luma.len() as i32
    } else {
        0
    };
    let actual_left = if left_avail {
        left_luma.len() as i32
    } else {
        0
    };
    let above_is4 = if left_avail { 0 } else { 1 };
    let left_is4 = if above_avail { 0 } else { 1 };
    let start_t = actual_top >> (2 + above_is4);
    let step_t = (actual_top >> (1 + above_is4)).max(1);
    let start_l = actual_left >> (2 + left_is4);
    let step_l = (actual_left >> (1 + left_is4)).max(1);

    let mut cnt_t = 0usize;
    if above_avail {
        cnt_t = (actual_top.min((1 + above_is4) << 1)) as usize;
        let mut pos = start_t;
        for c in sel_l.iter_mut().zip(sel_c.iter_mut()).take(cnt_t) {
            *c.0 = top_luma[pos as usize];
            *c.1 = top_chroma[pos as usize];
            pos += step_t;
        }
    }
    let mut cnt_l = 0usize;
    if left_avail {
        cnt_l = (actual_left.min((1 + left_is4) << 1)) as usize;
        let mut pos = start_l;
        for k in 0..cnt_l {
            sel_l[k + cnt_t] = left_luma[pos as usize];
            sel_c[k + cnt_t] = left_chroma[pos as usize];
            pos += step_l;
        }
    }
    if cnt_t + cnt_l == 2 {
        sel_l[3] = sel_l[0];
        sel_c[3] = sel_c[0];
        sel_l[2] = sel_l[1];
        sel_c[2] = sel_c[1];
        sel_l[0] = sel_l[1];
        sel_c[0] = sel_c[1];
        sel_l[1] = sel_l[3];
        sel_c[1] = sel_c[3];
    }
    (sel_l, sel_c, true)
}

pub(crate) fn predict_cclm(luma_ds: &[i32], a: i32, b: i32, shift: u32, max_val: i32) -> Vec<i32> {
    luma_ds
        .iter()
        .map(|&l| (((a * l) >> shift) + b).clamp(0, max_val))
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cclm_luma<F: Fn(isize, isize) -> i32>(
    luma: F,
    lx: usize,
    ly: usize,
    ccw: usize,
    cch: usize,
    sub_w: usize,
    sub_h: usize,
    top_len: usize,
    left_len: usize,
    above_avail: bool,
    left_avail: bool,
    first_row_of_ctu: bool,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let (lx, ly) = (lx as isize, ly as isize);
    let l = |x: isize, y: isize| luma(x, y);
    let is444 = sub_w == 1 && sub_h == 1;
    let is422 = sub_w == 2 && sub_h == 1;

    let mut block = vec![0i32; ccw * cch];
    for j in 0..cch as isize {
        for i in 0..ccw as isize {
            let xx = lx + i * sub_w as isize;
            let yy = ly + j * sub_h as isize;
            let lp = i == 0 && !left_avail;
            let lo = if lp { 0 } else { 1 }; // left tap offset
            let v = if is444 {
                l(xx, yy)
            } else if is422 {
                (2 + l(xx, yy) * 2 + l(xx + 1, yy) + l(xx - lo, yy)) >> 2
            } else {
                (4 + l(xx, yy) * 2
                    + l(xx + 1, yy)
                    + l(xx - lo, yy)
                    + l(xx, yy + 1) * 2
                    + l(xx + 1, yy + 1)
                    + l(xx - lo, yy + 1))
                    >> 3
            };
            block[(j as usize) * ccw + i as usize] = v;
        }
    }

    let mut top = Vec::new();
    if above_avail && top_len > 0 {
        top = vec![0i32; top_len];
        for i in 0..top_len as isize {
            let xx = lx + i * sub_w as isize;
            let lp = i == 0 && !left_avail;
            let lo = if lp { 0 } else { 1 };
            let v = if is444 {
                l(xx, ly - 1)
            } else if first_row_of_ctu {
                // Only the single row immediately above is in this CTU.
                (l(xx, ly - 1) * 2 + l(xx - lo, ly - 1) + l(xx + 1, ly - 1) + 2) >> 2
            } else if is422 {
                (2 + l(xx, ly - 1) * 2 + l(xx - lo, ly - 1) + l(xx + 1, ly - 1)) >> 2
            } else {
                // 4:2:0 6-tap on the two rows above (ly-2, ly-1).
                (4 + l(xx, ly - 2) * 2
                    + l(xx + 1, ly - 2)
                    + l(xx - lo, ly - 2)
                    + l(xx, ly - 1) * 2
                    + l(xx + 1, ly - 1)
                    + l(xx - lo, ly - 1))
                    >> 3
            };
            top[i as usize] = v;
        }
    }

    // ---- left template (one chroma column left of the block) ----
    let mut left = Vec::new();
    if left_avail && left_len > 0 {
        left = vec![0i32; left_len];
        for j in 0..left_len as isize {
            let yy = ly + j * sub_h as isize;
            let v = if is444 {
                l(lx - 1, yy)
            } else if is422 {
                (2 + l(lx - 2, yy) * 2 + l(lx - 1, yy) + l(lx - 3, yy)) >> 2
            } else {
                // 4:2:0 6-tap on the column pair (lx-2 centre, lx-1 / lx-3 sides).
                (4 + l(lx - 2, yy) * 2
                    + l(lx - 1, yy)
                    + l(lx - 3, yy)
                    + l(lx - 2, yy + 1) * 2
                    + l(lx - 1, yy + 1)
                    + l(lx - 3, yy + 1))
                    >> 3
            };
            left[j as usize] = v;
        }
    }

    (block, top, left)
}

/// Full CCLM predictor for one chroma block (LT mode): downsample the
/// co-located reconstructed luma (block + top/left templates), gather the
/// reconstructed chroma neighbour templates, fit a separate model for Cb and Cr,
/// and predict both. `luma`/`cb`/`cr` are clamping accessors over the recon
/// planes at absolute coordinates. Returns `(pred_cb, pred_cr)`, each `ccw·cch`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cclm_predict<L, Cb, Cr>(
    luma: L,
    cb: Cb,
    cr: Cr,
    lx: usize,
    ly: usize,
    cx: usize,
    cy: usize,
    ccw: usize,
    cch: usize,
    sub_w: usize,
    sub_h: usize,
    above_avail: bool,
    left_avail: bool,
    first_row_of_ctu: bool,
    mode: u8,
    avai_ar_units: usize,
    avai_bl_units: usize,
    max_val: i32,
    bd: u8,
) -> (Vec<i32>, Vec<i32>)
where
    L: Fn(isize, isize) -> i32,
    Cb: Fn(isize, isize) -> i32,
    Cr: Fn(isize, isize) -> i32,
{
    // Chroma neighbour "unit" sizes (VTM: baseUnit 4 >> chroma scale).
    let uw = 4 >> if sub_w == 2 { 1 } else { 0 };
    let uh = 4 >> if sub_h == 2 { 1 } else { 0 };
    // Per-mode effective availability + template lengths. MDLM_T uses only the
    // (extended) above template; MDLM_L only the (extended) left template.
    let (eff_above, eff_left, top_len, left_len) = match mode {
        crate::intra::CCLM_T_MODE => {
            let cap = cch / uw; // VTM caps avaiAboveRight at cHeight/unitWidth
            let ar = avai_ar_units.min(cap);
            let tlen = if above_avail { ccw + uw * ar } else { 0 };
            (above_avail, false, tlen, 0)
        }
        crate::intra::CCLM_L_MODE => {
            let cap = ccw / uh;
            let bl = avai_bl_units.min(cap);
            let llen = if left_avail { cch + uh * bl } else { 0 };
            (false, left_avail, 0, llen)
        }
        _ => {
            // CCLM_LT
            let tlen = if above_avail { ccw } else { 0 };
            let llen = if left_avail { cch } else { 0 };
            (above_avail, left_avail, tlen, llen)
        }
    };
    let (block, top_l, left_l) = cclm_luma(
        luma,
        lx,
        ly,
        ccw,
        cch,
        sub_w,
        sub_h,
        top_len,
        left_len,
        above_avail,
        left_avail,
        first_row_of_ctu,
    );
    let top = |c: &dyn Fn(isize, isize) -> i32| -> Vec<i32> {
        (0..top_len)
            .map(|i| c((cx + i) as isize, cy as isize - 1))
            .collect()
    };
    let left = |c: &dyn Fn(isize, isize) -> i32| -> Vec<i32> {
        (0..left_len)
            .map(|j| c(cx as isize - 1, (cy + j) as isize))
            .collect()
    };
    let fit = |top_c: &[i32], left_c: &[i32]| -> Vec<i32> {
        let (sl, sc, have) = cclm_select(&top_l, top_c, &left_l, left_c, eff_above, eff_left);
        let (a, b, shift) = cclm_model(sl, sc, have, bd);
        predict_cclm(&block, a, b, shift, max_val)
    };
    let pred_cb = fit(&top(&cb), &left(&cb));
    let pred_cr = fit(&top(&cr), &left(&cr));
    (pred_cb, pred_cr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_constant_chroma_gives_zero_slope() {
        // All chroma equal -> diffC = 0 -> a = 0, b = that chroma value.
        // diffC == 0 -> a = 0 (shift is unconstrained since a*L = 0), b = chroma.
        let (a, b, _shift) = cclm_model([10, 20, 30, 40], [50, 50, 50, 50], true, 8);
        assert_eq!(a, 0);
        assert_eq!(b, 50);
    }

    #[test]
    fn model_equal_luma_gives_dc() {
        // diff (max-min luma) == 0 -> a = 0, b = min chroma.
        let (a, b, _) = cclm_model([100, 100, 100, 100], [30, 40, 50, 60], true, 8);
        assert_eq!(a, 0);
        // diff == 0 -> b = min-group chroma average; min group indices are [0,2].
        assert_eq!(b, (30 + 50 + 1) >> 1);
    }

    #[test]
    fn model_no_neighbours_is_dc_midrange() {
        let (a, b, shift) = cclm_model([0; 4], [0; 4], false, 10);
        assert_eq!((a, b, shift), (0, 1 << 9, 0));
    }

    #[test]
    fn model_linear_relation_recovered() {
        // Construct luma/chroma with an exact linear relation C = 2*L + 5 and
        // check the fitted model reproduces it on the endpoints.
        let lum = [16i32, 32, 200, 216];
        let chr = lum.map(|l| 2 * l + 5);
        let (a, b, shift) = cclm_model(lum, chr, true, 8);
        // Min group avg luma=(16+32+1)>>1=24, chroma=(37+69+1)>>1=53.
        // Max group avg luma=(200+216+1)>>1=208, chroma=(405+437+1)>>1=421.
        // Slope ~ (421-53)/(208-24)=368/184=2. Predict at L=24 -> ~53, L=208 -> ~421.
        let lo = ((a * 24) >> shift) + b;
        let hi = ((a * 208) >> shift) + b;
        assert!((lo - 53).abs() <= 1, "lo={lo}");
        assert!((hi - 421).abs() <= 1, "hi={hi}");
    }

    #[test]
    fn select_both_available_picks_two_each() {
        let tl: Vec<i32> = (0..32).collect();
        let tc: Vec<i32> = (0..32).map(|x| x + 100).collect();
        let ll: Vec<i32> = (0..32).map(|x| x + 200).collect();
        let lc: Vec<i32> = (0..32).map(|x| x + 300).collect();
        let (sl, sc, ok) = cclm_select(&tl, &tc, &ll, &lc, true, true);
        assert!(ok);
        // top: start=32>>2=8, step=max(1,32>>1)=16 -> positions 8,24.
        assert_eq!(sl[0], 8);
        assert_eq!(sl[1], 24);
        assert_eq!(sc[0], 108);
        // left: positions 8,24 -> 208,224.
        assert_eq!(sl[2], 208);
        assert_eq!(sl[3], 224);
        assert_eq!(sc[2], 308);
    }

    #[test]
    fn select_left_only_picks_four() {
        let ll: Vec<i32> = (0..16).collect();
        let lc: Vec<i32> = (0..16).map(|x| x + 50).collect();
        let (sl, _sc, ok) = cclm_select(&[], &[], &ll, &lc, false, true);
        assert!(ok);
        // left_is4=1: start=16>>3=2, step=max(1,16>>2)=4 -> 2,6,10,14.
        assert_eq!([sl[0], sl[1], sl[2], sl[3]], [2, 6, 10, 14]);
    }

    #[test]
    fn downsample_444_is_identity() {
        // 4:4:4: block copies luma directly.
        let luma = [[5, 6], [7, 8]];
        let acc = |x: isize, y: isize| luma[y as usize][x as usize];
        let (block, _t, _l) = cclm_luma(acc, 0, 0, 2, 2, 1, 1, 0, 0, false, false, false);
        assert_eq!(block, vec![5, 6, 7, 8]);
    }

    #[test]
    fn downsample_420_six_tap() {
        let rows = [[10, 20, 30, 40], [10, 20, 30, 40]];
        let acc = |x: isize, y: isize| rows[y as usize][x as usize];
        let (block, top, left) = cclm_luma(acc, 0, 0, 2, 1, 2, 2, 0, 0, false, false, false);
        let c0 = (4 + 2 * 10 + 20 + 10 + 2 * 10 + 20 + 10) >> 3;
        let c1 = (4 + 2 * 30 + 40 + 20 + 2 * 30 + 40 + 20) >> 3;
        assert_eq!(block, vec![c0, c1]);
        assert!(top.is_empty() && left.is_empty());
    }

    #[test]
    fn predict_constant_luma_yields_neighbour_chroma() {
        // Constant luma -> downsample is that constant, diff==0 -> a=0,
        // b = min-group chroma = the constant chroma neighbour value. So the
        // predictor is the chroma neighbour constant everywhere. Exercises the
        // full path: luma DS + neighbour gather + select + model + predict.
        let luma = |_x: isize, _y: isize| 128;
        let cb = |_x: isize, _y: isize| 90;
        let cr = |_x: isize, _y: isize| 200;
        let (pcb, pcr) = cclm_predict(
            luma,
            cb,
            cr,
            64,
            64,
            32,
            32,
            32,
            32,
            2,
            2,
            true,
            true,
            false,
            crate::intra::CCLM_LT_MODE,
            0,
            0,
            255,
            8,
        );
        assert!(
            pcb.iter().all(|&v| v == 90),
            "Cb should be the neighbour constant"
        );
        assert!(
            pcr.iter().all(|&v| v == 200),
            "Cr should be the neighbour constant"
        );
        assert_eq!(pcb.len(), 32 * 32);
    }
}
