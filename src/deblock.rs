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

const MAX_QP: i32 = 63;
const DEFAULT_INTRA_TC_OFFSET: i32 = 2;

/// tc' table, indexed by `clip(0, MAX_QP + 2, qp + 2)` for intra (bs = 2).
static TC_TABLE: [u16; (MAX_QP + 1 + DEFAULT_INTRA_TC_OFFSET) as usize] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 4, 4, 5, 5, 5, 5, 7, 7, 8, 9,
    10, 10, 11, 13, 14, 15, 17, 19, 21, 24, 25, 29, 33, 36, 41, 45, 51, 57, 64, 71, 80, 89, 100,
    112, 125, 141, 157, 177, 198, 222, 250, 280, 314, 352, 395,
];

/// beta' table, indexed by `clip(0, MAX_QP, qp)`.
static BETA_TABLE: [u8; (MAX_QP + 1) as usize] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66,
    68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88,
];

#[derive(Clone, Copy, Default)]
pub(crate) struct Blk {
    /// CU top-left in luma samples (to detect which 4-block starts a CU edge).
    pub(crate) cux: u16,
    pub(crate) cuy: u16,
    /// CU size in luma samples.
    pub(crate) cuw: u16,
    pub(crate) cuh: u16,
    /// Luma QP of the CU (deblock chroma QP == luma QP in this codec).
    pub(crate) qp: u8,
}

/// Picture block grid on the 4×4 luma raster.
pub(crate) struct Grid {
    pub(crate) cols: usize, // width / 4
    pub(crate) rows: usize, // height / 4
    pub(crate) blk: Vec<Blk>,
}

impl Grid {
    pub(crate) fn new(width: usize, height: usize) -> Self {
        let cols = width / 4;
        let rows = height / 4;
        Grid {
            cols,
            rows,
            blk: vec![Blk::default(); cols * rows],
        }
    }
    #[inline]
    fn at(&self, bx: usize, by: usize) -> Blk {
        self.blk[by * self.cols + bx]
    }
    /// Fill the 4-block coverage of one CU.
    pub(crate) fn set_cu(&mut self, x: usize, y: usize, w: usize, h: usize, qp: u8) {
        let b = Blk {
            cux: x as u16,
            cuy: y as u16,
            cuw: w as u16,
            cuh: h as u16,
            qp,
        };
        for by in (y / 4)..((y + h) / 4) {
            for bx in (x / 4)..((x + w) / 4) {
                self.blk[by * self.cols + bx] = b;
            }
        }
    }
}

#[inline]
fn clip3(lo: i32, hi: i32, v: i32) -> i32 {
    v.max(lo).min(hi)
}

#[inline]
fn db_coeffs(len: u8) -> &'static [i32] {
    match len {
        3 => &[53, 32, 11],
        5 => &[58, 45, 32, 19, 6],
        7 => &[59, 50, 41, 32, 23, 14, 5],
        _ => &[],
    }
}
#[inline]
fn tc_coeffs(len: u8) -> &'static [i32] {
    match len {
        3 => &[6, 4, 2],
        5 => &[6, 5, 4, 3, 2],
        7 => &[6, 5, 4, 3, 2, 1, 1],
        _ => &[],
    }
}

/// `xCalcDP`: |p2 − 2·p1 + p0| (or the chroma CTB-boundary variant).
#[inline]
fn calc_dp(b: &[i32], i: isize, off: isize, chroma_ctb: bool) -> i32 {
    let g = |k: isize| b[(i + k) as usize];
    if chroma_ctb {
        (g(-off * 2) - 2 * g(-off * 2) + g(-off)).abs()
    } else {
        (g(-off * 3) - 2 * g(-off * 2) + g(-off)).abs()
    }
}
/// `xCalcDQ`: |q0 − 2·q1 + q2|.
#[inline]
fn calc_dq(b: &[i32], i: isize, off: isize) -> i32 {
    let g = |k: isize| b[(i + k) as usize];
    (g(0) - 2 * g(off) + g(off * 2)).abs()
}

/// `xUseStrongFiltering`.
#[allow(clippy::too_many_arguments)]
fn use_strong(
    b: &[i32],
    i: isize,
    off: isize,
    d: i32,
    beta: i32,
    tc: i32,
    side_p_large: bool,
    side_q_large: bool,
    max_p: u8,
    max_q: u8,
    chroma_ctb: bool,
) -> bool {
    let g = |k: isize| b[(i + k) as usize];
    let m4 = g(0);
    let m3 = g(-off);
    let m7 = g(off * 3);
    let m0 = g(-off * 4);
    let m2 = g(-off * 2);
    let mut sp3 = if chroma_ctb {
        (m2 - m3).abs()
    } else {
        (m0 - m3).abs()
    };
    let mut sq3 = (m7 - m4).abs();
    if side_p_large || side_q_large {
        if side_p_large {
            let mp4;
            if max_p == 7 {
                let mp5 = g(-off * 5);
                let mp6 = g(-off * 6);
                let mp7 = g(-off * 7);
                mp4 = g(-off * 8);
                sp3 += (mp5 - mp6 - mp7 + mp4).abs();
            } else {
                mp4 = g(-off * 6);
            }
            sp3 = (sp3 + (m0 - mp4).abs() + 1) >> 1;
        }
        if side_q_large {
            let m11;
            if max_q == 7 {
                let m8 = g(off * 4);
                let m9 = g(off * 5);
                let m10 = g(off * 6);
                m11 = g(off * 7);
                sq3 += (m8 - m9 - m10 + m11).abs();
            } else {
                m11 = g(off * 5);
            }
            sq3 = (sq3 + (m11 - m7).abs() + 1) >> 1;
        }
        sp3 + sq3 < ((beta * 3) >> 5) && d < (beta >> 4) && (m3 - m4).abs() < ((tc * 5 + 1) >> 1)
    } else {
        sp3 + sq3 < (beta >> 3) && d < (beta >> 2) && (m3 - m4).abs() < ((tc * 5 + 1) >> 1)
    }
}

/// `xFilteringPandQ`: the long (5/7-tap) luma filter.
fn filtering_p_and_q(b: &mut [i32], i: isize, off: isize, len_p: u8, len_q: u8, tc: i32) {
    let g = |b: &[i32], k: isize| b[(i + k) as usize];
    let cp = db_coeffs(len_p);
    let cq = db_coeffs(len_q);
    let np = cp.len() as isize;
    let nq = cq.len() as isize;
    // refP/refQ from the outermost samples.
    let ref_p = (g(b, -(np * off - off)) + g(b, -(np * off)) + 1) >> 1;
    let ref_q = (g(b, nq * off - off) + g(b, nq * off) + 1) >> 1;

    let s = |k: isize| g(b, k); // sample accessor (all reads happen before writes)
    let ref_middle: i32 = if len_p == len_q {
        if len_p == 5 {
            (2 * (s(-off) + s(0) + s(-2 * off) + s(off) + s(-3 * off) + s(2 * off))
                + s(-4 * off)
                + s(3 * off)
                + s(-5 * off)
                + s(4 * off)
                + 8)
                >> 4
        } else {
            // _7 == _7
            (2 * (s(-off) + s(0))
                + s(-2 * off)
                + s(off)
                + s(-3 * off)
                + s(2 * off)
                + s(-4 * off)
                + s(3 * off)
                + s(-5 * off)
                + s(4 * off)
                + s(-6 * off)
                + s(5 * off)
                + s(-7 * off)
                + s(6 * off)
                + 8)
                >> 4
        }
    } else {
        // asymmetric: orient so P is the longer side
        let (lp, lq) = (len_p.max(len_q), len_p.min(len_q));
        // srcPt walks toward the longer side. Using p side = src-off direction.
        // Build with explicit sample positions relative to the boundary.
        if lp == 7 && lq == 5 {
            (2 * (s(-off) + s(0) + s(-2 * off) + s(off))
                + s(-3 * off)
                + s(2 * off)
                + s(-4 * off)
                + s(3 * off)
                + s(-5 * off)
                + s(4 * off)
                + s(-6 * off)
                + s(5 * off)
                + 8)
                >> 4
        } else if lp == 7 {
            // _7 / _3 : orient pt toward the longer (7) side, qt toward shorter.
            // P longer:  pt(k)=p_k=s(-(k+1)off), qt(k)=q_k=s(k off)
            // Q longer:  pt(k)=q_k=s(k off),     qt(k)=p_k=s(-(k+1)off)
            let longer_is_q = len_q > len_p;
            let pt = |k: isize| {
                if longer_is_q {
                    s(off * k)
                } else {
                    s(-off * (k + 1))
                }
            };
            let qt = |k: isize| {
                if longer_is_q {
                    s(-off * (k + 1))
                } else {
                    s(off * k)
                }
            };
            (2 * (pt(0) + qt(0))
                + qt(0)
                + 2 * (qt(1) + qt(2))
                + pt(1)
                + qt(1)
                + pt(2)
                + pt(3)
                + pt(4)
                + pt(5)
                + pt(6)
                + 8)
                >> 4
        } else {
            // _5 / _3
            (s(-off)
                + s(0)
                + s(-2 * off)
                + s(off)
                + s(-3 * off)
                + s(2 * off)
                + s(-4 * off)
                + s(3 * off)
                + 4)
                >> 3
        }
    };

    let tcp = tc_coeffs(len_p);
    let tcq = tc_coeffs(len_q);
    // P side: positions 0..nP at -off*pos
    for pos in 0..tcp.len() as isize {
        let src = b[(i - off * pos) as usize];
        let cvalue = (tc * tcp[pos as usize]) >> 1;
        let cp_pos = cp[pos as usize];
        b[(i - off * pos) as usize] = clip3(
            src - cvalue,
            src + cvalue,
            (ref_middle * cp_pos + ref_p * (64 - cp_pos) + 32) >> 6,
        );
    }
    for pos in 0..tcq.len() as isize {
        let src = b[(i + off * pos) as usize];
        let cvalue = (tc * tcq[pos as usize]) >> 1;
        let cq_pos = cq[pos as usize];
        b[(i + off * pos) as usize] = clip3(
            src - cvalue,
            src + cvalue,
            (ref_middle * cq_pos + ref_q * (64 - cq_pos) + 32) >> 6,
        );
    }
}

/// `xPelFilterLuma` for one line.
#[allow(clippy::too_many_arguments)]
fn pel_filter_luma(
    b: &mut [i32],
    i: isize,
    off: isize,
    tc: i32,
    sw: bool,
    thr_cut: i32,
    filter_p: bool,
    filter_q: bool,
    side_p_large: bool,
    side_q_large: bool,
    max_p: u8,
    max_q: u8,
    max_val: i32,
) {
    let g = |b: &[i32], k: isize| b[(i + k) as usize];
    let m4 = g(b, 0);
    let m3 = g(b, -off);
    let m5 = g(b, off);
    let m2 = g(b, -off * 2);
    let m6 = g(b, off * 2);
    let m1 = g(b, -off * 3);
    let m7 = g(b, off * 3);
    let m0 = g(b, -off * 4);

    if sw {
        if side_p_large || side_q_large {
            let lp = if side_p_large { max_p } else { 3 };
            let lq = if side_q_large { max_q } else { 3 };
            filtering_p_and_q(b, i, off, lp, lq, tc);
        } else {
            b[(i - off) as usize] = clip3(
                m3 - 3 * tc,
                m3 + 3 * tc,
                (m1 + 2 * m2 + 2 * m3 + 2 * m4 + m5 + 4) >> 3,
            );
            b[i as usize] = clip3(
                m4 - 3 * tc,
                m4 + 3 * tc,
                (m2 + 2 * m3 + 2 * m4 + 2 * m5 + m6 + 4) >> 3,
            );
            b[(i - off * 2) as usize] =
                clip3(m2 - 2 * tc, m2 + 2 * tc, (m1 + m2 + m3 + m4 + 2) >> 2);
            b[(i + off) as usize] = clip3(m5 - 2 * tc, m5 + 2 * tc, (m3 + m4 + m5 + m6 + 2) >> 2);
            b[(i - off * 3) as usize] =
                clip3(m1 - tc, m1 + tc, (2 * m0 + 3 * m1 + m2 + m3 + m4 + 4) >> 3);
            b[(i + off * 2) as usize] =
                clip3(m6 - tc, m6 + tc, (m3 + m4 + m5 + 3 * m6 + 2 * m7 + 4) >> 3);
        }
    } else {
        let delta = (9 * (m4 - m3) - 3 * (m5 - m2) + 8) >> 4;
        if delta.abs() < thr_cut {
            let delta = clip3(-tc, tc, delta);
            b[(i - off) as usize] = clip3(0, max_val, m3 + delta);
            b[i as usize] = clip3(0, max_val, m4 - delta);
            let tc2 = tc >> 1;
            if filter_p {
                let d1 = clip3(-tc2, tc2, (((m1 + m3 + 1) >> 1) - m2 + delta) >> 1);
                b[(i - off * 2) as usize] = clip3(0, max_val, m2 + d1);
            }
            if filter_q {
                let d2 = clip3(-tc2, tc2, (((m6 + m4 + 1) >> 1) - m5 - delta) >> 1);
                b[(i + off) as usize] = clip3(0, max_val, m5 + d2);
            }
        }
    }
}

/// `xPelFilterChroma` for one line.
#[allow(clippy::too_many_arguments)]
fn pel_filter_chroma(
    b: &mut [i32],
    i: isize,
    off: isize,
    tc: i32,
    sw: bool,
    chroma_ctb: bool,
    max_val: i32,
) {
    let len = b.len() as isize;
    // Clamped read: VTM reads p3..q3 into locals even when the short filter uses
    // only p1..q1, relying on picture margins. The extra samples are unused in
    // the branches where they'd fall outside the plane, so clamping is exact.
    let g = |b: &[i32], k: isize| b[(i + k).clamp(0, len - 1) as usize];
    let m0 = g(b, -off * 4);
    let m1 = g(b, -off * 3);
    let m2 = g(b, -off * 2);
    let m3 = g(b, -off);
    let m4 = g(b, 0);
    let m5 = g(b, off);
    let m6 = g(b, off * 2);
    let m7 = g(b, off * 3);
    if sw {
        if chroma_ctb {
            b[(i - off) as usize] =
                clip3(m3 - tc, m3 + tc, (3 * m2 + 2 * m3 + m4 + m5 + m6 + 4) >> 3);
            b[i as usize] = clip3(
                m4 - tc,
                m4 + tc,
                (2 * m2 + m3 + 2 * m4 + m5 + m6 + m7 + 4) >> 3,
            );
            b[(i + off) as usize] = clip3(
                m5 - tc,
                m5 + tc,
                (m2 + m3 + m4 + 2 * m5 + m6 + 2 * m7 + 4) >> 3,
            );
            b[(i + off * 2) as usize] =
                clip3(m6 - tc, m6 + tc, (m3 + m4 + m5 + 2 * m6 + 3 * m7 + 4) >> 3);
        } else {
            b[(i - off * 3) as usize] =
                clip3(m1 - tc, m1 + tc, (3 * m0 + 2 * m1 + m2 + m3 + m4 + 4) >> 3);
            b[(i - off * 2) as usize] = clip3(
                m2 - tc,
                m2 + tc,
                (2 * m0 + m1 + 2 * m2 + m3 + m4 + m5 + 4) >> 3,
            );
            b[(i - off) as usize] = clip3(
                m3 - tc,
                m3 + tc,
                (m0 + m1 + m2 + 2 * m3 + m4 + m5 + m6 + 4) >> 3,
            );
            b[i as usize] = clip3(
                m4 - tc,
                m4 + tc,
                (m1 + m2 + m3 + 2 * m4 + m5 + m6 + m7 + 4) >> 3,
            );
            b[(i + off) as usize] = clip3(
                m5 - tc,
                m5 + tc,
                (m2 + m3 + m4 + 2 * m5 + m6 + 2 * m7 + 4) >> 3,
            );
            b[(i + off * 2) as usize] =
                clip3(m6 - tc, m6 + tc, (m3 + m4 + m5 + 2 * m6 + 3 * m7 + 4) >> 3);
        }
    } else {
        let delta = clip3(-tc, tc, (4 * (m4 - m3) + m2 - m5 + 4) >> 3);
        b[(i - off) as usize] = clip3(0, max_val, m3 + delta);
        b[i as usize] = clip3(0, max_val, m4 - delta);
    }
}

/// Derive luma tc and beta from the averaged QP.
fn luma_tc_beta(qp: i32, bit_depth: u8) -> (i32, i32) {
    let index_tc = clip3(
        0,
        MAX_QP + DEFAULT_INTRA_TC_OFFSET,
        qp + DEFAULT_INTRA_TC_OFFSET,
    ); // bs=2
    let index_b = clip3(0, MAX_QP, qp);
    let bd = bit_depth as i32;
    let tc = if bd < 10 {
        (TC_TABLE[index_tc as usize] as i32 + (1 << (9 - bd))) >> (10 - bd)
    } else {
        (TC_TABLE[index_tc as usize] as i32) << (bd - 10)
    };
    let beta = BETA_TABLE[index_b as usize] as i32 * (1 << (bd - 8));
    (tc, beta)
}

#[inline]
fn luma_max_len(size_p: i32, size_q: i32) -> (u8, u8) {
    if size_p <= 4 || size_q <= 4 {
        (1, 1)
    } else {
        (
            if size_p >= 32 { 7 } else { 3 },
            if size_q >= 32 { 7 } else { 3 },
        )
    }
}

/// Filter one 4-sample luma edge segment (the q0 sample of line 0 at `i0`,
/// stepping `along` per line for 4 lines; `off` crosses the edge).
#[allow(clippy::too_many_arguments)]
fn luma_segment(
    b: &mut [i32],
    i0: isize,
    off: isize,
    along: isize,
    size_p: i32,
    size_q: i32,
    qp: i32,
    bit_depth: u8,
    ctu_hor_p: bool,
) {
    let (tc, beta) = luma_tc_beta(qp, bit_depth);
    if tc == 0 && beta == 0 {
        return;
    }
    let max_val = (1 << bit_depth) - 1;
    let (max_p0, max_q) = luma_max_len(size_p, size_q);
    let mut max_p = max_p0;
    let mut side_p_large = max_p > 3;
    let side_q_large = max_q > 3;
    // affine restriction n/a (intra). CTU horizontal boundary: no large P.
    if ctu_hor_p {
        side_p_large = false;
    }
    let side_threshold = (beta + (beta >> 1)) >> 3;
    let thr_cut = tc * 10;

    let i3 = i0 + along * 3;
    let dp0 = calc_dp(b, i0, off, false);
    let dq0 = calc_dq(b, i0, off);
    let dp3 = calc_dp(b, i3, off, false);
    let dq3 = calc_dq(b, i3, off);

    let mut used_long = false;
    if side_p_large || side_q_large {
        if !side_p_large {
            max_p = if max_p > 3 { 3 } else { max_p };
        }
        let dp0l = if side_p_large {
            (dp0 + calc_dp(b, i0 - 3 * off, off, false) + 1) >> 1
        } else {
            dp0
        };
        let dp3l = if side_p_large {
            (dp3 + calc_dp(b, i3 - 3 * off, off, false) + 1) >> 1
        } else {
            dp3
        };
        let dq0l = if side_q_large {
            (dq0 + calc_dq(b, i0 + 3 * off, off) + 1) >> 1
        } else {
            dq0
        };
        let dq3l = if side_q_large {
            (dq3 + calc_dq(b, i3 + 3 * off, off) + 1) >> 1
        } else {
            dq3
        };
        let d0l = dp0l + dq0l;
        let d3l = dp3l + dq3l;
        let dpl = dp0l + dp3l;
        let dql = dq0l + dq3l;
        let dl = d0l + d3l;
        if dl < beta {
            let filter_p = dpl < side_threshold;
            let filter_q = dql < side_threshold;
            used_long = use_strong(
                b,
                i0,
                off,
                2 * d0l,
                beta,
                tc,
                side_p_large,
                side_q_large,
                max_p,
                max_q,
                false,
            ) && use_strong(
                b,
                i3,
                off,
                2 * d3l,
                beta,
                tc,
                side_p_large,
                side_q_large,
                max_p,
                max_q,
                false,
            );
            if used_long {
                for k in 0..4 {
                    pel_filter_luma(
                        b,
                        i0 + along * k,
                        off,
                        tc,
                        true,
                        thr_cut,
                        filter_p,
                        filter_q,
                        side_p_large,
                        side_q_large,
                        max_p,
                        max_q,
                        max_val,
                    );
                }
            }
        }
    }
    if !used_long {
        let d0 = dp0 + dq0;
        let d3 = dp3 + dq3;
        let dp = dp0 + dp3;
        let dq = dq0 + dq3;
        let d = d0 + d3;
        if d < beta {
            let larger1 = max_p > 1 && max_q > 1;
            let larger2 = max_p > 2 && max_q > 2;
            let filter_p = larger1 && dp < side_threshold;
            let filter_q = larger1 && dq < side_threshold;
            let sw = larger2
                && use_strong(
                    b,
                    i0,
                    off,
                    2 * d0,
                    beta,
                    tc,
                    false,
                    false,
                    max_p,
                    max_q,
                    false,
                )
                && use_strong(
                    b,
                    i3,
                    off,
                    2 * d3,
                    beta,
                    tc,
                    false,
                    false,
                    max_p,
                    max_q,
                    false,
                );
            for k in 0..4 {
                pel_filter_luma(
                    b,
                    i0 + along * k,
                    off,
                    tc,
                    sw,
                    thr_cut,
                    filter_p,
                    filter_q,
                    false,
                    false,
                    max_p,
                    max_q,
                    max_val,
                );
            }
        }
    }
}

/// Deblock one luma plane in place. `buf` is `width*height` i32 samples.
pub(crate) fn deblock_luma(
    buf: &mut [i32],
    width: usize,
    _height: usize,
    grid: &Grid,
    bit_depth: u8,
    ctu_size: usize,
) {
    let stride = width as isize;
    // Vertical edges (cross = horizontal, off = 1; along = stride).
    for by in 0..grid.rows {
        for bx in 1..grid.cols {
            let q = grid.at(bx, by);
            // left edge of this CU?
            if q.cux as usize != bx * 4 {
                continue;
            }
            let p = grid.at(bx - 1, by);
            let qp = (p.qp as i32 + q.qp as i32 + 1) >> 1;
            let i0 = (by * 4) as isize * stride + (bx * 4) as isize;
            luma_segment(
                buf,
                i0,
                1,
                stride,
                p.cuw as i32,
                q.cuw as i32,
                qp,
                bit_depth,
                false,
            );
        }
    }
    // Horizontal edges (cross = vertical, off = stride; along = 1).
    for by in 1..grid.rows {
        for bx in 0..grid.cols {
            let q = grid.at(bx, by);
            if q.cuy as usize != by * 4 {
                continue;
            }
            let p = grid.at(bx, by - 1);
            let qp = (p.qp as i32 + q.qp as i32 + 1) >> 1;
            let i0 = (by * 4) as isize * stride + (bx * 4) as isize;
            let ctu_hor_p = (by * 4) % ctu_size == 0;
            luma_segment(
                buf,
                i0,
                stride,
                1,
                p.cuh as i32,
                q.cuh as i32,
                qp,
                bit_depth,
                ctu_hor_p,
            );
        }
    }
}

#[inline]
fn chroma_max_large(size_p: i32, size_q: i32) -> bool {
    size_p >= 8 && size_q >= 8
}

/// Filter one chroma edge run (`run_len` samples along the edge, decision taken
/// at run offsets 0 and `dec3`).
#[allow(clippy::too_many_arguments)]
fn chroma_run(
    b: &mut [i32],
    i0: isize,
    off: isize,
    along: isize,
    dec3: isize,
    run_len: isize,
    size_p: i32,
    size_q: i32,
    qp: i32,
    bit_depth: u8,
    chroma_ctb: bool,
) {
    let large = chroma_max_large(size_p, size_q);
    // tc (chroma uses the same table; bs = 2, offsets 0).
    let index_tc = clip3(
        0,
        MAX_QP + DEFAULT_INTRA_TC_OFFSET,
        qp + DEFAULT_INTRA_TC_OFFSET,
    );
    let bd = bit_depth as i32;
    let tc = if bd < 10 {
        (TC_TABLE[index_tc as usize] as i32 + (1 << (9 - bd))) >> (10 - bd)
    } else {
        (TC_TABLE[index_tc as usize] as i32) << (bd - 10)
    };
    let max_val = (1 << bit_depth) - 1;
    let mut used_long = false;
    if large {
        let index_b = clip3(0, MAX_QP, qp);
        let beta = BETA_TABLE[index_b as usize] as i32 * (1 << (bd - 8));
        let i3 = i0 + dec3;
        let dp0 = calc_dp(b, i0, off, chroma_ctb);
        let dq0 = calc_dq(b, i0, off);
        let dp3 = calc_dp(b, i3, off, chroma_ctb);
        let dq3 = calc_dq(b, i3, off);
        let d0 = dp0 + dq0;
        let d3 = dp3 + dq3;
        let d = d0 + d3;
        if d < beta {
            used_long = true;
            let sw = use_strong(b, i0, off, 2 * d0, beta, tc, false, false, 7, 7, chroma_ctb)
                && use_strong(b, i3, off, 2 * d3, beta, tc, false, false, 7, 7, chroma_ctb);
            for k in 0..run_len {
                pel_filter_chroma(b, i0 + along * k, off, tc, sw, chroma_ctb, max_val);
            }
        }
    }
    if !used_long {
        for k in 0..run_len {
            pel_filter_chroma(b, i0 + along * k, off, tc, false, chroma_ctb, max_val);
        }
    }
}

/// Deblock one chroma plane (`cw*ch` samples) in place, using the luma `grid`.
/// `subx`/`suby` are the chroma subsampling shifts (4:2:0 → 1,1; 4:2:2 → 1,0;
/// 4:4:4 → 0,0).
#[allow(clippy::too_many_arguments)]
pub(crate) fn deblock_chroma(
    plane: &mut [i32],
    cw: usize,
    ch: usize,
    grid: &Grid,
    subx: usize,
    suby: usize,
    bit_depth: u8,
    ctu_size: usize,
) {
    let stride = cw as isize;
    let lpc_v = (4 >> suby) as isize; // chroma rows per luma 4-part (VER run length)
    let lpc_h = (4 >> subx) as isize; // chroma cols per luma 4-part (HOR run length)
    let dec3_v = (3 >> suby) as isize;
    let dec3_h = (3 >> subx) as isize;
    let chroma_ctu_rows = ctu_size >> suby;

    // Vertical chroma edges: chroma column ccx that is a CU left boundary and on
    // the 8-chroma grid.
    let mut ccx = 8;
    while ccx < cw {
        let lx = ccx << subx; // luma column
        let bx = lx / 4;
        if bx < grid.cols {
            let mut ccy = 0usize;
            while ccy < ch {
                let ly = ccy << suby;
                let by = (ly / 4).min(grid.rows - 1);
                let q = grid.at(bx, by);
                if (q.cux as usize) == lx && bx >= 1 {
                    let p = grid.at(bx - 1, by);
                    let qp = (p.qp as i32 + q.qp as i32 + 1) >> 1;
                    let size_p = (p.cuw as i32) >> subx;
                    let size_q = (q.cuw as i32) >> subx;
                    let i0 = ccy as isize * stride + ccx as isize;
                    chroma_run(
                        plane,
                        i0,
                        1,
                        stride,
                        dec3_v * stride,
                        lpc_v,
                        size_p,
                        size_q,
                        qp,
                        bit_depth,
                        false,
                    );
                    ccy += lpc_v as usize;
                } else {
                    ccy += lpc_v as usize;
                }
            }
        }
        ccx += 8;
    }

    // Horizontal chroma edges.
    let mut ccy = 8;
    while ccy < ch {
        let ly = ccy << suby;
        let by = ly / 4;
        if by < grid.rows {
            let chroma_ctb = chroma_ctu_rows != 0 && ccy % chroma_ctu_rows == 0;
            let mut ccx = 0usize;
            while ccx < cw {
                let lx = ccx << subx;
                let bx = (lx / 4).min(grid.cols - 1);
                let q = grid.at(bx, by);
                if (q.cuy as usize) == ly && by >= 1 {
                    let p = grid.at(bx, by - 1);
                    let qp = (p.qp as i32 + q.qp as i32 + 1) >> 1;
                    let size_p = (p.cuh as i32) >> suby;
                    let size_q = (q.cuh as i32) >> suby;
                    let i0 = ccy as isize * stride + ccx as isize;
                    chroma_run(
                        plane, i0, stride, 1, dec3_h, lpc_h, size_p, size_q, qp, bit_depth,
                        chroma_ctb,
                    );
                    ccx += lpc_h as usize;
                } else {
                    ccx += lpc_h as usize;
                }
            }
        }
        ccy += 8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_uniform(w: usize, h: usize, cu: usize, qp: u8) -> Grid {
        let mut g = Grid::new(w, h);
        for y in (0..h).step_by(cu) {
            for x in (0..w).step_by(cu) {
                g.set_cu(x, y, cu.min(w - x), cu.min(h - y), qp);
            }
        }
        g
    }

    #[test]
    fn flat_plane_unchanged() {
        // A constant plane has zero gradient everywhere, so every edge decision
        // yields delta 0: deblocking must be a no-op.
        let (w, h) = (64usize, 64usize);
        let g = grid_uniform(w, h, 16, 37);
        let mut luma = vec![128i32; w * h];
        deblock_luma(&mut luma, w, h, &g, 8, 128);
        assert!(luma.iter().all(|&v| v == 128));
        let (cw, ch) = (w / 2, h / 2);
        let mut cb = vec![128i32; cw * ch];
        deblock_chroma(&mut cb, cw, ch, &g, 1, 1, 8, 128);
        assert!(cb.iter().all(|&v| v == 128));
    }

    #[test]
    fn partitioned_plane_no_panic_and_bounded() {
        // Exercise all edge/boundary paths on a ramp with block structure and
        // confirm no out-of-bounds and output stays in range.
        for &(cu, sub) in &[(8usize, 1usize), (16, 1), (32, 0), (64, 0), (8, 0)] {
            let (w, h) = (128usize, 128usize);
            let g = grid_uniform(w, h, cu, 41);
            let mut luma: Vec<i32> = (0..w * h)
                .map(|k| ((k * 7 + (k / w) * 13) % 256) as i32)
                .collect();
            deblock_luma(&mut luma, w, h, &g, 8, 128);
            assert!(luma.iter().all(|&v| (0..=255).contains(&v)));
            let (cw, ch) = (w >> sub, h >> sub);
            let mut cb: Vec<i32> = (0..cw * ch).map(|k| ((k * 5) % 256) as i32).collect();
            deblock_chroma(&mut cb, cw, ch, &g, sub, sub, 8, 128);
            assert!(cb.iter().all(|&v| (0..=255).contains(&v)));
        }
    }
}
