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

use crate::lfnst_tables::{LFNST_4X4, LFNST_8X8, LFNST_LUT};

const NUM_LUMA_MODE: i32 = 67;
const NUM_EXT_LUMA_MODE: i32 = 28;
const DIA_IDX: i32 = 34;

/// Remap a (possibly wide-angle) intra prediction mode into the index space of
/// [`LFNST_LUT`] (VTM `getLFNSTIntraMode`). Wide-angle modes (`< 0` or
/// `>= NUM_LUMA_MODE`) fold into the extended-mode range.
pub(crate) fn lfnst_intra_mode(wide_ang: i32) -> usize {
    if wide_ang < 0 {
        (wide_ang + (NUM_EXT_LUMA_MODE >> 1) + NUM_LUMA_MODE) as usize
    } else if wide_ang >= NUM_LUMA_MODE {
        (wide_ang + (NUM_EXT_LUMA_MODE >> 1)) as usize
    } else {
        wide_ang as usize
    }
}

/// Whether the coefficient sub-block must be transposed before LFNST (VTM
/// `getTransposeFlag`): modes beyond the diagonal use the transposed layout.
pub(crate) fn lfnst_transpose(intra_mode_lut_idx: usize) -> bool {
    let m = intra_mode_lut_idx as i32;
    (m >= NUM_LUMA_MODE + (NUM_EXT_LUMA_MODE >> 1)) || (m < NUM_LUMA_MODE && m > DIA_IDX)
}

/// LFNST transform set (0..4) for a remapped intra mode index.
pub(crate) fn lfnst_set(intra_mode_lut_idx: usize) -> usize {
    LFNST_LUT[intra_mode_lut_idx] as usize
}

pub(crate) fn fwd_lfnst_nxn(
    src: &[i32],
    set: usize,
    idx: usize,
    size: usize,
    zero_out: usize,
) -> [i32; 48] {
    debug_assert!(idx < 2);
    let mut out = [0i32; 48];
    if size > 4 {
        let mat = &LFNST_8X8[set][idx];
        for j in 0..zero_out {
            let mut coef = 0i64;
            for i in 0..48 {
                coef += src[i] as i64 * mat[j][i] as i64;
            }
            out[j] = ((coef + 64) >> 7) as i32;
        }
    } else {
        let mat = &LFNST_4X4[set][idx];
        for j in 0..zero_out {
            let mut coef = 0i64;
            for i in 0..16 {
                coef += src[i] as i64 * mat[j][i] as i64;
            }
            out[j] = ((coef + 64) >> 7) as i32;
        }
    }
    out
}

/// Inverse LFNST. `src` holds `zero_out` secondary coefficients; returns
/// `tr_size` reconstructed primary coefficients, clamped to the transform
/// dynamic range `±2^max_log2`.
pub(crate) fn inv_lfnst_nxn(
    src: &[i32],
    set: usize,
    idx: usize,
    size: usize,
    zero_out: usize,
    max_log2: i32,
) -> [i32; 48] {
    debug_assert!(idx < 2);
    let lo = -(1i64 << max_log2);
    let hi = (1i64 << max_log2) - 1;
    let mut out = [0i32; 48];
    if size > 4 {
        let mat = &LFNST_8X8[set][idx];
        for j in 0..48 {
            let mut resi = 0i64;
            for i in 0..zero_out {
                resi += src[i] as i64 * mat[i][j] as i64;
            }
            out[j] = (((resi + 64) >> 7).clamp(lo, hi)) as i32;
        }
    } else {
        let mat = &LFNST_4X4[set][idx];
        for j in 0..16 {
            let mut resi = 0i64;
            for i in 0..zero_out {
                resi += src[i] as i64 * mat[i][j] as i64;
            }
            out[j] = (((resi + 64) >> 7).clamp(lo, hi)) as i32;
        }
    }
    out
}

/// (x, y) positions of the first 48 entries of VTM's `g_auiXYDiagScan8x8`: the
/// up-right diagonal scan over the top-left 8x8 region, excluding the bottom-right
/// 4x4 quadrant (the L-shape covered by the 8x8 LFNST). Flat index is `x + y*w`.
#[rustfmt::skip]
const XY_DIAG_8X8: [(usize, usize); 48] = [
    (0,0),(0,1),(1,0),(0,2),(1,1),(2,0),(0,3),(1,2),
    (2,1),(3,0),(1,3),(2,2),(3,1),(2,3),(3,2),(3,3),
    (0,4),(0,5),(1,4),(0,6),(1,5),(2,4),(0,7),(1,6),
    (2,5),(3,4),(1,7),(2,6),(3,5),(2,7),(3,6),(3,7),
    (4,0),(4,1),(5,0),(4,2),(5,1),(6,0),(4,3),(5,2),
    (6,1),(7,0),(5,3),(6,2),(7,1),(6,3),(7,2),(7,3),
];

/// Low-frequency LFNST scan as block coordinates `(x, y)`, returned as a static
/// slice (no allocation): the 48-entry 8×8 L-shape for `sb == 8`, else the
/// grouped 4×4 diagonal scan (callers take the first 16). The caller computes
/// the flat buffer index `y*w + x`.
fn lfnst_scan_coords(sb: usize, _w: usize, _h: usize) -> &'static [(usize, usize)] {
    if sb > 4 {
        &XY_DIAG_8X8
    } else {
        // RST4x4 outputs always occupy the top-left 4×4 sub-block in the fixed
        // 4×4 diagonal scan, independent of the (possibly rectangular) block size.
        crate::residual::scan_coords(4, 4)
    }
}

/// Apply the forward LFNST in place to a primary-transform coefficient block
/// (`coeff`, row-major stride `w`). `mode` is the LFNST-remapped intra mode index
/// (see [`lfnst_intra_mode`]); `lfnst_idx` is 1 or 2. After this, only the
/// low-frequency LFNST outputs are non-zero; every other coefficient is zeroed
/// (the high-frequency "zero-out" region H.266 requires).
pub(crate) fn apply_fwd_lfnst(
    coeff: &mut [i32],
    w: usize,
    h: usize,
    mode: usize,
    lfnst_idx: usize,
) {
    let whge3 = w >= 8 && h >= 8;
    let sb = if whge3 { 8 } else { 4 };
    let set = lfnst_set(mode);
    let transpose = lfnst_transpose(mode);
    let zero_out = if (w == 4 && h == 4) || (w == 8 && h == 8) {
        8
    } else {
        16
    };
    let coeff_num = if sb == 4 { 16 } else { 48 };

    // Gather the L-shape (rows 0..3 full sub-block width, rows 4..7 only 4 wide),
    // transposed when the intra mode is beyond the diagonal.
    let mut inv = [0i32; 48];
    if transpose {
        for y in 0..sb {
            inv[y] = coeff[y * w];
            inv[sb + y] = coeff[y * w + 1];
            inv[2 * sb + y] = coeff[y * w + 2];
            inv[3 * sb + y] = coeff[y * w + 3];
            if sb == 8 && y < 4 {
                inv[32 + y] = coeff[y * w + 4];
                inv[36 + y] = coeff[y * w + 5];
                inv[40 + y] = coeff[y * w + 6];
                inv[44 + y] = coeff[y * w + 7];
            }
        }
    } else {
        let mut idx = 0;
        for y in 0..sb {
            let stride = if y < 4 { sb } else { 4 };
            for x in 0..stride {
                inv[idx] = coeff[y * w + x];
                idx += 1;
            }
        }
    }

    let out = fwd_lfnst_nxn(&inv, set, lfnst_idx - 1, sb, zero_out);

    // Everything outside the LFNST output is zero; scatter the outputs into the
    // low-frequency scan positions.
    for c in coeff[..w * h].iter_mut() {
        *c = 0;
    }
    let scan = lfnst_scan_coords(sb, w, h);
    for (k, &(x, y)) in scan.iter().take(coeff_num).enumerate() {
        coeff[y * w + x] = out[k];
    }
}

/// Apply the inverse LFNST in place to a dequantized coefficient block, the
/// mirror of [`apply_fwd_lfnst`]. `max_log2` is the transform dynamic range.
pub(crate) fn apply_inv_lfnst(
    coeff: &mut [i32],
    w: usize,
    h: usize,
    mode: usize,
    lfnst_idx: usize,
    max_log2: i32,
) {
    let whge3 = w >= 8 && h >= 8;
    let sb = if whge3 { 8 } else { 4 };
    let set = lfnst_set(mode);
    let transpose = lfnst_transpose(mode);
    let zero_out = if (w == 4 && h == 4) || (w == 8 && h == 8) {
        8
    } else {
        16
    };

    // Inverse spectral rearrangement: read the 16 lowest-frequency scan positions.
    let scan = lfnst_scan_coords(sb, w, h);
    let mut inv = [0i32; 16];
    for (k, &(x, y)) in scan.iter().take(16).enumerate() {
        inv[k] = coeff[y * w + x];
    }

    let out = inv_lfnst_nxn(&inv, set, lfnst_idx - 1, sb, zero_out, max_log2);

    // Scatter the reconstructed primary coefficients back into the L-shape.
    if transpose {
        for y in 0..sb {
            coeff[y * w] = out[y];
            coeff[y * w + 1] = out[sb + y];
            coeff[y * w + 2] = out[2 * sb + y];
            coeff[y * w + 3] = out[3 * sb + y];
            if sb == 8 && y < 4 {
                coeff[y * w + 4] = out[32 + y];
                coeff[y * w + 5] = out[36 + y];
                coeff[y * w + 6] = out[40 + y];
                coeff[y * w + 7] = out[44 + y];
            }
        }
    } else {
        let mut idx = 0;
        for y in 0..sb {
            let stride = if y < 4 { sb } else { 4 };
            for x in 0..stride {
                coeff[y * w + x] = out[idx];
                idx += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Each LFNST matrix row should be near-orthonormal at scale 128: the VTM
    // matrices are integer approximations of orthonormal bases, so a row's
    // squared norm is ~128^2 and distinct rows are ~orthogonal.
    #[test]
    fn lfnst_matrices_are_near_orthonormal() {
        for set in 0..4 {
            for idx in 0..2 {
                for r in 0..16 {
                    let row = &LFNST_8X8[set][idx][r];
                    let norm: i64 = row.iter().map(|&v| v as i64 * v as i64).sum();
                    assert!(
                        (14000..=19000).contains(&norm),
                        "8x8 set{set} idx{idx} row{r} norm {norm}"
                    );
                    if r + 1 < 16 {
                        let next = &LFNST_8X8[set][idx][r + 1];
                        let dot: i64 = row
                            .iter()
                            .zip(next)
                            .map(|(&a, &b)| a as i64 * b as i64)
                            .sum();
                        assert!(
                            dot.abs() < 2500,
                            "8x8 set{set} idx{idx} rows {r}/{} dot {dot}",
                            r + 1
                        );
                    }
                }
            }
        }
    }

    // inverse then forward recovers an arbitrary LFNST coefficient vector
    // (exercising the full gather/scatter/transpose path on a coeff buffer).
    // Modes are chosen to hit both transpose=false and transpose=true.
    #[test]
    fn lfnst_buffer_inverse_forward_round_trips() {
        for &(w, h) in &[
            (4usize, 4usize),
            (8, 8),
            (16, 16),
            (8, 16),
            (4, 8),
            (32, 32),
        ] {
            for &mode in &[0usize, 10, 40, 66] {
                for lfnst_idx in 1..=2 {
                    let zero_out = if (w == 4 && h == 4) || (w == 8 && h == 8) {
                        8
                    } else {
                        16
                    };
                    let scan = lfnst_scan_coords(if w >= 8 && h >= 8 { 8 } else { 4 }, w, h);
                    // Seed the LFNST-domain coefficients (first `zero_out` scan positions).
                    let mut coeff = vec![0i32; w * h];
                    let seed = [
                        70i32, -45, 33, -22, 18, -12, 9, -6, 14, -11, 8, -5, 7, -4, 3, -2,
                    ];
                    for k in 0..zero_out {
                        let (x, y) = scan[k];
                        coeff[y * w + x] = seed[k];
                    }
                    let mut buf = coeff.clone();
                    apply_inv_lfnst(&mut buf, w, h, mode, lfnst_idx, 15);
                    apply_fwd_lfnst(&mut buf, w, h, mode, lfnst_idx);
                    let err: i64 = (0..zero_out)
                        .map(|k| {
                            let (x, y) = scan[k];
                            let p = y * w + x;
                            (buf[p] as i64 - coeff[p] as i64).abs()
                        })
                        .sum();
                    assert!(
                        err <= zero_out as i64 * 3,
                        "{w}x{h} mode{mode} idx{lfnst_idx} L1 err {err}"
                    );
                }
            }
        }
    }

    #[test]
    fn lfnst_composes_with_transform_and_quant() {
        use crate::transform::{
            MAX_TB, dequantize_wh, fwd_transform_wh_into, inv_transform_wh, quantize_wh,
        };
        for &(w, h) in &[(8usize, 8usize), (16, 16), (4, 4), (8, 16)] {
            let nn = w * h;
            let (bd, qp) = (8u8, 24u8);
            // Directional residual (diagonal ramp) — the energy LFNST compacts.
            let mut res = vec![0i32; nn];
            for y in 0..h {
                for x in 0..w {
                    res[y * w + x] = (x as i32 - y as i32) * 5;
                }
            }
            let mode = 40usize; // angular mode -> non-trivial set + transpose
            let chain = |lfnst_idx: usize| -> Vec<i32> {
                let mut coeff = [0i32; MAX_TB];
                fwd_transform_wh_into(&mut coeff[..nn], &res, w, h, bd);
                if lfnst_idx > 0 {
                    apply_fwd_lfnst(&mut coeff[..nn], w, h, mode, lfnst_idx);
                }
                let levels = quantize_wh(&coeff[..nn], w, h, qp, bd);
                let deq = dequantize_wh(&levels[..nn], w, h, qp, bd);
                let mut deqv = deq[..nn].to_vec();
                if lfnst_idx > 0 {
                    apply_inv_lfnst(&mut deqv, w, h, mode, lfnst_idx, 15);
                }
                inv_transform_wh(&deqv[..nn], w, h, bd)[..nn].to_vec()
            };
            let r0 = chain(0);
            let r1 = chain(1);
            assert_ne!(r0, r1, "{w}x{h}: LFNST had no effect on the reconstruction");
            let err: i64 = r1
                .iter()
                .zip(&res)
                .map(|(&a, &b)| (a as i64 - b as i64).abs())
                .sum();
            assert!(
                err < nn as i64 * 60,
                "{w}x{h}: LFNST reconstruction off by {err}"
            );
        }
    }

    // Inverse(forward(x)) recovers a low-frequency input within integer-rounding
    // tolerance when no coefficients are zeroed (zero_out == tr_size).
    #[test]
    fn lfnst_forward_inverse_round_trips_4x4() {
        let input: [i32; 16] = [
            200, -150, 90, -30, 110, -60, 25, -8, 40, -20, 10, -3, 12, -5, 2, -1,
        ];
        for set in 0..4 {
            for idx in 0..2 {
                let fwd = fwd_lfnst_nxn(&input, set, idx, 4, 16);
                let inv = inv_lfnst_nxn(&fwd[..16], set, idx, 4, 16, 15);
                let err: i64 = input
                    .iter()
                    .zip(&inv[..16])
                    .map(|(&a, &b)| (a as i64 - b as i64).abs())
                    .sum();
                assert!(
                    err < 16 * 4,
                    "set{set} idx{idx} round-trip L1 error {err} too high"
                );
            }
        }
    }
}
