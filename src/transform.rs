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

//! Spec-faithful VVC core transform (integer DCT-II), quantization, and
//! dequantization for square transform blocks of size 4, 8, 16, and 32.
//!
//! For DCT-II at these sizes the VVC transform matrices are identical to HEVC's
//! (same 6-bit-precision integer matrices), and with `COM16_C806_TRANS_PREC = 0`
//! and `MAX_TR_DYNAMIC_RANGE = 15` the VVM forward shifts reduce to
//! `log2(W) + bitDepth - 9` then `log2(H) + 6`, and the inverse shifts to `7`
//! then `20 - bitDepth` — matching HEVC exactly. The flat-scaling-list, no
//! dependent-quant, no transform-skip dequantization is likewise algebraically
//! identical to HEVC's. garnetash v1 uses none of MTS, LFNST, transform-skip,
//! scaling lists, or dependent quantization, so this single DCT-II path applies.
#![allow(dead_code)]

/// 4×4 DCT-II matrix (H.266 == H.265).
#[rustfmt::skip]
static T4: [[i8; 4]; 4] = [
    [64, 64, 64, 64],
    [83, 36, -36, -83],
    [64, -64, -64, 64],
    [36, -83, 83, -36],
];

/// 8×8 DCT-II matrix.
#[rustfmt::skip]
static T8: [[i8; 8]; 8] = [
    [64, 64, 64, 64, 64, 64, 64, 64],
    [89, 75, 50, 18, -18, -50, -75, -89],
    [83, 36, -36, -83, -83, -36, 36, 83],
    [75, -18, -89, -50, 50, 89, 18, -75],
    [64, -64, -64, 64, 64, -64, -64, 64],
    [50, -89, 18, 75, -75, -18, 89, -50],
    [36, -83, 83, -36, -36, 83, -83, 36],
    [18, -50, 75, -89, 89, -75, 50, -18],
];

/// 16×16 DCT-II matrix.
#[rustfmt::skip]
static T16: [[i8; 16]; 16] = [
    [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    [90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90],
    [89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89],
    [87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87],
    [83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83],
    [80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80],
    [75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75],
    [70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70],
    [64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64],
    [57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57],
    [50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50],
    [43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43],
    [36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36],
    [25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25],
    [18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18],
    [9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9],
];

#[rustfmt::skip]
static T32: [[i8; 32]; 32] = [
    [64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64],
    [90,90,88,85,82,78,73,67,61,54,46,38,31,22,13,4,-4,-13,-22,-31,-38,-46,-54,-61,-67,-73,-78,-82,-85,-88,-90,-90],
    [90,87,80,70,57,43,25,9,-9,-25,-43,-57,-70,-80,-87,-90,-90,-87,-80,-70,-57,-43,-25,-9,9,25,43,57,70,80,87,90],
    [90,82,67,46,22,-4,-31,-54,-73,-85,-90,-88,-78,-61,-38,-13,13,38,61,78,88,90,85,73,54,31,4,-22,-46,-67,-82,-90],
    [89,75,50,18,-18,-50,-75,-89,-89,-75,-50,-18,18,50,75,89,89,75,50,18,-18,-50,-75,-89,-89,-75,-50,-18,18,50,75,89],
    [88,67,31,-13,-54,-82,-90,-78,-46,-4,38,73,90,85,61,22,-22,-61,-85,-90,-73,-38,4,46,78,90,82,54,13,-31,-67,-88],
    [87,57,9,-43,-80,-90,-70,-25,25,70,90,80,43,-9,-57,-87,-87,-57,-9,43,80,90,70,25,-25,-70,-90,-80,-43,9,57,87],
    [85,46,-13,-67,-90,-73,-22,38,82,88,54,-4,-61,-90,-78,-31,31,78,90,61,4,-54,-88,-82,-38,22,73,90,67,13,-46,-85],
    [83,36,-36,-83,-83,-36,36,83,83,36,-36,-83,-83,-36,36,83,83,36,-36,-83,-83,-36,36,83,83,36,-36,-83,-83,-36,36,83],
    [82,22,-54,-90,-61,13,78,85,31,-46,-90,-67,4,73,88,38,-38,-88,-73,-4,67,90,46,-31,-85,-78,-13,61,90,54,-22,-82],
    [80,9,-70,-87,-25,57,90,43,-43,-90,-57,25,87,70,-9,-80,-80,-9,70,87,25,-57,-90,-43,43,90,57,-25,-87,-70,9,80],
    [78,-4,-82,-73,13,85,67,-22,-88,-61,31,90,54,-38,-90,-46,46,90,38,-54,-90,-31,61,88,22,-67,-85,-13,73,82,4,-78],
    [75,-18,-89,-50,50,89,18,-75,-75,18,89,50,-50,-89,-18,75,75,-18,-89,-50,50,89,18,-75,-75,18,89,50,-50,-89,-18,75],
    [73,-31,-90,-22,78,67,-38,-90,-13,82,61,-46,-88,-4,85,54,-54,-85,4,88,46,-61,-82,13,90,38,-67,-78,22,90,31,-73],
    [70,-43,-87,9,90,25,-80,-57,57,80,-25,-90,-9,87,43,-70,-70,43,87,-9,-90,-25,80,57,-57,-80,25,90,9,-87,-43,70],
    [67,-54,-78,38,85,-22,-90,4,90,13,-88,-31,82,46,-73,-61,61,73,-46,-82,31,88,-13,-90,-4,90,22,-85,-38,78,54,-67],
    [64,-64,-64,64,64,-64,-64,64,64,-64,-64,64,64,-64,-64,64,64,-64,-64,64,64,-64,-64,64,64,-64,-64,64,64,-64,-64,64],
    [61,-73,-46,82,31,-88,-13,90,-4,-90,22,85,-38,-78,54,67,-67,-54,78,38,-85,-22,90,4,-90,13,88,-31,-82,46,73,-61],
    [57,-80,-25,90,-9,-87,43,70,-70,-43,87,9,-90,25,80,-57,-57,80,25,-90,9,87,-43,-70,70,43,-87,-9,90,-25,-80,57],
    [54,-85,-4,88,-46,-61,82,13,-90,38,67,-78,-22,90,-31,-73,73,31,-90,22,78,-67,-38,90,-13,-82,61,46,-88,4,85,-54],
    [50,-89,18,75,-75,-18,89,-50,-50,89,-18,-75,75,18,-89,50,50,-89,18,75,-75,-18,89,-50,-50,89,-18,-75,75,18,-89,50],
    [46,-90,38,54,-90,31,61,-88,22,67,-85,13,73,-82,4,78,-78,-4,82,-73,-13,85,-67,-22,88,-61,-31,90,-54,-38,90,-46],
    [43,-90,57,25,-87,70,9,-80,80,-9,-70,87,-25,-57,90,-43,-43,90,-57,-25,87,-70,-9,80,-80,9,70,-87,25,57,-90,43],
    [38,-88,73,-4,-67,90,-46,-31,85,-78,13,61,-90,54,22,-82,82,-22,-54,90,-61,-13,78,-85,31,46,-90,67,4,-73,88,-38],
    [36,-83,83,-36,-36,83,-83,36,36,-83,83,-36,-36,83,-83,36,36,-83,83,-36,-36,83,-83,36,36,-83,83,-36,-36,83,-83,36],
    [31,-78,90,-61,4,54,-88,82,-38,-22,73,-90,67,-13,-46,85,-85,46,13,-67,90,-73,22,38,-82,88,-54,-4,61,-90,78,-31],
    [25,-70,90,-80,43,9,-57,87,-87,57,-9,-43,80,-90,70,-25,-25,70,-90,80,-43,-9,57,-87,87,-57,9,43,-80,90,-70,25],
    [22,-61,85,-90,73,-38,-4,46,-78,90,-82,54,-13,-31,67,-88,88,-67,31,13,-54,82,-90,78,-46,4,38,-73,90,-85,61,-22],
    [18,-50,75,-89,89,-75,50,-18,-18,50,-75,89,-89,75,-50,18,18,-50,75,-89,89,-75,50,-18,-18,50,-75,89,-89,75,-50,18],
    [13,-38,61,-78,88,-90,85,-73,54,-31,4,22,-46,67,-82,90,-90,82,-67,46,-22,-4,31,-54,73,-85,90,-88,78,-61,38,-13],
    [9,-25,43,-57,70,-80,87,-90,90,-87,80,-70,57,-43,25,-9,-9,25,-43,57,-70,80,-87,90,-90,87,-80,70,-57,43,-25,9],
    [4,-13,22,-31,38,-46,54,-61,67,-73,78,-82,85,-88,90,-90,90,-90,88,-85,82,-78,73,-67,61,-54,46,-38,31,-22,13,-4],
];

/// Forward quantization scales, indexed by `qp % 6` (`g_quantScales[0]`).
static QUANT_SCALE: [i64; 6] = [26214, 23302, 20560, 18396, 16384, 14564];
/// Dequantization scales, indexed by `qp % 6` (`g_invQuantScales[0]`).
static DEQUANT_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];

/// `g_quantScales[1]` / `g_InvQuantScales[1]`: the forward/inverse scales for a
/// "non-power-of-4" block — one whose `log2(width)+log2(height)` is odd. Such a
/// block's separable transform is off by a factor of √2 from the square
/// normalization, so VVC corrects it during (de)quantization by using these
/// √2-adjusted scales together with a one-bit shift change.
static QUANT_SCALE_SQRT: [i64; 6] = [18396, 16384, 14564, 13107, 11651, 10280];
static DEQUANT_SCALE_SQRT: [i64; 6] = [57, 64, 72, 80, 90, 102];

/// Row `k` of the DCT-II core matrix for transform size `size`. For the 64-point
/// matrix only the first 32 rows exist (high-frequency zero-out); callers never
/// request `k >= 32` there.
#[inline]
fn t_row(size: usize, k: usize) -> &'static [i8] {
    match size {
        4 => &T4[k],
        8 => &T8[k],
        16 => &T16[k],
        32 => &T32[k],
        64 => &T64[k],
        _ => panic!("unsupported transform size {size}"),
    }
}

/// Number of non-zero output coefficients along a dimension of length `n` after
/// VVC high-frequency zero-out (64 → 32, otherwise `n`).
#[inline]
fn kept(n: usize) -> usize {
    if n == 64 { 32 } else { n }
}

/// Largest supported transform: 32×32 → 1024 coefficients per fixed buffer.
/// VVC DCT-II 64-point core matrix, first 32 rows (extracted bit-exact from
/// vvdec's `DEFINE_DCT2_P64_MATRIX`). A 64-point transform zeroes all
/// coefficients with index >= 32, so only these 32 basis rows are ever used
/// (forward keeps outputs 0..32; inverse sums over inputs 0..32).
#[rustfmt::skip]
static T64: [[i8; 64]; 32] = [
    [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    [91, 90, 90, 90, 88, 87, 86, 84, 83, 81, 79, 77, 73, 71, 69, 65, 62, 59, 56, 52, 48, 44, 41, 37, 33, 28, 24, 20, 15, 11, 7, 2, -2, -7, -11, -15, -20, -24, -28, -33, -37, -41, -44, -48, -52, -56, -59, -62, -65, -69, -71, -73, -77, -79, -81, -83, -84, -86, -87, -88, -90, -90, -90, -91],
    [90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4, -4, -13, -22, -31, -38, -46, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90, -90, -90, -88, -85, -82, -78, -73, -67, -61, -54, -46, -38, -31, -22, -13, -4, 4, 13, 22, 31, 38, 46, 54, 61, 67, 73, 78, 82, 85, 88, 90, 90],
    [90, 88, 84, 79, 71, 62, 52, 41, 28, 15, 2, -11, -24, -37, -48, -59, -69, -77, -83, -87, -90, -91, -90, -86, -81, -73, -65, -56, -44, -33, -20, -7, 7, 20, 33, 44, 56, 65, 73, 81, 86, 90, 91, 90, 87, 83, 77, 69, 59, 48, 37, 24, 11, -2, -15, -28, -41, -52, -62, -71, -79, -84, -88, -90],
    [90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90, -90, -87, -80, -70, -57, -43, -25, -9, 9, 25, 43, 57, 70, 80, 87, 90, 90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90, -90, -87, -80, -70, -57, -43, -25, -9, 9, 25, 43, 57, 70, 80, 87, 90],
    [90, 84, 73, 59, 41, 20, -2, -24, -44, -62, -77, -86, -90, -90, -83, -71, -56, -37, -15, 7, 28, 48, 65, 79, 87, 91, 88, 81, 69, 52, 33, 11, -11, -33, -52, -69, -81, -88, -91, -87, -79, -65, -48, -28, -7, 15, 37, 56, 71, 83, 90, 90, 86, 77, 62, 44, 24, 2, -20, -41, -59, -73, -84, -90],
    [90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13, 13, 38, 61, 78, 88, 90, 85, 73, 54, 31, 4, -22, -46, -67, -82, -90, -90, -82, -67, -46, -22, 4, 31, 54, 73, 85, 90, 88, 78, 61, 38, 13, -13, -38, -61, -78, -88, -90, -85, -73, -54, -31, -4, 22, 46, 67, 82, 90],
    [90, 79, 59, 33, 2, -28, -56, -77, -88, -90, -81, -62, -37, -7, 24, 52, 73, 87, 90, 83, 65, 41, 11, -20, -48, -71, -86, -91, -84, -69, -44, -15, 15, 44, 69, 84, 91, 86, 71, 48, 20, -11, -41, -65, -83, -90, -87, -73, -52, -24, 7, 37, 62, 81, 90, 88, 77, 56, 28, -2, -33, -59, -79, -90],
    [89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89, 89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89, 89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89, 89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89],
    [88, 71, 41, 2, -37, -69, -87, -90, -73, -44, -7, 33, 65, 86, 90, 77, 48, 11, -28, -62, -84, -90, -79, -52, -15, 24, 59, 83, 91, 81, 56, 20, -20, -56, -81, -91, -83, -59, -24, 15, 52, 79, 90, 84, 62, 28, -11, -48, -77, -90, -86, -65, -33, 7, 44, 73, 90, 87, 69, 37, -2, -41, -71, -88],
    [88, 67, 31, -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85, 61, 22, -22, -61, -85, -90, -73, -38, 4, 46, 78, 90, 82, 54, 13, -31, -67, -88, -88, -67, -31, 13, 54, 82, 90, 78, 46, 4, -38, -73, -90, -85, -61, -22, 22, 61, 85, 90, 73, 38, -4, -46, -78, -90, -82, -54, -13, 31, 67, 88],
    [87, 62, 20, -28, -69, -90, -84, -56, -11, 37, 73, 90, 81, 48, 2, -44, -79, -91, -77, -41, 7, 52, 83, 90, 71, 33, -15, -59, -86, -88, -65, -24, 24, 65, 88, 86, 59, 15, -33, -71, -90, -83, -52, -7, 41, 77, 91, 79, 44, -2, -48, -81, -90, -73, -37, 11, 56, 84, 90, 69, 28, -20, -62, -87],
    [87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87, -87, -57, -9, 43, 80, 90, 70, 25, -25, -70, -90, -80, -43, 9, 57, 87, 87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87, -87, -57, -9, 43, 80, 90, 70, 25, -25, -70, -90, -80, -43, 9, 57, 87],
    [86, 52, -2, -56, -87, -84, -48, 7, 59, 88, 83, 44, -11, -62, -90, -81, -41, 15, 65, 90, 79, 37, -20, -69, -90, -77, -33, 24, 71, 91, 73, 28, -28, -73, -91, -71, -24, 33, 77, 90, 69, 20, -37, -79, -90, -65, -15, 41, 81, 90, 62, 11, -44, -83, -88, -59, -7, 48, 84, 87, 56, 2, -52, -86],
    [85, 46, -13, -67, -90, -73, -22, 38, 82, 88, 54, -4, -61, -90, -78, -31, 31, 78, 90, 61, 4, -54, -88, -82, -38, 22, 73, 90, 67, 13, -46, -85, -85, -46, 13, 67, 90, 73, 22, -38, -82, -88, -54, 4, 61, 90, 78, 31, -31, -78, -90, -61, -4, 54, 88, 82, 38, -22, -73, -90, -67, -13, 46, 85],
    [84, 41, -24, -77, -90, -56, 7, 65, 91, 69, 11, -52, -88, -79, -28, 37, 83, 86, 44, -20, -73, -90, -59, 2, 62, 90, 71, 15, -48, -87, -81, -33, 33, 81, 87, 48, -15, -71, -90, -62, -2, 59, 90, 73, 20, -44, -86, -83, -37, 28, 79, 88, 52, -11, -69, -91, -65, -7, 56, 90, 77, 24, -41, -84],
    [83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83],
    [83, 28, -44, -88, -73, -11, 59, 91, 62, -7, -71, -90, -48, 24, 81, 84, 33, -41, -87, -77, -15, 56, 90, 65, -2, -69, -90, -52, 20, 79, 86, 37, -37, -86, -79, -20, 52, 90, 69, 2, -65, -90, -56, 15, 77, 87, 41, -33, -84, -81, -24, 48, 90, 71, 7, -62, -91, -59, 11, 73, 88, 44, -28, -83],
    [82, 22, -54, -90, -61, 13, 78, 85, 31, -46, -90, -67, 4, 73, 88, 38, -38, -88, -73, -4, 67, 90, 46, -31, -85, -78, -13, 61, 90, 54, -22, -82, -82, -22, 54, 90, 61, -13, -78, -85, -31, 46, 90, 67, -4, -73, -88, -38, 38, 88, 73, 4, -67, -90, -46, 31, 85, 78, 13, -61, -90, -54, 22, 82],
    [81, 15, -62, -90, -44, 37, 88, 69, -7, -77, -84, -24, 56, 91, 52, -28, -86, -73, -2, 71, 87, 33, -48, -90, -59, 20, 83, 79, 11, -65, -90, -41, 41, 90, 65, -11, -79, -83, -20, 59, 90, 48, -33, -87, -71, 2, 73, 86, 28, -52, -91, -56, 24, 84, 77, 7, -69, -88, -37, 44, 90, 62, -15, -81],
    [80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80, -80, -9, 70, 87, 25, -57, -90, -43, 43, 90, 57, -25, -87, -70, 9, 80, 80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80, -80, -9, 70, 87, 25, -57, -90, -43, 43, 90, 57, -25, -87, -70, 9, 80],
    [79, 2, -77, -81, -7, 73, 83, 11, -71, -84, -15, 69, 86, 20, -65, -87, -24, 62, 88, 28, -59, -90, -33, 56, 90, 37, -52, -90, -41, 48, 91, 44, -44, -91, -48, 41, 90, 52, -37, -90, -56, 33, 90, 59, -28, -88, -62, 24, 87, 65, -20, -86, -69, 15, 84, 71, -11, -83, -73, 7, 81, 77, -2, -79],
    [78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 31, 90, 54, -38, -90, -46, 46, 90, 38, -54, -90, -31, 61, 88, 22, -67, -85, -13, 73, 82, 4, -78, -78, 4, 82, 73, -13, -85, -67, 22, 88, 61, -31, -90, -54, 38, 90, 46, -46, -90, -38, 54, 90, 31, -61, -88, -22, 67, 85, 13, -73, -82, -4, 78],
    [77, -11, -86, -62, 33, 90, 44, -52, -90, -24, 69, 83, 2, -81, -71, 20, 88, 56, -41, -91, -37, 59, 87, 15, -73, -79, 7, 84, 65, -28, -90, -48, 48, 90, 28, -65, -84, -7, 79, 73, -15, -87, -59, 37, 91, 41, -56, -88, -20, 71, 81, -2, -83, -69, 24, 90, 52, -44, -90, -33, 62, 86, 11, -77],
    [75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75, 75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75, 75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75, 75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75],
    [73, -24, -90, -37, 65, 81, -11, -88, -48, 56, 86, 2, -84, -59, 44, 90, 15, -79, -69, 33, 91, 28, -71, -77, 20, 90, 41, -62, -83, 7, 87, 52, -52, -87, -7, 83, 62, -41, -90, -20, 77, 71, -28, -91, -33, 69, 79, -15, -90, -44, 59, 84, -2, -86, -56, 48, 88, 11, -81, -65, 37, 90, 24, -73],
    [73, -31, -90, -22, 78, 67, -38, -90, -13, 82, 61, -46, -88, -4, 85, 54, -54, -85, 4, 88, 46, -61, -82, 13, 90, 38, -67, -78, 22, 90, 31, -73, -73, 31, 90, 22, -78, -67, 38, 90, 13, -82, -61, 46, 88, 4, -85, -54, 54, 85, -4, -88, -46, 61, 82, -13, -90, -38, 67, 78, -22, -90, -31, 73],
    [71, -37, -90, -7, 86, 48, -62, -79, 24, 91, 20, -81, -59, 52, 84, -11, -90, -33, 73, 69, -41, -88, -2, 87, 44, -65, -77, 28, 90, 15, -83, -56, 56, 83, -15, -90, -28, 77, 65, -44, -87, 2, 88, 41, -69, -73, 33, 90, 11, -84, -52, 59, 81, -20, -91, -24, 79, 62, -48, -86, 7, 90, 37, -71],
    [70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70, -70, 43, 87, -9, -90, -25, 80, 57, -57, -80, 25, 90, 9, -87, -43, 70, 70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70, -70, 43, 87, -9, -90, -25, 80, 57, -57, -80, 25, 90, 9, -87, -43, 70],
    [69, -48, -83, 24, 90, 2, -90, -28, 81, 52, -65, -71, 44, 84, -20, -90, -7, 88, 33, -79, -56, 62, 73, -41, -86, 15, 91, 11, -87, -37, 77, 59, -59, -77, 37, 87, -11, -91, -15, 86, 41, -73, -62, 56, 79, -33, -88, 7, 90, 20, -84, -44, 71, 65, -52, -81, 28, 90, -2, -90, -24, 83, 48, -69],
    [67, -54, -78, 38, 85, -22, -90, 4, 90, 13, -88, -31, 82, 46, -73, -61, 61, 73, -46, -82, 31, 88, -13, -90, -4, 90, 22, -85, -38, 78, 54, -67, -67, 54, 78, -38, -85, 22, 90, -4, -90, -13, 88, 31, -82, -46, 73, 61, -61, -73, 46, 82, -31, -88, 13, 90, 4, -90, -22, 85, 38, -78, -54, 67],
    [65, -59, -71, 52, 77, -44, -81, 37, 84, -28, -87, 20, 90, -11, -90, 2, 91, 7, -90, -15, 88, 24, -86, -33, 83, 41, -79, -48, 73, 56, -69, -62, 62, 69, -56, -73, 48, 79, -41, -83, 33, 86, -24, -88, 15, 90, -7, -91, -2, 90, 11, -90, -20, 87, 28, -84, -37, 81, 44, -77, -52, 71, 59, -65],
];

pub(crate) const MAX_TB: usize = 4096;

/// Forward integer transform of an N×N residual block (N ∈ {4,8,16,32}).
/// Returns a fixed buffer; only the first `n * n` entries are written.
pub(crate) fn fwd_transform(res: &[i32], n: usize, bit_depth: u8) -> [i32; MAX_TB] {
    let mut out = [0i32; MAX_TB];
    fwd_transform_into(&mut out, res, n, bit_depth);
    out
}

/// Forward transform writing coefficients into `out[..n*n]` without allocating a
/// `MAX_TB` buffer. For `n ≤ 32` the whole `n×n` is written; the 64-point case
/// only fills the surviving 32×32, so the high-frequency zero-out is applied
/// explicitly here (the caller's buffer may be reused and non-zero).
pub(crate) fn fwd_transform_into(out: &mut [i32], res: &[i32], n: usize, bit_depth: u8) {
    match n {
        4 => fwd_transform_n::<4>(res, &T4, bit_depth, out),
        8 => fwd_transform_n::<8>(res, &T8, bit_depth, out),
        16 => fwd_transform_n::<16>(res, &T16, bit_depth, out),
        32 => fwd_transform_n::<32>(res, &T32, bit_depth, out),
        64 => {
            out[..64 * 64].fill(0);
            fwd_transform_64(res, bit_depth, out);
        }
        _ => panic!("unsupported transform size {n}"),
    }
}

/// Forward 64-point DCT-II with VVC high-frequency zeroing: coefficients with
/// row or column index >= 32 are zeroed, so the result is a 64×64 block whose
/// only non-zero region is the top-left 32×32. Mirrors the generic shift
/// schedule with log2n = 6.
fn fwd_transform_64(res: &[i32], bit_depth: u8, out: &mut [i32]) {
    const N: usize = 64;
    let log2n = 6i32;
    let shift1 = log2n + bit_depth as i32 - 9;
    let add1 = if shift1 > 0 { 1i32 << (shift1 - 1) } else { 0 };
    let mut tmp = [0i32; MAX_TB];
    // pass 1 (rows) — keep only the first 32 outputs (the rest are zeroed).
    for j in 0..N {
        let row = &res[j * N..j * N + N];
        for i in 0..32 {
            let trow = &T64[i];
            let s: i32 = trow.iter().zip(row).map(|(&t, &r)| t as i32 * r).sum();
            tmp[j * N + i] = if shift1 > 0 { (s + add1) >> shift1 } else { s };
        }
    }
    // pass 2 (columns) — only the 32 surviving columns, 32 outputs each.
    let shift2 = log2n + 6;
    let add2 = 1i32 << (shift2 - 1);
    let mut colv = [0i32; N];
    for j in 0..32 {
        for (k, cv) in colv.iter_mut().enumerate() {
            *cv = tmp[k * N + j];
        }
        for i in 0..32 {
            let trow = &T64[i];
            let s: i32 = trow.iter().zip(&colv).map(|(&t, &c)| t as i32 * c).sum();
            out[i * N + j] = (s + add2) >> shift2;
        }
    }
}

#[inline]
fn fwd_transform_n<const N: usize>(res: &[i32], t: &[[i8; N]; N], bit_depth: u8, out: &mut [i32]) {
    let log2n = N.trailing_zeros() as i32;
    let shift1 = log2n + bit_depth as i32 - 9;
    let add1 = if shift1 > 0 { 1i32 << (shift1 - 1) } else { 0 };
    let mut tmp = [[0i32; N]; N];
    // pass 1 (rows): tmp[j][i] = (Σ_k T[i][k]·res[j*N+k]) >> shift1. The dense
    // inner product auto-vectorizes well under opt-level 3 + LTO; an even/odd
    // partial butterfly was tried and measured neutral, so the simpler form
    // is kept.
    for j in 0..N {
        let row = &res[j * N..j * N + N];
        for (i, trow) in t.iter().enumerate() {
            let s: i32 = trow.iter().zip(row).map(|(&t, &r)| t as i32 * r).sum();
            tmp[j][i] = if shift1 > 0 { (s + add1) >> shift1 } else { s };
        }
    }
    // pass 2 (columns): out[i*N+j] = (Σ_k T[i][k]·tmp[k][j]) >> shift2
    let shift2 = log2n + 6;
    let add2 = 1i32 << (shift2 - 1);
    let mut colv = [0i32; N];
    for j in 0..N {
        for (k, cv) in colv.iter_mut().enumerate() {
            *cv = tmp[k][j];
        }
        for (i, trow) in t.iter().enumerate() {
            let s: i32 = trow.iter().zip(&colv).map(|(&t, &c)| t as i32 * c).sum();
            out[i * N + j] = (s + add2) >> shift2;
        }
    }
}

/// Forward quantization: transform coefficient → level (intra rounding offset).
/// Forward integer transform of a `w × h` residual block, `w` and `h` each in
/// {4,8,16,32,64} (possibly unequal — chroma blocks under 4:2:2 are half-width).
/// The horizontal pass uses the size-`w` matrix, the vertical pass the size-`h`
/// matrix; a 64-length dimension keeps only its first 32 coefficients. The
/// coefficient block is stored with stride `w`.
pub(crate) fn fwd_transform_wh(res: &[i32], w: usize, h: usize, bit_depth: u8) -> [i32; MAX_TB] {
    let mut out = [0i32; MAX_TB];
    fwd_transform_wh_into(&mut out, res, w, h, bit_depth);
    out
}

/// Forward transform writing coefficients into `out[..w*h]` without allocating a
/// `MAX_TB` buffer. Square blocks take the fast const-generic path; the runtime
/// rectangular path zero-fills first so non-kept (high-frequency) positions in a
/// reused buffer are correct.
pub(crate) fn fwd_transform_wh_into(
    out: &mut [i32],
    res: &[i32],
    w: usize,
    h: usize,
    bit_depth: u8,
) {
    // Square blocks (every luma TU and all 4:2:0 chroma) dominate, and the
    // const-generic path fully unrolls/vectorises with a compile-time size —
    // measured ~1.7–2.4× faster than the runtime loop and bit-identical.
    if w == h {
        fwd_transform_into(out, res, w, bit_depth);
        return;
    }
    out[..w * h].fill(0);
    let (kw, kh) = (kept(w), kept(h));
    let log2w = w.trailing_zeros() as i32;
    let log2h = h.trailing_zeros() as i32;
    // Pass 1 — horizontal (size w), keep kw outputs per row.
    let shift1 = log2w + bit_depth as i32 - 9;
    let add1 = if shift1 > 0 { 1i32 << (shift1 - 1) } else { 0 };
    let mut tmp = [0i32; MAX_TB];
    for j in 0..h {
        let row = &res[j * w..j * w + w];
        for i in 0..kw {
            let trow = t_row(w, i);
            let mut s = 0i32;
            for k in 0..w {
                s += trow[k] as i32 * row[k];
            }
            tmp[j * w + i] = if shift1 > 0 { (s + add1) >> shift1 } else { s };
        }
    }
    // Pass 2 — vertical (size h), over the kw surviving columns, keep kh outputs.
    let shift2 = log2h + 6;
    let add2 = 1i32 << (shift2 - 1);
    let mut colv = [0i32; 64];
    for i in 0..kw {
        for (k, cv) in colv.iter_mut().enumerate().take(h) {
            *cv = tmp[k * w + i];
        }
        for o in 0..kh {
            let trow = t_row(h, o);
            let mut s = 0i32;
            for k in 0..h {
                s += trow[k] as i32 * colv[k];
            }
            out[o * w + i] = (s + add2) >> shift2;
        }
    }
}

/// Inverse integer transform of a `w × h` coefficient block whose non-zero
/// region is the top-left `kept(w) × kept(h)`. Mirrors [`fwd_transform_wh`];
/// stage shifts are size-independent (7, then 20−bitDepth) as in VVC.
/// Inverse transform dispatcher. Square sizes (the common 4:2:0 / 4:4:4 / luma
/// case) go through a const-generic kernel so the compiler knows every loop
/// bound at compile time: it elides bounds checks and auto-vectorises the inner
/// accumulation without any `unsafe`. Rectangular blocks (4:2:2 chroma) use the
/// dynamic kernel. Both are bit-identical to each other and to the prior code.
pub(crate) fn inv_transform_wh(coeff: &[i32], w: usize, h: usize, bit_depth: u8) -> [i32; MAX_TB] {
    let mut out = [0i32; MAX_TB];
    inv_transform_wh_into(&mut out, coeff, w, h, bit_depth);
    out
}

/// Inverse transform writing the `w×h` residual into `out[..w*h]` (fully
/// overwritten, so no pre-zeroing is required). Square blocks take the fast
/// const-generic path; rectangular 4:2:2 blocks use the dynamic loop.
pub(crate) fn inv_transform_wh_into(
    out: &mut [i32],
    coeff: &[i32],
    w: usize,
    h: usize,
    bit_depth: u8,
) {
    if w == h {
        inv_transform_into(out, coeff, w, bit_depth);
        return;
    }
    inv_transform_dyn(coeff, w, h, bit_depth, out);
}

/// Const-generic square inverse transform. `rows` is the forward DCT-II matrix
/// (its first `kept(N)` rows are the only basis vectors a conformant stream can
/// excite; for N=64 the array already holds exactly those 32 rows). Two
/// separable passes, transposing through `tmp`, identical in arithmetic to
/// [`inv_transform_dyn`]. Every array here has a compile-time length `N`, so the
/// hot `acc[n] += rows[k][n] * c` loop carries no bounds checks.
/// Dynamic (any `w×h`) inverse transform; used for rectangular 4:2:2 blocks.
fn inv_transform_dyn(coeff: &[i32], w: usize, h: usize, bit_depth: u8, out: &mut [i32]) {
    let (kw, kh) = (kept(w), kept(h));
    let clip = |v: i64| v.clamp(-32768, 32767) as i32;
    let shift1 = 7i32;
    let add1 = 1i32 << (shift1 - 1);
    let mut tmp = [0i32; MAX_TB];
    for i in 0..kw {
        let mut acc = [0i32; 64];
        for k in 0..kh {
            let c = coeff[k * w + i];
            if c != 0 {
                let trow = t_row(h, k);
                for (n, a) in acc.iter_mut().enumerate().take(h) {
                    *a += trow[n] as i32 * c;
                }
            }
        }
        for n in 0..h {
            tmp[n * w + i] = clip(((acc[n] + add1) >> shift1) as i64);
        }
    }
    let shift2 = 20 - bit_depth as i32;
    let add2 = 1i32 << (shift2 - 1);
    for row in 0..h {
        let mut acc = [0i32; 64];
        for k in 0..kw {
            let c = tmp[row * w + k];
            if c != 0 {
                let trow = t_row(w, k);
                for (n, a) in acc.iter_mut().enumerate().take(w) {
                    *a += trow[n] as i32 * c;
                }
            }
        }
        for n in 0..w {
            out[row * w + n] = (acc[n] + add2) >> shift2;
        }
    }
}

pub(crate) fn quantize_wh(
    coeff: &[i32],
    w: usize,
    h: usize,
    qp: u8,
    bit_depth: u8,
) -> [i16; MAX_TB] {
    let log2w = w.trailing_zeros() as i64;
    let log2h = h.trailing_zeros() as i64;
    let sqrt2 = (log2w + log2h) & 1 == 1;
    let avg = (log2w + log2h) >> 1;
    let q_bits = 14 + (qp as i64) / 6 + (15 - bit_depth as i64 - avg) - i64::from(sqrt2);
    let q_scale = if sqrt2 { QUANT_SCALE_SQRT } else { QUANT_SCALE }[(qp % 6) as usize];
    let offset = 171i64 << (q_bits - 9);
    let mut out = [0i16; MAX_TB];
    for (o, &c) in out[..w * h].iter_mut().zip(coeff) {
        let c = c as i64;
        let level = (c.abs() * q_scale + offset) >> q_bits;
        let level = if c < 0 { -level } else { level };
        *o = level.clamp(-32768, 32767) as i16;
    }
    out
}

/// Spatial SSD produced by a unit coefficient-domain error, measured from the
/// actual inverse transform (impulse-response energy, averaged over a spread of
/// frequency positions). Because the transform is linear and near-orthonormal,
/// quantization SSD ≈ Σ (coeff − dequant(level))² · `err_scale`, so RDOQ can use
/// the same spatial λ as mode decision without ever reconstructing a block.
/// QP-independent; depends only on `(w, h, bit_depth)`, so it is memoized.
pub(crate) fn err_scale(w: usize, h: usize, bit_depth: u8) -> f64 {
    thread_local! {
        static CACHE: std::cell::RefCell<std::collections::HashMap<(usize, usize, u8), f64>> =
            std::cell::RefCell::new(std::collections::HashMap::new());
    }
    CACHE.with(|c| {
        *c.borrow_mut().entry((w, h, bit_depth)).or_insert_with(|| {
            // A moderate amplitude keeps the integer inverse transform in its
            // linear range (no rounding-to-zero, no clipping).
            let a = 1024i32;
            let positions = [
                (0, 0),
                (w / 2, h / 2),
                (w - 1, 0),
                (0, h - 1),
                (w - 1, h - 1),
                ((w / 4).max(1), (h / 4).max(1)),
            ];
            let mut acc = 0.0f64;
            for &(u, v) in &positions {
                let mut coeff = [0i32; MAX_TB];
                coeff[v * w + u] = a;
                let spatial = inv_transform_wh(&coeff[..w * h], w, h, bit_depth);
                let e: f64 = spatial[..w * h]
                    .iter()
                    .map(|&s| (s as f64) * (s as f64))
                    .sum();
                acc += e / (a as f64 * a as f64);
            }
            acc / positions.len() as f64
        })
    })
}

fn est_coeff_bits(level: i64) -> f64 {
    if level <= 0 {
        return 0.0;
    }
    let mut b = 2.0; // sig_coeff_flag + sign
    b += 1.0; // abs_level_gt1_flag
    if level == 1 {
        return b;
    }
    b += 2.0; // par_level_flag + abs_level_gt3_flag
    if level <= 3 {
        return b;
    }
    let rem = (level - 4) as f64;
    b += 2.0 * (rem + 1.0).log2().floor() + 1.0; // Golomb-Rice remainder
    b
}

fn last_pos_bits(x: usize, y: usize) -> f64 {
    fn coord(c: usize) -> f64 {
        let g = crate::residual::last_group_idx(c) as f64;
        // ~g prefix bins + bypass suffix bits when g > 3
        g + if g > 3.0 {
            ((g - 2.0) * 0.5).floor()
        } else {
            0.0
        }
    }
    coord(x) + coord(y)
}

pub(crate) fn rdoq_wh(
    coeff: &[i32],
    w: usize,
    h: usize,
    qp: u8,
    bit_depth: u8,
    lambda: f64,
) -> [i16; MAX_TB] {
    let log2w = w.trailing_zeros() as i64;
    let log2h = h.trailing_zeros() as i64;
    let sqrt2 = (log2w + log2h) & 1 == 1;
    let avg = (log2w + log2h) >> 1;
    let q_bits = 14 + (qp as i64) / 6 + (15 - bit_depth as i64 - avg) - i64::from(sqrt2);
    let q_scale = if sqrt2 { QUANT_SCALE_SQRT } else { QUANT_SCALE }[(qp % 6) as usize];
    let offset = 171i64 << (q_bits - 9);
    let bd_shift = bit_depth as i64 + avg - 5 + i64::from(sqrt2);
    let dq_add = 1i64 << (bd_shift - 1);
    let dq_scale = if sqrt2 {
        DEQUANT_SCALE_SQRT
    } else {
        DEQUANT_SCALE
    }[(qp % 6) as usize];
    let dq_factor = dq_scale * (1i64 << ((qp as i64) / 6)) * 16;
    let deq = |level: i64| -> f64 { ((level * dq_factor + dq_add) >> bd_shift) as f64 };
    let scale = err_scale(w, h, bit_depth);
    let nn = w * h;

    // Pass 1: per-coefficient level, plus distortion of keep vs zero.
    let mut level = [0i64; MAX_TB];
    let mut dist_keep = [0f64; MAX_TB];
    let mut dist_zero = [0f64; MAX_TB];
    let mut bits_keep = [0f64; MAX_TB];
    for i in 0..nn {
        let c = coeff[i] as i64;
        let cf = c.abs() as f64;
        let l_plain = ((c.abs() * q_scale + offset) >> q_bits).min(32767);
        let dz = cf * cf * scale;
        let mut bl = 0i64;
        let mut bcost = dz; // level 0
        for cand in [l_plain - 1, l_plain] {
            if cand <= 0 {
                continue;
            }
            let e = cf - deq(cand);
            let cost = e * e * scale + lambda * est_coeff_bits(cand);
            if cost < bcost {
                bcost = cost;
                bl = cand;
            }
        }
        level[i] = if c < 0 { -bl } else { bl };
        dist_zero[i] = dz;
        if bl > 0 {
            let e = cf - deq(bl);
            dist_keep[i] = e * e * scale;
            bits_keep[i] = est_coeff_bits(bl);
        } else {
            dist_keep[i] = dz;
        }
    }

    // Pass 2: last-significant-position RD. Walk scan order; the RD of declaring
    // scan index L the last kept coefficient is (prefix kept cost) + (tail forced
    // to zero) + λ·last-position bits. Everything after the best L is zeroed.
    let scan = crate::residual::scan_coords(w, h);
    let m = scan.len();
    // tail_zero[j] = Σ_{p>=j} dist_zero of scanned coeff (cost if zeroed from j on)
    let mut tail_zero_buf = [0f64; MAX_TB + 1];
    let tail_zero = &mut tail_zero_buf[..m + 1];
    for j in (0..m).rev() {
        let (x, y) = scan[j];
        tail_zero[j] = tail_zero[j + 1] + dist_zero[y * w + x];
    }
    let mut prefix_dist = 0f64; // Σ_{p<=L} chosen-level distortion
    let mut prefix_bits = 0f64; // Σ_{p<=L} level bits
    let mut best_rd = tail_zero[0]; // all-zero block (no last position coded)
    let mut best_last: isize = -1;
    for (j, &(x, y)) in scan.iter().enumerate() {
        let idx = y * w + x;
        prefix_dist += dist_keep[idx];
        prefix_bits += bits_keep[idx];
        if level[idx] != 0 {
            let rd = prefix_dist + tail_zero[j + 1] + lambda * (prefix_bits + last_pos_bits(x, y));
            if rd < best_rd {
                best_rd = rd;
                best_last = j as isize;
            }
        }
    }
    // Zero every coefficient after the chosen last position.
    for &(x, y) in &scan[((best_last + 1) as usize)..m] {
        level[y * w + x] = 0;
    }

    let mut out = [0i16; MAX_TB];
    for (o, &l) in out[..nn].iter_mut().zip(&level[..nn]) {
        *o = l.clamp(-32768, 32767) as i16;
    }
    out
}

/// Dependent-quantization trellis (encoder-only): chooses the level sequence
/// minimising joint `Σ SSD·errScale + λ·rate` over the VVC 4-state dependent
/// quantizer, by Viterbi search in reverse scan order.
///
/// The quantizer at each position is `state>>1` (Q0/Q1), where the state is
/// driven by the parity of the levels coded at higher scan indices. Because the
/// reconstruction of a coefficient therefore depends on its predecessors, the
/// rounding direction at one position changes the optimal quantizer downstream
/// — the joint optimum the trellis finds is what beats independent scalar
/// rounding. State inits 0 at the highest scan index (only state 0 is a valid
/// start); leading zeros keep it there via the parity-0 self-loop, so the
/// effective last-significant position falls out of the search automatically.
///
/// Output is an ordinary signed-level raster array; the decoder reconstructs it
/// with [`dequantize_dq_wh`], which walks the identical state machine — so the
/// two agree by construction regardless of how the search chose the levels.
pub(crate) fn dq_trellis_wh(
    coeff: &[i32],
    w: usize,
    h: usize,
    qp: u8,
    bit_depth: u8,
    lambda: f64,
) -> Vec<i32> {
    let nn = w * h;
    let scan = crate::residual::scan_coords(w, h);
    let n = scan.len();
    let mut levels = vec![0i32; nn];

    // Forward-quant scale (level grid). Dependent quant quantises on the
    // qpDQ = qp+1 grid (VTM DepQuant.cpp initQuantBlock: qpDQ = Qp()+1,
    // m_QScale = g_quantScales[qpRem], m_QShift uses qpPer = qpDQ/6), matching
    // the qp+1 reconstruction grid so forward and inverse are consistent.
    let log2w = w.trailing_zeros() as i64;
    let log2h = h.trailing_zeros() as i64;
    let sqrt2 = (log2w + log2h) & 1 == 1;
    let avg = (log2w + log2h) >> 1;
    let qp_dq = qp as i64 + 1;
    let q_bits = 14 + qp_dq / 6 + (15 - bit_depth as i64 - avg) - i64::from(sqrt2);
    let q_scale = if sqrt2 { QUANT_SCALE_SQRT } else { QUANT_SCALE }[(qp_dq % 6) as usize];
    // Dequant scale (reconstruction), mirroring `dequantize_dq_wh`.
    let bd_shift = bit_depth as i64 + avg - 5 + i64::from(sqrt2);
    let dq_add = 1i64 << bd_shift;
    let dq_scale = if sqrt2 {
        DEQUANT_SCALE_SQRT
    } else {
        DEQUANT_SCALE
    }[(qp_dq % 6) as usize];
    let dq_factor = dq_scale * (1i64 << (qp_dq / 6)) * 16;
    let scale = err_scale(w, h, bit_depth);

    // Reconstruction magnitude for an abs level `l` under quantizer `q` (0/1).
    let dq_recon = |l: i64, q: i64| -> i64 {
        let qidx = if l == 0 { 0 } else { 2 * l - q };
        (qidx * dq_factor + dq_add) >> (bd_shift + 1)
    };

    const INF: f64 = f64::INFINITY;
    // dp[s] = best cost of a path whose state *arriving* at the current position
    // is `s`. Only state 0 is reachable before the highest scan index.
    let mut dp = [0.0f64, INF, INF, INF];
    // back[pos][s'] = (incoming_state, abs_level) chosen to reach outgoing s'.
    let mut back = vec![[(0u8, 0i32); 4]; n];

    for pos in (0..n).rev() {
        let (x, y) = scan[pos];
        let c = coeff[y * w + x] as i64;
        let ac = c.abs();
        let scaled = ac * q_scale;
        let mut ndp = [INF; 4];
        let bp = &mut back[pos];
        #[allow(clippy::needless_range_loop)]
        for s in 0..4usize {
            let base = dp[s];
            if base == INF {
                continue;
            }
            let q = (s >> 1) as i64;
            // Candidate abs levels for this quantizer: zero, plus the two grid
            // levels bracketing the scaled coefficient (Q1 grid is offset half a
            // step). RD picks among them.
            let l_lo = ((scaled + (q << (q_bits - 1).max(0))) >> q_bits).max(0);
            let cands = [0i64, l_lo, l_lo + 1];
            for &l in &cands {
                let recon = dq_recon(l, q);
                let e = (ac - recon) as f64;
                let dist = e * e * scale;
                let rate = est_coeff_bits(l);
                let cost = base + dist + lambda * rate;
                let s2 = crate::depquant::next_state(s as u8, l as i32) as usize;
                if cost < ndp[s2] {
                    ndp[s2] = cost;
                    bp[s2] = (s as u8, l as i32);
                }
            }
        }
        dp = ndp;
    }

    // Best final state, then backtrack low->high recovering the level sequence.
    let mut s_cur = (0..4)
        .min_by(|&a, &b| dp[a].partial_cmp(&dp[b]).unwrap())
        .unwrap();
    for pos in 0..n {
        let (incoming, lvl) = back[pos][s_cur];
        let (x, y) = scan[pos];
        let c = coeff[y * w + x];
        levels[y * w + x] = if c < 0 { -lvl } else { lvl };
        s_cur = incoming as usize;
    }
    levels
}

/// Inverse quantization of a `w × h` block (companion to [`quantize_wh`]).
pub(crate) fn dequantize_wh(
    level: &[i16],
    w: usize,
    h: usize,
    qp: u8,
    bit_depth: u8,
) -> [i32; MAX_TB] {
    let mut out = [0i32; MAX_TB];
    dequantize_wh_into(&mut out, level, w, h, qp, bit_depth);
    out
}

/// As [`dequantize_wh`] but writes the `w*h` coefficients into a caller-provided
/// buffer, avoiding the full 16 KiB zero-fill + return of the fixed-size array on
/// every RDO candidate. Only `out[..w*h]` is touched.
pub(crate) fn dequantize_wh_into(
    out: &mut [i32],
    level: &[i16],
    w: usize,
    h: usize,
    qp: u8,
    bit_depth: u8,
) {
    let log2w = w.trailing_zeros() as i64;
    let log2h = h.trailing_zeros() as i64;
    let sqrt2 = (log2w + log2h) & 1 == 1;
    let avg = (log2w + log2h) >> 1;
    let bd_shift = bit_depth as i64 + avg - 5 + i64::from(sqrt2);
    let add = 1i64 << (bd_shift - 1);
    let scale = if sqrt2 {
        DEQUANT_SCALE_SQRT
    } else {
        DEQUANT_SCALE
    }[(qp % 6) as usize];
    let per = 1i64 << ((qp as i64) / 6);
    let factor = scale * per * 16;
    let nn = w * h;
    // Positions beyond `level` (if any) and the unused part of the scan must read
    // as zero, matching the array version's initial zero-fill.
    for o in out[..nn].iter_mut() {
        *o = 0;
    }
    for (o, &l) in out[..nn].iter_mut().zip(level) {
        *o = ((l as i64 * factor + add) >> bd_shift).clamp(-32768, 32767) as i32;
    }
}

/// Dependent-quantization dequantizer. Reconstructs transform coefficients from
/// coded levels by walking the scan in reverse, maintaining the 4-state DQ
/// quantizer (state inits 0 at the highest scan index; leading zeros keep it
/// there via the parity-0 self-loop). Reuses `dequantize_wh`'s scale, but with
/// the doubled reconstruction index `qIdx = 2·level − sign·(state>>1)` and a
/// shift of `bd_shift+1` — so a Q0 (even-state) coefficient reconstructs
/// bit-identically to scalar dequant, while Q1 (odd-state) shifts by half a
/// step. `scan` must be the same scan residual coding uses (`scan_coords`).
pub(crate) fn dequantize_dq_wh(
    level: &[i32],
    scan: &[(usize, usize)],
    w: usize,
    h: usize,
    qp: u8,
    bit_depth: u8,
) -> [i32; MAX_TB] {
    let mut out = [0i32; MAX_TB];
    dequantize_dq_wh_into(&mut out, level, scan, w, h, qp, bit_depth);
    out
}

/// As [`dequantize_dq_wh`] but writes into a caller-provided buffer. `scan` is a
/// permutation of every `w*h` position, so all of `out[..w*h]` is written.
pub(crate) fn dequantize_dq_wh_into(
    out: &mut [i32],
    level: &[i32],
    scan: &[(usize, usize)],
    w: usize,
    h: usize,
    qp: u8,
    bit_depth: u8,
) {
    let log2w = w.trailing_zeros() as i64;
    let log2h = h.trailing_zeros() as i64;
    let sqrt2 = (log2w + log2h) & 1 == 1;
    let avg = (log2w + log2h) >> 1;
    let bd_shift = bit_depth as i64 + avg - 5 + i64::from(sqrt2);
    let add = 1i64 << bd_shift; // (1 << (bd_shift+1)) >> 1
    let qp_dq = qp as i64 + 1;
    let scale = if sqrt2 {
        DEQUANT_SCALE_SQRT
    } else {
        DEQUANT_SCALE
    }[(qp_dq % 6) as usize];
    let per = 1i64 << (qp_dq / 6);
    let factor = scale * per * 16;
    let mut state = 0u8;
    for &(x, y) in scan.iter().rev() {
        let l = level[y * w + x];
        let qidx = crate::depquant::recon_qidx(l, state) as i64;
        out[y * w + x] = ((qidx * factor + add) >> (bd_shift + 1)).clamp(-32768, 32767) as i32;
        state = crate::depquant::next_state(state, l);
    }
}

/// Forward quantization: transform coefficient → level (intra rounding offset).
pub(crate) fn quantize(coeff: &[i32], n: usize, qp: u8, bit_depth: u8) -> [i16; MAX_TB] {
    let log2n = n.trailing_zeros() as i64;
    let q_bits = 14 + (qp as i64) / 6 + (15 - bit_depth as i64 - log2n);
    let q_scale = QUANT_SCALE[(qp % 6) as usize];
    let offset = 171i64 << (q_bits - 9); // intra deadzone offset (171/512)
    let mut out = [0i16; MAX_TB];
    for (o, &c) in out[..n * n].iter_mut().zip(coeff) {
        let c = c as i64;
        let level = (c.abs() * q_scale + offset) >> q_bits;
        let level = if c < 0 { -level } else { level };
        *o = level.clamp(-32768, 32767) as i16;
    }
    out
}

/// Dequantization: level → transform coefficient (flat scaling, no dep-quant).
pub(crate) fn dequantize(level: &[i16], n: usize, qp: u8, bit_depth: u8) -> [i32; MAX_TB] {
    let mut out = [0i32; MAX_TB];
    dequantize_into(&mut out, level, n, qp, bit_depth);
    out
}

/// Dequantize `level[..n*n]` into `out[..n*n]` without allocating or zeroing a
/// full `MAX_TB` buffer — for the per-candidate RDO loop, where this runs many
/// times per leaf.
pub(crate) fn dequantize_into(out: &mut [i32], level: &[i16], n: usize, qp: u8, bit_depth: u8) {
    let log2n = n.trailing_zeros() as i64;
    let bd_shift = bit_depth as i64 + log2n - 5;
    let add = 1i64 << (bd_shift - 1);
    let scale = DEQUANT_SCALE[(qp % 6) as usize];
    let per = 1i64 << ((qp as i64) / 6);
    let factor = scale * per * 16;
    for (o, &l) in out[..n * n].iter_mut().zip(level) {
        // VVC clips dequantized coefficients to ±2^15 (MAX_TR_DYNAMIC_RANGE = 15).
        *o = ((l as i64 * factor + add) >> bd_shift).clamp(-32768, 32767) as i32;
    }
}

/// Forward quantization for a transform-skip block: the spatial residual is
/// quantized directly with no transform, so the shift schedule drops the
/// forward-transform term and is bit-depth-independent (VVC §8.7.3 transform
/// skip). The inverse is [`dequantize_ts`]. Conformance only requires that the
/// encoder reconstruct with the matching dequant; the rounding offset affects
/// rate–distortion, not bit-exactness.
pub(crate) fn quantize_ts(res: &[i32], n: usize, qp: u8) -> [i16; MAX_TB] {
    let per = (qp as i64) / 6;
    let q_bits = 14 + per; // regular q_bits minus the transform shift (15-bd-log2n)
    let q_scale = QUANT_SCALE[(qp % 6) as usize];
    let offset = 171i64 << (q_bits - 9); // intra deadzone offset (171/512)
    let mut out = [0i16; MAX_TB];
    for (o, &r) in out[..n * n].iter_mut().zip(res) {
        let r = r as i64;
        let level = (r.abs() * q_scale + offset) >> q_bits;
        let level = if r < 0 { -level } else { level };
        *o = level.clamp(-32768, 32767) as i16;
    }
    out
}

/// Dequantization for a transform-skip block, returning the spatial residual
/// directly (no inverse transform). Mirrors vvdec's `DeQuant` for transform
/// skip: `rightShift = IQUANT_SHIFT(6) − QP/6` with no bit-depth term, so at the
/// lossless QP (4) this is the identity `(level·64 + 32) >> 6 == level`.
pub(crate) fn dequantize_ts(level: &[i16], n: usize, qp: u8) -> [i32; MAX_TB] {
    dequantize_ts_wh(level, n, n, qp)
}

/// Rectangular transform-skip dequantization. Transform-skip never takes the
/// √2 non-square adjustment (it has no transform), so this differs from the
/// square version only in the element count.
pub(crate) fn dequantize_ts_wh(level: &[i16], w: usize, h: usize, qp: u8) -> [i32; MAX_TB] {
    let per = (qp as i32) / 6;
    let scale = DEQUANT_SCALE[(qp % 6) as usize];
    let right_shift = 6 - per;
    let mut out = [0i32; MAX_TB];
    for (o, &l) in out[..w * h].iter_mut().zip(level) {
        let v = l as i64 * scale;
        let r = if right_shift > 0 {
            (v + (1i64 << (right_shift - 1))) >> right_shift
        } else {
            v << (-right_shift)
        };
        *o = r.clamp(-32768, 32767) as i32;
    }
    out
}

/// Rectangular transform-skip forward quantization (companion to
/// [`dequantize_ts_wh`]).
pub(crate) fn quantize_ts_wh(res: &[i32], w: usize, h: usize, qp: u8) -> [i16; MAX_TB] {
    let per = (qp as i64) / 6;
    let q_bits = 14 + per;
    let q_scale = QUANT_SCALE[(qp % 6) as usize];
    let offset = 171i64 << (q_bits - 9);
    let mut out = [0i16; MAX_TB];
    for (o, &r) in out[..w * h].iter_mut().zip(res) {
        let r = r as i64;
        let level = (r.abs() * q_scale + offset) >> q_bits;
        let level = if r < 0 { -level } else { level };
        *o = level.clamp(-32768, 32767) as i16;
    }
    out
}

#[allow(dead_code)]
/// Inverse integer transform of an N×N coefficient block (N ∈ {4,8,16,32,64}),
/// returning a fresh `MAX_TB` buffer. Hot encoder paths should prefer
/// [`inv_transform_into`], which writes into reused scratch and avoids the
/// per-call zero-fill and return copy of a 16 KiB array.
pub(crate) fn inv_transform(coeff: &[i32], n: usize, bit_depth: u8) -> [i32; MAX_TB] {
    let mut out = [0i32; MAX_TB];
    inv_transform_into(&mut out, coeff, n, bit_depth);
    out
}

/// Inverse transform writing the `n×n` spatial residual into `out[..n*n]`
/// (which the inverse fully overwrites, so `out` needs no pre-zeroing).
pub(crate) fn inv_transform_into(out: &mut [i32], coeff: &[i32], n: usize, bit_depth: u8) {
    match n {
        4 => inv_transform_n::<4>(coeff, &T4, bit_depth, out),
        8 => inv_transform_n::<8>(coeff, &T8, bit_depth, out),
        16 => inv_transform_n::<16>(coeff, &T16, bit_depth, out),
        32 => inv_transform_n::<32>(coeff, &T32, bit_depth, out),
        64 => inv_transform_64(coeff, bit_depth, out),
        _ => panic!("unsupported transform size {n}"),
    }
}

/// Inverse 64-point DCT-II for a block whose only non-zero coefficients lie in
/// the top-left 32×32 (VVC high-frequency zeroing). Both passes therefore sum
/// over only the 32 surviving basis rows, producing a full 64×64 residual.
fn inv_transform_64(coeff: &[i32], bit_depth: u8, out: &mut [i32]) {
    const N: usize = 64;
    let clip = |v: i64| v.clamp(-32768, 32767) as i32;
    let shift1 = 7i32;
    let add1 = 1i32 << (shift1 - 1);
    let mut tmp = [0i32; MAX_TB];
    // pass 1 (columns) — only the 32 columns that can hold coefficients.
    for j in 0..32 {
        let mut acc = [0i32; N];
        for k in 0..32 {
            let c = coeff[k * N + j];
            if c != 0 {
                let trow = &T64[k];
                for n in 0..N {
                    acc[n] += trow[n] as i32 * c;
                }
            }
        }
        for n in 0..N {
            tmp[n * N + j] = clip(((acc[n] + add1) >> shift1) as i64);
        }
    }
    // pass 2 (rows) — sum over the 32 surviving columns.
    let shift2 = 20 - bit_depth as i32;
    let add2 = 1i32 << (shift2 - 1);
    for i in 0..N {
        let mut acc = [0i32; N];
        for k in 0..32 {
            let c = tmp[i * N + k];
            if c != 0 {
                let trow = &T64[k];
                for n in 0..N {
                    acc[n] += trow[n] as i32 * c;
                }
            }
        }
        for n in 0..N {
            out[i * N + n] = (acc[n] + add2) >> shift2;
        }
    }
}

#[inline]
fn inv_transform_n<const N: usize>(
    coeff: &[i32],
    t: &[[i8; N]; N],
    bit_depth: u8,
    out: &mut [i32],
) {
    // Inverse uses the transpose: x[n] = Σ_k T[k][n]·X[k].
    let clip = |v: i64| v.clamp(-32768, 32767) as i32; // intermediate range ±2^15
    let shift1 = 7i32;
    let add1 = 1i32 << (shift1 - 1);
    let mut tmp = [[0i32; N]; N];
    // pass 1 (columns of coeff). Accumulate by matrix row so the inner loop
    // walks a contiguous `t[k]` (cache-friendly) and whole rows are skipped when
    // the coefficient is zero — common after quantization. Result is identical:
    // omitted terms are exactly `t[k][n] * 0`, and i32 sums never overflow (the
    // pre-shift magnitude is bounded by the clip range).
    for j in 0..N {
        let mut acc = [0i32; N];
        for k in 0..N {
            let c = coeff[k * N + j];
            if c != 0 {
                let trow = &t[k];
                for n in 0..N {
                    acc[n] += trow[n] as i32 * c;
                }
            }
        }
        for n in 0..N {
            tmp[n][j] = clip(((acc[n] + add1) >> shift1) as i64);
        }
    }
    // pass 2 (rows)
    let shift2 = 20 - bit_depth as i32;
    let add2 = 1i32 << (shift2 - 1);
    let mut colv = [0i32; N];
    for i in 0..N {
        for (k, cv) in colv.iter_mut().enumerate() {
            *cv = tmp[i][k];
        }
        let mut acc = [0i32; N];
        for k in 0..N {
            let c = colv[k];
            if c != 0 {
                let trow = &t[k];
                for n in 0..N {
                    acc[n] += trow[n] as i32 * c;
                }
            }
        }
        for n in 0..N {
            out[i * N + n] = (acc[n] + add2) >> shift2;
        }
    }
}

/// One-dimensional 8-point Walsh–Hadamard transform (fast butterfly). The output
/// is sequency-permuted, but SATD only sums magnitudes so the order is moot.
#[inline]
fn hadamard_1d_8(a: [i32; 8]) -> [i32; 8] {
    let b0 = a[0] + a[4];
    let b1 = a[1] + a[5];
    let b2 = a[2] + a[6];
    let b3 = a[3] + a[7];
    let b4 = a[0] - a[4];
    let b5 = a[1] - a[5];
    let b6 = a[2] - a[6];
    let b7 = a[3] - a[7];
    let c0 = b0 + b2;
    let c1 = b1 + b3;
    let c2 = b0 - b2;
    let c3 = b1 - b3;
    let c4 = b4 + b6;
    let c5 = b5 + b7;
    let c6 = b4 - b6;
    let c7 = b5 - b7;
    [
        c0 + c1,
        c0 - c1,
        c2 + c3,
        c2 - c3,
        c4 + c5,
        c4 - c5,
        c6 + c7,
        c6 - c7,
    ]
}

/// Sum of absolute Hadamard-transformed coefficients of one 8×8 sub-block at
/// `(bx, by)` within an `n`-wide residual buffer.
fn hadamard_8x8(res: &[i32], n: usize, bx: usize, by: usize) -> i64 {
    let mut m = [[0i32; 8]; 8];
    // Horizontal pass.
    for (r, row) in m.iter_mut().enumerate() {
        let mut a = [0i32; 8];
        for (c, slot) in a.iter_mut().enumerate() {
            *slot = res[(by + r) * n + bx + c];
        }
        *row = hadamard_1d_8(a);
    }
    // Vertical pass.
    let mut sum = 0i64;
    #[allow(clippy::needless_range_loop)]
    for c in 0..8 {
        let col = [
            m[0][c], m[1][c], m[2][c], m[3][c], m[4][c], m[5][c], m[6][c], m[7][c],
        ];
        for v in hadamard_1d_8(col) {
            sum += v.unsigned_abs() as i64;
        }
    }
    sum
}

/// One-dimensional 4-point Walsh–Hadamard transform.
#[inline]
fn hadamard_1d_4(a: [i32; 4]) -> [i32; 4] {
    let b0 = a[0] + a[2];
    let b1 = a[1] + a[3];
    let b2 = a[0] - a[2];
    let b3 = a[1] - a[3];
    [b0 + b1, b0 - b1, b2 + b3, b2 - b3]
}

/// Sum of absolute Hadamard coefficients of a 4×4 residual.
fn hadamard_4x4(res: &[i32]) -> i64 {
    let mut m = [[0i32; 4]; 4];
    for (r, row) in m.iter_mut().enumerate() {
        *row = hadamard_1d_4([res[r * 4], res[r * 4 + 1], res[r * 4 + 2], res[r * 4 + 3]]);
    }
    let mut sum = 0i64;
    #[allow(clippy::needless_range_loop)]
    for c in 0..4 {
        for v in hadamard_1d_4([m[0][c], m[1][c], m[2][c], m[3][c]]) {
            sum += v.unsigned_abs() as i64;
        }
    }
    sum
}

/// Sum of absolute transformed differences for an `n`×`n` residual. Luma blocks
/// (`n` ∈ {8,16,32}) tile into 8×8 Hadamard blocks; the 4×4 case serves chroma
/// blocks at the minimum size. SATD correlates with post-transform coding cost
/// far better than a plain SAD, so it is the preferred metric for fast intra
/// mode decision (cf. VTM `xCalcHADs8x8`).
///
/// The raw Hadamard magnitude sum is returned without the 1/√(2N) normalisation:
/// every candidate for a given block size shares the same scale, so the argmin
/// is unaffected.
pub(crate) fn satd(res: &[i32], n: usize) -> i64 {
    if n == 4 {
        return hadamard_4x4(res);
    }
    debug_assert!(n.is_multiple_of(8) && n >= 8);
    let mut total = 0i64;
    let mut by = 0;
    while by < n {
        let mut bx = 0;
        while bx < n {
            total += hadamard_8x8(res, n, bx, by);
            bx += 8;
        }
        by += 8;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mts_transforms_round_trip_and_zero_out() {
        // Forward∘inverse of the near-orthonormal 8-bit DST7/DCT8 cores recovers
        // the residual (small integer rounding error), and a 32-point non-DCT2
        // transform zeroes everything outside the top-left 16×16.
        let combos = [(0u8, 0u8), (1, 1), (2, 1), (1, 2), (2, 2)];
        for n in [4usize, 8, 16] {
            for &(th, tv) in &combos {
                let mut res = vec![0i32; n * n];
                let mut st = 0x1234_567u32;
                for r in res.iter_mut() {
                    st = st.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    *r = ((st >> 16) % 401) as i32 - 200;
                }
                let mut coeff = [0i32; MAX_TB];
                fwd_transform_mts_into(&mut coeff, &res, n, n, 8, th, tv);
                let mut rec = [0i32; MAX_TB];
                inv_transform_mts_into(&mut rec, &coeff, n, n, 8, th, tv);
                let sse: i64 = (0..n * n).map(|i| ((res[i] - rec[i]) as i64).pow(2)).sum();
                // 8-bit cores are not perfectly orthonormal; a few units of SSE
                // per sample is expected, none of it from a structural error.
                assert!(
                    sse <= (n * n) as i64,
                    "{n}x{n} tr=({th},{tv}) SSE={sse} too high"
                );
            }
        }
        // 32-point DST7/DCT8 high-frequency zero-out: only 16×16 may be non-zero.
        let res: Vec<i32> = (0..32 * 32)
            .map(|i| ((i * 37 % 401) as i32) - 200)
            .collect();
        let mut coeff = [0i32; MAX_TB];
        fwd_transform_mts_into(&mut coeff, &res, 32, 32, 8, 1, 1);
        for y in 0..32 {
            for x in 0..32 {
                if x >= 16 || y >= 16 {
                    assert_eq!(
                        coeff[y * 32 + x],
                        0,
                        "coeff ({x},{y}) outside 16×16 not zeroed"
                    );
                }
            }
        }
    }

    #[test]
    fn transform_skip_dequant_is_identity_at_lossless_qp() {
        // At QP 4 the transform-skip dequant is (level·64 + 32) >> 6 == level,
        // independent of bit depth, so a raw residual round-trips exactly.
        let levels: Vec<i16> = (-300..=300).step_by(7).map(|v| v as i16).collect();
        let mut buf = [0i16; MAX_TB];
        buf[..levels.len()].copy_from_slice(&levels);
        let out = dequantize_ts(&buf, 16, 4);
        for (i, &l) in levels.iter().enumerate() {
            assert_eq!(out[i], l as i32, "level {l} not identity at QP4");
        }
    }

    #[test]
    fn transform_64_preserves_flat_and_zeroes_high_frequency() {
        // A flat (DC-only) 64×64 residual must survive forward→inverse exactly.
        let flat = vec![37i32; 64 * 64];
        let coeff = fwd_transform(&flat, 64, 8);
        // Only the top-left 32×32 may be non-zero (high-frequency zeroing).
        for y in 0..64 {
            for x in 0..64 {
                if x >= 32 || y >= 32 {
                    assert_eq!(coeff[y * 64 + x], 0, "coeff ({x},{y}) not zeroed");
                }
            }
        }
        let rec = inv_transform(&coeff, 64, 8);
        for (i, &r) in rec[..64 * 64].iter().enumerate() {
            assert_eq!(r, 37, "flat sample {i} = {r}");
        }
    }

    #[test]
    fn rect_transform_matches_square_for_square_sizes() {
        // The (w,h) path must be bit-identical to the optimized square path.
        let mut seed = 0x51ed_u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as i32 % 200 - 100
        };
        for &n in &[4usize, 8, 16, 32, 64] {
            let res: Vec<i32> = (0..n * n).map(|_| next()).collect();
            let sq = fwd_transform(&res, n, 8);
            let rect = fwd_transform_wh(&res, n, n, 8);
            assert_eq!(&sq[..n * n], &rect[..n * n], "fwd n={n}");
            let isq = inv_transform(&sq, n, 8);
            let irect = inv_transform_wh(&rect, n, n, 8);
            assert_eq!(&isq[..n * n], &irect[..n * n], "inv n={n}");
            let qsq = quantize(&sq, n, 26, 8);
            let qr = quantize_wh(&rect, n, n, 26, 8);
            assert_eq!(&qsq[..n * n], &qr[..n * n], "quant n={n}");
            let dsq = dequantize(&qsq, n, 26, 8);
            let dr = dequantize_wh(&qr, n, n, 26, 8);
            assert_eq!(&dsq[..n * n], &dr[..n * n], "dequant n={n}");
        }
    }

    #[test]
    fn rect_transform_preserves_flat_block() {
        // A flat rectangular residual must survive forward→inverse exactly,
        // exercising the √2 normalization on non-power-of-4 blocks (e.g. 4×8).
        for &(w, h) in &[
            (4usize, 8usize),
            (8, 4),
            (8, 16),
            (16, 8),
            (16, 32),
            (32, 64),
            (32, 16),
        ] {
            let flat = vec![29i32; w * h];
            let coeff = fwd_transform_wh(&flat, w, h, 8);
            let rec = inv_transform_wh(&coeff, w, h, 8);
            for (i, &r) in rec[..w * h].iter().enumerate() {
                assert_eq!(r, 29, "flat {w}x{h} sample {i} = {r}");
            }
        }
    }

    #[test]
    fn rect_quant_round_trips_to_grid() {
        // Forward then inverse (de)quant of a rectangular block lands on the
        // quantizer grid and is stable on re-quantization.
        for &(w, h) in &[(4usize, 8usize), (8, 16), (16, 32)] {
            for &qp in &[18u8, 27, 37] {
                let mut seed = 0xa5a5_u32 ^ (w as u32) ^ ((h as u32) << 8) ^ ((qp as u32) << 16);
                let mut next = || {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 9) as i32 % 80 - 40
                };
                let res: Vec<i32> = (0..w * h).map(|_| next()).collect();
                let coeff = fwd_transform_wh(&res, w, h, 8);
                let lv = quantize_wh(&coeff[..w * h], w, h, qp, 8);
                let deq = dequantize_wh(&lv, w, h, qp, 8);
                let lv2 = quantize_wh(&deq[..w * h], w, h, qp, 8);
                assert_eq!(&lv[..w * h], &lv2[..w * h], "unstable {w}x{h} qp{qp}");
            }
        }
    }
}

#[cfg(test)]
mod transform_tests {
    use super::*;

    #[test]
    fn satd_of_zero_residual_is_zero() {
        for &n in &[8usize, 16, 32] {
            assert_eq!(satd(&vec![0i32; n * n], n), 0);
        }
    }

    #[test]
    fn satd_of_constant_residual_equals_dc_energy() {
        // A flat residual k has a single non-zero Hadamard coefficient (the DC),
        // of magnitude k·N² per 8×8 tile -> 64·|k| each, summed over (n/8)² tiles.
        for &n in &[8usize, 16, 32] {
            let k = 7;
            let got = satd(&vec![k; n * n], n);
            let tiles = (n / 8) * (n / 8);
            assert_eq!(got, (64 * k.unsigned_abs() as i64) * tiles as i64);
        }
    }

    #[test]
    fn satd_is_nonnegative_and_sign_symmetric() {
        let mut res = vec![0i32; 64];
        for (i, v) in res.iter_mut().enumerate() {
            *v = ((i as i32 * 37) % 71) - 35;
        }
        let s = satd(&res, 8);
        assert!(s > 0);
        let neg: Vec<i32> = res.iter().map(|&v| -v).collect();
        assert_eq!(satd(&neg, 8), s, "SATD must be sign-symmetric");
    }

    fn round_trip_block(input: &[i32], n: usize, qp: u8, bd: u8) -> Vec<i32> {
        let coeff = fwd_transform(input, n, bd);
        let level = quantize(&coeff[..n * n], n, qp, bd);
        let deq = dequantize(&level[..n * n], n, qp, bd);
        let rec = inv_transform(&deq[..n * n], n, bd);
        rec[..n * n].to_vec()
    }

    #[test]
    fn dc_only_block_is_flat_after_round_trip() {
        // A constant residual transforms to a single DC coefficient; after a
        // round trip it must stay (very nearly) constant and close in value.
        for &n in &[4usize, 8, 16, 32] {
            let input = vec![40i32; n * n];
            let rec = round_trip_block(&input, n, 8, 8);
            let mn = *rec.iter().min().unwrap();
            let mx = *rec.iter().max().unwrap();
            assert!(mx - mn <= 1, "n={n} not flat: {mn}..{mx}");
            assert!((rec[0] - 40).abs() <= 3, "n={n} dc off: {}", rec[0]);
        }
    }

    #[test]
    fn lossless_at_qp4_small_signals() {
        // At low QP the quantizer is near-lossless for small residuals; the
        // reconstruction should match the input closely.
        for &n in &[4usize, 8, 16] {
            let mut input = vec![0i32; n * n];
            for (i, v) in input.iter_mut().enumerate() {
                *v = ((i as i32 * 7) % 11) - 5; // small pattern in [-5,5]
            }
            let rec = round_trip_block(&input, n, 4, 8);
            let max_err = input
                .iter()
                .zip(&rec)
                .map(|(a, b)| (a - b).abs())
                .max()
                .unwrap();
            assert!(max_err <= 2, "n={n} qp4 max_err={max_err}");
        }
    }

    #[test]
    fn error_grows_with_qp_but_stays_bounded() {
        // Reconstruction error should remain bounded and grow with QP.
        let n = 16;
        let mut input = vec![0i32; n * n];
        for (i, v) in input.iter_mut().enumerate() {
            *v = (((i * 131 + 7) % 200) as i32) - 100; // pseudo-random [-100,99]
        }
        let mut prev = 0;
        for &qp in &[10u8, 22, 37, 51] {
            let rec = round_trip_block(&input, n, qp, 8);
            let max_err = input
                .iter()
                .zip(&rec)
                .map(|(a, b)| (a - b).abs())
                .max()
                .unwrap();
            // Coarse step at QP q is ~ scale; reconstruction stays well within it.
            let bound = 4 + (1i32 << (qp as i32 / 6));
            assert!(max_err <= bound, "qp={qp} err={max_err} bound={bound}");
            prev = prev.max(max_err);
        }
        assert!(prev > 0);
    }

    #[test]
    fn transform_normalization_reconstructs_without_quant() {
        // inverse(forward(x)) ≈ x directly (the shift schedule is normalized so
        // the cascade is near-identity), and a flat block has ~zero AC energy.
        for &n in &[4usize, 8, 16, 32] {
            let c = 50i32;
            let coeff = fwd_transform(&vec![c; n * n], n, 8);
            assert!(coeff[0] > 0, "n={n} DC should be positive");
            assert!(
                coeff[1..n * n].iter().all(|&x| x.abs() <= 1),
                "n={n} AC not ~0"
            );
            let rec = inv_transform(&coeff[..n * n], n, 8);
            assert!(
                rec[..n * n].iter().all(|&v| (v - c).abs() <= 2),
                "n={n} not reconstructed"
            );
        }
    }

    #[test]
    fn high_bit_depth_round_trip() {
        let n = 8;
        let input: Vec<i32> = (0..n * n).map(|i| ((i as i32 * 53) % 800) - 400).collect();
        let rec = round_trip_block(&input, n, 16, 10);
        let max_err = input
            .iter()
            .zip(&rec)
            .map(|(a, b)| (a - b).abs())
            .max()
            .unwrap();
        assert!(max_err <= 4 + (1 << (16 / 6)), "10-bit err={max_err}");
    }
}

#[cfg(test)]
mod hf_check {
    use super::*;
    #[test]
    fn fwd_wh_64_zeroes_highfreq() {
        let mut res = vec![0i32; 64 * 64];
        let mut s = 1u64;
        for v in res.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *v = (s & 0xff) as i32 - 128;
        }
        let out = fwd_transform_wh(&res, 64, 64, 8);
        let mut bad = 0;
        for y in 0..64 {
            for x in 0..64 {
                if (x >= 32 || y >= 32) && out[y * 64 + x] != 0 {
                    bad += 1;
                }
            }
        }
        let lv = quantize_wh(&out[..64 * 64], 64, 64, 38, 8);
        let mut badq = 0;
        for y in 0..64 {
            for x in 0..64 {
                if (x >= 32 || y >= 32) && lv[y * 64 + x] != 0 {
                    badq += 1;
                }
            }
        }
        assert_eq!(bad, 0, "fwd_transform_wh leaves {} high-freq nonzero", bad);
        assert_eq!(badq, 0, "quantize_wh leaves {} high-freq nonzero", badq);
    }

    #[test]
    fn dq_dequant_state0_matches_scalar() {
        // Dependent quant dequantises on the qpDQ = qp+1 grid (VTM
        // DepQuant.cpp), NOT the scalar qp grid — so even a lone DC coefficient
        // (which stays in state 0 / Q0) must reconstruct via the explicit qp+1
        // formula, generally differing from scalar dequant.
        let (w, h) = (8usize, 8usize);
        let scan = crate::residual::scan_coords(w, h);
        let avg = 3i64; // (3+3)>>1 for 8x8
        let bd_shift = 8i64 + avg - 5; // bit_depth 8, sqrt2 0
        for &qp in &[12u8, 22, 28, 37, 42] {
            let qp_dq = qp as i64 + 1;
            let scale = DEQUANT_SCALE[(qp_dq % 6) as usize];
            let factor = scale * (1i64 << (qp_dq / 6)) * 16;
            let add = 1i64 << bd_shift;
            for &lvl in &[1i32, 2, 3, 7, -5, 19] {
                let mut dq_lv = [0i32; 64];
                dq_lv[0] = lvl;
                let dq = dequantize_dq_wh(&dq_lv, scan, w, h, qp, 8);
                // state 0 => Q0 => qIdx = 2*level
                let expect =
                    ((2 * lvl as i64 * factor + add) >> (bd_shift + 1)).clamp(-32768, 32767) as i32;
                assert_eq!(dq[0], expect, "qp{qp} lvl{lvl}");
            }
        }
    }

    #[test]
    fn dq_dequant_tracks_state() {
        // Two coefficients along the scan. State inits 0 at the highest scan
        // index; an odd level there flips the next (lower) position into Q1,
        // shifting its reconstruction relative to the scalar value.
        let (w, h) = (4usize, 4usize);
        let scan = crate::residual::scan_coords(w, h);
        // scan[1] = highest-but-one; place an odd level at scan[1] and a level
        // at DC (scan[0]). next_state(0, odd) = 2 -> DC sees Q1 (state>>1==1).
        let (x1, y1) = scan[1];
        let (x0, y0) = scan[0];
        let mut lv = [0i32; 16];
        lv[y1 * w + x1] = 3; // odd -> advances state 0->2
        lv[y0 * w + x0] = 4;
        let qp = 28u8;
        let dq = dequantize_dq_wh(&lv, scan, w, h, qp, 8);
        // DC level 4 reconstructed at state 2 (Q1): qIdx = 2*4 - 1 = 7, vs scalar
        // qIdx 8 -> must be strictly smaller in magnitude than scalar dequant.
        let mut sc_lv = [0i16; 16];
        sc_lv[y0 * w + x0] = 4;
        let sc = dequantize_wh(&sc_lv, w, h, qp, 8);
        assert!(
            dq[y0 * w + x0].unsigned_abs() < sc[y0 * w + x0].unsigned_abs(),
            "Q1 should shift DC down: dq={} sc={}",
            dq[y0 * w + x0],
            sc[y0 * w + x0]
        );
        assert!(dq[y0 * w + x0] > 0, "sign preserved");
    }
}

// ===========================================================================
// MTS (Multiple Transform Selection) — DST-VII and DCT-VIII 8-bit core matrices
// (VTM RomTr.cpp g_trCore{DST7,DCT8}P{4,8,16,32}[TRANSFORM_INVERSE], expanded
// from the DEFINE_*_MATRIX macros). Same 6-bit normalisation and shift schedule
// as the DCT-II tables, so they slot into the existing transform machinery; the
// inverse is Mᵀ·coeff exactly as VTM's fastInverse{DST7,DCT8} butterflies.
// ===========================================================================
pub(crate) const DST74: [[i8; 4]; 4] = [
    [29, 55, 74, 84],
    [74, 74, 0, -74],
    [84, -29, -74, 55],
    [55, -84, 74, -29],
];

pub(crate) const DST78: [[i8; 8]; 8] = [
    [17, 32, 46, 60, 71, 78, 85, 86],
    [46, 78, 86, 71, 32, -17, -60, -85],
    [71, 85, 32, -46, -86, -60, 17, 78],
    [85, 46, -60, -78, 17, 86, 32, -71],
    [86, -17, -85, 32, 78, -46, -71, 60],
    [78, -71, -17, 85, -60, -32, 86, -46],
    [60, -86, 71, -17, -46, 85, -78, 32],
    [32, -60, 78, -86, 85, -71, 46, -17],
];

pub(crate) const DST716: [[i8; 16]; 16] = [
    [
        8, 17, 25, 33, 40, 48, 55, 62, 68, 73, 77, 81, 85, 87, 88, 88,
    ],
    [
        25, 48, 68, 81, 88, 88, 81, 68, 48, 25, 0, -25, -48, -68, -81, -88,
    ],
    [
        40, 73, 88, 85, 62, 25, -17, -55, -81, -88, -77, -48, -8, 33, 68, 87,
    ],
    [
        55, 87, 81, 40, -17, -68, -88, -73, -25, 33, 77, 88, 62, 8, -48, -85,
    ],
    [
        68, 88, 48, -25, -81, -81, -25, 48, 88, 68, 0, -68, -88, -48, 25, 81,
    ],
    [
        77, 77, 0, -77, -77, 0, 77, 77, 0, -77, -77, 0, 77, 77, 0, -77,
    ],
    [
        85, 55, -48, -87, -8, 81, 62, -40, -88, -17, 77, 68, -33, -88, -25, 73,
    ],
    [
        88, 25, -81, -48, 68, 68, -48, -81, 25, 88, 0, -88, -25, 81, 48, -68,
    ],
    [
        88, -8, -88, 17, 87, -25, -85, 33, 81, -40, -77, 48, 73, -55, -68, 62,
    ],
    [
        87, -40, -68, 73, 33, -88, 8, 85, -48, -62, 77, 25, -88, 17, 81, -55,
    ],
    [
        81, -68, -25, 88, -48, -48, 88, -25, -68, 81, 0, -81, 68, 25, -88, 48,
    ],
    [
        73, -85, 25, 55, -88, 48, 33, -87, 68, 8, -77, 81, -17, -62, 88, -40,
    ],
    [
        62, -88, 68, -8, -55, 88, -73, 17, 48, -87, 77, -25, -40, 85, -81, 33,
    ],
    [
        48, -81, 88, -68, 25, 25, -68, 88, -81, 48, 0, -48, 81, -88, 68, -25,
    ],
    [
        33, -62, 81, -88, 85, -68, 40, -8, -25, 55, -77, 88, -87, 73, -48, 17,
    ],
    [
        17, -33, 48, -62, 73, -81, 87, -88, 88, -85, 77, -68, 55, -40, 25, -8,
    ],
];

pub(crate) const DST732: [[i8; 32]; 32] = [
    [
        4, 9, 13, 17, 21, 26, 30, 34, 38, 42, 46, 50, 53, 56, 60, 63, 66, 68, 72, 74, 77, 78, 80,
        82, 84, 85, 86, 87, 88, 89, 90, 90,
    ],
    [
        13, 26, 38, 50, 60, 68, 77, 82, 86, 89, 90, 88, 85, 80, 74, 66, 56, 46, 34, 21, 9, -4, -17,
        -30, -42, -53, -63, -72, -78, -84, -87, -90,
    ],
    [
        21, 42, 60, 74, 84, 89, 89, 84, 74, 60, 42, 21, 0, -21, -42, -60, -74, -84, -89, -89, -84,
        -74, -60, -42, -21, 0, 21, 42, 60, 74, 84, 89,
    ],
    [
        30, 56, 77, 87, 89, 80, 63, 38, 9, -21, -50, -72, -85, -90, -84, -68, -46, -17, 13, 42, 66,
        82, 90, 86, 74, 53, 26, -4, -34, -60, -78, -88,
    ],
    [
        38, 68, 86, 88, 74, 46, 9, -30, -63, -84, -90, -78, -53, -17, 21, 56, 80, 90, 82, 60, 26,
        -13, -50, -77, -89, -85, -66, -34, 4, 42, 72, 87,
    ],
    [
        46, 78, 90, 77, 42, -4, -50, -80, -90, -74, -38, 9, 53, 82, 89, 72, 34, -13, -56, -84, -88,
        -68, -30, 17, 60, 85, 87, 66, 26, -21, -63, -86,
    ],
    [
        53, 85, 85, 53, 0, -53, -85, -85, -53, 0, 53, 85, 85, 53, 0, -53, -85, -85, -53, 0, 53, 85,
        85, 53, 0, -53, -85, -85, -53, 0, 53, 85,
    ],
    [
        60, 89, 74, 21, -42, -84, -84, -42, 21, 74, 89, 60, 0, -60, -89, -74, -21, 42, 84, 84, 42,
        -21, -74, -89, -60, 0, 60, 89, 74, 21, -42, -84,
    ],
    [
        66, 90, 56, -13, -74, -87, -46, 26, 80, 84, 34, -38, -85, -78, -21, 50, 88, 72, 9, -60,
        -90, -63, 4, 68, 89, 53, -17, -77, -86, -42, 30, 82,
    ],
    [
        72, 86, 34, -46, -89, -63, 13, 78, 82, 21, -56, -90, -53, 26, 84, 77, 9, -66, -88, -42, 38,
        87, 68, -4, -74, -85, -30, 50, 90, 60, -17, -80,
    ],
    [
        77, 80, 9, -72, -84, -17, 66, 86, 26, -60, -88, -34, 53, 90, 42, -46, -90, -50, 38, 89, 56,
        -30, -87, -63, 21, 85, 68, -13, -82, -74, 4, 78,
    ],
    [
        80, 72, -17, -86, -60, 34, 90, 46, -50, -89, -30, 63, 85, 13, -74, -78, 4, 82, 68, -21,
        -87, -56, 38, 90, 42, -53, -88, -26, 66, 84, 9, -77,
    ],
    [
        84, 60, -42, -89, -21, 74, 74, -21, -89, -42, 60, 84, 0, -84, -60, 42, 89, 21, -74, -74,
        21, 89, 42, -60, -84, 0, 84, 60, -42, -89, -21, 74,
    ],
    [
        86, 46, -63, -78, 21, 90, 26, -77, -66, 42, 87, 4, -85, -50, 60, 80, -17, -90, -30, 74, 68,
        -38, -88, -9, 84, 53, -56, -82, 13, 89, 34, -72,
    ],
    [
        88, 30, -78, -56, 60, 77, -34, -87, 4, 89, 26, -80, -53, 63, 74, -38, -86, 9, 90, 21, -82,
        -50, 66, 72, -42, -85, 13, 90, 17, -84, -46, 68,
    ],
    [
        90, 13, -87, -26, 84, 38, -78, -50, 72, 60, -63, -68, 53, 77, -42, -82, 30, 86, -17, -89,
        4, 90, 9, -88, -21, 85, 34, -80, -46, 74, 56, -66,
    ],
    [
        90, -4, -90, 9, 89, -13, -88, 17, 87, -21, -86, 26, 85, -30, -84, 34, 82, -38, -80, 42, 78,
        -46, -77, 50, 74, -53, -72, 56, 68, -60, -66, 63,
    ],
    [
        89, -21, -84, 42, 74, -60, -60, 74, 42, -84, -21, 89, 0, -89, 21, 84, -42, -74, 60, 60,
        -74, -42, 84, 21, -89, 0, 89, -21, -84, 42, 74, -60,
    ],
    [
        87, -38, -72, 68, 42, -86, -4, 88, -34, -74, 66, 46, -85, -9, 89, -30, -77, 63, 50, -84,
        -13, 90, -26, -78, 60, 53, -82, -17, 90, -21, -80, 56,
    ],
    [
        85, -53, -53, 85, 0, -85, 53, 53, -85, 0, 85, -53, -53, 85, 0, -85, 53, 53, -85, 0, 85,
        -53, -53, 85, 0, -85, 53, 53, -85, 0, 85, -53,
    ],
    [
        82, -66, -30, 90, -42, -56, 86, -13, -77, 74, 17, -87, 53, 46, -89, 26, 68, -80, -4, 84,
        -63, -34, 90, -38, -60, 85, -9, -78, 72, 21, -88, 50,
    ],
    [
        78, -77, -4, 80, -74, -9, 82, -72, -13, 84, -68, -17, 85, -66, -21, 86, -63, -26, 87, -60,
        -30, 88, -56, -34, 89, -53, -38, 90, -50, -42, 90, -46,
    ],
    [
        74, -84, 21, 60, -89, 42, 42, -89, 60, 21, -84, 74, 0, -74, 84, -21, -60, 89, -42, -42, 89,
        -60, -21, 84, -74, 0, 74, -84, 21, 60, -89, 42,
    ],
    [
        68, -88, 46, 30, -84, 78, -17, -56, 90, -60, -13, 77, -85, 34, 42, -87, 72, -4, -66, 89,
        -50, -26, 82, -80, 21, 53, -90, 63, 9, -74, 86, -38,
    ],
    [
        63, -90, 66, -4, -60, 90, -68, 9, 56, -89, 72, -13, -53, 88, -74, 17, 50, -87, 77, -21,
        -46, 86, -78, 26, 42, -85, 80, -30, -38, 84, -82, 34,
    ],
    [
        56, -87, 80, -38, -21, 72, -90, 68, -17, -42, 82, -86, 53, 4, -60, 88, -78, 34, 26, -74,
        90, -66, 13, 46, -84, 85, -50, -9, 63, -89, 77, -30,
    ],
    [
        50, -82, 88, -66, 21, 30, -72, 90, -78, 42, 9, -56, 85, -86, 60, -13, -38, 77, -90, 74,
        -34, -17, 63, -87, 84, -53, 4, 46, -80, 89, -68, 26,
    ],
    [
        42, -74, 89, -84, 60, -21, -21, 60, -84, 89, -74, 42, 0, -42, 74, -89, 84, -60, 21, 21,
        -60, 84, -89, 74, -42, 0, 42, -74, 89, -84, 60, -21,
    ],
    [
        34, -63, 82, -90, 84, -66, 38, -4, -30, 60, -80, 90, -85, 68, -42, 9, 26, -56, 78, -89, 86,
        -72, 46, -13, -21, 53, -77, 88, -87, 74, -50, 17,
    ],
    [
        26, -50, 68, -82, 89, -88, 80, -66, 46, -21, -4, 30, -53, 72, -84, 90, -87, 78, -63, 42,
        -17, -9, 34, -56, 74, -85, 90, -86, 77, -60, 38, -13,
    ],
    [
        17, -34, 50, -63, 74, -82, 87, -90, 88, -84, 77, -66, 53, -38, 21, -4, -13, 30, -46, 60,
        -72, 80, -86, 90, -89, 85, -78, 68, -56, 42, -26, 9,
    ],
    [
        9, -17, 26, -34, 42, -50, 56, -63, 68, -74, 78, -82, 85, -87, 89, -90, 90, -88, 86, -84,
        80, -77, 72, -66, 60, -53, 46, -38, 30, -21, 13, -4,
    ],
];

pub(crate) const DCT84: [[i8; 4]; 4] = [
    [84, 74, 55, 29],
    [74, 0, -74, -74],
    [55, -74, -29, 84],
    [29, -74, 84, -55],
];

pub(crate) const DCT88: [[i8; 8]; 8] = [
    [86, 85, 78, 71, 60, 46, 32, 17],
    [85, 60, 17, -32, -71, -86, -78, -46],
    [78, 17, -60, -86, -46, 32, 85, 71],
    [71, -32, -86, -17, 78, 60, -46, -85],
    [60, -71, -46, 78, 32, -85, -17, 86],
    [46, -86, 32, 60, -85, 17, 71, -78],
    [32, -78, 85, -46, -17, 71, -86, 60],
    [17, -46, 71, -85, 86, -78, 60, -32],
];

pub(crate) const DCT816: [[i8; 16]; 16] = [
    [
        88, 88, 87, 85, 81, 77, 73, 68, 62, 55, 48, 40, 33, 25, 17, 8,
    ],
    [
        88, 81, 68, 48, 25, 0, -25, -48, -68, -81, -88, -88, -81, -68, -48, -25,
    ],
    [
        87, 68, 33, -8, -48, -77, -88, -81, -55, -17, 25, 62, 85, 88, 73, 40,
    ],
    [
        85, 48, -8, -62, -88, -77, -33, 25, 73, 88, 68, 17, -40, -81, -87, -55,
    ],
    [
        81, 25, -48, -88, -68, 0, 68, 88, 48, -25, -81, -81, -25, 48, 88, 68,
    ],
    [
        77, 0, -77, -77, 0, 77, 77, 0, -77, -77, 0, 77, 77, 0, -77, -77,
    ],
    [
        73, -25, -88, -33, 68, 77, -17, -88, -40, 62, 81, -8, -87, -48, 55, 85,
    ],
    [
        68, -48, -81, 25, 88, 0, -88, -25, 81, 48, -68, -68, 48, 81, -25, -88,
    ],
    [
        62, -68, -55, 73, 48, -77, -40, 81, 33, -85, -25, 87, 17, -88, -8, 88,
    ],
    [
        55, -81, -17, 88, -25, -77, 62, 48, -85, -8, 88, -33, -73, 68, 40, -87,
    ],
    [
        48, -88, 25, 68, -81, 0, 81, -68, -25, 88, -48, -48, 88, -25, -68, 81,
    ],
    [
        40, -88, 62, 17, -81, 77, -8, -68, 87, -33, -48, 88, -55, -25, 85, -73,
    ],
    [
        33, -81, 85, -40, -25, 77, -87, 48, 17, -73, 88, -55, -8, 68, -88, 62,
    ],
    [
        25, -68, 88, -81, 48, 0, -48, 81, -88, 68, -25, -25, 68, -88, 81, -48,
    ],
    [
        17, -48, 73, -87, 88, -77, 55, -25, -8, 40, -68, 85, -88, 81, -62, 33,
    ],
    [
        8, -25, 40, -55, 68, -77, 85, -88, 88, -87, 81, -73, 62, -48, 33, -17,
    ],
];

pub(crate) const DCT832: [[i8; 32]; 32] = [
    [
        90, 90, 89, 88, 87, 86, 85, 84, 82, 80, 78, 77, 74, 72, 68, 66, 63, 60, 56, 53, 50, 46, 42,
        38, 34, 30, 26, 21, 17, 13, 9, 4,
    ],
    [
        90, 87, 84, 78, 72, 63, 53, 42, 30, 17, 4, -9, -21, -34, -46, -56, -66, -74, -80, -85, -88,
        -90, -89, -86, -82, -77, -68, -60, -50, -38, -26, -13,
    ],
    [
        89, 84, 74, 60, 42, 21, 0, -21, -42, -60, -74, -84, -89, -89, -84, -74, -60, -42, -21, 0,
        21, 42, 60, 74, 84, 89, 89, 84, 74, 60, 42, 21,
    ],
    [
        88, 78, 60, 34, 4, -26, -53, -74, -86, -90, -82, -66, -42, -13, 17, 46, 68, 84, 90, 85, 72,
        50, 21, -9, -38, -63, -80, -89, -87, -77, -56, -30,
    ],
    [
        87, 72, 42, 4, -34, -66, -85, -89, -77, -50, -13, 26, 60, 82, 90, 80, 56, 21, -17, -53,
        -78, -90, -84, -63, -30, 9, 46, 74, 88, 86, 68, 38,
    ],
    [
        86, 63, 21, -26, -66, -87, -85, -60, -17, 30, 68, 88, 84, 56, 13, -34, -72, -89, -82, -53,
        -9, 38, 74, 90, 80, 50, 4, -42, -77, -90, -78, -46,
    ],
    [
        85, 53, 0, -53, -85, -85, -53, 0, 53, 85, 85, 53, 0, -53, -85, -85, -53, 0, 53, 85, 85, 53,
        0, -53, -85, -85, -53, 0, 53, 85, 85, 53,
    ],
    [
        84, 42, -21, -74, -89, -60, 0, 60, 89, 74, 21, -42, -84, -84, -42, 21, 74, 89, 60, 0, -60,
        -89, -74, -21, 42, 84, 84, 42, -21, -74, -89, -60,
    ],
    [
        82, 30, -42, -86, -77, -17, 53, 89, 68, 4, -63, -90, -60, 9, 72, 88, 50, -21, -78, -85,
        -38, 34, 84, 80, 26, -46, -87, -74, -13, 56, 90, 66,
    ],
    [
        80, 17, -60, -90, -50, 30, 85, 74, 4, -68, -87, -38, 42, 88, 66, -9, -77, -84, -26, 53, 90,
        56, -21, -82, -78, -13, 63, 89, 46, -34, -86, -72,
    ],
    [
        78, 4, -74, -82, -13, 68, 85, 21, -63, -87, -30, 56, 89, 38, -50, -90, -46, 42, 90, 53,
        -34, -88, -60, 26, 86, 66, -17, -84, -72, 9, 80, 77,
    ],
    [
        77, -9, -84, -66, 26, 88, 53, -42, -90, -38, 56, 87, 21, -68, -82, -4, 78, 74, -13, -85,
        -63, 30, 89, 50, -46, -90, -34, 60, 86, 17, -72, -80,
    ],
    [
        74, -21, -89, -42, 60, 84, 0, -84, -60, 42, 89, 21, -74, -74, 21, 89, 42, -60, -84, 0, 84,
        60, -42, -89, -21, 74, 74, -21, -89, -42, 60, 84,
    ],
    [
        72, -34, -89, -13, 82, 56, -53, -84, 9, 88, 38, -68, -74, 30, 90, 17, -80, -60, 50, 85, -4,
        -87, -42, 66, 77, -26, -90, -21, 78, 63, -46, -86,
    ],
    [
        68, -46, -84, 17, 90, 13, -85, -42, 72, 66, -50, -82, 21, 90, 9, -86, -38, 74, 63, -53,
        -80, 26, 89, 4, -87, -34, 77, 60, -56, -78, 30, 88,
    ],
    [
        66, -56, -74, 46, 80, -34, -85, 21, 88, -9, -90, -4, 89, 17, -86, -30, 82, 42, -77, -53,
        68, 63, -60, -72, 50, 78, -38, -84, 26, 87, -13, -90,
    ],
    [
        63, -66, -60, 68, 56, -72, -53, 74, 50, -77, -46, 78, 42, -80, -38, 82, 34, -84, -30, 85,
        26, -86, -21, 87, 17, -88, -13, 89, 9, -90, -4, 90,
    ],
    [
        60, -74, -42, 84, 21, -89, 0, 89, -21, -84, 42, 74, -60, -60, 74, 42, -84, -21, 89, 0, -89,
        21, 84, -42, -74, 60, 60, -74, -42, 84, 21, -89,
    ],
    [
        56, -80, -21, 90, -17, -82, 53, 60, -78, -26, 90, -13, -84, 50, 63, -77, -30, 89, -9, -85,
        46, 66, -74, -34, 88, -4, -86, 42, 68, -72, -38, 87,
    ],
    [
        53, -85, 0, 85, -53, -53, 85, 0, -85, 53, 53, -85, 0, 85, -53, -53, 85, 0, -85, 53, 53,
        -85, 0, 85, -53, -53, 85, 0, -85, 53, 53, -85,
    ],
    [
        50, -88, 21, 72, -78, -9, 85, -60, -38, 90, -34, -63, 84, -4, -80, 68, 26, -89, 46, 53,
        -87, 17, 74, -77, -13, 86, -56, -42, 90, -30, -66, 82,
    ],
    [
        46, -90, 42, 50, -90, 38, 53, -89, 34, 56, -88, 30, 60, -87, 26, 63, -86, 21, 66, -85, 17,
        68, -84, 13, 72, -82, 9, 74, -80, 4, 77, -78,
    ],
    [
        42, -89, 60, 21, -84, 74, 0, -74, 84, -21, -60, 89, -42, -42, 89, -60, -21, 84, -74, 0, 74,
        -84, 21, 60, -89, 42, 42, -89, 60, 21, -84, 74,
    ],
    [
        38, -86, 74, -9, -63, 90, -53, -21, 80, -82, 26, 50, -89, 66, 4, -72, 87, -42, -34, 85,
        -77, 13, 60, -90, 56, 17, -78, 84, -30, -46, 88, -68,
    ],
    [
        34, -82, 84, -38, -30, 80, -85, 42, 26, -78, 86, -46, -21, 77, -87, 50, 17, -74, 88, -53,
        -13, 72, -89, 56, 9, -68, 90, -60, -4, 66, -90, 63,
    ],
    [
        30, -77, 89, -63, 9, 50, -85, 84, -46, -13, 66, -90, 74, -26, -34, 78, -88, 60, -4, -53,
        86, -82, 42, 17, -68, 90, -72, 21, 38, -80, 87, -56,
    ],
    [
        26, -68, 89, -80, 46, 4, -53, 84, -87, 63, -17, -34, 74, -90, 77, -38, -13, 60, -86, 85,
        -56, 9, 42, -78, 90, -72, 30, 21, -66, 88, -82, 50,
    ],
    [
        21, -60, 84, -89, 74, -42, 0, 42, -74, 89, -84, 60, -21, -21, 60, -84, 89, -74, 42, 0, -42,
        74, -89, 84, -60, 21, 21, -60, 84, -89, 74, -42,
    ],
    [
        17, -50, 74, -87, 88, -77, 53, -21, -13, 46, -72, 86, -89, 78, -56, 26, 9, -42, 68, -85,
        90, -80, 60, -30, -4, 38, -66, 84, -90, 82, -63, 34,
    ],
    [
        13, -38, 60, -77, 86, -90, 85, -74, 56, -34, 9, 17, -42, 63, -78, 87, -90, 84, -72, 53,
        -30, 4, 21, -46, 66, -80, 88, -89, 82, -68, 50, -26,
    ],
    [
        9, -26, 42, -56, 68, -78, 85, -89, 90, -86, 80, -72, 60, -46, 30, -13, -4, 21, -38, 53,
        -66, 77, -84, 88, -90, 87, -82, 74, -63, 50, -34, 17,
    ],
    [
        4, -13, 21, -30, 38, -46, 53, -60, 66, -72, 77, -80, 84, -86, 88, -90, 90, -89, 87, -85,
        82, -78, 74, -68, 63, -56, 50, -42, 34, -26, 17, -9,
    ],
];

/// Row `k` of the DST-VII core matrix of the given `size` (4/8/16/32).
#[inline]
fn dst7_row(size: usize, k: usize) -> &'static [i8] {
    match size {
        4 => &DST74[k],
        8 => &DST78[k],
        16 => &DST716[k],
        32 => &DST732[k],
        _ => panic!("unsupported DST7 size {size}"),
    }
}

/// Row `k` of the DCT-VIII core matrix of the given `size` (4/8/16/32).
#[inline]
fn dct8_row(size: usize, k: usize) -> &'static [i8] {
    match size {
        4 => &DCT84[k],
        8 => &DCT88[k],
        16 => &DCT816[k],
        32 => &DCT832[k],
        _ => panic!("unsupported DCT8 size {size}"),
    }
}

/// Row `k` of the core matrix for transform type `tr` (0 = DCT-II, 1 = DST-VII,
/// 2 = DCT-VIII) and `size`.
#[inline]
fn tr_row(tr: u8, size: usize, k: usize) -> &'static [i8] {
    match tr {
        0 => t_row(size, k),
        1 => dst7_row(size, k),
        _ => dct8_row(size, k),
    }
}

/// Non-zero output count along a dimension of length `n` under transform type
/// `tr`. VVC zeroes the high half of a 32-point DST-VII/DCT-VIII transform
/// (`TrQuant.cpp`: `skip = (trType != DCT2 && size == 32) ? 16 : ...`); DCT-II
/// follows the usual 64→32 rule.
#[inline]
fn kept_mts(n: usize, tr: u8) -> usize {
    if tr != 0 && n == 32 { 16 } else { kept(n) }
}

/// Forward separable transform with independent horizontal/vertical transform
/// types (MTS). `tr_hor`/`tr_ver` are 0 = DCT-II, 1 = DST-VII, 2 = DCT-VIII.
/// DCT-II/DCT-II delegates to the optimised default path; other combinations
/// take the generic two-pass path with MTS high-frequency zeroing.
pub(crate) fn fwd_transform_mts_into(
    out: &mut [i32],
    res: &[i32],
    w: usize,
    h: usize,
    bit_depth: u8,
    tr_hor: u8,
    tr_ver: u8,
) {
    if tr_hor == 0 && tr_ver == 0 {
        fwd_transform_wh_into(out, res, w, h, bit_depth);
        return;
    }
    let mut tmp = [0i32; MAX_TB];
    fwd_mts_pass1_into(&mut tmp, res, w, h, bit_depth, tr_hor);
    fwd_mts_pass2_into(out, &tmp, w, h, tr_hor, tr_ver);
}

/// Forward MTS pass 1 — horizontal transform `tr_hor` (a non-DCT-II type) over
/// size `w`, keeping the first `kept_mts(w, tr_hor)` outputs per row into `tmp`.
/// Split out so the encoder can compute it once per distinct horizontal type and
/// reuse it for every vertical pairing (the result is bit-identical to running
/// the full transform per pair).
pub(crate) fn fwd_mts_pass1_into(
    tmp: &mut [i32],
    res: &[i32],
    w: usize,
    h: usize,
    bit_depth: u8,
    tr_hor: u8,
) {
    let kw = kept_mts(w, tr_hor);
    let log2w = w.trailing_zeros() as i32;
    let shift1 = log2w + bit_depth as i32 - 9;
    let add1 = if shift1 > 0 { 1i32 << (shift1 - 1) } else { 0 };
    for j in 0..h {
        let row = &res[j * w..j * w + w];
        for i in 0..kw {
            let trow = tr_row(tr_hor, w, i);
            let s: i32 = trow.iter().zip(row).map(|(&t, &r)| t as i32 * r).sum();
            tmp[j * w + i] = if shift1 > 0 { (s + add1) >> shift1 } else { s };
        }
    }
}

/// Forward MTS pass 2 — vertical transform `tr_ver` over size `h` across the
/// `kept_mts(w, tr_hor)` surviving columns of a pass-1 buffer `tmp`.
pub(crate) fn fwd_mts_pass2_into(
    out: &mut [i32],
    tmp: &[i32],
    w: usize,
    h: usize,
    tr_hor: u8,
    tr_ver: u8,
) {
    out[..w * h].fill(0);
    let kw = kept_mts(w, tr_hor);
    let kh = kept_mts(h, tr_ver);
    let log2h = h.trailing_zeros() as i32;
    let shift2 = log2h + 6;
    let add2 = 1i32 << (shift2 - 1);
    let mut colv = [0i32; 64];
    for i in 0..kw {
        for (k, cv) in colv.iter_mut().enumerate().take(h) {
            *cv = tmp[k * w + i];
        }
        for o in 0..kh {
            let trow = tr_row(tr_ver, h, o);
            let s: i32 = (0..h).map(|k| trow[k] as i32 * colv[k]).sum();
            out[o * w + i] = (s + add2) >> shift2;
        }
    }
}

/// Inverse separable transform with independent horizontal/vertical transform
/// types (MTS). Mirrors [`fwd_transform_mts_into`]; the vertical pass runs first
/// then the horizontal, matching VTM's `xIT`. Only the top-left
/// `kept_mts(w)×kept_mts(h)` coefficients are read (the rest are zeroed).
pub(crate) fn inv_transform_mts_into(
    out: &mut [i32],
    coeff: &[i32],
    w: usize,
    h: usize,
    bit_depth: u8,
    tr_hor: u8,
    tr_ver: u8,
) {
    if tr_hor == 0 && tr_ver == 0 {
        inv_transform_wh_into(out, coeff, w, h, bit_depth);
        return;
    }
    let (kw, kh) = (kept_mts(w, tr_hor), kept_mts(h, tr_ver));
    let clip = |v: i64| v.clamp(-32768, 32767) as i32;
    let shift1 = 7i32;
    let add1 = 1i32 << (shift1 - 1);
    let mut tmp = [0i32; MAX_TB];
    // Pass 1 — vertical (trTypeVer, size h) over the kw surviving columns.
    for i in 0..kw {
        let mut acc = [0i32; 64];
        for k in 0..kh {
            let c = coeff[k * w + i];
            if c != 0 {
                let trow = tr_row(tr_ver, h, k);
                for (n, a) in acc.iter_mut().enumerate().take(h) {
                    *a += trow[n] as i32 * c;
                }
            }
        }
        for n in 0..h {
            tmp[n * w + i] = clip(((acc[n] + add1) >> shift1) as i64);
        }
    }
    // Pass 2 — horizontal (trTypeHor, size w).
    let shift2 = 20 - bit_depth as i32;
    let add2 = 1i32 << (shift2 - 1);
    for row in 0..h {
        let mut acc = [0i32; 64];
        for k in 0..kw {
            let c = tmp[row * w + k];
            if c != 0 {
                let trow = tr_row(tr_hor, w, k);
                for (n, a) in acc.iter_mut().enumerate().take(w) {
                    *a += trow[n] as i32 * c;
                }
            }
        }
        for n in 0..w {
            out[row * w + n] = (acc[n] + add2) >> shift2;
        }
    }
}

/// Map a coded `mts_idx` (0 = DCT-II/DCT-II; 1..=4 = the four DST-VII/DCT-VIII
/// pairs) to `(tr_hor, tr_ver)` transform-type codes (0 = DCT-II, 1 = DST-VII,
/// 2 = DCT-VIII), per VVC `TrQuant.cpp`: `indHor=(idx-1)&1`, `indVer=(idx-1)>>1`.
#[inline]
pub(crate) fn mts_to_types(mts_idx: u8) -> (u8, u8) {
    if mts_idx == 0 {
        return (0, 0);
    }
    let k = mts_idx - 1;
    let hor = if k & 1 != 0 { 2 } else { 1 };
    let ver = if k & 2 != 0 { 2 } else { 1 };
    (hor, ver)
}

/// Inverse MTS transform returning a fresh `[i32; MAX_TB]` (companion to
/// [`inv_transform_wh`], selecting DST-VII/DCT-VIII per dimension).
pub(crate) fn inv_transform_mts_wh(
    coeff: &[i32],
    w: usize,
    h: usize,
    bit_depth: u8,
    mts_idx: u8,
) -> [i32; MAX_TB] {
    let (th, tv) = mts_to_types(mts_idx);
    let mut out = [0i32; MAX_TB];
    inv_transform_mts_into(&mut out, coeff, w, h, bit_depth, th, tv);
    out
}
