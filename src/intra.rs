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

//! Intra luma prediction-mode signalling: the most-probable-mode (MPM) list
//! derivation (H.266 §8.4.2 / `PU::getIntraMPMs`) and the `intra_luma_pred_mode`
//! syntax (`intra_luma_mpm_flag`, `intra_luma_not_planar_flag`, the MPM index,
//! and the truncated-binary remainder). MIP, MRL, ISP, and BDPCM are disabled in
//! garnetash v1, so this is the plain MPM path.
#![allow(dead_code)]

use crate::cabac::{CabacEncoder, Contexts};

/// Planar intra mode index.
pub(crate) const PLANAR_IDX: u8 = 0;
/// DC intra mode index.
pub(crate) const DC_IDX: u8 = 1;
/// Horizontal angular mode index.
pub(crate) const HOR_IDX: u8 = 18;
/// Vertical angular mode index.
pub(crate) const VER_IDX: u8 = 50;
/// Top-right diagonal angular mode index (VDIA). Used as the chroma substitute
/// when a fixed chroma candidate coincides with the luma (DM) mode.
pub(crate) const VDIA_IDX: u8 = 66;
/// Number of luma modes (planar, DC, and 65 angular).
pub(crate) const NUM_LUMA_MODE: u8 = 67;
/// Number of most-probable modes.
pub(crate) const NUM_MPM: usize = 6;
/// Number of non-MPM modes coded by the truncated-binary remainder.
pub(crate) const NUM_REM_MODES: u32 = (NUM_LUMA_MODE as u32) - (NUM_MPM as u32); // 61

/// Build the 6-entry MPM list from the left and above neighbour luma modes
/// (`None` when unavailable, treated as planar). Port of `PU::getIntraMPMs`.
pub(crate) fn build_mpm(left: Option<u8>, above: Option<u8>) -> [u8; NUM_MPM] {
    let left = left.unwrap_or(PLANAR_IDX) as i32;
    let above = above.unwrap_or(PLANAR_IDX) as i32;
    let dc = DC_IDX as i32;
    let offset = NUM_LUMA_MODE as i32 - 6; // 61
    let modv = offset + 3; // 64

    // Default list {PLANAR, DC, VER, HOR, VER-4, VER+4}, overwritten below
    // whenever a neighbour is angular (H.266 `PU::getIntraMPMs`).
    let mut mpm = [
        PLANAR_IDX,
        DC_IDX,
        VER_IDX,
        HOR_IDX,
        VER_IDX - 4,
        VER_IDX + 4,
    ];
    // Modular angular wrap, matching VTM's `((m + delta) % mod) + 2` form. The
    // mode passed in is the raw value (no angular-base subtraction).
    let w = |m: i32| ((m % modv) + 2) as u8;

    if left == above {
        if left > dc {
            mpm[1] = left as u8;
            mpm[2] = w(left + offset);
            mpm[3] = w(left - 1);
            mpm[4] = w(left + offset - 1);
            mpm[5] = w(left);
        }
    } else if left > dc && above > dc {
        mpm[1] = left as u8;
        mpm[2] = above as u8;
        let (max_c, min_c) = if left > above {
            (left, above)
        } else {
            (above, left)
        };
        let diff = max_c - min_c;
        if diff == 1 {
            mpm[3] = w(min_c + offset);
            mpm[4] = w(max_c - 1);
            mpm[5] = w(min_c + offset - 1);
        } else if diff >= 62 {
            mpm[3] = w(min_c - 1);
            mpm[4] = w(max_c + offset);
            mpm[5] = w(min_c);
        } else if diff == 2 {
            mpm[3] = w(min_c - 1);
            mpm[4] = w(min_c + offset);
            mpm[5] = w(max_c - 1);
        } else {
            mpm[3] = w(min_c + offset);
            mpm[4] = w(min_c - 1);
            mpm[5] = w(max_c + offset);
        }
    } else if left + above >= 2 {
        // Exactly one neighbour is angular.
        let m1 = left.max(above);
        mpm[1] = m1 as u8;
        mpm[2] = w(m1 + offset);
        mpm[3] = w(m1 - 1);
        mpm[4] = w(m1 + offset - 1);
        mpm[5] = w(m1);
    }
    mpm
}

/// Threshold (`floorLog2`) for a positive value.
#[inline]
fn floor_log2(x: u32) -> u32 {
    31 - x.leading_zeros()
}

/// Encode `symbol` in `[0, num_symbols)` with a truncated-binary code
/// (`xWriteTruncBinCode`), via equal-probability bypass bins.
fn encode_trunc_bin(enc: &mut CabacEncoder, symbol: u32, num_symbols: u32) {
    let thresh = floor_log2(num_symbols);
    let val = 1u32 << thresh;
    let b = num_symbols - val;
    if symbol < val - b {
        enc.encode_bypass_bits(symbol, thresh);
    } else {
        enc.encode_bypass_bits(symbol + val - b, thresh + 1);
    }
}

/// Code `intra_bdpcm_*_flag` and, when set, `intra_bdpcm_*_dir_flag`
/// (H.266 7.3.11.5). `mode` is 0 = none, 1 = horizontal, 2 = vertical; the
/// direction bin is 0 for horizontal, 1 for vertical. `is_luma` selects the
/// context pair (0/1 for luma, 2/3 for chroma).
pub(crate) fn encode_bdpcm_mode(
    enc: &mut CabacEncoder,
    ctx: &mut Contexts,
    mode: u8,
    is_luma: bool,
) {
    let c = if is_luma { 0 } else { 2 };
    enc.encode_bin((mode != 0) as u8, &mut ctx.bdpcm_mode[c]);
    if mode != 0 {
        enc.encode_bin((mode == 2) as u8, &mut ctx.bdpcm_mode[c + 1]);
    }
}

/// Decode `intra_bdpcm_*_flag` / `intra_bdpcm_*_dir_flag`, the inverse of
/// [`encode_bdpcm_mode`]. Returns 0 = none, 1 = horizontal, 2 = vertical.
pub(crate) fn decode_bdpcm_mode(
    dec: &mut crate::cabac::engine::CabacDecoder,
    ctx: &mut Contexts,
    is_luma: bool,
) -> u8 {
    let c = if is_luma { 0 } else { 2 };
    if dec.decode_bin(&mut ctx.bdpcm_mode[c]) == 0 {
        return 0;
    }
    if dec.decode_bin(&mut ctx.bdpcm_mode[c + 1]) == 1 {
        2
    } else {
        1
    }
}

/// Encode the luma intra mode for one CU given its MPM list.
pub(crate) fn encode_luma_mode(
    enc: &mut CabacEncoder,
    ctx: &mut Contexts,
    mpm: &[u8; NUM_MPM],
    mode: u8,
) {
    let mpm_idx = mpm.iter().position(|&m| m == mode);
    match mpm_idx {
        Some(idx) => {
            enc.encode_bin(1, &mut ctx.intra_mpm_flag[0]); // in MPM list
            // not-planar flag (ISP disabled -> context 1)
            enc.encode_bin((idx > 0) as u8, &mut ctx.intra_planar_flag[1]);
            if idx >= 1 {
                enc.encode_bypass((idx > 1) as u8);
            }
            if idx >= 2 {
                enc.encode_bypass((idx > 2) as u8);
            }
            if idx >= 3 {
                enc.encode_bypass((idx > 3) as u8);
            }
            if idx >= 4 {
                enc.encode_bypass((idx > 4) as u8);
            }
        }
        None => {
            enc.encode_bin(0, &mut ctx.intra_mpm_flag[0]); // not in MPM list
            // Remainder: rank of `mode` among the non-MPM modes.
            let mut sorted = *mpm;
            sorted.sort_unstable();
            let mut rem = mode as i32;
            for idx in (0..NUM_MPM).rev() {
                if rem > sorted[idx] as i32 {
                    rem -= 1;
                }
            }
            encode_trunc_bin(enc, rem as u32, NUM_REM_MODES);
        }
    }
}

pub(crate) fn decode_luma_mode(
    dec: &mut crate::cabac::engine::CabacDecoder,
    ctx: &mut Contexts,
    mpm: &[u8; NUM_MPM],
) -> u8 {
    if dec.decode_bin(&mut ctx.intra_mpm_flag[0]) != 0 {
        // In MPM list.
        let not_planar = dec.decode_bin(&mut ctx.intra_planar_flag[1]) != 0;
        if !not_planar {
            return mpm[0];
        }
        let mut idx = 1usize;
        // Up to four bypass bins extend the index (truncated unary).
        if dec.decode_bypass() != 0 {
            idx = 2;
            if dec.decode_bypass() != 0 {
                idx = 3;
                if dec.decode_bypass() != 0 {
                    idx = 4;
                    if dec.decode_bypass() != 0 {
                        idx = 5;
                    }
                }
            }
        }
        mpm[idx]
    } else {
        // Remainder: truncated-binary, then re-insert MPM gaps.
        let thresh = floor_log2(NUM_REM_MODES);
        let val = 1u32 << thresh;
        let b = NUM_REM_MODES - val;
        let prefix = dec_bypass_bits(dec, thresh);
        let rem = if prefix < val - b {
            prefix
        } else {
            ((prefix << 1) | dec.decode_bypass() as u32) - (val - b)
        };
        let mut sorted = *mpm;
        sorted.sort_unstable();
        let mut mode = rem as i32;
        for &s in sorted.iter() {
            if mode >= s as i32 {
                mode += 1;
            }
        }
        mode as u8
    }
}

fn dec_bypass_bits(dec: &mut crate::cabac::engine::CabacDecoder, n: u32) -> u32 {
    let mut v = 0;
    for _ in 0..n {
        v = (v << 1) | dec.decode_bypass() as u32;
    }
    v
}

/// Code `intra_chroma_pred_mode` for the derived-mode (DM) case: chroma reuses
/// the luma mode. With CCLM disabled this is a single context-coded `0` bin
/// (H.266 `intra_chroma_pred_mode`).
/// The four fixed chroma intra candidates {planar, vertical, horizontal, DC},
/// with the one coinciding with the luma (DM) mode replaced by VDIA so that the
/// derived mode and the signalled candidates never alias (H.266
/// `PU::getIntraChromaCandModes`, CCLM disabled).
/// 4:2:2 chroma intra-mode angle mapping (H.266 Table 8-3,
/// `g_chroma422IntraAngleMappingTable`). Because 4:2:2 chroma is subsampled 2:1
/// horizontally only, a chroma block predicting along a luma direction must use
/// a re-angled mode so the predicted direction matches the physical geometry.
/// Indexed by the resolved chroma mode (0..=66); planar/DC/pure-H/pure-V map to
/// themselves. The *coded* mode is unchanged — only the predictor is remapped.
#[rustfmt::skip]
static CHROMA_422_MAP: [u8; 67] = [
    0, 1, 61, 62, 63, 64, 65, 66, 2, 3, 5, 6, 8, 10, 12, 13, 14, 16, 18, 20, 22,
    23, 24, 26, 28, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 41, 42, 43, 43,
    44, 44, 45, 45, 46, 47, 48, 48, 49, 49, 50, 51, 51, 52, 52, 53, 54, 55, 55,
    56, 56, 57, 57, 58, 59, 59, 60,
];

/// Map a resolved chroma intra mode to the direction actually used for 4:2:2
/// prediction. Returns the input unchanged for non-angular modes.
pub(crate) fn chroma_422_mode(mode: u8) -> u8 {
    if (mode as usize) < CHROMA_422_MAP.len() {
        CHROMA_422_MAP[mode as usize]
    } else {
        mode
    }
}

pub(crate) fn chroma_cand_modes(luma_mode: u8) -> [u8; 4] {
    let mut list = [PLANAR_IDX, VER_IDX, HOR_IDX, DC_IDX];
    for m in list.iter_mut() {
        if *m == luma_mode {
            *m = VDIA_IDX;
            break;
        }
    }
    list
}

/// Code `intra_chroma_pred_mode`: a single context bin selects the derived mode
/// (DM = luma), otherwise two bypass bins pick one of the fixed candidates.
/// Chroma-mode markers for the three CCLM modes (outside the 0..66 intra-dir
/// range and the DM case): LT (LM_CHROMA), L (MDLM_L), T (MDLM_T).
pub(crate) const CCLM_LT_MODE: u8 = 81;
pub(crate) const CCLM_L_MODE: u8 = 82;
pub(crate) const CCLM_T_MODE: u8 = 83;

/// True if `m` is one of the CCLM chroma modes.
pub(crate) fn is_cclm_mode(m: u8) -> bool {
    (CCLM_LT_MODE..=CCLM_T_MODE).contains(&m)
}

pub(crate) fn encode_chroma_mode(
    enc: &mut CabacEncoder,
    ctx: &mut Contexts,
    luma_mode: u8,
    chroma_mode: u8,
    cclm_enabled: bool,
) {
    if cclm_enabled {
        // cclm_mode_flag, then for CCLM the mode index (LT via a context bin = 0,
        // else context bin = 1 + a bypass bin selecting L vs T). H.266 §7.3.9.5.
        let is_lmc = is_cclm_mode(chroma_mode);
        enc.encode_bin(is_lmc as u8, &mut ctx.cclm_mode_flag[0]);
        if is_lmc {
            let symbol = chroma_mode - CCLM_LT_MODE; // 0 = LT, 1 = L, 2 = T
            enc.encode_bin((symbol != 0) as u8, &mut ctx.cclm_mode_idx[0]);
            if symbol > 0 {
                enc.encode_bypass(symbol - 1);
            }
            return;
        }
    }
    if chroma_mode == luma_mode {
        enc.encode_bin(0, &mut ctx.intra_chroma_mode[0]);
        return;
    }
    enc.encode_bin(1, &mut ctx.intra_chroma_mode[0]);
    let list = chroma_cand_modes(luma_mode);
    let cand_id = list
        .iter()
        .position(|&m| m == chroma_mode)
        .expect("chroma mode must be DM or a fixed candidate") as u8;
    // Two bypass bins, most-significant first.
    enc.encode_bypass((cand_id >> 1) & 1);
    enc.encode_bypass(cand_id & 1);
}

pub(crate) fn decode_chroma_mode(
    dec: &mut crate::cabac::engine::CabacDecoder,
    ctx: &mut Contexts,
    luma_mode: u8,
    cclm_enabled: bool,
) -> u8 {
    if cclm_enabled && dec.decode_bin(&mut ctx.cclm_mode_flag[0]) == 1 {
        if dec.decode_bin(&mut ctx.cclm_mode_idx[0]) == 0 {
            return CCLM_LT_MODE;
        }
        return if dec.decode_bypass() == 0 {
            CCLM_L_MODE
        } else {
            CCLM_T_MODE
        };
    }
    if dec.decode_bin(&mut ctx.intra_chroma_mode[0]) == 0 {
        return luma_mode; // DM
    }
    let hi = dec.decode_bypass();
    let lo = dec.decode_bypass();
    let cand_id = ((hi << 1) | lo) as usize;
    chroma_cand_modes(luma_mode)[cand_id]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cclm_mode_binarization_round_trips() {
        use crate::cabac::engine::{CabacDecoder, CabacEncoder};
        // With CCLM enabled, every chroma mode — the three CCLM modes, the DM
        // case, and each non-DM candidate — must survive encode→decode, and the
        // CCLM modes must not collide with the normal path.
        let luma = 25u8; // an arbitrary angular luma mode (the DM source)
        let mut modes = vec![CCLM_LT_MODE, CCLM_L_MODE, CCLM_T_MODE, luma];
        modes.extend_from_slice(&chroma_cand_modes(luma));
        let mut enc = CabacEncoder::new();
        let mut ectx = Contexts::new_intra(26);
        for &m in &modes {
            encode_chroma_mode(&mut enc, &mut ectx, luma, m, true);
        }
        enc.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = CabacDecoder::new(&bytes);
        let mut dctx = Contexts::new_intra(26);
        for &m in &modes {
            assert_eq!(
                decode_chroma_mode(&mut dec, &mut dctx, luma, true),
                m,
                "mode {m}"
            );
        }
    }

    #[test]
    fn chroma_422_mode_matches_vvc_table() {
        // Non-angular and the pure axes map to themselves.
        assert_eq!(chroma_422_mode(PLANAR_IDX), PLANAR_IDX);
        assert_eq!(chroma_422_mode(DC_IDX), DC_IDX);
        assert_eq!(chroma_422_mode(HOR_IDX), HOR_IDX); // 18 -> 18
        assert_eq!(chroma_422_mode(VER_IDX), VER_IDX); // 50 -> 50
        // Representative angular remappings from H.266 Table 8-3.
        assert_eq!(chroma_422_mode(2), 61);
        assert_eq!(chroma_422_mode(7), 66);
        assert_eq!(chroma_422_mode(64), 59);
        assert_eq!(chroma_422_mode(VDIA_IDX), 60); // 66 -> 60
        // Idempotent only where the table maps to a fixed point; just bounds-check.
        for m in 0..=VDIA_IDX {
            assert!(chroma_422_mode(m) <= VDIA_IDX);
        }
    }

    #[test]
    fn mpm_always_starts_with_planar() {
        for l in [None, Some(0u8), Some(1), Some(18), Some(50), Some(66)] {
            for a in [None, Some(0u8), Some(1), Some(34), Some(60)] {
                let m = build_mpm(l, a);
                assert_eq!(m[0], PLANAR_IDX);
                assert!(m.iter().all(|&x| (x as u8) < NUM_LUMA_MODE));
                // All six entries distinct (a property VVC's derivation guarantees).
                for i in 0..NUM_MPM {
                    for j in (i + 1)..NUM_MPM {
                        assert_ne!(m[i], m[j], "dup in MPM {m:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn both_nonangular_uses_fixed_set() {
        let m = build_mpm(Some(DC_IDX), Some(PLANAR_IDX));
        assert_eq!(
            m,
            [
                PLANAR_IDX,
                DC_IDX,
                VER_IDX,
                HOR_IDX,
                VER_IDX - 4,
                VER_IDX + 4
            ]
        );
    }

    #[test]
    fn mpm_matches_vtm_reference_values() {
        // Reference lists produced by VTM/vvdec `PU::getIntraMPMs` (offset 61,
        // mod 64). These pin the modular wrap exactly; the diff==62 case is the
        // boundary that a `>= 63` threshold gets wrong.
        // L=66, A=4 -> diff 62 wrap branch.
        assert_eq!(build_mpm(Some(66), Some(4)), [0, 66, 4, 5, 65, 6]);
        // L==A angular: neighbours of the shared mode.
        assert_eq!(build_mpm(Some(50), Some(50)), [0, 50, 49, 51, 48, 52]);
        // diff == 1.
        assert_eq!(build_mpm(Some(30), Some(31)), [0, 30, 31, 29, 32, 28]);
        // diff == 2.
        assert_eq!(build_mpm(Some(30), Some(32)), [0, 30, 32, 31, 29, 33]);
        // One angular, one planar/DC.
        assert_eq!(build_mpm(Some(40), Some(DC_IDX)), [0, 40, 39, 41, 38, 42]);
    }

    #[test]
    fn trunc_bin_threshold() {
        assert_eq!(floor_log2(61), 5);
        assert_eq!(floor_log2(64), 6);
        assert_eq!(floor_log2(1), 0);
    }
    #[test]
    fn chroma_mode_round_trips_all_candidates() {
        use crate::cabac::engine::CabacDecoder;
        // For several luma modes, every selectable chroma mode (DM plus the four
        // fixed candidates after VDIA substitution) must encode and decode back.
        for &luma in &[0u8, 1, 18, 30, 50, 66] {
            let mut targets = vec![luma]; // DM
            targets.extend_from_slice(&chroma_cand_modes(luma));
            for &cm in &targets {
                let mut enc = CabacEncoder::new();
                let mut ectx = Contexts::new_intra(30);
                encode_chroma_mode(&mut enc, &mut ectx, luma, cm, false);
                enc.encode_terminate(1);
                let bytes = enc.finish();
                let mut dec = CabacDecoder::new(&bytes);
                let mut dctx = Contexts::new_intra(30);
                assert_eq!(
                    decode_chroma_mode(&mut dec, &mut dctx, luma, false),
                    cm,
                    "luma={luma} cm={cm}"
                );
                assert_eq!(dec.decode_terminate(), 1);
            }
        }
    }

    #[test]
    fn chroma_candidates_avoid_dm_alias() {
        // When luma is one of the fixed candidates, that slot becomes VDIA.
        assert_eq!(
            chroma_cand_modes(VER_IDX),
            [PLANAR_IDX, VDIA_IDX, HOR_IDX, DC_IDX]
        );
        assert_eq!(
            chroma_cand_modes(PLANAR_IDX),
            [VDIA_IDX, VER_IDX, HOR_IDX, DC_IDX]
        );
        // An angular luma mode not in the set leaves the list untouched.
        assert_eq!(
            chroma_cand_modes(30),
            [PLANAR_IDX, VER_IDX, HOR_IDX, DC_IDX]
        );
    }
}
