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

use crate::cabac::CtxModel;

static SPLIT_FLAG_INIT_I: [u8; 9] = [19, 28, 38, 27, 29, 38, 20, 30, 31];
static SPLIT_FLAG_RATE: [u8; 9] = [12, 13, 8, 8, 13, 12, 5, 9, 9];

static SPLIT_QT_FLAG_INIT_I: [u8; 6] = [27, 6, 15, 25, 19, 37];
static SPLIT_QT_FLAG_RATE: [u8; 6] = [0, 8, 8, 12, 12, 8];

// Selects vertical vs horizontal once an MTT (binary/ternary) split is chosen.
static SPLIT_HV_FLAG_INIT_I: [u8; 5] = [43, 42, 29, 27, 44];
static SPLIT_HV_FLAG_RATE: [u8; 5] = [9, 8, 9, 8, 5];

// Selects binary (1:1) vs ternary (1:2:1) for the chosen MTT orientation.
static SPLIT_12_FLAG_INIT_I: [u8; 4] = [36, 45, 36, 45];
/// LFNST index (lfnst_idx): I-slice init + rate, VTM ContextSetCfg::LFNSTIdx.
static LFNST_IDX_INIT_I: [u8; 3] = [28, 52, 42];
static LFNST_IDX_RATE: [u8; 3] = [9, 9, 10];
static MTS_IDX_INIT_I: [u8; 4] = [29, 0, 28, 0];
static MTS_IDX_RATE: [u8; 4] = [8, 0, 9, 0];
static SPLIT_12_FLAG_RATE: [u8; 4] = [12, 13, 12, 13];

static INTRA_MPM_FLAG_INIT_I: [u8; 1] = [45];
static INTRA_MPM_FLAG_RATE: [u8; 1] = [6];

static INTRA_PLANAR_FLAG_INIT_I: [u8; 2] = [13, 28];
static INTRA_PLANAR_FLAG_RATE: [u8; 2] = [1, 5];

// sig_coeff_group_flag (SigCoeffGroup, luma set), 2 contexts
static SIG_GROUP_INIT_I: [u8; 2] = [18, 31];
static SIG_GROUP_RATE: [u8; 2] = [8, 5];
// sig_coeff_flag (SigFlag, luma base set), 12 contexts
static SIG_FLAG_INIT_I: [u8; 12] = [25, 19, 28, 14, 25, 20, 29, 30, 19, 37, 30, 38];
static SIG_FLAG_RATE: [u8; 12] = [12, 9, 9, 10, 9, 9, 9, 10, 8, 8, 8, 10];
// par_level_flag (ParFlag, luma set), 21 contexts
static PAR_FLAG_INIT_I: [u8; 21] = [
    33, 25, 18, 26, 34, 27, 25, 26, 19, 42, 35, 33, 19, 27, 35, 35, 34, 42, 20, 43, 20,
];
static PAR_FLAG_RATE: [u8; 21] = [
    8, 9, 12, 13, 13, 13, 10, 13, 13, 13, 13, 13, 13, 13, 13, 13, 10, 13, 13, 13, 13,
];
// abs_level_gt1_flag (GtxFlag[chType+2], luma), 21 contexts
static GT1_FLAG_INIT_I: [u8; 21] = [
    25, 25, 11, 27, 20, 21, 33, 12, 28, 21, 22, 34, 28, 29, 29, 30, 36, 29, 45, 30, 23,
];
static GT1_FLAG_RATE: [u8; 21] = [
    9, 5, 10, 13, 13, 10, 9, 10, 13, 13, 13, 9, 10, 10, 10, 13, 8, 9, 10, 10, 13,
];
// abs_level_gt3_flag / "greater2" (GtxFlag[chType], luma), 21 contexts
static GT2_FLAG_INIT_I: [u8; 21] = [
    25, 1, 40, 25, 33, 11, 17, 25, 25, 18, 4, 17, 33, 26, 19, 13, 33, 19, 20, 28, 22,
];
static GT2_FLAG_RATE: [u8; 21] = [
    1, 5, 9, 9, 9, 6, 5, 9, 10, 10, 9, 9, 9, 9, 9, 9, 6, 8, 9, 9, 10,
];
// last_sig_coeff_x_prefix (LastX, luma set), 20 contexts
static LAST_X_INIT_I: [u8; 20] = [
    13, 5, 4, 21, 14, 4, 6, 14, 21, 11, 14, 7, 14, 5, 11, 21, 30, 22, 13, 42,
];
static LAST_X_RATE: [u8; 20] = [8, 5, 4, 5, 4, 4, 5, 4, 1, 0, 4, 1, 0, 0, 0, 0, 1, 0, 0, 0];
// last_sig_coeff_y_prefix (LastY, luma set), 20 contexts
static LAST_Y_INIT_I: [u8; 20] = [
    13, 5, 4, 6, 13, 11, 14, 6, 5, 3, 14, 22, 6, 4, 3, 6, 22, 29, 20, 34,
];
static LAST_Y_RATE: [u8; 20] = [8, 5, 8, 5, 5, 4, 5, 5, 4, 0, 5, 4, 1, 0, 0, 1, 4, 0, 0, 0];
static QT_CBF_LUMA_INIT_I: [u8; 4] = [15, 12, 5, 7];
static QT_CBF_LUMA_RATE: [u8; 4] = [5, 1, 8, 9];
static SIG_GROUP_C_INIT_I: [u8; 2] = [25, 15];
static SIG_GROUP_C_RATE: [u8; 2] = [5, 8];
static SIG_FLAG_C_INIT_I: [u8; 8] = [25, 27, 28, 37, 34, 53, 53, 46];
// Dependent-quantization state-dependent significance sets (VTM SigFlag[2..6]).
// State group selects the set: max(0,state-1) -> 0 (above), 1 (st1), 2 (st2).
static SIG_FLAG_ST1_INIT_I: [u8; 12] = [11, 38, 46, 54, 27, 39, 39, 39, 44, 39, 39, 39];
static SIG_FLAG_ST1_RATE: [u8; 12] = [9, 13, 8, 8, 8, 8, 8, 5, 8, 0, 0, 0];
static SIG_FLAG_ST2_INIT_I: [u8; 12] = [18, 39, 39, 39, 27, 39, 39, 39, 0, 39, 39, 39];
static SIG_FLAG_ST2_RATE: [u8; 12] = [8, 8, 8, 8, 8, 0, 4, 4, 0, 0, 0, 0];
static SIG_FLAG_C_ST1_INIT_I: [u8; 8] = [19, 46, 38, 39, 52, 39, 39, 39];
static SIG_FLAG_C_ST1_RATE: [u8; 8] = [8, 12, 12, 8, 4, 0, 0, 0];
static SIG_FLAG_C_ST2_INIT_I: [u8; 8] = [11, 39, 39, 39, 19, 39, 39, 39];
static SIG_FLAG_C_ST2_RATE: [u8; 8] = [8, 8, 8, 8, 4, 0, 0, 0];
static SIG_FLAG_C_RATE: [u8; 8] = [12, 12, 9, 13, 4, 5, 8, 9];
static PAR_FLAG_C_INIT_I: [u8; 11] = [33, 25, 26, 42, 19, 27, 26, 50, 35, 20, 43];
static PAR_FLAG_C_RATE: [u8; 11] = [8, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13];
static GT1_FLAG_C_INIT_I: [u8; 11] = [40, 33, 27, 28, 21, 37, 36, 37, 45, 38, 46];
static GT1_FLAG_C_RATE: [u8; 11] = [8, 8, 9, 12, 12, 10, 5, 9, 9, 9, 13];
static GT2_FLAG_C_INIT_I: [u8; 11] = [40, 9, 25, 18, 26, 35, 25, 26, 35, 28, 37];
static GT2_FLAG_C_RATE: [u8; 11] = [1, 5, 8, 8, 9, 6, 6, 9, 8, 8, 9];
static LAST_X_C_INIT_I: [u8; 3] = [12, 4, 3];
static LAST_X_C_RATE: [u8; 3] = [5, 4, 4];
static LAST_Y_C_INIT_I: [u8; 3] = [12, 4, 3];
static LAST_Y_C_RATE: [u8; 3] = [6, 5, 5];
static QT_CBF_CB_INIT_I: [u8; 2] = [12, 21];
static QT_CBF_CB_RATE: [u8; 2] = [5, 0];
static QT_CBF_CR_INIT_I: [u8; 3] = [33, 28, 36];
static QT_CBF_CR_RATE: [u8; 3] = [2, 1, 0];
static INTRA_CHROMA_MODE_INIT_I: [u8; 1] = [34];
static INTRA_CHROMA_MODE_RATE: [u8; 1] = [5];
// cclm_mode_flag (CclmModeFlag) + cclm_mode_idx (CclmModeIdx), 1 context each.
static CCLM_MODE_FLAG_INIT_I: [u8; 1] = [59];
static CCLM_MODE_FLAG_RATE: [u8; 1] = [4];
static CCLM_MODE_IDX_INIT_I: [u8; 1] = [27];
static CCLM_MODE_IDX_RATE: [u8; 1] = [9];

// sig_coeff_group_flag (TsSigCoeffGroup), indexed by neighbor-group count 0..2
static TS_SIG_GROUP_INIT_I: [u8; 3] = [18, 20, 38];
static TS_SIG_GROUP_RATE: [u8; 3] = [5, 8, 8];
// sig_coeff_flag (TsSigFlag), indexed by significant-neighbor count 0..2
static TS_SIG_FLAG_INIT_I: [u8; 3] = [25, 28, 38];
static TS_SIG_FLAG_RATE: [u8; 3] = [13, 13, 8];
// par_level_flag (TsParFlag), 1 context
static TS_PAR_FLAG_INIT_I: [u8; 1] = [11];
static TS_PAR_FLAG_RATE: [u8; 1] = [6];
// abs_level_gtx_flag pass-2 (TsGtxFlag), index 0 unused (CNU/DWS)
static TS_GTX_FLAG_INIT_I: [u8; 5] = [35, 10, 3, 3, 3];
static TS_GTX_FLAG_RATE: [u8; 5] = [8, 1, 1, 1, 1];
// abs_level_gt1_flag pass-1 (TsLrg1Flag), indexed by neighbour count (3 = BDPCM)
static TS_LRG1_FLAG_INIT_I: [u8; 4] = [11, 5, 5, 14];
static TS_LRG1_FLAG_RATE: [u8; 4] = [4, 2, 1, 6];
// coeff_sign_flag (TsResidualSign), 6 contexts (sign pattern, +3 for BDPCM)
static TS_SIGN_FLAG_INIT_I: [u8; 6] = [12, 17, 46, 28, 25, 46];
static TS_SIGN_FLAG_RATE: [u8; 6] = [1, 4, 4, 5, 8, 8];
// mts_idx / transform_skip_flag (MTSIndex). Indices 4 (luma) and 5 (chroma)
// carry the transform_skip_flag; the rest belong to the (unused) MTS index.
static MTS_INDEX_INIT_I: [u8; 6] = [29, 0, 28, 0, 25, 9];
const MTS_INDEX_RATE: [u8; 6] = [8, 0, 9, 0, 1, 1];
// intra_bdpcm_*_flag / dir (BDPCMMode): 0/1 luma flag+dir, 2/3 chroma flag+dir
const BDPCM_MODE_INIT_I: [u8; 4] = [19, 35, 1, 27];
const BDPCM_MODE_RATE: [u8; 4] = [1, 4, 1, 0];
// cu_qp_delta_abs: VTM ContextSetCfg::DeltaQP, initValue=CNU(35), shiftIdx=DWS(8).
const CU_QP_DELTA_ABS_INIT_I: [u8; 2] = [35, 35];
const CU_QP_DELTA_ABS_RATE: [u8; 2] = [8, 8];

/// All CABAC contexts used while coding slice data, initialised for one slice.
/// Grows as more syntax elements (intra mode, CBF, residual) are implemented.
#[derive(Clone)]
pub(crate) struct Contexts {
    pub(crate) split_flag: [CtxModel; 9],
    pub(crate) split_qt_flag: [CtxModel; 6],
    pub(crate) mtt_split_vertical: [CtxModel; 5],
    pub(crate) mtt_split_binary: [CtxModel; 4],
    pub(crate) intra_mpm_flag: [CtxModel; 1],
    pub(crate) intra_planar_flag: [CtxModel; 2],
    pub(crate) sig_group: [CtxModel; 2],
    pub(crate) sig_flag: [CtxModel; 12],
    pub(crate) par_flag: [CtxModel; 21],
    pub(crate) gt1_flag: [CtxModel; 21],
    pub(crate) gt2_flag: [CtxModel; 21],
    pub(crate) last_x: [CtxModel; 20],
    pub(crate) last_y: [CtxModel; 20],
    pub(crate) qt_cbf_luma: [CtxModel; 4],
    pub(crate) sig_group_c: [CtxModel; 2],
    pub(crate) sig_flag_c: [CtxModel; 8],
    pub(crate) sig_flag_st1: [CtxModel; 12],
    pub(crate) sig_flag_st2: [CtxModel; 12],
    pub(crate) sig_flag_c_st1: [CtxModel; 8],
    pub(crate) sig_flag_c_st2: [CtxModel; 8],
    pub(crate) par_flag_c: [CtxModel; 11],
    pub(crate) gt1_flag_c: [CtxModel; 11],
    pub(crate) gt2_flag_c: [CtxModel; 11],
    pub(crate) last_x_c: [CtxModel; 3],
    pub(crate) last_y_c: [CtxModel; 3],
    pub(crate) qt_cbf_cb: [CtxModel; 2],
    pub(crate) qt_cbf_cr: [CtxModel; 3],
    pub(crate) intra_chroma_mode: [CtxModel; 1],
    pub(crate) cclm_mode_flag: [CtxModel; 1],
    pub(crate) cclm_mode_idx: [CtxModel; 1],
    pub(crate) ts_sig_group: [CtxModel; 3],
    pub(crate) ts_sig_flag: [CtxModel; 3],
    pub(crate) ts_par_flag: [CtxModel; 1],
    pub(crate) ts_gtx_flag: [CtxModel; 5],
    pub(crate) ts_lrg1_flag: [CtxModel; 4],
    pub(crate) ts_sign_flag: [CtxModel; 6],
    pub(crate) mts_index: [CtxModel; 6],
    pub(crate) bdpcm_mode: [CtxModel; 4],
    pub(crate) cu_qp_delta_abs: [CtxModel; 2],
    pub(crate) lfnst_idx: [CtxModel; 3],
    pub(crate) mts_idx: [CtxModel; 4],
}

/// Initialise a fixed-size context array from value + rate tables at `qp`.
fn init_set<const N: usize>(init: [u8; N], rate: [u8; N], qp: u8) -> [CtxModel; N] {
    std::array::from_fn(|i| CtxModel::init(init[i], qp, rate[i]))
}

impl Contexts {
    /// Significance-flag context model for offset `idx`, selecting the
    /// dependent-quantization state set: `group = max(0, state-1)` picks the
    /// base set (states 0/1), the state-1 set (state 2) or the state-2 set
    /// (state 3). With dependent quant off the state is pinned at 0, so this
    /// always returns the base set — identical to non-DQ coding.
    #[inline]
    pub(crate) fn sig_model(&mut self, luma: bool, state: u8, idx: usize) -> &mut CtxModel {
        let group = state.saturating_sub(1);
        match (luma, group) {
            (true, 0) => &mut self.sig_flag[idx],
            (true, 1) => &mut self.sig_flag_st1[idx],
            (true, _) => &mut self.sig_flag_st2[idx],
            (false, 0) => &mut self.sig_flag_c[idx],
            (false, 1) => &mut self.sig_flag_c_st1[idx],
            (false, _) => &mut self.sig_flag_c_st2[idx],
        }
    }
}

impl Contexts {
    /// Initialise every slice-data context for an I-slice at luma QP `qp`.
    pub(crate) fn new_intra(qp: u8) -> Self {
        Contexts {
            split_flag: init_set(SPLIT_FLAG_INIT_I, SPLIT_FLAG_RATE, qp),
            split_qt_flag: init_set(SPLIT_QT_FLAG_INIT_I, SPLIT_QT_FLAG_RATE, qp),
            mtt_split_vertical: init_set(SPLIT_HV_FLAG_INIT_I, SPLIT_HV_FLAG_RATE, qp),
            mtt_split_binary: init_set(SPLIT_12_FLAG_INIT_I, SPLIT_12_FLAG_RATE, qp),
            intra_mpm_flag: init_set(INTRA_MPM_FLAG_INIT_I, INTRA_MPM_FLAG_RATE, qp),
            intra_planar_flag: init_set(INTRA_PLANAR_FLAG_INIT_I, INTRA_PLANAR_FLAG_RATE, qp),
            sig_group: init_set(SIG_GROUP_INIT_I, SIG_GROUP_RATE, qp),
            sig_flag: init_set(SIG_FLAG_INIT_I, SIG_FLAG_RATE, qp),
            par_flag: init_set(PAR_FLAG_INIT_I, PAR_FLAG_RATE, qp),
            gt1_flag: init_set(GT1_FLAG_INIT_I, GT1_FLAG_RATE, qp),
            gt2_flag: init_set(GT2_FLAG_INIT_I, GT2_FLAG_RATE, qp),
            last_x: init_set(LAST_X_INIT_I, LAST_X_RATE, qp),
            last_y: init_set(LAST_Y_INIT_I, LAST_Y_RATE, qp),
            qt_cbf_luma: init_set(QT_CBF_LUMA_INIT_I, QT_CBF_LUMA_RATE, qp),
            sig_group_c: init_set(SIG_GROUP_C_INIT_I, SIG_GROUP_C_RATE, qp),
            sig_flag_c: init_set(SIG_FLAG_C_INIT_I, SIG_FLAG_C_RATE, qp),
            sig_flag_st1: init_set(SIG_FLAG_ST1_INIT_I, SIG_FLAG_ST1_RATE, qp),
            sig_flag_st2: init_set(SIG_FLAG_ST2_INIT_I, SIG_FLAG_ST2_RATE, qp),
            sig_flag_c_st1: init_set(SIG_FLAG_C_ST1_INIT_I, SIG_FLAG_C_ST1_RATE, qp),
            sig_flag_c_st2: init_set(SIG_FLAG_C_ST2_INIT_I, SIG_FLAG_C_ST2_RATE, qp),
            par_flag_c: init_set(PAR_FLAG_C_INIT_I, PAR_FLAG_C_RATE, qp),
            gt1_flag_c: init_set(GT1_FLAG_C_INIT_I, GT1_FLAG_C_RATE, qp),
            gt2_flag_c: init_set(GT2_FLAG_C_INIT_I, GT2_FLAG_C_RATE, qp),
            last_x_c: init_set(LAST_X_C_INIT_I, LAST_X_C_RATE, qp),
            last_y_c: init_set(LAST_Y_C_INIT_I, LAST_Y_C_RATE, qp),
            qt_cbf_cb: init_set(QT_CBF_CB_INIT_I, QT_CBF_CB_RATE, qp),
            qt_cbf_cr: init_set(QT_CBF_CR_INIT_I, QT_CBF_CR_RATE, qp),
            intra_chroma_mode: init_set(INTRA_CHROMA_MODE_INIT_I, INTRA_CHROMA_MODE_RATE, qp),
            cclm_mode_flag: init_set(CCLM_MODE_FLAG_INIT_I, CCLM_MODE_FLAG_RATE, qp),
            cclm_mode_idx: init_set(CCLM_MODE_IDX_INIT_I, CCLM_MODE_IDX_RATE, qp),
            ts_sig_group: init_set(TS_SIG_GROUP_INIT_I, TS_SIG_GROUP_RATE, qp),
            ts_sig_flag: init_set(TS_SIG_FLAG_INIT_I, TS_SIG_FLAG_RATE, qp),
            ts_par_flag: init_set(TS_PAR_FLAG_INIT_I, TS_PAR_FLAG_RATE, qp),
            ts_gtx_flag: init_set(TS_GTX_FLAG_INIT_I, TS_GTX_FLAG_RATE, qp),
            ts_lrg1_flag: init_set(TS_LRG1_FLAG_INIT_I, TS_LRG1_FLAG_RATE, qp),
            ts_sign_flag: init_set(TS_SIGN_FLAG_INIT_I, TS_SIGN_FLAG_RATE, qp),
            mts_index: init_set(MTS_INDEX_INIT_I, MTS_INDEX_RATE, qp),
            bdpcm_mode: init_set(BDPCM_MODE_INIT_I, BDPCM_MODE_RATE, qp),
            cu_qp_delta_abs: init_set(CU_QP_DELTA_ABS_INIT_I, CU_QP_DELTA_ABS_RATE, qp),
            lfnst_idx: init_set(LFNST_IDX_INIT_I, LFNST_IDX_RATE, qp),
            mts_idx: init_set(MTS_IDX_INIT_I, MTS_IDX_RATE, qp),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contexts_initialise_in_range() {
        let c = Contexts::new_intra(32);
        for ctx in c
            .split_flag
            .iter()
            .chain(c.split_qt_flag.iter())
            .chain(c.mtt_split_vertical.iter())
            .chain(c.mtt_split_binary.iter())
            .chain(c.sig_flag_st1.iter())
            .chain(c.sig_flag_st2.iter())
            .chain(c.sig_flag_c_st1.iter())
            .chain(c.sig_flag_c_st2.iter())
        {
            // mps() must be a valid bit; this also exercises the internal state.
            assert!(ctx.mps() <= 1);
        }
    }

    #[test]
    fn init_is_qp_dependent() {
        let a = Contexts::new_intra(12);
        let b = Contexts::new_intra(51);
        // At least one split_flag context should differ between QPs.
        let differ = a
            .split_flag
            .iter()
            .zip(b.split_flag.iter())
            .any(|(x, y)| x.get_lps(384) != y.get_lps(384));
        assert!(differ, "context init should depend on QP");
    }
}
