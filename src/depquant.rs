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

#![allow(dead_code)] // wired in subsequent stages

const STATE_TRANS_PACKED: u32 = 32040;

pub(crate) static STATE_TRANS: [[u8; 2]; 4] = [[0, 2], [2, 0], [1, 3], [3, 1]];

/// Next dependent-quant state after coding `level` (any sign) from `state`.
#[inline]
pub(crate) fn next_state(state: u8, level: i32) -> u8 {
    ((STATE_TRANS_PACKED >> (((state as u32) << 2) + (((level & 1) as u32) << 1))) & 3) as u8
}

#[inline]
pub(crate) fn quantizer(state: u8) -> u8 {
    state >> 1
}

#[inline]
pub(crate) fn recon_qidx(level: i32, state: u8) -> i32 {
    if level == 0 {
        return 0; // a zero coefficient reconstructs to zero (never dequantized in VTM)
    }
    let q = (state >> 1) as i32;
    2 * level + if level > 0 { -q } else { q }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_matches_explicit_table() {
        for state in 0u8..4 {
            for parity in 0i32..2 {
                // next_state only looks at level&1, so any level of that parity works.
                let got = next_state(state, parity);
                assert_eq!(
                    got, STATE_TRANS[state as usize][parity as usize],
                    "state {state} parity {parity}"
                );
            }
        }
    }

    #[test]
    fn quantizer_partitions_states() {
        // States {0,1} use Q0, {2,3} use Q1 (state >> 1).
        assert_eq!(quantizer(0), 0);
        assert_eq!(quantizer(1), 0);
        assert_eq!(quantizer(2), 1);
        assert_eq!(quantizer(3), 1);
    }

    #[test]
    fn recon_qidx_grid() {
        // Q0 (state 0/1): qIdx == 2*level (even grid).
        for &s in &[0u8, 1] {
            assert_eq!(recon_qidx(3, s), 6);
            assert_eq!(recon_qidx(-3, s), -6);
            assert_eq!(recon_qidx(0, s), 0);
        }
        // Q1 (state 2/3): magnitude reduced by one (odd grid), sign preserved.
        for &s in &[2u8, 3] {
            assert_eq!(recon_qidx(3, s), 5); // 2*3 - 1
            assert_eq!(recon_qidx(-3, s), -5); // -6 + 1
            assert_eq!(recon_qidx(1, s), 1); // 2 - 1
            assert_eq!(recon_qidx(0, s), 0);
        }
    }

    #[test]
    fn state_walk_is_deterministic_and_bounded() {
        // A scan of arbitrary levels keeps the state in 0..4 and is reproducible.
        let levels = [0i32, 1, 2, -3, 4, 0, 7, -1, 1, 1, 2, 2];
        let mut state = 0u8;
        let mut trace = Vec::new();
        for &l in &levels {
            assert!(state < 4);
            trace.push(state);
            state = next_state(state, l);
        }
        // Re-running yields the identical trace (no hidden state).
        let mut s2 = 0u8;
        for (k, &l) in levels.iter().enumerate() {
            assert_eq!(s2, trace[k]);
            s2 = next_state(s2, l);
        }
    }
}
