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

//! Encoder error type. Kept small and `std::error::Error`-compatible so callers
//! can use `?` without pulling in a dependency.

use std::fmt;

use crate::fmt::BitDepth;

/// Errors returned by the `garnetash` VVC image encoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodeError {
    /// Image width or height was 0 or exceeded the supported maximum.
    InvalidDimensions { width: u32, height: u32 },
    /// `quality` was outside the valid 1..=100 range.
    InvalidQuality(u8),
    /// The supplied pixel buffer length did not match `width * height * channels`.
    BufferSize { expected: usize, found: usize },
    /// A sample exceeded the maximum value permitted by the configured bit depth.
    SampleOutOfRange { value: u16, depth: BitDepth },
    /// A requested feature is recognised but not yet implemented in this build.
    Unsupported(&'static str),
    /// A bitstream could not be decoded (malformed or unsupported structure).
    Decode(&'static str),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::InvalidDimensions { width, height } => {
                write!(f, "invalid image dimensions: {width}x{height}")
            }
            EncodeError::InvalidQuality(q) => {
                write!(f, "quality {q} out of range (expected 1..=100)")
            }
            EncodeError::BufferSize { expected, found } => {
                write!(
                    f,
                    "pixel buffer size mismatch: expected {expected} bytes, found {found}"
                )
            }
            EncodeError::SampleOutOfRange { value, depth } => {
                write!(
                    f,
                    "sample {value} exceeds maximum {} for {}-bit depth",
                    depth.max_val(),
                    depth.bits()
                )
            }
            EncodeError::Unsupported(what) => write!(f, "unsupported: {what}"),
            EncodeError::Decode(what) => write!(f, "decode error: {what}"),
        }
    }
}

impl std::error::Error for EncodeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_human_readable() {
        let e = EncodeError::InvalidDimensions {
            width: 0,
            height: 10,
        };
        assert!(e.to_string().contains("0x10"));
        let e = EncodeError::InvalidQuality(200);
        assert!(e.to_string().contains("200"));
    }
}
