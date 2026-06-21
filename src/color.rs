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

/// color primaries (CICP `colorPrimaries`, H.273 Table 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum Primaries {
    /// BT.709 / sRGB.
    Bt709 = 1,
    /// Unspecified.
    Unspecified = 2,
    /// BT.470 System M.
    Bt470M = 4,
    /// BT.470 System B,G / BT.601-625.
    Bt470Bg = 5,
    /// SMPTE 170M / BT.601-525.
    Bt601 = 6,
    /// SMPTE 240M.
    Smpte240 = 7,
    /// Generic film.
    GenericFilm = 8,
    /// BT.2020 / BT.2100.
    Bt2020 = 9,
    /// CIE XYZ.
    Xyz = 10,
    /// SMPTE RP 431-2 (DCI P3).
    Smpte431 = 11,
    /// SMPTE EG 432-1 (Display P3).
    Smpte432 = 12,
    /// EBU Tech 3213-E.
    Ebu3213 = 22,
}

/// Transfer characteristics (CICP `TransferCharacteristics`, H.273 Table 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum TransferFunction {
    /// BT.709.
    Bt709 = 1,
    /// Unspecified.
    Unspecified = 2,
    /// BT.470 System M (gamma 2.2).
    Bt470M = 4,
    /// BT.470 System B,G (gamma 2.8).
    Bt470Bg = 5,
    /// SMPTE 170M / BT.601.
    Bt601 = 6,
    /// SMPTE 240M.
    Smpte240 = 7,
    /// Linear.
    Linear = 8,
    /// Logarithmic (100:1).
    Log100 = 9,
    /// Logarithmic (100*sqrt(10):1).
    Log100sqrt10 = 10,
    /// IEC 61966-2-4.
    Iec61966 = 11,
    /// BT.1361 extended.
    Bt1361 = 12,
    /// IEC 61966-2-1 (sRGB / sYCC).
    Srgb = 13,
    /// BT.2020 (10-bit).
    Bt202010bit = 14,
    /// BT.2020 (12-bit).
    Bt202012bit = 15,
    /// SMPTE ST 2084 (PQ).
    Smpte2084 = 16,
    /// SMPTE ST 428-1.
    Smpte428 = 17,
    /// ARIB STD-B67 (HLG).
    Hlg = 18,
}

/// Matrix coefficients (CICP `MatrixCoefficients`, H.273 Table 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum MatrixCoefficients {
    /// Identity (RGB / GBR).
    Identity = 0,
    /// BT.709.
    Bt709 = 1,
    /// Unspecified.
    Unspecified = 2,
    /// FCC.
    Fcc = 4,
    /// BT.470BG / BT.601-625.
    Bt470Bg = 5,
    /// SMPTE 170M / BT.601-525.
    Smpte170m = 6,
    /// SMPTE 240M.
    Smpte240m = 7,
    /// YCgCo.
    YCgCo = 8,
    /// BT.2020 non-constant luminance.
    Bt2020Ncl = 9,
    /// BT.2020 constant luminance.
    Bt2020Cl = 10,
    /// SMPTE ST 2085.
    Smpte2085 = 11,
    /// Chromaticity-derived non-constant luminance.
    ChromaticityDerivedNcl = 12,
    /// Chromaticity-derived constant luminance.
    ChromaticityDerivedCl = 13,
    /// ICtCp.
    ICtCp = 14,
    /// IPT-C2 (SMPTE IPT-PQ-C2).
    IPtC2 = 15,
    /// YCgCo-Re (reversible YCgCo-R, even bit-depth increase).
    YCgCoRe = 16,
    /// YCgCo-Ro (reversible YCgCo-R, odd bit-depth increase).
    YCgCoRo = 17,
}

impl Primaries {
    /// Parse a CICP `colorPrimaries` value; unknown codes map to `Unspecified`.
    pub fn from_u16(v: u16) -> Self {
        match v {
            1 => Primaries::Bt709,
            4 => Primaries::Bt470M,
            5 => Primaries::Bt470Bg,
            6 => Primaries::Bt601,
            7 => Primaries::Smpte240,
            8 => Primaries::GenericFilm,
            9 => Primaries::Bt2020,
            10 => Primaries::Xyz,
            11 => Primaries::Smpte431,
            12 => Primaries::Smpte432,
            22 => Primaries::Ebu3213,
            _ => Primaries::Unspecified,
        }
    }
}

impl TransferFunction {
    /// Parse a CICP `TransferCharacteristics` value; unknown codes → `Unspecified`.
    pub fn from_u16(v: u16) -> Self {
        match v {
            1 => TransferFunction::Bt709,
            4 => TransferFunction::Bt470M,
            5 => TransferFunction::Bt470Bg,
            6 => TransferFunction::Bt601,
            7 => TransferFunction::Smpte240,
            8 => TransferFunction::Linear,
            9 => TransferFunction::Log100,
            10 => TransferFunction::Log100sqrt10,
            11 => TransferFunction::Iec61966,
            12 => TransferFunction::Bt1361,
            13 => TransferFunction::Srgb,
            14 => TransferFunction::Bt202010bit,
            15 => TransferFunction::Bt202012bit,
            16 => TransferFunction::Smpte2084,
            17 => TransferFunction::Smpte428,
            18 => TransferFunction::Hlg,
            _ => TransferFunction::Unspecified,
        }
    }
}

impl MatrixCoefficients {
    /// Parse a CICP `MatrixCoefficients` value; unknown codes → `Unspecified`.
    pub fn from_u16(v: u16) -> Self {
        match v {
            0 => MatrixCoefficients::Identity,
            1 => MatrixCoefficients::Bt709,
            4 => MatrixCoefficients::Fcc,
            5 => MatrixCoefficients::Bt470Bg,
            6 => MatrixCoefficients::Smpte170m,
            7 => MatrixCoefficients::Smpte240m,
            8 => MatrixCoefficients::YCgCo,
            9 => MatrixCoefficients::Bt2020Ncl,
            10 => MatrixCoefficients::Bt2020Cl,
            11 => MatrixCoefficients::Smpte2085,
            12 => MatrixCoefficients::ChromaticityDerivedNcl,
            13 => MatrixCoefficients::ChromaticityDerivedCl,
            14 => MatrixCoefficients::ICtCp,
            15 => MatrixCoefficients::IPtC2,
            16 => MatrixCoefficients::YCgCoRe,
            17 => MatrixCoefficients::YCgCoRo,
            _ => MatrixCoefficients::Unspecified,
        }
    }
}

/// A CICP color description: primaries, transfer function, matrix coefficients
/// and the full-range flag, exactly as carried by an `nclx` `colr` box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cicp {
    pub primaries: Primaries,
    pub transfer: TransferFunction,
    pub matrix: MatrixCoefficients,
    pub full_range: bool,
}

impl Cicp {
    /// The color space the encoder actually uses: BT.709 primaries, sRGB
    /// transfer, BT.601 (SMPTE 170M) matrix, full range. Matches the SPS VUI.
    pub const fn garnetash_default() -> Self {
        Cicp {
            primaries: Primaries::Bt709,
            transfer: TransferFunction::Srgb,
            matrix: MatrixCoefficients::Smpte170m,
            full_range: true,
        }
    }

    /// sRGB: BT.709 primaries, sRGB transfer, BT.709 matrix, full range.
    pub const fn srgb() -> Self {
        Cicp {
            primaries: Primaries::Bt709,
            transfer: TransferFunction::Srgb,
            matrix: MatrixCoefficients::Bt709,
            full_range: true,
        }
    }

    /// BT.709 video: BT.709 primaries/transfer/matrix, full range.
    pub const fn bt709() -> Self {
        Cicp {
            primaries: Primaries::Bt709,
            transfer: TransferFunction::Bt709,
            matrix: MatrixCoefficients::Bt709,
            full_range: true,
        }
    }

    /// BT.2020 PQ (HDR10): BT.2020 primaries, PQ transfer, BT.2020 NCL matrix.
    pub const fn bt2020_pq() -> Self {
        Cicp {
            primaries: Primaries::Bt2020,
            transfer: TransferFunction::Smpte2084,
            matrix: MatrixCoefficients::Bt2020Ncl,
            full_range: true,
        }
    }

    /// Unspecified colorimetry (CICP value 2 for primaries/transfer/matrix).
    pub const fn unspecified() -> Self {
        Cicp {
            primaries: Primaries::Unspecified,
            transfer: TransferFunction::Unspecified,
            matrix: MatrixCoefficients::Unspecified,
            full_range: true,
        }
    }

    /// The `nclx` payload for a `colr` box (without the box header): the
    /// `color_type` (`nclx`) plus the four CICP fields. `full_range_flag`
    /// occupies the top bit of the final byte; the low 7 bits are reserved zero.
    pub(crate) fn nclx_payload(&self) -> Vec<u8> {
        let mut p = Vec::with_capacity(11);
        p.extend_from_slice(b"nclx");
        p.extend_from_slice(&(self.primaries as u16).to_be_bytes());
        p.extend_from_slice(&(self.transfer as u16).to_be_bytes());
        p.extend_from_slice(&(self.matrix as u16).to_be_bytes());
        p.push(if self.full_range { 0x80 } else { 0x00 });
        p
    }

    /// Parse an `nclx` `colr` payload (the bytes following the box header):
    /// `'nclx'` then three big-endian `u16` CICP codes and a full-range flag
    /// byte. Returns `None` if the payload is too short or not an `nclx` type.
    pub(crate) fn from_nclx_payload(p: &[u8]) -> Option<Cicp> {
        if p.len() < 11 || &p[0..4] != b"nclx" {
            return None;
        }
        Some(Cicp {
            primaries: Primaries::from_u16(u16::from_be_bytes([p[4], p[5]])),
            transfer: TransferFunction::from_u16(u16::from_be_bytes([p[6], p[7]])),
            matrix: MatrixCoefficients::from_u16(u16::from_be_bytes([p[8], p[9]])),
            full_range: p[10] & 0x80 != 0,
        })
    }
}

impl Default for Cicp {
    fn default() -> Self {
        Cicp::garnetash_default()
    }
}

/// How the output color space is described in the file: a CICP code-point set
/// (`nclx`) and/or an embedded ICC profile (`prof`). Both may be present.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ColorMetadata {
    pub cicp: Option<Cicp>,
    pub icc: Option<Vec<u8>>,
}

impl ColorMetadata {
    /// A CICP-only description.
    pub fn cicp(c: Cicp) -> Self {
        ColorMetadata {
            cicp: Some(c),
            icc: None,
        }
    }

    /// An ICC-profile-only description.
    pub fn icc(profile: Vec<u8>) -> Self {
        ColorMetadata {
            cicp: None,
            icc: Some(profile),
        }
    }

    /// Both a CICP description and an ICC profile.
    pub fn cicp_and_icc(c: Cicp, profile: Vec<u8>) -> Self {
        ColorMetadata {
            cicp: Some(c),
            icc: Some(profile),
        }
    }

    pub fn with_cicp(mut self, c: Cicp) -> Self {
        self.cicp = Some(c);
        self
    }

    pub fn with_icc(mut self, profile: Vec<u8>) -> Self {
        self.icc = Some(profile);
        self
    }

    /// The CICP to write into the primary `nclx` `colr` box, falling back to
    /// the encoder's working color space when none was set.
    pub(crate) fn effective_cicp(&self) -> Cicp {
        self.cicp.unwrap_or_else(Cicp::garnetash_default)
    }

    /// True when both a CICP set and an ICC profile are present, so a second
    /// `colr` (`prof`) box is needed alongside the `nclx` one.
    pub(crate) fn has_secondary_colr(&self) -> bool {
        self.cicp.is_some() && self.icc.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nclx_payload_matches_vui() {
        // The default CICP must equal the SPS VUI: primaries 1, transfer 13,
        // matrix 6, full range.
        let p = Cicp::garnetash_default().nclx_payload();
        assert_eq!(&p[0..4], b"nclx");
        assert_eq!(u16::from_be_bytes([p[4], p[5]]), 1); // primaries BT.709
        assert_eq!(u16::from_be_bytes([p[6], p[7]]), 13); // transfer sRGB
        assert_eq!(u16::from_be_bytes([p[8], p[9]]), 6); // matrix BT.601
        assert_eq!(p[10], 0x80); // full range
    }

    #[test]
    fn secondary_colr_only_when_both() {
        assert!(!ColorMetadata::cicp(Cicp::srgb()).has_secondary_colr());
        assert!(!ColorMetadata::icc(vec![0; 4]).has_secondary_colr());
        assert!(ColorMetadata::cicp_and_icc(Cicp::srgb(), vec![0; 4]).has_secondary_colr());
    }

    #[test]
    fn matrix_coefficients_extended_round_trip() {
        for (v, m) in [
            (15u16, MatrixCoefficients::IPtC2),
            (16, MatrixCoefficients::YCgCoRe),
            (17, MatrixCoefficients::YCgCoRo),
        ] {
            assert_eq!(MatrixCoefficients::from_u16(v), m);
            assert_eq!(m as u16, v);
        }
        // Reserved / unknown codes still fall back to Unspecified.
        assert_eq!(
            MatrixCoefficients::from_u16(3),
            MatrixCoefficients::Unspecified
        );
        assert_eq!(
            MatrixCoefficients::from_u16(99),
            MatrixCoefficients::Unspecified
        );
    }
}
