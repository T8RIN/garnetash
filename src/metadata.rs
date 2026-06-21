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

/// Display orientation, using the eight EXIF orientation values. HEIF expresses
/// orientation with a rotation property (`irot`, anticlockwise multiples of 90°)
/// and a mirror property (`imir`); the diagonal EXIF values map to a mirror
/// followed by a rotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Orientation {
    /// 1 — upright, no transform.
    #[default]
    Normal,
    /// 2 — mirrored horizontally.
    FlipH,
    /// 3 — rotated 180°.
    Rotate180,
    /// 4 — mirrored vertically.
    FlipV,
    /// 5 — transpose (mirror H then rotate 90° clockwise).
    Transpose,
    /// 6 — rotated 90° clockwise.
    Rotate90,
    /// 7 — transverse (mirror H then rotate 90° anticlockwise).
    Transverse,
    /// 8 — rotated 90° anticlockwise.
    Rotate270,
}

impl Orientation {
    /// Map a raw EXIF Orientation value (1..=8) to an [`Orientation`]; anything
    /// out of range is treated as `Normal`.
    pub fn from_exif(v: u16) -> Self {
        match v {
            2 => Orientation::FlipH,
            3 => Orientation::Rotate180,
            4 => Orientation::FlipV,
            5 => Orientation::Transpose,
            6 => Orientation::Rotate90,
            7 => Orientation::Transverse,
            8 => Orientation::Rotate270,
            _ => Orientation::Normal,
        }
    }

    /// True when no orientation transform is needed (so neither `irot` nor
    /// `imir` is written).
    pub fn is_identity(self) -> bool {
        self.irot_steps() == 0 && self.imir_axis().is_none()
    }

    /// The `imir` axis when a mirror is part of the transform: `Some(false)` for
    /// a vertical mirroring axis (left-right flip), `Some(true)` for a
    /// horizontal mirroring axis (top-bottom flip), or `None` for no mirror.
    /// (HEIF `imir`: `axis == 0` mirrors about a vertical axis.)
    pub(crate) fn imir_axis(self) -> Option<bool> {
        match self {
            Orientation::FlipH | Orientation::Transpose | Orientation::Transverse => Some(false),
            Orientation::FlipV => Some(true),
            _ => None,
        }
    }

    /// `irot` rotation in anticlockwise 90° steps (0..=3).
    pub(crate) fn irot_steps(self) -> u8 {
        match self {
            Orientation::Normal | Orientation::FlipH | Orientation::FlipV => 0,
            Orientation::Rotate180 => 2,
            Orientation::Rotate90 => 3,
            Orientation::Rotate270 => 1,
            Orientation::Transpose => 3,
            Orientation::Transverse => 1,
        }
    }

    /// Reconstruct an [`Orientation`] from the `irot` step count (0..=3) and the
    /// `imir` axis (`None`, `Some(false)` = vertical axis / left-right flip,
    /// `Some(true)` = horizontal axis / top-bottom flip). Inverse of the
    /// `irot_steps` / `imir_axis` pair this crate writes.
    pub(crate) fn from_irot_imir(steps: u8, axis: Option<bool>) -> Self {
        match (steps & 3, axis) {
            (0, None) => Orientation::Normal,
            (2, None) => Orientation::Rotate180,
            (3, None) => Orientation::Rotate90,
            (1, None) => Orientation::Rotate270,
            (0, Some(false)) => Orientation::FlipH,
            (0, Some(true)) => Orientation::FlipV,
            (3, Some(false)) => Orientation::Transpose,
            (1, Some(false)) => Orientation::Transverse,
            _ => Orientation::Normal,
        }
    }
}

/// HDR content light level, written as a `ContentLightLevelBox` (`clli`): the
/// maximum content light level and maximum picture-average light level, in nits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContentLightLevel {
    pub max_content_light_level: u16,
    pub max_pic_average_light_level: u16,
}

impl ContentLightLevel {
    pub fn new(max_cll: u16, max_fall: u16) -> Self {
        ContentLightLevel {
            max_content_light_level: max_cll,
            max_pic_average_light_level: max_fall,
        }
    }

    /// The 4-byte `clli` payload: MaxCLL then MaxPALL, big-endian.
    pub(crate) fn clli_payload(&self) -> [u8; 4] {
        let cll = self.max_content_light_level.to_be_bytes();
        let fall = self.max_pic_average_light_level.to_be_bytes();
        [cll[0], cll[1], fall[0], fall[1]]
    }
}

/// Bundle of image-level metadata written into the `meta` box.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImageMetadata {
    pub orientation: Orientation,
    pub content_light_level: Option<ContentLightLevel>,
    /// Raw Exif payload (the TIFF/Exif block, without the 4-byte
    /// `exif_tiff_header_offset` prefix; that prefix is added when writing).
    pub exif: Option<Vec<u8>>,
}

impl ImageMetadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_orientation(mut self, o: Orientation) -> Self {
        self.orientation = o;
        self
    }

    pub fn with_content_light_level(mut self, cll: ContentLightLevel) -> Self {
        self.content_light_level = Some(cll);
        self
    }

    pub fn with_exif(mut self, exif: Vec<u8>) -> Self {
        self.exif = Some(exif);
        self
    }

    /// True when nothing here needs writing.
    pub fn is_empty(&self) -> bool {
        self.orientation.is_identity() && self.content_light_level.is_none() && self.exif.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exif_mapping() {
        assert_eq!(Orientation::from_exif(1), Orientation::Normal);
        assert_eq!(Orientation::from_exif(6), Orientation::Rotate90);
        assert_eq!(Orientation::from_exif(99), Orientation::Normal);
    }

    #[test]
    fn rotate90_has_irot_no_imir() {
        assert_eq!(Orientation::Rotate90.irot_steps(), 3);
        assert_eq!(Orientation::Rotate90.imir_axis(), None);
        assert!(!Orientation::Rotate90.is_identity());
        assert!(Orientation::Normal.is_identity());
    }

    #[test]
    fn clli_payload_layout() {
        let c = ContentLightLevel::new(1000, 400);
        assert_eq!(c.clli_payload(), [0x03, 0xE8, 0x01, 0x90]);
    }
}
