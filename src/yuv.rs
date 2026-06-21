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

use crate::{
    BitDepth, ChromaFormat, EncodeConfig, EncodeError, encode_yuv8, encode_yuv8_266, encode_yuv10,
    encode_yuv10_266, encode_yuv12, encode_yuv12_266, encode_yuva8_with_alpha,
    encode_yuva10_with_alpha, encode_yuva12_with_alpha,
};

#[derive(Clone, Copy, Debug)]
pub struct Yuv<'a, T> {
    /// Luma plane, `y_stride` samples per row, at least `height` rows.
    pub y: &'a [T],
    /// Samples per luma row (≥ `width`).
    pub y_stride: u32,
    /// Cb plane at chroma resolution (`uv_stride` samples per row).
    pub u: &'a [T],
    /// Cr plane at chroma resolution (`uv_stride` samples per row).
    pub v: &'a [T],
    /// Samples per chroma row (≥ `⌈width/sub_w⌉`); `0` for monochrome.
    pub uv_stride: u32,
    /// Optional alpha plane at full `width × height` resolution.
    pub alpha: Option<&'a [T]>,
    /// Samples per alpha row (≥ `width`); ignored when `alpha` is `None`.
    pub alpha_stride: u32,
    /// Display width in luma samples.
    pub width: u32,
    /// Display height in luma samples.
    pub height: u32,
    /// Chroma subsampling format.
    pub chroma: ChromaFormat,
}

impl<'a, T> Yuv<'a, T> {
    /// Build from three tightly packed planes (luma stride `width`, chroma stride
    /// `⌈width/sub_w⌉`), no alpha.
    pub fn new(
        y: &'a [T],
        u: &'a [T],
        v: &'a [T],
        width: u32,
        height: u32,
        chroma: ChromaFormat,
    ) -> Self {
        let uv_stride = if chroma.is_monochrome() {
            0
        } else {
            (width as usize).div_ceil(chroma.sub_w()) as u32
        };
        Self {
            y,
            y_stride: width,
            u,
            v,
            uv_stride,
            alpha: None,
            alpha_stride: 0,
            width,
            height,
            chroma,
        }
    }

    /// Build a monochrome (4:0:0) image from a single luma plane.
    pub fn mono(y: &'a [T], width: u32, height: u32) -> Self {
        Self::new(y, &[], &[], width, height, ChromaFormat::Monochrome)
    }

    /// Override the luma and chroma strides (samples per row) for non-tight planes.
    pub fn with_strides(mut self, y_stride: u32, uv_stride: u32) -> Self {
        self.y_stride = y_stride;
        self.uv_stride = uv_stride;
        self
    }

    /// Attach a tightly packed (`stride = width`) alpha plane.
    pub fn with_alpha(mut self, alpha: &'a [T]) -> Self {
        self.alpha = Some(alpha);
        self.alpha_stride = self.width;
        self
    }

    /// Attach an alpha plane with an explicit stride (samples per row).
    pub fn with_alpha_strided(mut self, alpha: &'a [T], stride: u32) -> Self {
        self.alpha = Some(alpha);
        self.alpha_stride = stride;
        self
    }

    /// Chroma plane dimensions `(⌈width/sub_w⌉, ⌈height/sub_h⌉)`, or `(0, 0)`
    /// for monochrome.
    fn chroma_dims(&self) -> (usize, usize) {
        if self.chroma.is_monochrome() {
            (0, 0)
        } else {
            (
                (self.width as usize).div_ceil(self.chroma.sub_w()),
                (self.height as usize).div_ceil(self.chroma.sub_h()),
            )
        }
    }
}

/// A plane must be wide enough for its stride and tall enough for its rows.
fn check_plane<T>(p: &[T], stride: usize, cols: usize, rows: usize) -> Result<(), EncodeError> {
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    if stride < cols || p.len() < (rows - 1) * stride + cols {
        return Err(EncodeError::Unsupported(
            "a YUV/alpha plane is too small for its stride and dimensions",
        ));
    }
    Ok(())
}

/// Copy `rows × cols` samples out of a strided plane into a tight `Vec`.
fn tighten<T: Copy>(out: &mut Vec<T>, p: &[T], stride: usize, cols: usize, rows: usize) {
    for r in 0..rows {
        let start = r * stride;
        out.extend_from_slice(&p[start..start + cols]);
    }
}

impl<T: Copy> Yuv<'_, T> {
    /// Pack the (possibly strided) color planes into one tight `Y‖Cb‖Cr` buffer
    /// in display resolution — exactly the layout the packed encoders expect.
    fn pack_color(&self) -> Result<Vec<T>, EncodeError> {
        let (w, h) = (self.width as usize, self.height as usize);
        let (dcw, dch) = self.chroma_dims();
        let ys = self.y_stride as usize;
        check_plane(self.y, ys, w, h)?;
        let mut out = Vec::with_capacity(w * h + 2 * dcw * dch);
        tighten(&mut out, self.y, ys, w, h);
        if !self.chroma.is_monochrome() {
            let cs = self.uv_stride as usize;
            check_plane(self.u, cs, dcw, dch)?;
            check_plane(self.v, cs, dcw, dch)?;
            tighten(&mut out, self.u, cs, dcw, dch);
            tighten(&mut out, self.v, cs, dcw, dch);
        }
        Ok(out)
    }

    /// Pack the alpha plane (full resolution) into a tight `Vec`.
    fn pack_alpha(&self, alpha: &[T]) -> Result<Vec<T>, EncodeError> {
        let (w, h) = (self.width as usize, self.height as usize);
        let st = self.alpha_stride as usize;
        check_plane(alpha, st, w, h)?;
        let mut out = Vec::with_capacity(w * h);
        tighten(&mut out, alpha, st, w, h);
        Ok(out)
    }
}

impl Yuv<'_, u8> {
    /// Encode to a raw Annex-B VVC stream (no container, so alpha is ignored;
    /// use [`encode`](Self::encode) for an alpha-carrying HEIF file).
    pub fn encode_266(&self, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
        let cfg = cfg.clone().with_chroma(self.chroma);
        let packed = self.pack_color()?;
        encode_yuv8_266(&packed, self.width, self.height, &cfg)
    }

    /// Encode to a HEIF file, carrying alpha as a monochrome auxiliary image when
    /// present.
    pub fn encode(&self, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
        let cfg = cfg.clone().with_chroma(self.chroma);
        let packed = self.pack_color()?;
        match self.alpha {
            None => encode_yuv8(&packed, self.width, self.height, &cfg),
            Some(a) => {
                let ap = self.pack_alpha(a)?;
                encode_yuva8_with_alpha(&packed, &ap, self.width, self.height, &cfg)
            }
        }
    }
}

impl Yuv<'_, u16> {
    /// Encode to a raw Annex-B VVC stream at the config's bit depth (must be 10
    /// or 12). Alpha is ignored (raw streams carry no auxiliary); use
    /// [`encode`](Self::encode) for an alpha-carrying HEIF file.
    pub fn encode_266(&self, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
        let cfg = cfg.clone().with_chroma(self.chroma);
        let packed = self.pack_color()?;
        match cfg.bit_depth {
            BitDepth::Ten => encode_yuv10_266(&packed, self.width, self.height, &cfg),
            BitDepth::Twelve => encode_yuv12_266(&packed, self.width, self.height, &cfg),
            BitDepth::Eight => Err(EncodeError::Unsupported(
                "u16 Yuv requires a 10- or 12-bit EncodeConfig",
            )),
        }
    }

    /// Encode to a HEIF file at the config's bit depth (must be 10 or 12),
    /// carrying alpha as a monochrome auxiliary image when present.
    pub fn encode(&self, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
        let cfg = cfg.clone().with_chroma(self.chroma);
        let packed = self.pack_color()?;
        let (w, h) = (self.width, self.height);
        match (cfg.bit_depth, self.alpha) {
            (BitDepth::Ten, None) => encode_yuv10(&packed, w, h, &cfg),
            (BitDepth::Twelve, None) => encode_yuv12(&packed, w, h, &cfg),
            (BitDepth::Ten, Some(a)) => {
                let ap = self.pack_alpha(a)?;
                encode_yuva10_with_alpha(&packed, &ap, w, h, &cfg)
            }
            (BitDepth::Twelve, Some(a)) => {
                let ap = self.pack_alpha(a)?;
                encode_yuva12_with_alpha(&packed, &ap, w, h, &cfg)
            }
            (BitDepth::Eight, _) => Err(EncodeError::Unsupported(
                "u16 Yuv requires a 10- or 12-bit EncodeConfig",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        decode, encode_yuv8_266, encode_yuv10_266, encode_yuv12_266, encode_yuva8_with_alpha,
        encode_yuva10_with_alpha, encode_yuva12_with_alpha,
    };

    fn dims(w: usize, h: usize, c: ChromaFormat) -> (usize, usize) {
        if c.is_monochrome() {
            (0, 0)
        } else {
            (w.div_ceil(c.sub_w()), h.div_ceil(c.sub_h()))
        }
    }

    // Deterministic planes (tight) for a given chroma format and max sample value.
    fn make<T: From<u8> + Copy>(
        w: usize,
        h: usize,
        c: ChromaFormat,
        mask: u32,
    ) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let _ = std::marker::PhantomData::<T>;
        let (cw, ch) = dims(w, h, c);
        let y: Vec<u16> = (0..w * h)
            .map(|i| ((i as u32 * 37 + 11) & mask) as u16)
            .collect();
        let cb: Vec<u16> = (0..cw * ch)
            .map(|i| ((i as u32 * 53 + 7) & mask) as u16)
            .collect();
        let cr: Vec<u16> = (0..cw * ch)
            .map(|i| ((i as u32 * 29 + 19) & mask) as u16)
            .collect();
        let a: Vec<u16> = (0..w * h)
            .map(|i| ((i as u32 * 17 + 3) & mask) as u16)
            .collect();
        (y, cb, cr, a)
    }

    fn packed16(y: &[u16], cb: &[u16], cr: &[u16]) -> Vec<u16> {
        let mut p = Vec::with_capacity(y.len() + cb.len() + cr.len());
        p.extend_from_slice(y);
        p.extend_from_slice(cb);
        p.extend_from_slice(cr);
        p
    }

    #[test]
    fn struct_matches_packed_8bit_all_formats() {
        let (w, h) = (24usize, 16usize);
        let cfg = EncodeConfig::new().with_quality(80);
        for c in [
            ChromaFormat::Yuv444,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv420,
            ChromaFormat::Monochrome,
        ] {
            let (y16, cb16, cr16, _) = make::<u8>(w, h, c, 0xFF);
            let (y, cb, cr): (Vec<u8>, Vec<u8>, Vec<u8>) = (
                y16.iter().map(|&v| v as u8).collect(),
                cb16.iter().map(|&v| v as u8).collect(),
                cr16.iter().map(|&v| v as u8).collect(),
            );
            let mut packed = y.clone();
            if !c.is_monochrome() {
                packed.extend_from_slice(&cb);
                packed.extend_from_slice(&cr);
            }
            let reference =
                encode_yuv8_266(&packed, w as u32, h as u32, &cfg.clone().with_chroma(c)).unwrap();

            let img = if c.is_monochrome() {
                Yuv::mono(&y, w as u32, h as u32)
            } else {
                Yuv::new(&y, &cb, &cr, w as u32, h as u32, c)
            };
            let got = img.encode_266(&cfg).unwrap();
            assert_eq!(got, reference, "8-bit {c:?} struct vs packed");
        }
    }

    #[test]
    fn struct_matches_packed_high_bit() {
        let (w, h) = (16usize, 16usize);
        for (bd, mask) in [(BitDepth::Ten, 0x3FFu32), (BitDepth::Twelve, 0xFFF)] {
            let cfg = EncodeConfig::new().with_quality(85).with_bit_depth(bd);
            for c in [
                ChromaFormat::Yuv444,
                ChromaFormat::Yuv422,
                ChromaFormat::Yuv420,
            ] {
                let (y, cb, cr, _) = make::<u16>(w, h, c, mask);
                let packed = packed16(&y, &cb, &cr);
                let reference = match bd {
                    BitDepth::Ten => {
                        encode_yuv10_266(&packed, w as u32, h as u32, &cfg.clone().with_chroma(c))
                            .unwrap()
                    }
                    _ => encode_yuv12_266(&packed, w as u32, h as u32, &cfg.clone().with_chroma(c))
                        .unwrap(),
                };
                let got = Yuv::new(&y, &cb, &cr, w as u32, h as u32, c)
                    .encode_266(&cfg)
                    .unwrap();
                assert_eq!(got, reference, "{bd:?} {c:?} struct vs packed");
            }
        }
    }

    #[test]
    fn strided_planes_match_tight() {
        let (w, h) = (20usize, 12usize);
        let c = ChromaFormat::Yuv420;
        let (cw, ch) = dims(w, h, c);
        let cfg = EncodeConfig::new().with_quality(75);
        let (y16, cb16, cr16, _) = make::<u8>(w, h, c, 0xFF);
        let (y, cb, cr): (Vec<u8>, Vec<u8>, Vec<u8>) = (
            y16.iter().map(|&v| v as u8).collect(),
            cb16.iter().map(|&v| v as u8).collect(),
            cr16.iter().map(|&v| v as u8).collect(),
        );

        let tight = Yuv::new(&y, &cb, &cr, w as u32, h as u32, c)
            .encode_266(&cfg)
            .unwrap();

        // Re-lay the same data into padded planes (extra junk columns per row).
        let (ypad, cpad) = (7usize, 5usize);
        let mut ys = vec![0u8; (w + ypad) * h];
        for r in 0..h {
            ys[r * (w + ypad)..r * (w + ypad) + w].copy_from_slice(&y[r * w..r * w + w]);
        }
        let mut cbs = vec![0u8; (cw + cpad) * ch];
        let mut crs = vec![0u8; (cw + cpad) * ch];
        for r in 0..ch {
            cbs[r * (cw + cpad)..r * (cw + cpad) + cw].copy_from_slice(&cb[r * cw..r * cw + cw]);
            crs[r * (cw + cpad)..r * (cw + cpad) + cw].copy_from_slice(&cr[r * cw..r * cw + cw]);
        }
        let strided = Yuv::new(&ys, &cbs, &crs, w as u32, h as u32, c)
            .with_strides((w + ypad) as u32, (cw + cpad) as u32)
            .encode_266(&cfg)
            .unwrap();
        assert_eq!(
            strided, tight,
            "strided planes must produce the tight stream"
        );
    }

    #[test]
    fn alpha_struct_matches_packed_all_depths() {
        let (w, h) = (16usize, 16usize);
        let c = ChromaFormat::Yuv444;
        // 8-bit
        {
            let cfg = EncodeConfig::new().with_quality(70);
            let (y16, cb16, cr16, a16) = make::<u8>(w, h, c, 0xFF);
            let (y, cb, cr, a): (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) = (
                y16.iter().map(|&v| v as u8).collect(),
                cb16.iter().map(|&v| v as u8).collect(),
                cr16.iter().map(|&v| v as u8).collect(),
                a16.iter().map(|&v| v as u8).collect(),
            );
            let mut packed = y.clone();
            packed.extend_from_slice(&cb);
            packed.extend_from_slice(&cr);
            let reference = encode_yuva8_with_alpha(
                &packed,
                &a,
                w as u32,
                h as u32,
                &cfg.clone().with_chroma(c),
            )
            .unwrap();
            let got = Yuv::new(&y, &cb, &cr, w as u32, h as u32, c)
                .with_alpha(&a)
                .encode(&cfg)
                .unwrap();
            assert_eq!(got, reference, "8-bit alpha struct vs packed");
        }
        // 10/12-bit
        for (bd, mask) in [(BitDepth::Ten, 0x3FFu32), (BitDepth::Twelve, 0xFFF)] {
            let cfg = EncodeConfig::new().with_quality(70).with_bit_depth(bd);
            let (y, cb, cr, a) = make::<u16>(w, h, c, mask);
            let packed = packed16(&y, &cb, &cr);
            let reference = match bd {
                BitDepth::Ten => encode_yuva10_with_alpha(
                    &packed,
                    &a,
                    w as u32,
                    h as u32,
                    &cfg.clone().with_chroma(c),
                )
                .unwrap(),
                _ => encode_yuva12_with_alpha(
                    &packed,
                    &a,
                    w as u32,
                    h as u32,
                    &cfg.clone().with_chroma(c),
                )
                .unwrap(),
            };
            let got = Yuv::new(&y, &cb, &cr, w as u32, h as u32, c)
                .with_alpha(&a)
                .encode(&cfg)
                .unwrap();
            assert_eq!(got, reference, "{bd:?} alpha struct vs packed");
        }
    }

    #[test]
    fn lossless_roundtrip_through_struct() {
        let (w, h) = (16usize, 16usize);
        let cfg = EncodeConfig::new().with_lossless(true);
        for c in [
            ChromaFormat::Yuv444,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv420,
        ] {
            let (y16, cb16, cr16, _) = make::<u8>(w, h, c, 0xFF);
            let (y, cb, cr): (Vec<u8>, Vec<u8>, Vec<u8>) = (
                y16.iter().map(|&v| v as u8).collect(),
                cb16.iter().map(|&v| v as u8).collect(),
                cr16.iter().map(|&v| v as u8).collect(),
            );
            // round-trip via the HEIF path
            let heif = Yuv::new(&y, &cb, &cr, w as u32, h as u32, c)
                .encode(&cfg)
                .unwrap();
            let dec = decode(&heif).unwrap();
            let (cw, ch) = dims(w, h, c);
            let n = w * h;
            assert_eq!(&dec.planes[..n], &y[..], "luma {c:?}");
            assert_eq!(&dec.planes[n..n + cw * ch], &cb[..], "Cb {c:?}");
            assert_eq!(
                &dec.planes[n + cw * ch..n + 2 * cw * ch],
                &cr[..],
                "Cr {c:?}"
            );
        }
    }
}
