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

use garnetash::{EncodeConfig, encode_rgb};
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("garnetash: {e}");
            ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "\
garnetash — a tiny VVC/H.266 intra still-image encoder

USAGE:
    garnetash [OPTIONS] <input.ppm> <output.266>

ARGS:
    <input.ppm>     Source image as netpbm PPM (binary P6 or ASCII P3, 8-bit)
    <output.266>    Destination VVC/H.266 Annex-B bitstream

OPTIONS:
    -q, --quality <1..=100>   Visual quality; higher is better (default: 90)
    -L, --lossless            Mathematically lossless (ignores --quality)
    -h, --help                Print this help

Convert other formats first, e.g. `ffmpeg -i in.png in.ppm`.";

struct Args {
    input: String,
    output: String,
    quality: u8,
    lossless: bool,
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut quality: u8 = 90;
    let mut lossless = false;
    let mut positional: Vec<String> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => return Ok(None),
            "-q" | "--quality" => {
                let v = it.next().ok_or_else(|| format!("{a} requires a value"))?;
                quality = v
                    .parse::<u8>()
                    .ok()
                    .filter(|&q| (1..=100).contains(&q))
                    .ok_or_else(|| format!("invalid quality '{v}' (expected 1..=100)"))?;
            }
            "-L" | "--lossless" => lossless = true,
            s if s.starts_with('-') && s.len() > 1 => {
                return Err(format!("unknown option '{s}' (try --help)"));
            }
            _ => positional.push(a),
        }
    }
    if positional.len() != 2 {
        return Err(format!(
            "expected <input.ppm> and <output.266>, got {} argument(s) (try --help)",
            positional.len()
        ));
    }
    Ok(Some(Args {
        input: positional[0].clone(),
        output: positional[1].clone(),
        quality,
        lossless,
    }))
}

fn run() -> Result<(), String> {
    let args = match parse_args()? {
        Some(a) => a,
        None => {
            println!("{USAGE}");
            return Ok(());
        }
    };

    let bytes =
        std::fs::read(&args.input).map_err(|e| format!("cannot read '{}': {e}", args.input))?;
    let img = read_file(&bytes).map_err(|e| format!("'{}': {e}", args.input))?;

    let cfg = EncodeConfig::new()
        .with_quality(args.quality)
        .with_lossless(args.lossless)
        .with_threads(12)
        .with_aq(true)
        .with_mtt(true)
        .with_lfnst(true)
        .with_mts(true)
        .with_cclm(true)
        .with_dual_tree(true);
    let time = std::time::Instant::now();
    let stream = encode_rgb(&img.rgb, img.width, img.height, &cfg)
        .map_err(|e| format!("encode failed: {e:?}"))?;
    println!("elapsed: {:?}", time.elapsed());

    std::fs::write(&args.output, &stream)
        .map_err(|e| format!("cannot write '{}': {e}", args.output))?;

    let mode = if args.lossless {
        "lossless".to_string()
    } else {
        format!("q{}", args.quality)
    };
    eprintln!(
        "garnetash: {}x{} {} -> {} ({} bytes)",
        img.width,
        img.height,
        mode,
        args.output,
        stream.len()
    );
    Ok(())
}

/// A decoded 8-bit RGB image.
struct Image {
    rgb: Vec<u8>,
    width: u32,
    height: u32,
}

fn read_file(data: &[u8]) -> Result<Image, String> {
    let img = image::load_from_memory(data).map_err(|e| "cannot open file".to_string())?;
    Ok(Image {
        rgb: img.to_rgb8().as_raw().to_vec(),
        width: img.width(),
        height: img.height(),
    })
}

#[inline]
fn scale_to_u8(v: u32, maxval: u32) -> u8 {
    if maxval == 255 {
        v.min(255) as u8
    } else {
        ((v.min(maxval) * 255 + maxval / 2) / maxval) as u8
    }
}

/// Cursor over a PPM header that yields whitespace-separated tokens while
/// skipping `#` comments to end-of-line.
struct HeaderScan<'a> {
    data: &'a [u8],
    pos: usize,
}

impl HeaderScan<'_> {
    fn skip_ws_and_comments(&mut self) {
        while let Some(&b) = self.data.get(self.pos) {
            if b == b'#' {
                while let Some(&c) = self.data.get(self.pos) {
                    self.pos += 1;
                    if c == b'\n' {
                        break;
                    }
                }
            } else if b.is_ascii_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn token(&mut self) -> Result<Vec<u8>, String> {
        self.skip_ws_and_comments();
        let start = self.pos;
        while let Some(&b) = self.data.get(self.pos) {
            if b.is_ascii_whitespace() || b == b'#' {
                break;
            }
            self.pos += 1;
        }
        if self.pos == start {
            return Err("unexpected end of header".into());
        }
        Ok(self.data[start..self.pos].to_vec())
    }

    fn uint(&mut self, what: &str) -> Result<u32, String> {
        let t = self.token()?;
        std::str::from_utf8(&t)
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .ok_or_else(|| {
                format!(
                    "expected integer for {what}, got {:?}",
                    String::from_utf8_lossy(&t)
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_p6() {
        // 2x1 image: red, green.
        let mut v = b"P6\n2 1\n255\n".to_vec();
        v.extend_from_slice(&[255, 0, 0, 0, 255, 0]);
        let img = read_file(&v).unwrap();
        assert_eq!((img.width, img.height), (2, 1));
        assert_eq!(img.rgb, vec![255, 0, 0, 0, 255, 0]);
    }

    #[test]
    fn handles_comments_and_odd_whitespace() {
        let mut v = b"P6 # magic\n# a comment\n  2\t1  # dims\n255\n".to_vec();
        v.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
        let img = read_file(&v).unwrap();
        assert_eq!((img.width, img.height), (2, 1));
        assert_eq!(img.rgb, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn parses_ascii_p3() {
        let v = b"P3\n2 1\n255\n255 0 0  0 128 64\n".to_vec();
        let img = read_file(&v).unwrap();
        assert_eq!(img.rgb, vec![255, 0, 0, 0, 128, 64]);
    }

    #[test]
    fn rescales_low_maxval() {
        // maxval=15: sample 15 -> 255, sample 0 -> 0.
        let mut v = b"P6\n1 1\n15\n".to_vec();
        v.extend_from_slice(&[15, 0, 8]);
        let img = read_file(&v).unwrap();
        assert_eq!(img.rgb[0], 255);
        assert_eq!(img.rgb[1], 0);
        // 8/15*255 ~= 136
        assert_eq!(img.rgb[2], 136);
    }

    #[test]
    fn rejects_truncated_and_16bit() {
        let v = b"P6\n4 4\n255\nshort".to_vec();
        assert!(read_file(&v).is_err());
        let v16 = b"P6\n1 1\n65535\n".to_vec();
        assert!(read_file(&v16).is_err());
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(read_file(b"P5\n1 1\n255\n\0").is_err());
    }
}
