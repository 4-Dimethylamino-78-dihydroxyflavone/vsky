//! Per-image sky probability, and the binary mask built from it.
//!
//! Two model layouts are accepted, told apart by the output's channel count:
//! one channel is a sky probability (skyseg, U2-Net), 150 channels are ADE20K
//! class logits (SegFormer), of which "sky" is index 2. Anything else is
//! refused rather than guessed at: a Cityscapes model's channel 2 is
//! "building", and silently masking buildings is the worst possible failure.

use image::codecs::png::{CompressionType, FilterType as PngFilter, PngEncoder};
use image::imageops::{self, FilterType};
use image::metadata::Orientation;
use image::{DynamicImage, ExtendedColorType, ImageDecoder, ImageEncoder, ImageReader, RgbImage};
use ort::execution_providers::{CUDAExecutionProvider, DirectMLExecutionProvider};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{Tensor, ValueType};
use std::path::Path;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
/// Short edge the image is decoded to; JPEG DCT scaling takes the largest
/// 1/2^k reduction that keeps at least this. Masks are built, culled against
/// and upsampled from this resolution: 756x504 for a 6048x4032 frame, already
/// finer than dvlt's 504x336 grid.
const WORK: u32 = 480;
/// Long edge fed to models with a dynamic input shape (SegFormer). Its decoder
/// runs at 1/4 scale, so 768 gives a 192-wide logit map. Multiple of 32.
const DYN_LONG: u32 = 768;
const ADE20K_CLASSES: usize = 150;
const ADE20K_SKY: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cuda,
    DirectMl,
    Cpu,
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Device::Cuda => "CUDA",
            Device::DirectMl => "DirectML",
            Device::Cpu => "CPU",
        })
    }
}

/// Upright image at working resolution, and the upright size of the original.
pub struct Decoded {
    pub rgb: RgbImage,
    pub src_w: u32,
    pub src_h: u32,
}

/// Decode as dvlt sees the frame: EXIF orientation applied for JPEG only,
/// because dvlt's loader reads the tag from JPEG APP1 and nothing else.
pub fn decode(path: &Path) -> Result<Decoded, String> {
    let err = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
    let is_jpeg = matches!(
        path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("jpg" | "jpeg")
    );
    let mut dec = ImageReader::open(path).map_err(|e| err(&e))?.into_decoder().map_err(|e| err(&e))?;
    let orient = if is_jpeg { dec.orientation().unwrap_or(Orientation::NoTransforms) } else { Orientation::NoTransforms };
    let (w0, h0) = dec.dimensions();

    let fast = if is_jpeg { decode_jpeg_scaled(path, WORK).ok() } else { None };
    let rgb = match fast {
        Some(rgb) => rgb,
        // Progressive or unusual JPEGs, PNG: full decode, then shrink.
        None => {
            let full = DynamicImage::from_decoder(dec).map_err(|e| err(&e))?.into_rgb8();
            let s = WORK as f32 / w0.min(h0) as f32;
            if s < 1.0 {
                let (w, h) = ((w0 as f32 * s).round() as u32, (h0 as f32 * s).round() as u32);
                imageops::resize(&full, w, h, FilterType::Triangle)
            } else {
                full
            }
        }
    };

    let mut img = DynamicImage::ImageRgb8(rgb);
    img.apply_orientation(orient);
    let swap = matches!(
        orient,
        Orientation::Rotate90 | Orientation::Rotate270 | Orientation::Rotate90FlipH | Orientation::Rotate270FlipH
    );
    let (src_w, src_h) = if swap { (h0, w0) } else { (w0, h0) };
    Ok(Decoded { rgb: img.into_rgb8(), src_w, src_h })
}

fn decode_jpeg_scaled(path: &Path, target: u32) -> Result<RgbImage, String> {
    use jpeg_decoder::{Decoder, PixelFormat};
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut dec = Decoder::new(std::io::BufReader::new(file));
    dec.read_info().map_err(|e| e.to_string())?;
    let info = dec.info().ok_or("no jpeg info")?;
    let short = info.width.min(info.height) as u32;
    let mut denom = 1u32;
    while denom < 8 && short / (denom * 2) >= target {
        denom *= 2;
    }
    let want = ((info.width as u32 / denom) as u16, (info.height as u32 / denom) as u16);
    let (w, h) = dec.scale(want.0, want.1).map_err(|e| e.to_string())?;
    let px = dec.decode().map_err(|e| e.to_string())?;
    let px = match dec.info().ok_or("no jpeg info")?.pixel_format {
        PixelFormat::RGB24 => px,
        PixelFormat::L8 => px.iter().flat_map(|&g| [g, g, g]).collect(),
        other => return Err(format!("unsupported jpeg pixel format {other:?}")),
    };
    RgbImage::from_raw(w as u32, h as u32, px).ok_or_else(|| "bad jpeg buffer".into())
}

/// Sky probability at the model's output resolution, row-major.
pub struct Prob {
    w: usize,
    h: usize,
    p: Vec<f32>,
}

pub struct Segmenter {
    session: Session,
    output: String,
    /// Input (w, h) when the graph fixes it; None means any size is accepted.
    fixed: Option<(u32, u32)>,
    pub device: Device,
}

impl Segmenter {
    /// `auto` tries CUDA then CPU, each rung hard-failed so the reported device
    /// is the one actually running (ort's default is a silent CPU fallback).
    /// DirectML is opt-in, as in vfold, until verified on these exports.
    pub fn new(model: &Path, pref: &str) -> Result<Self, String> {
        let ladder: &[Device] = match pref {
            "auto" => &[Device::Cuda, Device::Cpu],
            "cuda" => &[Device::Cuda],
            "dml" | "directml" => &[Device::DirectMl],
            "cpu" => &[Device::Cpu],
            other => return Err(format!("unknown --device '{other}' (auto|cuda|dml|cpu)")),
        };
        let mut last = String::new();
        for &dev in ladder {
            match session(model, dev) {
                Ok(session) => {
                    if dev != ladder[0] {
                        eprintln!("[vsky] {} unavailable: {last}", ladder[0]);
                    }
                    let output = session.outputs()[0].name().to_string();
                    let fixed = match session.inputs()[0].dtype() {
                        ValueType::Tensor { shape, .. } if shape.len() == 4 && shape[2] > 0 && shape[3] > 0 => {
                            Some((shape[3] as u32, shape[2] as u32))
                        }
                        _ => None,
                    };
                    return Ok(Self { session, output, fixed, device: dev });
                }
                Err(e) => last = e.lines().next().unwrap_or("").to_string(),
            }
        }
        Err(format!("no execution provider initialised: {last}"))
    }

    pub fn prob(&mut self, rgb: &RgbImage) -> Result<Prob, String> {
        let (w, h) = self.fixed.unwrap_or_else(|| dyn_size(rgb.width(), rgb.height()));
        let x = imageops::resize(rgb, w, h, FilterType::Triangle);
        let n = (w * h) as usize;
        let mut t = vec![0f32; 3 * n];
        for (i, p) in x.pixels().enumerate() {
            for c in 0..3 {
                t[c * n + i] = (p[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
        let input = Tensor::from_array(([1i64, 3, h as i64, w as i64], t)).map_err(|e| e.to_string())?;
        let out = self.session.run(ort::inputs![input]).map_err(|e| e.to_string())?;
        let (shape, data) = out[self.output.as_str()].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
        let (c, ph, pw) = match shape.len() {
            4 => (shape[1] as usize, shape[2] as usize, shape[3] as usize),
            3 => (1, shape[1] as usize, shape[2] as usize),
            _ => return Err(format!("unexpected model output shape {:?}", &shape[..])),
        };
        let m = ph * pw;
        let p = match c {
            1 if data[..m].iter().all(|v| (0.0..=1.0).contains(v)) => data[..m].to_vec(),
            1 => data[..m].iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
            ADE20K_CLASSES => (0..m)
                .map(|i| {
                    let mx = (0..c).map(|k| data[k * m + i]).fold(f32::MIN, f32::max);
                    let sum: f32 = (0..c).map(|k| (data[k * m + i] - mx).exp()).sum();
                    (data[ADE20K_SKY * m + i] - mx).exp() / sum
                })
                .collect(),
            _ => {
                return Err(format!(
                    "model output has {c} channels: expected 1 (sky probability) or {ADE20K_CLASSES} (ADE20K)"
                ))
            }
        };
        Ok(Prob { w: pw, h: ph, p })
    }
}

fn session(model: &Path, dev: Device) -> Result<Session, String> {
    let mut b = Session::builder()
        .map_err(|e| e.to_string())?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| e.to_string())?;
    b = match dev {
        Device::Cuda => b
            .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
            .map_err(|e| e.to_string())?,
        Device::DirectMl => b
            .with_execution_providers([DirectMLExecutionProvider::default().build().error_on_failure()])
            .map_err(|e| e.to_string())?,
        Device::Cpu => b,
    };
    b.commit_from_file(model).map_err(|e| format!("loading {}: {e}", model.display()))
}

fn dyn_size(w: u32, h: u32) -> (u32, u32) {
    let r32 = |v: f32| ((v / 32.0).round() as u32).max(1) * 32;
    let s = DYN_LONG as f32 / w.max(h) as f32;
    (r32(w as f32 * s), r32(h as f32 * s))
}

/// What is known about one image at working resolution, upright: the sky
/// mask (1 = sky) and the colours, which the cull's photo-consistency test reads.
pub struct View {
    pub w: u32,
    pub h: u32,
    pub src_w: u32,
    pub src_h: u32,
    sky: Vec<u8>,
    rgb: Vec<u8>,
}

impl View {
    /// Threshold the probability map at working resolution, then widen the sky
    /// by `grow` source pixels.
    pub fn build(prob: &Prob, d: Decoded, thresh: f32, grow: u32) -> View {
        let (w, h) = (d.rgb.width() as usize, d.rgb.height() as usize);
        let mut sky = vec![0u8; w * h];
        for y in 0..h {
            let (y0, y1, fy) = tap(y, h, prob.h);
            for x in 0..w {
                let (x0, x1, fx) = tap(x, w, prob.w);
                let at = |yy: usize, xx: usize| prob.p[yy * prob.w + xx];
                let top = at(y0, x0) + (at(y0, x1) - at(y0, x0)) * fx;
                let bot = at(y1, x0) + (at(y1, x1) - at(y1, x0)) * fx;
                sky[y * w + x] = (top + (bot - top) * fy > thresh) as u8;
            }
        }
        let r = (grow as f32 * w as f32 / d.src_w as f32).round() as usize;
        dilate(&mut sky, w, h, r);
        View { w: w as u32, h: h as u32, src_w: d.src_w, src_h: d.src_h, sky, rgb: d.rgb.into_raw() }
    }

    pub fn sky_frac(&self) -> f32 {
        self.sky.iter().map(|&v| v as u32).sum::<u32>() as f32 / self.sky.len() as f32
    }

    /// Pixel index of normalised image position (xn, yn), both in [0, 1).
    #[inline]
    pub fn index(&self, xn: f32, yn: f32) -> usize {
        let x = ((xn * self.w as f32) as usize).min(self.w as usize - 1);
        let y = ((yn * self.h as f32) as usize).min(self.h as usize - 1);
        y * self.w as usize + x
    }

    #[inline]
    pub fn is_sky(&self, i: usize) -> bool {
        self.sky[i] != 0
    }

    #[inline]
    pub fn rgb(&self, i: usize) -> [u8; 3] {
        [self.rgb[3 * i], self.rgb[3 * i + 1], self.rgb[3 * i + 2]]
    }

    /// Full source resolution, upright, white = keep, black = sky: the
    /// polarity COLMAP, nerfstudio and Spirula read as "ignore the black".
    pub fn save_png(&self, path: &Path) -> Result<(), String> {
        let px: Vec<u8> = self.full_mask().iter().map(|&s| if s { 0 } else { 255 }).collect();
        let f = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        PngEncoder::new_with_quality(std::io::BufWriter::new(f), CompressionType::Fast, PngFilter::Sub)
            .write_image(&px, self.src_w, self.src_h, ExtendedColorType::L8)
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The source image, upright, with its sky painted pure black. Fed to dvlt,
    /// every sky point comes back exactly (0,0,0): the provenance the PLY
    /// otherwise lacks, carried through the colour channel (see `--strip`).
    pub fn save_keyed(&self, src: &Path, path: &Path, quality: u8) -> Result<(), String> {
        let mut img = decode_full(src)?;
        if (img.width(), img.height()) != (self.src_w, self.src_h) {
            return Err(format!("{}: changed size since masking", src.display()));
        }
        for (p, s) in img.pixels_mut().zip(self.full_mask()) {
            if s {
                p.0 = [0, 0, 0];
            }
        }
        jpeg_encoder::Encoder::new_file(path, quality)
            .and_then(|e| e.encode(img.as_raw(), self.src_w as u16, self.src_h as u16, jpeg_encoder::ColorType::Rgb))
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The mask at full source resolution, bilinearly upsampled, true = sky.
    fn full_mask(&self) -> Vec<bool> {
        let (w, h) = (self.w as usize, self.h as usize);
        let (sw, sh) = (self.src_w as usize, self.src_h as usize);
        let cols: Vec<_> = (0..sw).map(|x| tap(x, sw, w)).collect();
        let mut px = vec![false; sw * sh];
        for (y, row) in px.chunks_mut(sw).enumerate() {
            let (y0, y1, fy) = tap(y, sh, h);
            let (r0, r1) = (&self.sky[y0 * w..(y0 + 1) * w], &self.sky[y1 * w..(y1 + 1) * w]);
            for (o, &(x0, x1, fx)) in row.iter_mut().zip(&cols) {
                let top = r0[x0] as f32 + (r0[x1] as f32 - r0[x0] as f32) * fx;
                let bot = r1[x0] as f32 + (r1[x1] as f32 - r1[x0] as f32) * fx;
                *o = top + (bot - top) * fy >= 0.5;
            }
        }
        px
    }
}

/// Upright full-resolution image, as dvlt loads it (EXIF orientation for JPEG only).
fn decode_full(path: &Path) -> Result<RgbImage, String> {
    let err = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
    let is_jpeg = matches!(
        path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("jpg" | "jpeg")
    );
    let mut dec = ImageReader::open(path).map_err(|e| err(&e))?.into_decoder().map_err(|e| err(&e))?;
    let orient = if is_jpeg { dec.orientation().unwrap_or(Orientation::NoTransforms) } else { Orientation::NoTransforms };
    let mut img = DynamicImage::from_decoder(dec).map_err(|e| err(&e))?;
    img.apply_orientation(orient);
    Ok(img.into_rgb8())
}

/// Bilinear taps for output index `i` of `n`, sampling an axis of length `m`
/// (pixel-centre aligned).
#[inline]
fn tap(i: usize, n: usize, m: usize) -> (usize, usize, f32) {
    let s = ((i as f32 + 0.5) * m as f32 / n as f32 - 0.5).clamp(0.0, (m - 1) as f32);
    let i0 = s as usize;
    (i0, (i0 + 1).min(m - 1), s - i0 as f32)
}

/// Square dilation of the 1s by radius `r`, separable.
fn dilate(m: &mut [u8], w: usize, h: usize, r: usize) {
    if r == 0 {
        return;
    }
    let mut tmp = vec![0u8; m.len()];
    for y in 0..h {
        let row = &m[y * w..(y + 1) * w];
        for x in 0..w {
            tmp[y * w + x] = row[x.saturating_sub(r)..=(x + r).min(w - 1)].iter().copied().max().unwrap_or(0);
        }
    }
    for y in 0..h {
        for x in 0..w {
            m[y * w + x] = (y.saturating_sub(r)..=(y + r).min(h - 1)).map(|yy| tmp[yy * w + x]).max().unwrap_or(0);
        }
    }
}
