//! Dropping the points of a dvlt reconstruction that land on sky.
//!
//! dvlt.exe writes no pixel index per point, so correspondence is recovered by
//! projecting every point into every camera of poses.json and voting.
//!
//! In principle one vote settles it: sky is at infinity, so a real surface
//! point, occluded or not, can never project onto sky. In practice dvlt's
//! per-view depths disagree. Measured on a 64-view capture, structure points
//! landed on sky in a median 3% of the views that frame them, up to 14% at the
//! 95th percentile, and an any-view veto deleted 56% of the structure. Sky
//! points fail the other way: hallucinated at finite depth, they sit in front
//! of buildings from most other viewpoints.
//!
//! Both failures are views that disagree with the point, so a view only votes
//! when it is photo-consistent: the image colour where the point lands matches
//! the point's colour, as in MVS depth fusion. A structure point drifting onto
//! blue sky stops voting sky; a sky floater landing on brick stops counting as
//! seen. The gate only removes evidence, so at worst it is the plain vote.

use crate::sky::View;
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};

/// dvlt's ViT patch edge: each frame is center-cropped to a multiple of it.
const PATCH: u32 = 14;
/// Mean absolute RGB difference (0-255) under which a view agrees with a
/// point. Point colours come from dvlt's 504 px resample, the view's from the
/// ~756 px working decode, so exact equality is never expected. At 20 a point
/// is photo-consistent in a median 4 of the 26 views that frame it.
const PHOTO_TOL: u32 = 20;

pub struct Stats {
    pub source: PathBuf,
    pub points: usize,
    pub dropped: usize,
    /// What three candidate rules would drop, so the choice is made on data.
    pub any: usize,
    pub two: usize,
    pub majority: usize,
}

/// Is a point sky, given its photo-consistent votes?
///
/// `seen`: views that frame the point and agree with its colour.
/// `sky`:  those of them in which it lands on a sky pixel (`sky <= seen`).
fn is_sky(sky: u32, seen: u32) -> bool {
    // Survey quality: losing structure is worse than keeping a sky speck, so
    // sky must be most of the agreeing evidence. 0.16% ground loss measured.
    2 * sky > seen
}

struct Cam<'a> {
    /// World -> camera rotation (R transposed), row-major.
    rt: [f32; 9],
    /// Camera centre in world.
    c: [f32; 3],
    /// fx, fy, cx, cy in dvlt's padded batch frame.
    k: [f32; 4],
    /// Padded-batch pixel -> resized-source pixel.
    off: [f32; 2],
    /// Resized-source pixel -> [0, 1).
    inv: [f32; 2],
    view: &'a View,
}

/// `masks[i]` must belong to frame `i` of poses.json. `size` is the `-s` the
/// reconstruction ran at.
pub fn run(dir: &Path, masks: &[View], size: u32) -> Result<Stats, String> {
    let poses = read_poses(&dir.join("poses.json"))?;
    if poses.len() != masks.len() {
        return Err(format!(
            "{} images but poses.json has {} frames; pairing is positional, so pass \
             the directory or list the reconstruction ran on",
            masks.len(),
            poses.len()
        ));
    }

    // dvlt.cu load_frame + preprocess: resize longest edge to `size` (lroundf,
    // in f32), center-crop to PATCH, center-pad every frame to the batch max.
    let geo: Vec<(u32, u32, u32, u32)> = masks
        .iter()
        .map(|m| {
            let s = size as f32 / m.src_w.max(m.src_h) as f32;
            let (nw, nh) = ((m.src_w as f32 * s).round() as u32, (m.src_h as f32 * s).round() as u32);
            (nw, nh, nw / PATCH * PATCH, nh / PATCH * PATCH)
        })
        .collect();
    let bw = geo.iter().map(|g| g.2).max().unwrap_or(0);
    let bh = geo.iter().map(|g| g.3).max().unwrap_or(0);
    if let Some((rw, rh)) = report_resolution(dir) {
        if (rw, rh) != (bw, bh) {
            return Err(format!(
                "report.txt says dvlt ran at {rw}x{rh}, but --size {size} gives {bw}x{bh}; \
                 pass the -s the reconstruction used"
            ));
        }
    }

    let cams: Vec<Cam> = poses
        .iter()
        .zip(masks)
        .zip(&geo)
        .map(|(((e, k), mask), &(nw, nh, cw, ch))| Cam {
            rt: [e[0], e[4], e[8], e[1], e[5], e[9], e[2], e[6], e[10]],
            c: [e[3], e[7], e[11]],
            k: [k[0], k[4], k[2], k[5]],
            off: [
                ((nw - cw) / 2) as f32 - ((bw - cw) / 2) as f32,
                ((nh - ch) / 2) as f32 - ((bh - ch) / 2) as f32,
            ],
            inv: [1.0 / nw as f32, 1.0 / nh as f32],
            view: mask,
        })
        .collect();

    let source = original(dir)?;
    let bytes = fs::read(&source).map_err(|e| format!("{}: {e}", source.display()))?;
    let ply = Ply::parse(&bytes)?;
    let body = bytes
        .get(ply.body..ply.body + ply.count * ply.stride)
        .ok_or("scene.ply: truncated vertex data")?;

    let votes: Vec<(u32, u32)> = body
        .par_chunks(ply.stride)
        .map(|v| {
            let f = |o: usize| f32::from_le_bytes([v[o], v[o + 1], v[o + 2], v[o + 3]]);
            let p = [f(ply.xyz[0]), f(ply.xyz[1]), f(ply.xyz[2])];
            let rgb = [v[ply.rgb[0]], v[ply.rgb[1]], v[ply.rgb[2]]];
            let (mut sky, mut seen) = (0u32, 0u32);
            for c in &cams {
                let d = [p[0] - c.c[0], p[1] - c.c[1], p[2] - c.c[2]];
                let z = c.rt[6] * d[0] + c.rt[7] * d[1] + c.rt[8] * d[2];
                if z <= 1e-6 {
                    continue;
                }
                let x = c.rt[0] * d[0] + c.rt[1] * d[1] + c.rt[2] * d[2];
                let y = c.rt[3] * d[0] + c.rt[4] * d[1] + c.rt[5] * d[2];
                let xn = (c.k[0] * x / z + c.k[2] + c.off[0]) * c.inv[0];
                let yn = (c.k[1] * y / z + c.k[3] + c.off[1]) * c.inv[1];
                if !(0.0..1.0).contains(&xn) || !(0.0..1.0).contains(&yn) {
                    continue;
                }
                let i = c.view.index(xn, yn);
                let q = c.view.rgb(i);
                let diff: u32 = (0..3).map(|k| rgb[k].abs_diff(q[k]) as u32).sum();
                if diff > 3 * PHOTO_TOL {
                    continue;
                }
                seen += 1;
                sky += c.view.is_sky(i) as u32;
            }
            (sky, seen)
        })
        .collect();

    let count = |rule: fn(u32, u32) -> bool| votes.iter().filter(|&&(s, n)| rule(s, n)).count();
    let (any, two, majority) = (count(|s, _| s >= 1), count(|s, _| s >= 2), count(|s, n| 2 * s > n));
    let pct = |n: usize| 100.0 * n as f64 / ply.count.max(1) as f64;
    eprintln!(
        "[vsky] {} points; would drop  sky>=1: {:.2}%  sky>=2: {:.2}%  majority: {:.2}%",
        ply.count,
        pct(any),
        pct(two),
        pct(majority)
    );
    let keep: Vec<bool> = votes.par_iter().map(|&(s, n)| !is_sky(s, n)).collect();
    let kept = write(dir, &ply, body, &keep)?;
    Ok(Stats { source, points: ply.count, dropped: ply.count - kept, any, two, majority })
}

/// For a reconstruction of `vsky --key` images: every point that is exactly
/// (0,0,0) came from a keyed sky pixel, so this is the exact per-source-pixel
/// removal the vote approximates. Measured on the building capture against raw +
/// vote cull: 3.1% vs 5.0% sky-blue left, 14% more ground points kept.
pub fn strip(dir: &Path) -> Result<Stats, String> {
    let source = original(dir)?;
    let bytes = fs::read(&source).map_err(|e| format!("{}: {e}", source.display()))?;
    let ply = Ply::parse(&bytes)?;
    let body = bytes.get(ply.body..ply.body + ply.count * ply.stride).ok_or("scene.ply: truncated vertex data")?;
    let keep: Vec<bool> = body.par_chunks(ply.stride).map(|v| (v[ply.rgb[0]] | v[ply.rgb[1]] | v[ply.rgb[2]]) != 0).collect();
    let kept = write(dir, &ply, body, &keep)?;
    Ok(Stats { source, points: ply.count, dropped: ply.count - kept, any: 0, two: 0, majority: 0 })
}

/// The original cloud, always as scene_withsky.ply. A scene.ply vsky wrote
/// carries `comment vsky`; one without it is fresh dvlt output (perhaps over
/// an older culled run) and becomes the original. So reruns never cull a cull,
/// and a new reconstruction into the same folder is never mistaken for old.
fn original(dir: &Path) -> Result<PathBuf, String> {
    let cur = dir.join("scene.ply");
    let orig = dir.join("scene_withsky.ply");
    let mut head = [0u8; 256];
    let n = fs::File::open(&cur).and_then(|mut f| std::io::Read::read(&mut f, &mut head)).unwrap_or(0);
    let ours = head[..n].windows(STAMP.len()).any(|w| w == STAMP.as_bytes());
    if !ours {
        fs::rename(&cur, &orig).map_err(|e| format!("keeping original as {}: {e}", orig.display()))?;
    } else if !orig.exists() {
        return Err(format!("{} was written by vsky but {} is gone", cur.display(), orig.display()));
    }
    Ok(orig)
}

const STAMP: &str = "comment vsky";

/// Write the kept records as a stamped scene.ply; returns how many.
fn write(dir: &Path, ply: &Ply, body: &[u8], keep: &[bool]) -> Result<usize, String> {
    let kept = keep.iter().filter(|&&k| k).count();
    let mut out = ply.header_with(kept).into_bytes();
    out.reserve(kept * ply.stride);
    for (v, &k) in body.chunks(ply.stride).zip(keep) {
        if k {
            out.extend_from_slice(v);
        }
    }
    let cur = dir.join("scene.ply");
    fs::write(&cur, &out).map_err(|e| format!("{}: {e}", cur.display()))?;
    Ok(kept)
}

/// (c2w row-major 4x4, K row-major 3x3) per frame, in file order.
fn read_poses(path: &Path) -> Result<Vec<(Vec<f32>, Vec<f32>)>, String> {
    let s = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let e = arrays(&s, "\"extrinsics_c2w\"")?;
    let k = arrays(&s, "\"intrinsics\"")?;
    if e.len() != k.len() || e.iter().any(|a| a.len() != 16) || k.iter().any(|a| a.len() != 9) {
        return Err(format!("{}: expected 4x4 extrinsics_c2w + 3x3 intrinsics per frame", path.display()));
    }
    Ok(e.into_iter().zip(k).collect())
}

/// Every `[...]` number array that follows `key`. poses.json is flat enough
/// that this beats a JSON dependency.
fn arrays(s: &str, key: &str) -> Result<Vec<Vec<f32>>, String> {
    s.match_indices(key)
        .map(|(i, _)| {
            let rest = &s[i + key.len()..];
            let (a, b) = (rest.find('[').ok_or("poses.json: no '['")?, rest.find(']').ok_or("poses.json: no ']'")?);
            rest[a + 1..b]
                .split(',')
                .map(|t| t.trim().parse::<f32>().map_err(|e| format!("poses.json: {e}")))
                .collect()
        })
        .collect()
}

/// The `resolution WxH` line dvlt.exe writes into report.txt.
fn report_resolution(dir: &Path) -> Option<(u32, u32)> {
    let s = fs::read_to_string(dir.join("report.txt")).ok()?;
    let line = s.lines().find(|l| l.trim_start().starts_with("resolution"))?;
    let (w, h) = line.split_whitespace().last()?.split_once('x')?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

/// A binary little-endian PLY with one `vertex` element, as dvlt writes it.
/// Records are copied through verbatim, so extra properties survive.
struct Ply<'a> {
    header: &'a str,
    body: usize,
    count: usize,
    stride: usize,
    xyz: [usize; 3],
    rgb: [usize; 3],
}

impl<'a> Ply<'a> {
    fn parse(b: &'a [u8]) -> Result<Self, String> {
        const END: &[u8] = b"end_header\n";
        let body = b.windows(END.len()).position(|w| w == END).ok_or("scene.ply: no end_header")? + END.len();
        let header = std::str::from_utf8(&b[..body]).map_err(|_| "scene.ply: header is not text")?;
        let (mut count, mut stride, mut xyz, mut rgb) = (None, 0, [usize::MAX; 3], [usize::MAX; 3]);
        for line in header.lines() {
            match line.split_whitespace().collect::<Vec<_>>().as_slice() {
                ["format", f, _] if *f != "binary_little_endian" => {
                    return Err(format!("scene.ply: format {f}, expected binary_little_endian"))
                }
                ["element", "vertex", n] => count = Some(n.parse::<usize>().map_err(|e| format!("scene.ply: {e}"))?),
                ["element", other, _] => return Err(format!("scene.ply: element '{other}' unsupported")),
                ["property", "list", ..] => return Err("scene.ply: list properties unsupported".into()),
                ["property", ty, name] => {
                    let size = match *ty {
                        "char" | "uchar" | "int8" | "uint8" => 1,
                        "short" | "ushort" | "int16" | "uint16" => 2,
                        "int" | "uint" | "float" | "int32" | "uint32" | "float32" => 4,
                        "double" | "float64" => 8,
                        _ => return Err(format!("scene.ply: property type '{ty}'")),
                    };
                    if let Some(i) = ["x", "y", "z"].iter().position(|a| a == name) {
                        if !matches!(*ty, "float" | "float32") {
                            return Err(format!("scene.ply: {name} is {ty}, expected float"));
                        }
                        xyz[i] = stride;
                    }
                    if let Some(i) = ["red", "green", "blue"].iter().position(|a| a == name) {
                        if !matches!(*ty, "uchar" | "uint8") {
                            return Err(format!("scene.ply: {name} is {ty}, expected uchar"));
                        }
                        rgb[i] = stride;
                    }
                    stride += size;
                }
                _ => {}
            }
        }
        if xyz.contains(&usize::MAX) || rgb.contains(&usize::MAX) {
            return Err("scene.ply: needs x/y/z and red/green/blue".into());
        }
        Ok(Self { header, body, count: count.ok_or("scene.ply: no vertex element")?, stride, xyz, rgb })
    }

    fn header_with(&self, n: usize) -> String {
        self.header
            .lines()
            .filter(|l| *l != STAMP)
            .map(|l| match l {
                l if l.starts_with("element vertex") => format!("element vertex {n}\n"),
                l if l.starts_with("format") => format!("{l}\n{STAMP}\n"),
                l => format!("{l}\n"),
            })
            .collect()
    }
}
