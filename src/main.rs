//! vsky — take the sky out of a dvlt reconstruction.
//!
//! A feed-forward pointmap model predicts a depth for every pixel, sky
//! included, and dvlt's `conf_drop` is a percentile: at 0.1 it keeps 90% of
//! the points, most of the sky among them. vsky segments the sky in each
//! image, writes the masks beside the input, and with `--cull` drops the points
//! of a dvlt output that land on sky.
//!
//! Detached like vfold: it reads an image folder or list and writes files.
//! Delete the binary and the pipeline is unchanged.

mod cull;
mod sky;

use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "\
vsky - mask the sky out of an image set, and cull it from a dvlt reconstruction

USAGE:
    vsky <IMAGE_DIR | list.txt> [OPTIONS]
    vsky --strip <DVLT_OUT>

Writes one mask per image at full resolution, white = keep, black = sky, named
<stem>.png. Frame order and orientation follow dvlt.exe: a folder is sorted by
name, a list keeps its order, JPEG EXIF orientation is applied.

OPTIONS:
        --masks <DIR>    where masks go    [default: <input>_masks beside the input]
        --key <DIR>      also write each image with its sky painted pure black
                         (<stem>.jpg, upright). Reconstruct from these, then
                         --strip: exact sky removal, no voting.
        --strip <DIR>    dvlt output of keyed images: drop the points that are
                         exactly (0,0,0). Needs no images or model.
        --cull <DIR>     dvlt output folder: drop scene.ply points that land on
                         sky. The original is kept as scene_withsky.ply and is
                         what every later --cull reads. Rerun dvlt_postprocess after.
        --size <PX>      the -s that reconstruction ran at          [default: 504]
        --model <PATH>   ADE20K (150-class, e.g. SegFormer) or skyseg (1-channel) ONNX.
                         ADE20K knows 'ceiling': no false sky indoors
                         A bare file name is looked up like the default.
                         [default: segformer-b0-ade.onnx in <exe dir>/model/,
                          else ~/.cache/vsky/]
        --device <D>     auto | cuda | dml | cpu                [default: auto]
        --thresh <F>     sky probability cut                      [default: 0.5]
        --grow <PX>      widen the sky by PX source pixels          [default: 0]
        --json <PATH>    write diagnostics JSON
";

/// Frames decoded at once; bounds memory on large sets.
const CHUNK: usize = 16;

struct Args {
    input: PathBuf,
    masks: Option<PathBuf>,
    cull: Option<PathBuf>,
    size: u32,
    model: PathBuf,
    device: String,
    thresh: f32,
    grow: u32,
    json: Option<PathBuf>,
    key: Option<PathBuf>,
    strip: Option<PathBuf>,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("vsky: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let a = parse_args()?;
    if let Some(dir) = &a.strip {
        let t = Instant::now();
        let s = cull::strip(dir)?;
        eprintln!(
            "[vsky] stripped {} keyed-sky points of {} ({:.2}%, {:.1}s); original kept as {}",
            s.dropped,
            s.points,
            100.0 * s.dropped as f64 / s.points.max(1) as f64,
            t.elapsed().as_secs_f32(),
            s.source.display()
        );
        return Ok(());
    }
    let images = list_images(&a.input)?;
    let mut stems = HashMap::new();
    for p in &images {
        if let Some(prev) = stems.insert(p.file_stem(), p) {
            return Err(format!("{} and {} would write the same mask", prev.display(), p.display()));
        }
    }
    let masks_dir = a.masks.clone().unwrap_or_else(|| sibling(&a.input, "_masks"));
    std::fs::create_dir_all(&masks_dir).map_err(|e| format!("{}: {e}", masks_dir.display()))?;
    if let Some(k) = &a.key {
        std::fs::create_dir_all(k).map_err(|e| format!("{}: {e}", k.display()))?;
    }
    let mask_path = |p: &Path| {
        let mut name = p.file_stem().unwrap_or_default().to_os_string();
        name.push(".png");
        masks_dir.join(name)
    };

    let t = Instant::now();
    let mut seg = sky::Segmenter::new(&a.model, &a.device)?;
    eprintln!("[vsky] {} images  device {}  model {}", images.len(), seg.device, a.model.display());
    let mut masks = Vec::with_capacity(images.len());
    for chunk in images.chunks(CHUNK) {
        let dec = chunk.par_iter().map(|p| sky::decode(p)).collect::<Result<Vec<_>, _>>()?;
        let probs = chunk
            .iter()
            .zip(&dec)
            .map(|(p, d)| seg.prob(&d.rgb).map_err(|e| format!("{}: {e}", p.display())))
            .collect::<Result<Vec<_>, _>>()?;
        let built = chunk
            .par_iter()
            .zip(dec.into_par_iter().zip(probs.par_iter()))
            .map(|(p, (d, pr))| {
                let v = sky::View::build(pr, d, a.thresh, a.grow);
                v.save_png(&mask_path(p))?;
                if let Some(k) = &a.key {
                    let mut name = p.file_stem().unwrap_or_default().to_os_string();
                    name.push(".jpg");
                    v.save_keyed(p, &k.join(name), 95)?;
                }
                Ok::<_, String>(v)
            })
            .collect::<Result<Vec<_>, _>>()?;
        masks.extend(built);
        eprint!("\r[vsky] masks {}/{}", masks.len(), images.len());
    }
    if let Some(k) = &a.key {
        // The input's order, so a list input (line 1 = reference camera) keeps
        // it through the keyed folder: reconstruct from <key>/images.txt.
        write_order(k, &images)?;
    }
    let mut fr: Vec<f32> = masks.iter().map(|m| m.sky_frac()).collect();
    let fracs = fr.clone();
    fr.sort_by(f32::total_cmp);
    eprintln!(
        "\r[vsky] masks {}  sky min {:.1}%  median {:.1}%  max {:.1}%  ({:.1}s) -> {}",
        masks.len(),
        100.0 * fr[0],
        100.0 * fr[fr.len() / 2],
        100.0 * fr[fr.len() - 1],
        t.elapsed().as_secs_f32(),
        masks_dir.display()
    );

    let stats = match &a.cull {
        Some(dir) => {
            let t = Instant::now();
            let s = cull::run(dir, &masks, a.size)?;
            eprintln!(
                "[vsky] culled {} of {} points ({:.2}%, {:.1}s); original kept as {}",
                s.dropped,
                s.points,
                100.0 * s.dropped as f64 / s.points.max(1) as f64,
                t.elapsed().as_secs_f32(),
                dir.join("scene_withsky.ply").display()
            );
            Some(s)
        }
        None => None,
    };

    if let Some(path) = &a.json {
        let images: Vec<String> = images
            .iter()
            .zip(&fracs)
            .map(|(p, f)| format!("    {{\"path\": {}, \"sky\": {f:.4}}}", js(p)))
            .collect();
        let cull = match (&a.cull, &stats) {
            (Some(dir), Some(s)) => format!(
                "{{\"dir\": {}, \"source\": {}, \"size\": {}, \"points\": {}, \"dropped\": {}, \
                 \"would_drop\": {{\"any\": {}, \"two\": {}, \"majority\": {}}}}}",
                js(dir),
                js(&s.source),
                a.size,
                s.points,
                s.dropped,
                s.any,
                s.two,
                s.majority
            ),
            _ => "null".into(),
        };
        let doc = format!(
            "{{\n  \"input\": {},\n  \"masks\": {},\n  \"model\": {},\n  \"device\": \"{}\",\n  \
             \"thresh\": {},\n  \"grow\": {},\n  \"images\": [\n{}\n  ],\n  \"cull\": {}\n}}\n",
            js(&a.input),
            js(&masks_dir),
            js(&a.model),
            seg.device,
            a.thresh,
            a.grow,
            images.join(",\n"),
            cull
        );
        std::fs::write(path, doc).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        input: PathBuf::new(),
        masks: None,
        cull: None,
        size: 504,
        model: model_path("segformer-b0-ade.onnx"),
        device: "auto".into(),
        thresh: 0.5,
        grow: 0,
        json: None,
        key: None,
        strip: None,
    };
    let mut input = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--masks" => a.masks = Some(val()?.into()),
            "--cull" => a.cull = Some(val()?.into()),
            "--key" => a.key = Some(val()?.into()),
            "--strip" => a.strip = Some(val()?.into()),
            "--size" => a.size = num(&arg, val()?)?,
            "--model" => a.model = val()?.into(),
            "--device" => a.device = val()?,
            "--thresh" => a.thresh = num(&arg, val()?)?,
            "--grow" => a.grow = num(&arg, val()?)?,
            "--json" => a.json = Some(val()?.into()),
            s if s.starts_with('-') => return Err(format!("unknown option {s}\n\n{USAGE}")),
            _ if input.is_none() => input = Some(PathBuf::from(&arg)),
            _ => return Err(format!("unexpected argument {arg}")),
        }
    }
    if a.model.parent().is_some_and(|p| p.as_os_str().is_empty()) {
        a.model = model_path(&a.model.to_string_lossy());
    }
    match input {
        Some(i) => a.input = i,
        None if a.strip.is_some() => {}
        None => return Err(USAGE.to_string()),
    }
    Ok(a)
}

/// `<exe dir>\model\<name>` when present (a self-contained install), else the
/// user cache `~/.cache/vsky/<name>`.
fn model_path(name: &str) -> PathBuf {
    let local = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join("model").join(name)));
    match local {
        Some(p) if p.is_file() => p,
        _ => std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .map(PathBuf::from)
            .unwrap_or_default()
            .join(".cache")
            .join("vsky")
            .join(name),
    }
}

fn num<T: std::str::FromStr>(name: &str, v: String) -> Result<T, String> {
    v.parse().map_err(|_| format!("{name}: not a number: {v}"))
}

fn is_list(p: &Path) -> bool {
    p.is_file() && p.extension().is_some_and(|e| e.eq_ignore_ascii_case("txt"))
}

/// The frames dvlt.exe would read, in its order (dvlt.cu expand_dir/expand_list).
fn list_images(input: &Path) -> Result<Vec<PathBuf>, String> {
    let err = |e: std::io::Error| format!("{}: {e}", input.display());
    let out: Vec<PathBuf> = if is_list(input) {
        let text = std::fs::read_to_string(input).map_err(err)?;
        let base = input.parent().unwrap_or(Path::new(""));
        let mut out = Vec::new();
        for (n, line) in text.trim_start_matches('\u{feff}').lines().enumerate() {
            let s = line.trim();
            if s.is_empty() || s.starts_with('#') {
                continue;
            }
            let p = base.join(s);
            if !p.is_file() {
                return Err(format!("{} line {}: file not found: {}", input.display(), n + 1, p.display()));
            }
            out.push(p);
        }
        out
    } else {
        let mut out: Vec<PathBuf> = std::fs::read_dir(input)
            .map_err(err)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| ["jpg", "jpeg", "png"].contains(&e.to_ascii_lowercase().as_str()))
            })
            .collect();
        // dvlt sorts bytewise (std::sort on the path string), case-sensitive.
        out.sort_by(|x, y| x.file_name().cmp(&y.file_name()));
        out
    };
    if out.is_empty() {
        return Err(format!("{}: no .jpg/.jpeg/.png images", input.display()));
    }
    Ok(out)
}

/// `scene` -> `scene_masks`, `scene/vfold-60.txt` -> `scene/vfold-60_masks`:
/// beside the input, never inside an image folder that a trainer would ingest.
fn sibling(input: &Path, suffix: &str) -> PathBuf {
    let abs = std::path::absolute(input).unwrap_or_else(|_| input.to_path_buf());
    let mut name = if is_list(&abs) { abs.file_stem() } else { abs.file_name() }.unwrap_or_default().to_os_string();
    name.push(suffix);
    abs.with_file_name(name)
}

/// `<dir>/images.txt`: one `<stem>.jpg` per line, in `images` order.
fn write_order(dir: &Path, images: &[PathBuf]) -> Result<(), String> {
    let list: String = images.iter().map(|p| format!("{}.jpg\n", p.file_stem().unwrap_or_default().to_string_lossy())).collect();
    std::fs::write(dir.join("images.txt"), list).map_err(|e| format!("{}: {e}", dir.display()))
}

fn js(p: &Path) -> String {
    format!("\"{}\"", p.display().to_string().replace('\\', "\\\\").replace('"', "\\\""))
}
