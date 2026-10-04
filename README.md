# vsky

Take the sky out of a dvlt reconstruction.

A feed-forward pointmap model predicts a depth for every pixel, sky included,
and dvlt's `conf_drop` is a percentile: at 0.1 it keeps 90% of the points and
most of the sky with them. On the 64-view building capture, **one point in five was
sky**, a blue shell draped over the trees and roofs.

vsky segments the sky in each image and writes the masks beside the input. With
`--cull` it also drops the points of a dvlt output that land on sky. Detached
like vfold: it reads a folder or list and writes files. Delete the binary and
the pipeline is unchanged.

```
vsky <IMAGE_DIR | list.txt>                              # masks only
vsky <IMAGE_DIR | list.txt> --key <DIR>                  # + sky painted black
vsky --strip <dvlt_out>                                  # drop black points
vsky <IMAGE_DIR | list.txt> --cull <dvlt_out> --size 504 # vote cull (fallback)
```

**Prefer `--key` + `--strip`.** Reconstruct from the keyed images, and every
sky point comes back exactly (0,0,0): the per-pixel provenance the PLY
otherwise lacks, carried through its colour channel. Measured A/B on the
60-frame building set, keyed vs raw + vote cull: 3.1% vs 5.0% sky-blue points
left, 14% more ground points kept, multi-view photo-consistency 0.210 vs
0.206, and the most isolated camera's pose became more consistent, not less.
The vote cull stays for reconstructions already made from raw images.
`reconstruct.bat --sky` runs key → dvlt → strip.

Pass the same folder or list the reconstruction ran on: poses.json carries no
filenames, so frames pair by position. For a `dvlt_long` output that means
`<out>\merged\images.txt`.

## Outputs

- `<input>_masks\<stem>.png`: full resolution, upright (JPEG EXIF applied, as
  dvlt does), **white = keep, black = sky**, the polarity COLMAP, nerfstudio and
  Spirula read as "ignore". COLMAP's own `--ImageReader.mask_path` expects
  `<name>.jpg.png` instead; rename if you feed it directly.
- `--key <DIR>`: `<stem>.jpg` per image, upright, sky pure black, plus
  `images.txt` in input order (so a list keeps its reference camera).
- `--strip` / `--cull`: `scene.ply` is rewritten without the sky and stamped
  `comment vsky`, and the original is kept as `scene_withsky.ply`. A stamped
  scene.ply means the original is the backup; an unstamped one is fresh dvlt
  output, even over an older run, and becomes the new original. Reruns never
  cull a cull, and a new reconstruction is never mistaken for an old one. Rerun `dvlt_postprocess.py`
  afterwards to refresh the upright/GLB/COLMAP products.
- `--json`: per-image sky fraction, plus the cull counts.

## How the cull works, and why it votes

dvlt.exe writes points frame by frame but records no pixel index, so vsky
recovers correspondence by projecting every point into every camera. It replays
dvlt's own geometry exactly: `lroundf` resize to `-s`, centre crop to 14 px,
centre pad to the batch. `report.txt`'s resolution is checked against `--size`,
and a mismatch is a hard error, never a silent misalignment.

A projection-based method gets one argument for free. Sky is at infinity, so a
real point, occluded or not, can never land on sky. In practice that fails,
because dvlt's per-view depths disagree. These are measured over 200k sampled
points:

| rule | sky-blue points removed | ground lost¹ | non-blue lost |
|---|---|---|---|
| plain, any view says sky | 95.9% | 49.5% | 56.1% |
| plain, ≥25% of views | 62.5% | 0.01% | 1.06% |
| plain, majority | 32.9% | 0.00% | 0.09% |
| **photo-gated**, any view | 80.8% | 10.4% | 14.2% |
| **photo-gated**, ≥2 views | 62.8% | 1.45% | 2.74% |
| **photo-gated**, majority | 78.1% | 0.16% | 0.91% |

¹ Points below the lowest camera in dvlt's upright frame: structure by
construction, so this column is independent of colour. The "removed" column
uses colour (clear blue sky) as its proxy, and so flatters the gated rules.

*Photo-gated* means a view only votes when the image colour where the point
lands matches the point's own colour (mean |ΔRGB| ≤ 20, as in MVS fusion). That
filters both failure modes. Structure drifting onto sky stops voting sky, and
sky floaters landing on brick stop counting as seen. The gate only removes
evidence, so at worst it equals the plain vote. The rule itself is `is_sky()` in
`src/cull.rs`.

## Models

| | default | alternative |
|---|---|---|
| file | `skyseg.onnx` (176 MB) | `segformer-b0-ade.onnx` (15 MB) |
| source | huggingface.co/JianyuanWang/skyseg | Xenova/segformer-b0-finetuned-ade-512-512 |
| licence | MIT | NVIDIA non-commercial |
| kind | U2-Net, binary sky, 320² | 150-class ADE20K, sky = class 2 |
| building set, CPU | 0.65 s/frame | 0.5 s/frame |

Both live in `%USERPROFILE%\.cache\vsky\`. The output layout is detected (1
channel = probability, 150 = ADE20K logits) and anything else is refused,
because a Cityscapes model's channel 2 is "building". skyseg is the default: it
is the model VGGT's demo filters sky with, and on tree canopy it masks the sky
between bare branches that SegFormer labels as tree. Neither resolves the
smallest canopy gaps, which is the blue speckle left inside the tree crowns.

Device: `--device auto` tries CUDA then CPU, each rung hard-failed so the
printed device is the real one. DirectML is opt-in (`--device dml`) until
verified. Keep the `onnxruntime_providers_*.dll` and `DirectML.dll` beside
`vsky.exe`; they are the same rc.12 builds vfold ships.

## Wiring into reconstruct.bat (not applied)

Between the `dvlt.exe` call and post-processing:

```bat
if exist "%ROOT%vsky.exe" "%ROOT%vsky.exe" "%INPUT%" --cull "%OUTPUT%" --size %IMG_SIZE%
```

## The exact alternative

Voting exists only because the PLY drops pixel provenance. If dvlt.exe accepted
masks (zeroing them in its keep-mask, before the `conf_drop` percentile) or
wrote a per-point pixel index, sky removal would be exact per source pixel, as
in VGGT, with no voting and no colour gate. That is roughly 15 lines in
`dvlt.cu` and a rebuild.
