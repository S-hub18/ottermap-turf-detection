# Ottermap Turf Detection — Project Handoff / Continuation Brief

This document exists to hand off full context to whoever/whatever continues
this work (Copilot, another session, a teammate). Read it fully before
writing code — several early assumptions turned out to be wrong and were
corrected through investigation; don't repeat those mistakes.

---

## 1. The Task

72-hour technical challenge (see `task_brief.pdf` in repo root) for an
Ottermap ML Engineer Intern role. Build an end-to-end computer vision
pipeline that:
1. Learns to identify **turf/grass** from aerial GeoTIFF imagery + GeoJSON
   labels (single class only).
2. Generalizes to unseen imagery from a **different geographic location**.
3. Produces GIS-compatible outputs (GeoJSON/Shapefile).
4. Is packaged as a reproducible GitHub repo with a simple inference command.

**Chosen flow (agreed with user, follow this exactly):**
```
Data Preparation
  Load imagery and GeoJSON layers → Rasterize masks → Tile into 256×512px
  patches → Augment → Train/val split
Model Training
  Fine-tune a pretrained architecture (YOLOv8-seg, U-Net, SegFormer, SAM)
  → Log mIoU + per-class metrics → Save best checkpoint
Generalization Testing
  Source 2–3 new aerial images from a different location → Run inference
  → Produce overlay PNGs and GeoJSON outputs → Document successes and
  failure cases
Packaging & Submission
  Structure GitHub repo → Write README with setup + inference command →
  Complete validation_report.json → Submit link via portal
```

**Deliverables required by the brief:** GitHub repo link, model weights,
technical summary PDF (≤3 pages), sample outputs — submitted via portal,
not just a zip.

---

## 2. Critical Finding: There Are TWO Label Sources Per Image — Only One Is Usable

The assessment provides a `feature layers` folder with **two subfolders**,
`geojson/` and `shapefile/`, each containing a same-named `.geojson` file
per image (e.g. both have `1.geojson`) — **but their contents are
completely different**, and only one is correct.

### `shapefile/` folder version — DO NOT USE AS TRAINING TARGET
- Format: bare `{"geometry": {...}}` object (not a proper `FeatureCollection`)
- Single `MultiPolygon`, one blob per image
- **Visually verified against all 3 images: this is a coarse parcel/site
  boundary, NOT a turf outline.** It includes rooftops, driveways, pools,
  parking lots, tennis courts — anything inside the general property
  boundary, not just grass.
- Coverage was 28% (image 1), 73% (image 2), 45% (image 3) of the image —
  way too high to be turf-only.
- An entire label-refinement heuristic (Excess Green Index + texture
  filtering to strip out non-grass pixels within this boundary) was built
  to compensate for this. **That heuristic is now obsolete — do not use it.**
  It lived in `src/label_refine.py` in earlier iterations of this repo.

### `geojson/` folder version — USE THIS ONE
- Format: proper `FeatureCollection` with `type`, `name`, `crs`, `features`
- **Multiple individual polygons per image** — 95 features (image 1), 50
  features (image 2), 16 features (image 3)
- **Visually verified against all 3 images: these trace individual lawns /
  turf patches precisely** — excluding rooftops, driveways, pools, dirt
  infields (image 3's baseball diamonds), courts. This is real, usable
  ground truth.
- Coverage: 9.1% (image 1), 45.2% (image 2), 42.4% (image 3) — much more
  plausible for actual grass area.
- CRS declared as `urn:ogc:def:crs:OGC:1.3:CRS84` (i.e. lon/lat, same as
  EPSG:4326 axis order).

**Action for whoever continues this: confirm `data/raw/*.geojson` in the
repo are copies from the `geojson/` folder (FeatureCollection, multi-polygon
per image), NOT the `shapefile/` folder. User already copied the correct
ones in — verified via `95 features` printout — but double check if picking
this back up fresh.**

Because the labels are now trustworthy, **Data Prep no longer needs a label
refinement/heuristic step.** Load → rasterize → tile is sufficient. This
simplifies the pipeline significantly versus earlier plans.

---

## 3. Critical Finding: GeoTIFFs Have Non-Standard Georeferencing That GDAL/rasterio Fails to Auto-Detect

All 3 provided TIFFs (`1.tiff`, `2.tiff`, `3.tiff`) are:
- RGB, uint8, JPEG-compressed inside the TIFF container
- Georeferenced via `ModelTiepointTag` (4 corner tiepoints), NOT a full
  `ModelTransformationTag` (single affine matrix) or `ModelPixelScaleTag`
  combo
- CRS declared via `GeoKeyDirectoryTag`/`GeoAsciiParamsTag` as WGS84
  (EPSG:4326), confirmed via `tifffile` inspection
- Example tiepoint tag content (image 1):
  `(0,0,0, -121.868..., 39.7539..., 0,  3811,0,0, -121.863..., 39.7539...,
  0,  0,3407,0, -121.868..., 39.7503...,0,  3811,3407,0, -121.863...,
  39.7503...,0)` — 4 corner points mapping pixel (col,row) to (lon,lat)

**Problem discovered in this session:** on the user's machine, running:
```python
import rasterio
with rasterio.open('data/raw/1.tiff') as src:
    print(src.transform)  # -> identity matrix (1,0,0 / 0,1,0 / 0,0,1)
    print(src.crs)        # -> None
    print(src.tags())     # -> {'AREA_OR_POINT': 'Area'} only
```
rasterio/GDAL **completely fails to parse the georeferencing** — no CRS,
identity transform. This matches a known rough edge with tiepoint-only
(not full affine) GeoTIFFs combined with JPEG compression, depending on
libtiff/GDAL build.

**We were mid-investigation when session ended.** Next step was running:
```bash
gdalinfo data/raw/1.tiff 2>&1 | head -30
```
on the user's Mac to see if the GDAL CLI (separate from rasterio's
bindings) parses it any differently — **this output was never obtained,
pick up here.**

### Fallback plan if GDAL genuinely cannot read the geo tags (likely path)
Compute the affine transform **manually** from the tiepoints rather than
relying on `rasterio`'s auto-detection, since we already know the tiepoint
math is correct (verified repeatedly in this conversation via
successful/correct visual overlays):

```python
import tifffile

def get_manual_geotransform(tiff_path):
    tif = tifffile.TiffFile(tiff_path)
    page = tif.pages[0]
    tp = page.tags['ModelTiepointTag'].value
    # 4 tiepoints: (pixel_i, pixel_j, pixel_k, geo_x, geo_y, geo_z) x4
    corners = [(tp[i*6+3], tp[i*6+4]) for i in range(4)]
    xs = [c[0] for c in corners]; ys = [c[1] for c in corners]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    return minx, maxx, miny, maxy  # then compute px_w, px_h as (max-min)/dimension
```
This has been used and validated multiple times already in this session
(overlay checks all lined up correctly with visible lawns/features in the
imagery). It can either:
(a) be used standalone with `cv2.fillPoly` for rasterization (no
    rasterio dependency for this specific step), or
(b) be converted into a proper `rasterio.transform.Affine` object
    (`Affine(px_w, 0, minx, 0, -px_h, maxy)`) and passed explicitly to
    `rasterio.open(path, transform=..., crs='EPSG:4326')`-style APIs or
    `rasterio.features.rasterize(..., transform=manual_affine)` so the
    rest of the rasterio-based pipeline (rasterize, GeoJSON/Shapefile
    export via geopandas) still works normally downstream.

**Recommended: go with (b)** — construct the Affine manually, inject it
wherever rasterio needs a transform, keep everything else in the pipeline
using the standard rasterio/geopandas stack as originally planned. Don't
abandon rasterio entirely; it works fine for pixel I/O, just not for
auto-parsing this specific file's geo tags.

---

## 4. Environment State (User's Machine — Mac, Apple Silicon M-series)

Already installed and confirmed working, in a Python 3.11 venv (user's
system Python was 3.14, too new for PyTorch compatibility at time of
writing — 3.11 installed via `brew install python@3.11` specifically for
this project):

```
torch 2.12.1, torchvision 0.27.1   (MPS backend confirmed available: True)
rasterio 1.4.4, geopandas 1.1.4, shapely 2.1.2  (installed clean, no GDAL brew needed)
opencv-python 4.13.0.92, matplotlib 3.11.0
numpy 2.4.6, pillow 12.3.0
```

Venv setup (already done, for reference only):
```bash
brew install python@3.11
cd ottermap-turf-detection
python3.11 -m venv venv
source venv/bin/activate
pip install torch torchvision
pip install rasterio geopandas shapely
pip install opencv-python matplotlib
```

**Device selection code needed everywhere (MPS, not just cuda/cpu):**
```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

**Still needed, not yet installed in the project venv as of the latest
check:** `segmentation-models-pytorch`, `tifffile`. `albumentations`,
`rasterio`, `geopandas`, `cv2`, `torch`, and `torchvision` are installed.
`transformers` is only needed if the optional SAM/Grounding-DINO comparison
is pursued later.

---

## 5. Repo State / File Layout

Project root: `ottermap-turf-detection/` (already git-uninitialized on
user's machine — user explicitly does NOT want it auto-pushed, wants to
review/push manually themselves).

```
ottermap-turf-detection/
├── README.md                  (updated to current direct-rasterization plan)
├── requirements.txt           (lists needed dependencies; venv still lacks segmentation-models-pytorch/tifffile)
├── technical_summary.md       (draft with placeholders, needs real numbers post-training)
├── task_brief.pdf
├── data/raw/                  (1/2/3 .tiff + .geojson — VERIFY these are the geojson/-folder versions, not shapefile/-folder)
├── data/tiles/                (clean regenerable target; stale manifest/samples removed)
├── data/tiles_smoke/          (77 verified 256×512 tile pairs from image 1; usable for dataset smoke tests)
├── src/
│   ├── load_data.py           (uses shared geoutils loader; confirmed correct manual transform for image 1)
│   ├── geoutils.py            (manual tiepoint fallback + direct GeoJSON rasterization)
│   ├── tiling.py              (256×512 direct-rasterization tiler; ready to regenerate data/tiles)
│   ├── dataset.py             (batch loading tested on data/tiles_smoke)
│   ├── train.py               (untested — needs segmentation-models-pytorch; has MPS device fix already applied)
│   ├── inference.py            (untested — has MPS device fix already applied)
│   ├── vectorize.py            (needs re-test after predictions exist)
│   └── zero_shot_sam.py        (untested, optional — see Section 6)
├── weights/                    (empty, nothing trained yet)
├── results/{train,val,external}_preds/  (empty)
└── outputs_gis/                (stale sample output removed; regenerate after inference)
```

**Current cleanup status:** the obsolete `src/label_refine.py` module and
old generated `_refined_labels`, `_samples`, stale `data/tiles/manifest.json`,
and stale `outputs_gis/3_predictions.geojson` have been removed. Current
docs/configs point at direct FeatureCollection rasterization and 256×512
tiling.

---

## 6. Current Validation State

`src/load_data.py` now delegates to `src/geoutils.py`, which falls back to
manual GeoTIFF tiepoints when rasterio returns `crs=None` and identity
transform. Latest confirmed output (`cd src && ../venv/bin/python
load_data.py`):
```
Loaded ../data/raw/1.tiff: 3811x3407, CRS=EPSG:4326
Loaded ../data/raw/1.geojson: 95 polygons
Image shape: (3407, 3811, 3)
Image dtype: uint8
Transform: | 0.00, 0.00,-121.87|
| 0.00,-0.00, 39.75|
| 0.00, 0.00, 1.00|
Top-left corner (geo): (-121.86821784073192, 39.75390683152841)
Bottom-right corner (geo): (-121.86310715330987, 39.75039416847159)
Sample polygon coord: [-121.86547277245134, 39.75232989994444]
```

FeatureCollection label counts are verified:
- `1.geojson`: 95 features
- `2.geojson`: 50 features
- `3.geojson`: 16 features

Direct rasterization coverage with the manual transform:
- image 1: 8.94%
- image 2: 45.05%
- image 3: 42.31%

`data/tiles_smoke/manifest.json` has 77 verified tile pairs from image 1
(68 train / 9 val), all 256×512. `dataset.py` successfully loaded a batch:
images `[B, 3, 512, 256]`, masks `[B, 1, 512, 256]`.

Model-side smoke test completed:
```bash
venv/bin/python src/train.py \
  --manifest data/tiles_smoke/manifest.json \
  --out_weights weights/turf_unet_smoke.pt \
  --epochs 1 --warmup_epochs 1 --batch_size 4 --num_workers 0
```
Result:
- device: `mps`
- train tiles: 68
- val tiles: 9
- epoch 1 train loss: 0.6006
- epoch 1 val IoU: 0.3631
- checkpoint: `weights/turf_unet_smoke.pt`

This proves the U-Net model loads, pretrained encoder weights are cached,
loss/backprop works, rectangular 512×256 tensors flow through the model,
MPS selection works, and checkpoint saving works.

Full manifest regenerated and smoke-tested:
```bash
venv/bin/python src/tiling.py \
  --raw_dir data/raw --out_dir data/tiles \
  --tile_width 256 --tile_height 512 \
  --stride_x 192 --stride_y 384 --ids 1 2 3
venv/bin/python src/train.py \
  --manifest data/tiles/manifest.json \
  --out_weights weights/turf_unet_full_smoke.pt \
  --epochs 1 --warmup_epochs 1 --batch_size 4 --num_workers 0
```
Result:
- manifest: 538 tile pairs total
- train tiles: 473
- val tiles: 65
- epoch 1 train loss: 0.5387
- epoch 1 val IoU: 0.4908
- checkpoint: `weights/turf_unet_full_smoke.pt`
- device in the non-escalated shell: `cpu`

Short real training run completed on the full manifest:
```bash
venv/bin/python src/train.py \
  --manifest data/tiles/manifest.json \
  --out_weights weights/turf_unet_full_5ep.pt \
  --epochs 5 --warmup_epochs 1 --batch_size 4 --num_workers 0
```
Result:
- device: `mps`
- train tiles: 473
- val tiles: 65
- epoch 1 train loss: 0.4871  val IoU: 0.6162
- epoch 2 train loss: 0.4452  val IoU: 0.6379
- epoch 3 train loss: 0.4212  val IoU: 0.6553
- epoch 4 train loss: 0.3943  val IoU: 0.6871
- epoch 5 train loss: 0.3564  val IoU: 0.6316
- best checkpoint: `weights/turf_unet_full_5ep.pt`
- best recorded val IoU: `0.6871`

---

## 7. Immediate Next Steps (in order)

1. **Train longer or finalize** — either continue the full-manifest run
   toward the planned 40 epochs or lock `weights/turf_unet_full_5ep.pt` as
   the current best smoke-trained checkpoint. The 5-epoch pass already
   shows the model is learning and validation IoU is rising.

2. **Generalization testing** — source 2–3 external images from a
   different location (not yet done — user has not provided or sourced
   these yet). Run `inference.py`, produce overlay PNGs + GeoJSON, and
   explicitly document failure cases (this is a graded rubric item, don't
   skip the "document failures" part).

3. **Packaging** — build `validation_report.json` (this is a NEW
   deliverable the user specified that wasn't in earlier drafts — needs a
   schema designed; should probably include: per-image train/val
   metrics, external test image results, IoU/precision/recall, list of
   documented failure cases, model/config metadata). Finalize README with
   exact setup + inference commands (draft exists but needs updating to
   match final file layout/label source). User will handle the actual
   `git init`/push themselves — don't do it for them unprompted.

---

## 8. Working Style / Constraints to Respect

- **User wants to run everything themselves, one step at a time**, pasting
  back terminal output for verification before proceeding to the next
  step. Do not dump large multi-file scaffolds — build incrementally,
  confirm each piece runs, then move to the next.
- **User does not want anything git-pushed automatically** — they'll
  handle GitHub themselves once ready.
- User is on a **Mac with Apple Silicon (M-series)** — always account for
  MPS vs CUDA, and be aware Homebrew is available.
- Always **visually verify georeferencing/rasterization against the actual
  source image** before trusting any mask or overlay — this session
  caught two serious data-quality issues (wrong label source, broken geo
  tags) precisely by insisting on visual checks rather than trusting file
  structure alone. Keep doing this at every new step.
- Follow the user's exact 4-stage flow (Section 1) — don't reintroduce
  the abandoned label-refinement heuristic or restructure the plan
  without cause.
