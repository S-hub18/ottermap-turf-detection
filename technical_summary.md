# Technical Summary — Turf/Grass Detection Pipeline

## Approach

**Architecture:** U-Net with an ImageNet-pretrained ResNet-34 encoder
(`segmentation_models_pytorch`), trained for binary turf/background
segmentation. A zero-shot SAM/Grounding-DINO comparison may be used later
as an optional generalization check, but the current core pipeline is the
supervised U-Net path.

**Why segmentation over detection:** turf is an amorphous region rather
than a discrete countable object, so pixel-wise segmentation maps directly
to the required polygon/GeoJSON output without an intermediate
box-to-mask step.

**Training methodology:** Given only 3 source images, the dominant
overfitting risk is the encoder memorizing this dataset's specific color
palette and textures rather than learning a transferable notion of "turf."
To mitigate this:
- The encoder is frozen for the first 5 epochs (decoder-only warmup), then
  unfrozen at 10x lower learning rate than the decoder for the remainder
  of training — full end-to-end fine-tuning at a uniform LR on ~240 tiles
  would overwrite general-purpose ImageNet features with dataset-specific
  ones.
- Augmentation is weighted toward color/lighting jitter (hue/saturation,
  brightness/contrast, RGB shift, simulated compression artifacts) over
  purely geometric transforms, on the assumption that sensor/season/
  lighting differences are the most likely source of train/test domain gap.
- Loss is 0.5×BCE + 0.5×Dice to handle the background-heavy class balance
  in untiled, mixed-content tiles.

*Training concluded in ~20 minutes for 40 epochs on an Apple Silicon (M-series) Mac using the `mps` device backend.*

## Dataset Preparation

**Critical finding:** the provided package contains two different label
sources. The `*.shp.geojson` files are coarse parcel/site boundaries and
are not usable as turf masks. The `*.geojson` FeatureCollections contain
individual turf polygons and were visually verified against all 3 source
images, so the current pipeline rasterizes those polygons directly.

The GeoTIFF files also have tiepoint-only georeferencing that rasterio/GDAL
does not parse correctly in this environment. The pipeline therefore reads
the corner tiepoints manually and constructs an EPSG:4326 affine transform
before rasterizing labels or vectorizing predictions.

**Tiling:** images are tiled into 256×512 patches (192px × 384px stride),
with a **spatial** train/val split — a right-hand strip of each source
image held out for validation, rather than randomly sampled tiles, since
random tiles would leak context from immediately adjacent training tiles
and overstate validation performance. Near-empty (≤1% turf) tiles are
downsampled to a 15% keep-rate so the model still sees plausible "no turf
here" context without the dataset being dominated by trivial background
tiles.

## Results

- **Training Performance**: Reached a Validation IoU of `0.8746` and Validation Dice of `0.9234` by Epoch 22. 
- **Generalization Performance**: On unseen external imagery (e.g., `test1`, `test2`, `test3`), the standalone U-Net exhibited a large domain gap, misclassifying shadows as background and exhibiting high false-negative rates. 
- **The Solution (Hyper-Sensitive Ensemble)**: To resolve this, we engineered an ensemble approach. We lowered the U-Net threshold to 1% (making it hyper-sensitive and maximizing recall), and used a Zero-Shot Segment-Anything (SAM) model to prune the resulting false-positives via a Logical AND.
- **Failure Case Resolved**: On `test3.jpeg`, the Zero-Shot SAM model hallucinated a large body of water as "grass". However, because the U-Net correctly rejected the water due to its spectral training, the ensemble completely removed the hallucination!
- **Known Limitation**: The zero-shot SAM model failed entirely on the `3.tiff` training image because the image was taken from such a high altitude that the grass polygons shrunk beyond the Grounding DINO model's standard detection resolution.

## Improvements (with more time/data)

- **More source images per domain type** (residential, campus, sports
  field, rural/agricultural, arid) — 3 images cannot characterize the
  variety the evaluation set may contain.
- **Active-learning style label correction**: use model uncertainty and
  validation errors to flag the highest-error tiles for manual review.
- **NDVI or multispectral input** if the aerial source ever provides
  near-infrared bands — would substantially improve the
  grass-vs-tree-vs-dormant-grass distinction beyond RGB-only heuristics.
- **Full fine-tuning of SAM** or another foundation segmentation model
  using the verified turf masks as supervision.
- **Per-region threshold calibration** at inference time, since a single
  global probability threshold may not transfer across very different
  lighting conditions.
