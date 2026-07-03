"""
Tiles each image and its turf mask into rectangular patches for training.

The source labels are the correct FeatureCollection GeoJSON polygons. This
script rasterizes them directly into a binary mask, then slices image/mask
pairs into 256x512 tiles by default.

Spatial validation split: the right-most portion of each source image is
held out as validation so the train and val tiles do not share nearby
context.

Empty-tile handling: tiles with almost no turf are kept at a reduced rate
so the dataset still contains plausible background-only examples.
"""
import argparse
import json
import os
import random

import cv2

try:
    from .geoutils import load_image_and_transform, load_raw_label_geometry, rasterize_geometry
except ImportError:
    from geoutils import load_image_and_transform, load_raw_label_geometry, rasterize_geometry


def tile_image(
    image_id,
    raw_dir,
    out_dir,
    tile_width=256,
    tile_height=512,
    stride_x=192,
    stride_y=384,
    val_frac=0.2,
    keep_empty_ratio=0.15,
    seed=0,
):
    rng = random.Random(seed)
    tiff_path = f"{raw_dir}/{image_id}.tiff"
    geojson_path = f"{raw_dir}/{image_id}.geojson"

    img, geo, _ = load_image_and_transform(tiff_path)
    geometries = load_raw_label_geometry(geojson_path)
    mask = rasterize_geometry(geometries, geo, value=1)

    h, w = img.shape[:2]
    val_split_col = int(w * (1 - val_frac))

    manifest = []
    n_train = 0
    n_val = 0
    n_skipped = 0

    for top in range(0, h - tile_height + 1, stride_y):
        for left in range(0, w - tile_width + 1, stride_x):
            img_tile = img[top:top + tile_height, left:left + tile_width]
            mask_tile = mask[top:top + tile_height, left:left + tile_width]

            turf_frac = float(mask_tile.mean())
            split = "val" if left >= val_split_col else "train"

            if turf_frac < 0.01 and rng.random() > keep_empty_ratio:
                n_skipped += 1
                continue

            tile_id = f"{image_id}_{top}_{left}"
            img_path = f"{out_dir}/images/{tile_id}.png"
            mask_path = f"{out_dir}/masks/{tile_id}.png"

            cv2.imwrite(img_path, cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR))
            cv2.imwrite(mask_path, (mask_tile * 255).astype("uint8"))

            manifest.append(
                {
                    "tile_id": tile_id,
                    "source_image": image_id,
                    "split": split,
                    "turf_frac": turf_frac,
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "tile_width": tile_width,
                    "tile_height": tile_height,
                    "top": top,
                    "left": left,
                }
            )

            if split == "train":
                n_train += 1
            else:
                n_val += 1

    print(f"[{image_id}] {n_train} train tiles, {n_val} val tiles, {n_skipped} empty tiles skipped")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="../data/raw")
    ap.add_argument("--out_dir", default="../data/tiles")
    ap.add_argument("--tile_width", type=int, default=256)
    ap.add_argument("--tile_height", type=int, default=512)
    ap.add_argument("--stride_x", type=int, default=192)
    ap.add_argument("--stride_y", type=int, default=384)
    ap.add_argument("--ids", nargs="+", default=["1", "2", "3"])
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--keep_empty_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(f"{args.out_dir}/images", exist_ok=True)
    os.makedirs(f"{args.out_dir}/masks", exist_ok=True)

    full_manifest = []
    for image_id in args.ids:
        full_manifest += tile_image(
            image_id,
            args.raw_dir,
            args.out_dir,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            stride_x=args.stride_x,
            stride_y=args.stride_y,
            val_frac=args.val_frac,
            keep_empty_ratio=args.keep_empty_ratio,
            seed=args.seed,
        )

    with open(f"{args.out_dir}/manifest.json", "w") as f:
        json.dump(full_manifest, f, indent=2)

    n_train = sum(1 for m in full_manifest if m["split"] == "train")
    n_val = sum(1 for m in full_manifest if m["split"] == "val")
    print(f"\nTotal: {n_train} train tiles, {n_val} val tiles -> {args.out_dir}/manifest.json")
