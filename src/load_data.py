import json
import numpy as np

try:
    from .geoutils import load_image_and_transform, load_raw_label_geometry
except ImportError:
    from geoutils import load_image_and_transform, load_raw_label_geometry

def load_image_and_labels(tiff_path, geojson_path):
    img, geo, crs = load_image_and_transform(tiff_path)
    geometries = load_raw_label_geometry(geojson_path)
    h, w = img.shape[:2]
    print(f"Loaded {tiff_path}: {w}x{h}, CRS={crs}")
    print(f"Loaded {geojson_path}: {len(geometries) if isinstance(geometries, list) else 1} polygons")

    return img, geometries, geo["rio_transform"], crs, (h, w)

if __name__ == "__main__":
    img, geoms, transform, crs, (h, w) = load_image_and_labels(
        "../data/raw/1.tiff", "../data/raw/1.geojson"
    )
    print("Image shape:", img.shape)
    print("Image dtype:", img.dtype)
    print("Transform:", transform)

    # sanity check: convert the image's pixel corners to geo coords
    print("Top-left corner (geo):", transform * (0, 0))
    print("Bottom-right corner (geo):", transform * (w, h))

    # peek at one polygon's raw coordinates to compare
    sample_geometry = geoms[0] if isinstance(geoms, list) else geoms
    print("Sample polygon coord:", sample_geometry["coordinates"][0][0])