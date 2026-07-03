"""
Vectorizes a predicted binary turf mask into GIS-compatible outputs:
GeoJSON and Shapefile, in the original image's CRS, plus simplification
to avoid absurdly dense polygons from raster stair-stepping.

Usage:
    python vectorize.py --image ../data/raw/1.tiff --mask ../results/.../1_mask.png \
        --out_dir ../outputs_gis --min_area_px 30 --simplify_tol 0.00001
"""
import argparse
import os
import numpy as np
import cv2

from geoutils import load_image_and_transform, px_to_lonlat

try:
    import rasterio
    from rasterio.features import shapes as rio_shapes
    import geopandas as gpd
    from shapely.geometry import shape, mapping
    HAVE_GIS_STACK = True
except ImportError:
    HAVE_GIS_STACK = False


def mask_to_polygons_fallback(mask, geo, min_area_px=30):
    """cv2-only fallback (no rasterio/shapely) -> list of lon/lat polygon rings."""
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_px:
            continue
        ring = [px_to_lonlat(pt[0][0], pt[0][1], geo) for pt in cnt]
        if len(ring) >= 3:
            ring.append(ring[0])
            polygons.append(ring)
    return polygons


def vectorize(tiff_path, mask_path, out_dir, min_area_px=30, simplify_tol=None):
    image_id = os.path.splitext(os.path.basename(mask_path))[0].replace("_mask", "")
    img, geo, crs = load_image_and_transform(tiff_path)
    mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)

    os.makedirs(out_dir, exist_ok=True)

    if HAVE_GIS_STACK and "rio_transform" in geo:
        results = (
            {"properties": {"class": "turf"}, "geometry": s}
            for s, v in rio_shapes(mask, mask=mask.astype(bool), transform=geo["rio_transform"])
            if v == 1
        )
        geoms = [shape(r["geometry"]) for r in results]
        geoms = [g for g in geoms if g.area > 0]
        gdf = gpd.GeoDataFrame({"class": ["turf"] * len(geoms)}, geometry=geoms, crs=crs)
        px_area = abs(geo["rio_transform"][0] * geo["rio_transform"][4])
        gdf = gdf[gdf.geometry.area > min_area_px * px_area]
        if simplify_tol:
            gdf["geometry"] = gdf.geometry.simplify(simplify_tol, preserve_topology=True)

        geojson_path = f"{out_dir}/{image_id}_predictions.geojson"
        shp_path = f"{out_dir}/{image_id}_predictions.shp"
        gdf.to_file(geojson_path, driver="GeoJSON")
        gdf.to_file(shp_path, driver="ESRI Shapefile")
        print(f"[{image_id}] {len(gdf)} polygons -> {geojson_path}, {shp_path}")
        return geojson_path, shp_path

    else:
        # fallback: GeoJSON only, written by hand (no shapefile without pyshp/geopandas)
        polygons = mask_to_polygons_fallback(mask, geo, min_area_px)
        features = []
        for ring in polygons:
            features.append({
                "type": "Feature",
                "properties": {"class": "turf"},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            })
        fc = {"type": "FeatureCollection", "features": features}
        geojson_path = f"{out_dir}/{image_id}_predictions.geojson"
        import json
        with open(geojson_path, "w") as f:
            json.dump(fc, f)
        print(f"[{image_id}] {len(features)} polygons -> {geojson_path} "
              f"(install rasterio+geopandas for Shapefile export)")
        return geojson_path, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out_dir", default="../outputs_gis")
    ap.add_argument("--min_area_px", type=float, default=30)
    ap.add_argument("--simplify_tol", type=float, default=None)
    args = ap.parse_args()
    vectorize(args.image, args.mask, args.out_dir, args.min_area_px, args.simplify_tol)
