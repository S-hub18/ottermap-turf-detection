"""
Shared geospatial utilities.

Uses rasterio/geopandas/shapely where available (standard stack — install via
requirements.txt). Falls back to a lightweight tifffile+cv2 implementation
only for environments where rasterio can't be installed (e.g. sandboxed dev),
so the logic can still be sanity-checked without the full GIS stack.
"""
import json
import numpy as np
from affine import Affine

try:
    import rasterio
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import rowcol
    from rasterio.crs import CRS
    HAVE_RASTERIO = True
except ImportError:
    HAVE_RASTERIO = False
    import tifffile
    import cv2


def load_image_and_transform(tiff_path):
    """
    Returns (image as HxWx3 uint8 ndarray, geospatial metadata dict, crs).
    The metadata dict includes bounds, image dimensions, and a usable
    north-up affine transform under 'rio_transform'.
    """
    def _manual_geo_from_tiepoints(tiff_path, width, height):
        from PIL import Image

        img = Image.open(tiff_path)
        tiepoints = img.tag_v2[33922]
        corners = [(tiepoints[i * 6 + 3], tiepoints[i * 6 + 4]) for i in range(len(tiepoints) // 6)]
        xs = [corner[0] for corner in corners]
        ys = [corner[1] for corner in corners]
        minx = min(xs)
        maxx = max(xs)
        miny = min(ys)
        maxy = max(ys)
        px_w = (maxx - minx) / width
        px_h = (maxy - miny) / height
        geo = {
            "minx": minx,
            "maxx": maxx,
            "miny": miny,
            "maxy": maxy,
            "width": width,
            "height": height,
            "rio_transform": Affine(px_w, 0.0, minx, 0.0, -px_h, maxy),
        }
        crs = CRS.from_epsg(4326) if HAVE_RASTERIO else "EPSG:4326"
        return geo, crs

    if HAVE_RASTERIO:
        with rasterio.open(tiff_path) as src:
            img = src.read()
            img = np.moveaxis(img, 0, -1)
            if img.shape[2] > 3:
                img = img[:, :, :3]
                
            try:
                geo, crs = _manual_geo_from_tiepoints(tiff_path, src.width, src.height)
            except Exception:
                import logging
                logging.warning(f"No valid georeferencing/tiepoints found in {tiff_path}. Falling back to pixel coordinates.")
                geo, crs = {"width": src.width, "height": src.height}, None

            if src.transform and src.transform != Affine.identity():
                geo["rio_transform"] = src.transform
                crs = src.crs
                geo["px_area"] = abs(src.transform[0] * src.transform[4])
            elif "rio_transform" not in geo:
                geo["rio_transform"] = Affine.identity()

            return img, geo, crs
    else:
        from PIL import Image
        img = np.array(Image.open(tiff_path).convert("RGB"))
        h, w = img.shape[:2]
        try:
            geo, crs = _manual_geo_from_tiepoints(tiff_path, w, h)
        except Exception:
            import logging
            logging.warning(f"No valid georeferencing/tiepoints found in {tiff_path}. Falling back to pixel coordinates.")
            geo, crs = {"width": w, "height": h, "rio_transform": Affine.identity()}, None
        return img, geo, crs


def lonlat_to_px(lon, lat, geo):
    px_w = (geo["maxx"] - geo["minx"]) / geo["width"]
    px_h = (geo["maxy"] - geo["miny"]) / geo["height"]
    col = (lon - geo["minx"]) / px_w
    row = (geo["maxy"] - lat) / px_h
    return col, row


def px_to_lonlat(col, row, geo):
    px_w = (geo["maxx"] - geo["minx"]) / geo["width"]
    px_h = (geo["maxy"] - geo["miny"]) / geo["height"]
    lon = geo["minx"] + col * px_w
    lat = geo["maxy"] - row * px_h
    return lon, lat


def load_raw_label_geometry(geojson_path):
    """
    Loads either the Ottermap FeatureCollection format or the older bare
    {"geometry": {...}} format and returns normalized geometry objects.
    """
    with open(geojson_path) as f:
        data = json.load(f)

    if data.get("type") == "FeatureCollection":
        return [feature["geometry"] for feature in data.get("features", []) if feature.get("geometry")]

    if "geometry" in data:
        return data["geometry"]

    return data


def _iter_polygon_rings(geom):
    if geom.get("type") == "Polygon":
        yield geom["coordinates"]
    elif geom.get("type") == "MultiPolygon":
        for polygon in geom["coordinates"]:
            yield polygon


def rasterize_geometry(geom, geo, value=1):
    """
    Rasterize one geometry or a list of geometries (in lon/lat) onto a mask
    the size of the source image, honoring holes in polygon rings.
    """
    h, w = geo["height"], geo["width"]
    mask = np.zeros((h, w), dtype=np.uint8)

    geometries = geom if isinstance(geom, list) else [geom]

    if HAVE_RASTERIO and "rio_transform" in geo:
        shapes = []
        for geometry in geometries:
            for polygon in _iter_polygon_rings(geometry):
                shapes.append(({"type": "Polygon", "coordinates": polygon}, value))
        return rio_rasterize(shapes, out_shape=(h, w), transform=geo["rio_transform"],
                              fill=0, dtype=np.uint8)

    for geometry in geometries:
        for poly in _iter_polygon_rings(geometry):
            ext = poly[0]
            pts = np.array([lonlat_to_px(lon, lat, geo) for lon, lat in ext],
                            dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], value)
            for hole in poly[1:]:
                pts_h = np.array([lonlat_to_px(lon, lat, geo) for lon, lat in hole],
                                  dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts_h], 0)
    return mask
