from pathlib import Path
import geopandas as gpd
import geemap
import ee

from .io_utils import ensure_dir, shp_path

def export_gdf(gdf: gpd.GeoDataFrame, output_dir: Path, filename: str) -> Path:
    ensure_dir(output_dir)
    out = shp_path(output_dir, filename)
    gdf.to_file(out)
    return out

def export_ee_fc_as_shp(fc: ee.FeatureCollection, output_dir: Path, filename: str, to_epsg: int = 3857) -> Path:
    ensure_dir(output_dir)

    gdf = geemap.ee_to_gdf(fc)
    gdf = gdf.set_crs(epsg=4326, allow_override=True)
    if to_epsg:
        gdf = gdf.to_crs(epsg=to_epsg)

    out = shp_path(output_dir, filename)
    gdf.to_file(out)
    return out

def export_permanent_water_shp(
    permanent_water_bin_img: ee.Image,
    area: ee.Geometry,
    output_dir: Path,
    filename: str = "permanent_water.shp",
) -> Path:
    """
    permanent_water_bin_img: ee.Image (binary) where 1 means permanent water.
    """
    ensure_dir(output_dir)

    mask_img = permanent_water_bin_img.rename("mask").selfMask()
    mask_vectors = mask_img.reduceToVectors(
        geometry=area,
        scale=30,
        crs="EPSG:4326",
        geometryType="polygon",
        eightConnected=True,
        labelProperty="mask",
        maxPixels=1e10,
    )

    gdf = geemap.ee_to_gdf(mask_vectors)
    gdf = gdf.set_crs(epsg=4326, allow_override=True).to_crs(epsg=3857)

    out = shp_path(output_dir, filename)
    gdf.to_file(out)
    return out