from __future__ import annotations

import datetime
from dataclasses import dataclass
import ee


@dataclass
class FloodResult:
    # Maski (ee.Image)
    floodedD_raw: ee.Image
    floodedA_raw: ee.Image
    floodedD_10m: ee.Image
    floodedA_10m: ee.Image

    # Wektory (ee.FeatureCollection)
    flood_vectorsD: ee.FeatureCollection
    flood_vectorsA: ee.FeatureCollection
    flood_vectors_filteredD: ee.FeatureCollection

    # Dodatkowo (przydatne do eksportu / debug)
    diffD: ee.Image
    diffA: ee.Image
    permanent_water_bin: ee.Image
    slope: ee.Image


# dB <-> linear

def to_natural(img_db: ee.Image) -> ee.Image:
    return ee.Image(10.0).pow(img_db.divide(10.0))


def to_db(img_lin: ee.Image) -> ee.Image:
    return img_lin.log10().multiply(10.0)


# Refined Lee (na obrazie liniowym) 

def refined_lee(img_lin: ee.Image) -> ee.Image:
    """
    Refined Lee filter for speckle reduction.
    Input: linear scale (NOT dB). Output: linear scale.
    """
    weights3 = ee.List.repeat(ee.List.repeat(1, 3), 3)
    kernel3 = ee.Kernel.fixed(3, 3, weights3, 1, 1, False)

    mean3 = img_lin.reduceNeighborhood(ee.Reducer.mean(), kernel3)
    var3 = img_lin.reduceNeighborhood(ee.Reducer.variance(), kernel3)

    sample_weights = ee.List([
        [0,0,0,0,0,0,0],
        [0,1,0,1,0,1,0],
        [0,0,0,0,0,0,0],
        [0,1,0,1,0,1,0],
        [0,0,0,0,0,0,0],
        [0,1,0,1,0,1,0],
        [0,0,0,0,0,0,0],
    ])
    sample_kernel = ee.Kernel.fixed(7, 7, sample_weights, 3, 3, False)

    sample_mean = mean3.neighborhoodToBands(sample_kernel)
    sample_var = var3.neighborhoodToBands(sample_kernel)

    gradients = (
        sample_mean.select(1).subtract(sample_mean.select(7)).abs()
        .addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs())
        .addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs())
        .addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs())
    )

    max_gradient = gradients.reduce(ee.Reducer.max())
    grad1 = gradients.select(0).eq(max_gradient)
    grad2 = gradients.select(1).eq(max_gradient)
    grad3 = gradients.select(2).eq(max_gradient)
    grad4 = gradients.select(3).eq(max_gradient)

    direction = grad1.multiply(1).add(grad2.multiply(2)).add(grad3.multiply(3)).add(grad4.multiply(4))

    # Kernels for 4 directions
    rect0 = ee.List.repeat(ee.List.repeat(0, 7), 7)
    rect1 = ee.List.repeat(ee.List.repeat(0, 7), 7)
    diag0 = ee.List.repeat(ee.List.repeat(0, 7), 7)
    diag1 = ee.List.repeat(ee.List.repeat(0, 7), 7)

    def _set(k, r, c):
        return ee.List(k).set(r, ee.List(ee.List(k).get(r)).set(c, 1))

    # vertical
    for r in range(7):
        rect0 = _set(rect0, r, 3)
    # horizontal
    for c in range(7):
        rect1 = _set(rect1, 3, c)
    # diag \
    for i in range(7):
        diag0 = _set(diag0, i, i)
    # diag /
    for i in range(7):
        diag1 = _set(diag1, i, 6 - i)

    k_rect0 = ee.Kernel.fixed(7, 7, rect0, 3, 3, False)
    k_rect1 = ee.Kernel.fixed(7, 7, rect1, 3, 3, False)
    k_diag0 = ee.Kernel.fixed(7, 7, diag0, 3, 3, False)
    k_diag1 = ee.Kernel.fixed(7, 7, diag1, 3, 3, False)

    mean_rect0 = img_lin.reduceNeighborhood(ee.Reducer.mean(), k_rect0)
    var_rect0 = img_lin.reduceNeighborhood(ee.Reducer.variance(), k_rect0)

    mean_rect1 = img_lin.reduceNeighborhood(ee.Reducer.mean(), k_rect1)
    var_rect1 = img_lin.reduceNeighborhood(ee.Reducer.variance(), k_rect1)

    mean_diag0 = img_lin.reduceNeighborhood(ee.Reducer.mean(), k_diag0)
    var_diag0 = img_lin.reduceNeighborhood(ee.Reducer.variance(), k_diag0)

    mean_diag1 = img_lin.reduceNeighborhood(ee.Reducer.mean(), k_diag1)
    var_diag1 = img_lin.reduceNeighborhood(ee.Reducer.variance(), k_diag1)

    mean_dir = (
        mean_rect0.updateMask(direction.eq(1))
        .blend(mean_rect1.updateMask(direction.eq(2)))
        .blend(mean_diag0.updateMask(direction.eq(3)))
        .blend(mean_diag1.updateMask(direction.eq(4)))
    )
    var_dir = (
        var_rect0.updateMask(direction.eq(1))
        .blend(var_rect1.updateMask(direction.eq(2)))
        .blend(var_diag0.updateMask(direction.eq(3)))
        .blend(var_diag1.updateMask(direction.eq(4)))
    )

    sigmaV = sample_var.reduce(ee.Reducer.median())
    b = var_dir.subtract(sigmaV).divide(var_dir).max(0)

    return mean_dir.add(b.multiply(img_lin.subtract(mean_dir)))


def refined_lee_db(img_db: ee.Image) -> ee.Image:
    """Convenience wrapper: input dB -> output dB."""
    return to_db(refined_lee(to_natural(img_db)))


# ---------- GEE datasets: permanent water + slope ----------

def build_permanent_water_bin(area: ee.Geometry) -> ee.Image:
    water_surface = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(area)
    seasonality = water_surface.select("seasonality")
    permanent = seasonality.gte(6)
    return permanent.unmask(0)


def build_slope(area: ee.Geometry) -> ee.Image:
    dem = ee.Image("WWF/HydroSHEDS/03VFDEM").clip(area)
    terrain = ee.Algorithms.Terrain(dem)
    return terrain.select("slope")


# Collections & flood detection 

def _date_window(event_date_str: str, days_before: int, days_after: int) -> tuple[str, str, str, str]:
    event_date = datetime.date.fromisoformat(event_date_str)
    before_start = (event_date - datetime.timedelta(days=days_before)).isoformat()
    before_end = (event_date - datetime.timedelta(days=1)).isoformat()
    after_start = (event_date + datetime.timedelta(days=1)).isoformat()
    after_end = (event_date + datetime.timedelta(days=days_after)).isoformat()
    return before_start, before_end, after_start, after_end


def _s1_base_collection(area: ee.Geometry, polarizations: list[str] | None = None) -> ee.ImageCollection:
    pols = polarizations or ["VV", "VH"]
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filterBounds(area)
        .select(pols)
    )


def _mosaic(coll: ee.ImageCollection, area: ee.Geometry) -> ee.Image:
    return coll.mosaic().clip(area)


def detect_flood_from_s1(
    area: ee.Geometry,
    event_date_str: str,
    days_before: int,
    days_after: int,
    flood_ratio_threshold: float = 1.35,
    max_slope: float = 5.0,
    min_area_m2: float = 800.0,
) -> FloodResult:
    """
    Returns flood masks (raw + filtered+10m) and vectors.
    This matches the logic you had in the notebook, but in a reusable function.
    """
    before_start, before_end, after_start, after_end = _date_window(event_date_str, days_before, days_after)

    coll = _s1_base_collection(area, ["VV", "VH"])

    before = coll.filterDate(before_start, before_end)
    after = coll.filterDate(after_start, after_end)

    before_desc = _mosaic(before.filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING")), area)
    after_desc = _mosaic(after.filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING")), area)

    before_asc = _mosaic(before.filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING")), area)
    after_asc = _mosaic(after.filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING")), area)

    # Refined Lee (per band)
    beforeD_vh = refined_lee_db(before_desc.select("VH")).rename("VH")
    afterD_vh = refined_lee_db(after_desc.select("VH")).rename("VH")

    beforeA_vh = refined_lee_db(before_asc.select("VH")).rename("VH")
    afterA_vh = refined_lee_db(after_asc.select("VH")).rename("VH")

    diffD = to_natural(beforeD_vh).divide(to_natural(afterD_vh)).rename("ratio")
    diffA = to_natural(beforeA_vh).divide(to_natural(afterA_vh)).rename("ratio")

    floodedD_raw = diffD.gt(flood_ratio_threshold).rename("water").selfMask()
    floodedA_raw = diffA.gt(flood_ratio_threshold).rename("water").selfMask()

    # Masks: permanent water + slope
    permanent_water_bin = build_permanent_water_bin(area)  # 1 = permanent
    non_permanent = permanent_water_bin.Not()

    slope = build_slope(area)
    low_slope = slope.lt(max_slope)

    combined_mask = non_permanent.And(low_slope)

    floodedD_masked = floodedD_raw.updateMask(combined_mask)
    floodedA_masked = floodedA_raw.updateMask(combined_mask)

    # Morphological cleaning
    k10 = ee.Kernel.circle(10, "meters")
    k15 = ee.Kernel.circle(15, "meters")
    k5  = ee.Kernel.circle(5,  "meters")

    baseD = floodedD_masked.unmask(0)
    floodedD_m = (
        baseD.focal_max(kernel=k10)
             .focal_min(kernel=k15)
             .focal_max(kernel=k5)
    )
    floodedD_morf = floodedD_m.updateMask(floodedD_m)

    baseA = floodedA_masked.unmask(0)
    floodedA_m = (
        baseA.focal_max(kernel=k10)
             .focal_min(kernel=k15)
             .focal_max(kernel=k5)
    )
    floodedA_morf = floodedA_m.updateMask(floodedA_m)

    # 10m resampling
    floodedD_10m = (
        floodedD_morf.setDefaultProjection("EPSG:4326", None, 10)
        .reduceResolution(ee.Reducer.max(), maxPixels=1024)
        .reproject(crs="EPSG:4326", scale=10)
    )
    floodedA_10m = (
        floodedA_morf.setDefaultProjection("EPSG:4326", None, 10)
        .reduceResolution(ee.Reducer.max(), maxPixels=1024)
        .reproject(crs="EPSG:4326", scale=10)
    )

    # Vectorization
    flood_vectorsD = floodedD_10m.reduceToVectors(
        geometry=area,
        scale=10,
        crs="EPSG:4326",
        geometryType="polygon",
        eightConnected=True,
        labelProperty="water",
        maxPixels=1e10,
    )

    flood_vectorsA = floodedA_10m.reduceToVectors(
        geometry=area,
        scale=10,
        crs="EPSG:4326",
        geometryType="polygon",
        eightConnected=True,
        labelProperty="water",
        maxPixels=1e10,
    )

    flood_vectorsD_area = flood_vectorsD.map(lambda f: f.set("area_m2", f.geometry().area(1)))
    flood_vectors_filteredD = flood_vectorsD_area.filter(ee.Filter.gte("area_m2", min_area_m2))

    return FloodResult(
        floodedD_raw=floodedD_raw,
        floodedA_raw=floodedA_raw,
        floodedD_10m=floodedD_10m,
        floodedA_10m=floodedA_10m,
        flood_vectorsD=flood_vectorsD,
        flood_vectorsA=flood_vectorsA,
        flood_vectors_filteredD=flood_vectors_filteredD,
        diffD=diffD,
        diffA=diffA,
        permanent_water_bin=permanent_water_bin,
        slope=slope,
    )