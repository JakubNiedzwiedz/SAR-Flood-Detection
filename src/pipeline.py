from __future__ import annotations

from dataclasses import dataclass

from .config import UserParams, Paths
from .ee_utils import ee_init
from .gee_processing import detect_flood_from_s1
from .osm_processing import analyze_osm_roads_flood_intersections, build_db_url
from .export_layers import export_gdf, export_ee_fc_as_shp, export_permanent_water_shp

import ee


@dataclass
class PipelineResult:
    flood: object
    osm: object
    summary: dict


def get_area_geometry(params: UserParams) -> ee.Geometry:
    fc = ee.FeatureCollection(params.nysa_asset)
    return fc.geometry()

def build_result_summary(
    area: ee.Geometry,
    flood,
    osm,
) -> dict:
    """
    Liczy podstawowe metryki wyniku:
    - powierzchnia zalania,
    - procent powierzchni AOI,
    - długość dróg zalanych i niezalanych,
    - liczba barrier points.
    """

    # Powierzchnia całego AOI / gminy
    area_m2 = area.area(1).getInfo()
    area_ha = area_m2 / 10_000

    # Powierzchnia zalania z finalnej warstwy wektorowej.
    # flood_vectors_filteredD ma już property area_m2 dodane w gee_processing.py.
    flooded_area_m2 = flood.flood_vectors_filteredD.aggregate_sum("area_m2").getInfo()

    if flooded_area_m2 is None:
        flooded_area_m2 = 0.0

    flooded_area_ha = flooded_area_m2 / 10_000

    if area_m2 > 0:
        flooded_area_percent = (flooded_area_m2 / area_m2) * 100
    else:
        flooded_area_percent = 0.0

    flooded_roads_km = osm.flooded_length_m / 1000
    dry_roads_km = osm.dry_length_m / 1000
    total_roads_km = flooded_roads_km + dry_roads_km

    barrier_points_count = len(osm.intersection_points)

    return {
        "area_ha": round(area_ha, 2),
        "flooded_area_ha": round(flooded_area_ha, 2),
        "flooded_area_percent": round(flooded_area_percent, 3),
        "flooded_roads_km": round(flooded_roads_km, 3),
        "dry_roads_km": round(dry_roads_km, 3),
        "total_roads_km": round(total_roads_km, 3),
        "barrier_points_count": int(barrier_points_count),
    }

def run_pipeline(params: UserParams, paths: Paths) -> PipelineResult:
    """
    Produkcyjny pipeline:
    1) init GEE
    2) AOI
    3) flood detection (S1)
    4) roads analysis z bazy PostGIS
    5) export shapefiles
    """
    paths.ensure()
    ee_init(params.gee_project)

    area = get_area_geometry(params)

    # 1) Flood detection
    flood = detect_flood_from_s1(
        area=area,
        event_date_str=params.event_date_str,
        days_before=params.days_before,
        days_after=params.days_after,
        flood_ratio_threshold=params.flood_ratio_threshold,
        max_slope=5,
        min_area_m2=800,
    )

    # 2) DB connection
    db_url = build_db_url(
        host=params.db_host,
        port=params.db_port,
        database=params.db_name,
        user=params.db_user,
        password=params.db_password,
    )

    # 3) Roads analysis from PostGIS
    osm = analyze_osm_roads_flood_intersections(
        area=area,
        flood_vectors=flood.flood_vectors_filteredD,
        db_url=db_url,
        roads_schema=params.db_schema,
        roads_table=params.db_roads_table,
        roads_crs_projected="EPSG:3857",
        buffer_m=5.0,
        only_active=True,
    )

    # 4) Export outputs
    # export_gdf(osm.intersection_points, paths.outputs_dir, "intersection_points.shp")
    # export_gdf(osm.buffers_5m, paths.outputs_dir, "buffers_5m.shp")
    export_gdf(osm.roads_in_flood, paths.outputs_dir, "drogi_zalane.shp")
    export_gdf(osm.roads_outside_flood, paths.outputs_dir, "drogi_niezalane.shp")

    export_ee_fc_as_shp(flood.flood_vectors_filteredD, paths.outputs_dir, "zalane_sar.shp")
    #export_permanent_water_shp(flood.permanent_water_bin, area, paths.outputs_dir, "permanent_water.shp")

    summary = build_result_summary(
        area=area,
        flood=flood,
        osm=osm,
    )

    return PipelineResult(
        flood=flood,
        osm=osm,
        summary=summary,
    )