from __future__ import annotations

from dataclasses import dataclass

import ee
import geemap
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, GeometryCollection
from sqlalchemy import create_engine
from sqlalchemy.engine import URL


@dataclass
class OSMFloodRoadsResult:
    roads_all: gpd.GeoDataFrame
    roads_in_flood: gpd.GeoDataFrame
    roads_outside_flood: gpd.GeoDataFrame
    intersection_points: gpd.GeoDataFrame
    buffers_5m: gpd.GeoDataFrame

    flooded_length_m: float
    dry_length_m: float


def build_db_url(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
):
    return URL.create(
        drivername="postgresql+psycopg2",
        username=user,
        password=password,
        host=host,
        port=port,
        database=database,
    )


def _ee_geom_to_gdf(area: ee.Geometry) -> gpd.GeoDataFrame:
    """Convert ee.Geometry to GeoDataFrame (EPSG:4326)."""
    area_fc = ee.FeatureCollection([ee.Feature(area)])
    gdf = geemap.ee_to_gdf(area_fc)
    return gdf.set_crs(epsg=4326, allow_override=True)


def load_roads_from_postgis_for_area(
    area: ee.Geometry,
    db_url,
    *,
    schema: str = "public",
    table: str = "roads_osm",
    only_active: bool = True,
) -> gpd.GeoDataFrame:
    """
    Wczytuje drogi z PostGIS tylko dla obszaru AOI.
    Zakłada geometrię w kolumnie 'geom' w EPSG:4326.
    """
    area_gdf = _ee_geom_to_gdf(area)
    area_union = area_gdf.unary_union
    area_wkt = area_union.wkt

    active_filter = "AND is_active = TRUE" if only_active else ""

    sql = f"""
        SELECT
            road_id,
            osm_id,
            oneway,
            geom,
            highway_type_id,
            commune_id,
            length_m,
            is_active
        FROM {schema}.{table}
        WHERE ST_Intersects(
            geom,
            ST_GeomFromText(%(area_wkt)s, 4326)
        )
        {active_filter}
    """

    engine = create_engine(db_url)

    roads = gpd.read_postgis(
        sql=sql,
        con=engine,
        geom_col="geom",
        params={"area_wkt": area_wkt},
    )

    if roads.empty:
        return gpd.GeoDataFrame(
            columns=[
                "road_id",
                "osm_id",
                "oneway",
                "geom",
                "highway_type_id",
                "commune_id",
                "length_m",
                "is_active",
            ],
            geometry="geom",
            crs="EPSG:4326",
        )

    roads = roads.set_geometry("geom")
    if roads.crs is None:
        roads = roads.set_crs(epsg=4326, allow_override=True)

    return roads


def _intersection_points_with_boundary(
    roads_proj: gpd.GeoDataFrame,
    flood_polygons_proj: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Finds intersection points between road geometries and the boundary of flood polygons.
    Returns GDF in same CRS as inputs (projected).
    """
    boundaries = flood_polygons_proj.copy()
    boundaries["geometry"] = boundaries.geometry.boundary
    boundary_union = boundaries.unary_union

    geom_col = roads_proj.geometry.name

    points_list = []
    for _, row in roads_proj.iterrows():
        road_geom = row[geom_col]

        if road_geom is None or road_geom.is_empty:
            continue

        inter = road_geom.intersection(boundary_union)
        if inter.is_empty:
            continue

        road_id = row.get("road_id")

        if isinstance(inter, Point):
            points_list.append({"road_id": road_id, "geometry": inter})

        elif isinstance(inter, MultiPoint):
            for pt in inter.geoms:
                points_list.append({"road_id": road_id, "geometry": pt})

        elif isinstance(inter, GeometryCollection):
            for geom_part in inter.geoms:
                if isinstance(geom_part, Point):
                    points_list.append({"road_id": road_id, "geometry": geom_part})

    if not points_list:
        return gpd.GeoDataFrame(
            columns=["road_id", "geometry"],
            geometry="geometry",
            crs=roads_proj.crs,
        )

    return gpd.GeoDataFrame(points_list, geometry="geometry", crs=roads_proj.crs)


def analyze_osm_roads_flood_intersections(
    area: ee.Geometry,
    flood_vectors: ee.FeatureCollection,
    *,
    db_url,
    roads_schema: str = "public",
    roads_table: str = "roads_osm",
    roads_crs_projected: str = "EPSG:3857",
    buffer_m: float = 5.0,
    only_active: bool = True,
) -> OSMFloodRoadsResult:
    """
    Wczytuje drogi z PostGIS dla AOI, przecina z flood polygons,
    zwraca zalane i niezalane odcinki + intersection points + bufory.
    """
    roads_all = load_roads_from_postgis_for_area(
        area=area,
        db_url=db_url,
        schema=roads_schema,
        table=roads_table,
        only_active=only_active,
    )

    flood_gdf = geemap.ee_to_gdf(flood_vectors)
    flood_gdf = flood_gdf.set_crs(epsg=4326, allow_override=True)

    roads_proj = roads_all.to_crs(roads_crs_projected)
    flood_proj = flood_gdf.to_crs(roads_crs_projected)

    # Gdy brak dróg w AOI
    if roads_proj.empty:
        empty_roads = gpd.GeoDataFrame(
            columns=list(roads_proj.columns),
            geometry=roads_proj.geometry.name if roads_proj.geometry.name else "geom",
            crs=roads_proj.crs,
        )
        pts = gpd.GeoDataFrame(columns=["road_id", "geometry"], geometry="geometry", crs=roads_crs_projected)
        bufs = gpd.GeoDataFrame(columns=["road_id", "geometry"], geometry="geometry", crs=roads_crs_projected)

        return OSMFloodRoadsResult(
            roads_all=roads_proj,
            roads_in_flood=empty_roads,
            roads_outside_flood=empty_roads.copy(),
            intersection_points=pts,
            buffers_5m=bufs,
            flooded_length_m=0.0,
            dry_length_m=0.0,
        )

    # Gdy brak poligonów powodzi
    if flood_proj.empty:
        empty = gpd.GeoDataFrame(columns=list(roads_proj.columns), geometry=roads_proj.geometry.name, crs=roads_proj.crs)
        pts = gpd.GeoDataFrame(columns=["road_id", "geometry"], geometry="geometry", crs=roads_proj.crs)
        bufs = gpd.GeoDataFrame(columns=["road_id", "geometry"], geometry="geometry", crs=roads_proj.crs)

        roads_outside_flood = roads_proj.copy()
        roads_outside_flood["length_m"] = roads_outside_flood.geometry.length

        return OSMFloodRoadsResult(
            roads_all=roads_proj,
            roads_in_flood=empty,
            roads_outside_flood=roads_outside_flood,
            intersection_points=pts,
            buffers_5m=bufs,
            flooded_length_m=0.0,
            dry_length_m=float(roads_outside_flood["length_m"].sum()),
        )

    # Overlay
    roads_in_flood = gpd.overlay(roads_proj, flood_proj, how="intersection")
    roads_outside_flood = gpd.overlay(roads_proj, flood_proj, how="difference")

    roads_in_flood = roads_in_flood.copy()
    roads_outside_flood = roads_outside_flood.copy()

    roads_in_flood["length_m"] = roads_in_flood.geometry.length
    roads_outside_flood["length_m"] = roads_outside_flood.geometry.length

    flooded_length_m = float(roads_in_flood["length_m"].sum()) if len(roads_in_flood) else 0.0
    dry_length_m = float(roads_outside_flood["length_m"].sum()) if len(roads_outside_flood) else 0.0

    # Punkty przecięć z granicą powodzi + bufory
    points_gdf = _intersection_points_with_boundary(roads_proj, flood_proj)
    buffers_gdf = points_gdf.copy()
    if len(buffers_gdf):
        buffers_gdf["geometry"] = buffers_gdf.buffer(buffer_m)

    return OSMFloodRoadsResult(
        roads_all=roads_proj,
        roads_in_flood=roads_in_flood,
        roads_outside_flood=roads_outside_flood,
        intersection_points=points_gdf,
        buffers_5m=buffers_gdf,
        flooded_length_m=flooded_length_m,
        dry_length_m=dry_length_m,
    )