from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
import osmnx as ox
from shapely import simplify
from shapely.geometry import Point, MultiPoint, GeometryCollection
import ee
import geemap


@dataclass
class OSMFloodRoadsResult:
    roads_all: gpd.GeoDataFrame
    roads_in_flood: gpd.GeoDataFrame
    roads_outside_flood: gpd.GeoDataFrame
    intersection_points: gpd.GeoDataFrame
    buffers_5m: gpd.GeoDataFrame

    flooded_length_m: float
    dry_length_m: float


def _ee_geom_to_gdf(area: ee.Geometry) -> gpd.GeoDataFrame:
    """Convert ee.Geometry to GeoDataFrame (EPSG:4326)."""
    area_fc = ee.FeatureCollection([ee.Feature(area)])
    gdf = geemap.ee_to_gdf(area_fc)
    return gdf.set_crs(epsg=4326, allow_override=True)


def _bbox_from_gdf_wgs84(gdf_wgs84: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = gdf_wgs84.total_bounds
    return (minx, miny, maxx, maxy)


def _intersection_points_with_boundary(
    roads_proj: gpd.GeoDataFrame,
    flood_polygons_proj: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Finds intersection points between road geometries and the boundary of flood polygons.
    Returns GDF in same CRS as inputs (projected).
    """
    # boundary union of all flood polygons
    boundaries = flood_polygons_proj.copy()
    boundaries["geometry"] = boundaries.geometry.boundary
    boundary_union = boundaries.unary_union

    points_list = []
    for ridx, row in roads_proj.iterrows():
        inter = row.geometry.intersection(boundary_union)
        if inter.is_empty:
            continue

        if isinstance(inter, Point):
            points_list.append({"road_id": ridx, "geometry": inter})

        elif isinstance(inter, MultiPoint):
            for pt in inter.geoms:
                points_list.append({"road_id": ridx, "geometry": pt})

        elif isinstance(inter, GeometryCollection):
            for geom_part in inter.geoms:
                if isinstance(geom_part, Point):
                    points_list.append({"road_id": ridx, "geometry": geom_part})

    if not points_list:
        return gpd.GeoDataFrame(columns=["road_id", "geometry"], geometry="geometry", crs=roads_proj.crs)

    return gpd.GeoDataFrame(points_list, geometry="geometry", crs=roads_proj.crs)


def analyze_osm_roads_flood_intersections(
    area: ee.Geometry,
    flood_vectors: ee.FeatureCollection,
    *,
    network_type: str = "drive_service",
    roads_crs_projected: str = "EPSG:3857",
    buffer_m: float = 5.0,
    simplify: bool = True,
) -> OSMFloodRoadsResult:
    """
    Downloads OSM road network in AOI bbox, overlays with flood polygons,
    returns flooded & non-flooded road parts + intersection points & buffers.

    area: ee.Geometry AOI
    flood_vectors: ee.FeatureCollection polygons of flooded area
    """
    # AOI -> bbox
    area_gdf = _ee_geom_to_gdf(area)  # EPSG:4326
    minx, miny, maxx, maxy = _bbox_from_gdf_wgs84(area_gdf)

    # OSMnx bbox: prefer west, south, east, north (older/newer versions differ),
    # but graph_from_bbox currently accepts named args in most releases.
    bbox = (maxy, miny, maxx, minx)  # north, south, east, west
    G = ox.graph_from_bbox(bbox=bbox, network_type=network_type, simplify=simplify)
    roads_all = ox.graph_to_gdfs(G, nodes=False, edges=True)

    flood_gdf = geemap.ee_to_gdf(flood_vectors)
    flood_gdf = flood_gdf.set_crs(epsg=4326, allow_override=True)

    roads_proj = roads_all.to_crs(roads_crs_projected)
    flood_proj = flood_gdf.to_crs(roads_crs_projected)

    # Checking if empty
    if flood_proj.empty:
        empty = gpd.GeoDataFrame(columns=list(roads_proj.columns), geometry="geometry", crs=roads_proj.crs)
        pts = gpd.GeoDataFrame(columns=["road_id", "geometry"], geometry="geometry", crs=roads_proj.crs)
        bufs = gpd.GeoDataFrame(columns=["road_id", "geometry"], geometry="geometry", crs=roads_proj.crs)
        return OSMFloodRoadsResult(
            roads_all=roads_proj,
            roads_in_flood=empty,
            roads_outside_flood=roads_proj.copy(),
            intersection_points=pts,
            buffers_5m=bufs,
            flooded_length_m=0.0,
            dry_length_m=float(roads_proj.geometry.length.sum()),
        )

    # Roads flooded not flooded
    roads_in_flood = gpd.overlay(roads_proj, flood_proj, how="intersection")
    roads_outside_flood = gpd.overlay(roads_proj, flood_proj, how="difference")

    # Stats
    roads_in_flood = roads_in_flood.copy()
    roads_outside_flood = roads_outside_flood.copy()

    roads_in_flood["length_m"] = roads_in_flood.geometry.length
    roads_outside_flood["length_m"] = roads_outside_flood.geometry.length

    flooded_length_m = float(roads_in_flood["length_m"].sum()) if len(roads_in_flood) else 0.0
    dry_length_m = float(roads_outside_flood["length_m"].sum()) if len(roads_outside_flood) else 0.0

    # Intersection points with boundary + buffers
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