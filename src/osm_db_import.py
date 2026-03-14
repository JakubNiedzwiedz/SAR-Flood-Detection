from __future__ import annotations

import geopandas as gpd
import osmnx as ox
import ee

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from .config import UserParams 
from .ee_utils import ee_init
import geemap


def get_area_geometry(params: UserParams) -> ee.Geometry:
    fc = ee.FeatureCollection(params.nysa_asset)
    return fc.geometry()


def ee_geometry_to_gdf(area: ee.Geometry) -> gpd.GeoDataFrame:
    area_fc = ee.FeatureCollection([ee.Feature(area)])
    gdf = geemap.ee_to_gdf(area_fc)
    return gdf.set_crs(epsg=4326, allow_override=True)


def fetch_osm_roads_for_area(
    area_gdf: gpd.GeoDataFrame,
    network_type: str = "drive_service",
    simplify: bool = True,
) -> gpd.GeoDataFrame:
    """
    Pobiera drogi OSM dla bbox obszaru.
    Na start używamy tej samej logiki co w osm_processing.py.
    """
    minx, miny, maxx, maxy = area_gdf.total_bounds
    bbox = (maxy, miny, maxx, minx)  # north, south, east, west

    G = ox.graph_from_bbox(
        bbox=bbox,
        network_type=network_type,
        simplify=simplify,
    )

    roads = ox.graph_to_gdfs(G, nodes=False, edges=True)
    roads = roads.to_crs(epsg=4326)

    return roads


def prepare_roads_for_db(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Dopasowuje dane OSMnx do schematu tabeli roads_osm.
    """
    roads = roads.copy()

    wanted_columns = [
        "osmid",
        "highway",
        "name",
        "ref",
        "oneway",
        "bridge",
        "tunnel",
        "surface",
        "geometry",
    ]

    for col in wanted_columns:
        if col not in roads.columns:
            roads[col] = None

    # osmid / highway czasem bywają listą
    def normalize_value(value):
        if isinstance(value, list):
            return ",".join(str(v) for v in value)
        return value

    for col in ["osmid", "highway", "name", "ref", "oneway", "bridge", "tunnel", "surface"]:
        roads[col] = roads[col].apply(normalize_value)

    # geometria do MULTILINESTRING / LINESTRING zostaje w kolumnie geometry
    # nazwy kolumn zgodne z tabelą SQL
    roads_db = roads.rename(
        columns={
            "osmid": "osm_id",
            "geometry": "geom",
        }
    ).copy()

    roads_db["osm_type"] = "way"

    # tylko kolumny, które masz w tabeli
    roads_db = roads_db[
        [
            "osm_id",
            "osm_type",
            "highway",
            "name",
            "ref",
            "oneway",
            "bridge",
            "tunnel",
            "surface",
            "geom",
        ]
    ]

    roads_db = gpd.GeoDataFrame(roads_db, geometry="geom", crs="EPSG:4326")

    return roads_db


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


def full_reimport_roads_to_postgis(
    roads_gdf: gpd.GeoDataFrame,
    db_url: str,
    table_name: str = "roads_osm",
    schema: str = "public",
) -> None:
    """
    Pełny reimport:
    1. czyści tabelę
    2. wrzuca nowe dane
    """
    engine = create_engine(db_url)

    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {schema}.{table_name};"))

    roads_gdf.to_postgis(
        name=table_name,
        con=engine,
        schema=schema,
        if_exists="append",
        index=False,
    )


def import_nysa_roads_to_db() -> None:
    params = UserParams()

    # Uzupełnij swoimi danymi dostępowymi
    DB_HOST = "localhost"
    DB_PORT = 5432
    DB_NAME = "Drogi_OSM"
    DB_USER = "drogi_user"
    DB_PASSWORD = "drogi"

    ee_init(params.gee_project)

    area = get_area_geometry(params)
    area_gdf = ee_geometry_to_gdf(area)

    roads_raw = fetch_osm_roads_for_area(
        area_gdf=area_gdf,
        network_type="drive_service",
        simplify=True,
    )

    roads_db = prepare_roads_for_db(roads_raw)

    db_url = build_db_url(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )

    full_reimport_roads_to_postgis(
        roads_gdf=roads_db,
        db_url=db_url,
        table_name="roads_osm",
        schema="public",
    )

    print("OK ✅ Zaimportowano drogi OSM dla gminy Nysa do PostGIS.")
    print(f"Liczba rekordów: {len(roads_db)}")


if __name__ == "__main__":
    import_nysa_roads_to_db()