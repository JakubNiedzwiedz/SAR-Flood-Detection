from __future__ import annotations

from pathlib import Path

import os
import ee
import geemap
import geopandas as gpd
import osmnx as ox
from shapely.geometry import MultiLineString, LineString
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from .config import UserParams
from .ee_utils import ee_init


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


def load_communes_gdf(
    communes_path: str | Path,
    teryt_column: str,
) -> gpd.GeoDataFrame:
    communes = gpd.read_file(communes_path)

    if communes.crs is None:
        raise ValueError("Warstwa gmin nie ma zdefiniowanego CRS.")

    # Myślałem że mam gminy w 4326, ale jednak w 3857. Szybciej tu poprawić niż cały plik eksportować na nowo
    communes = communes.set_crs(epsg=3857, allow_override=True)
    communes = communes.to_crs(epsg=4326)

    if teryt_column not in communes.columns:
        raise ValueError(
            f"Nie znaleziono kolumny '{teryt_column}'. "
            f"Dostępne kolumny: {list(communes.columns)}"
        )
    
    os.environ["SHAPE_RESTORE_SHX"] = "YES"

    communes = communes[[teryt_column, "geometry"]].copy()
    communes = communes.rename(columns={teryt_column: "commune_id"})

    # TERYT musi zostać tekstem bo inaczej sie wysypie i utnie zera
    communes["commune_id"] = communes["commune_id"].astype(str).str.strip()

    return communes


def normalize_value(value):
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return value


def parse_osm_id(value) -> int | None:
    """
    OSMnx potrafi zwrócić pojedyncze ID albo listę ID.
    Do kolumny osm_id bierzemy pierwsze ID jako bigint.
    """
    if value is None:
        return None

    if isinstance(value, list):
        if not value:
            return None
        value = value[0]

    try:
        return int(value)
    except (TypeError, ValueError):
        # Gdyby trafił się string typu "123,456", bierzemy pierwszy element
        try:
            return int(str(value).split(",")[0])
        except (TypeError, ValueError):
            return None


def parse_osm_type(value) -> int:
    """
    Proste mapowanie typu OSM do bigint.
    1 = way
    2 = node
    3 = relation
    """
    mapping = {
        "way": 1,
        "node": 2,
        "relation": 3,
    }
    return mapping.get(str(value).lower(), 1)


def parse_oneway(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False

    value_str = str(value).strip().lower()
    return value_str in {"true", "yes", "1", "-1"}


def map_highway_to_type_id(highway_value) -> int | None:
    """
    Tymczasowe mapowanie highway -> highway_type_id.
    Docelowo najlepiej oprzeć to o osobną tabelę słownikową.
    """
    if highway_value is None:
        return None

    if isinstance(highway_value, list):
        highway_value = highway_value[0] if highway_value else None

    if highway_value is None:
        return None

    highway = str(highway_value).strip().lower()

    mapping = {
        "motorway": 1,
        "trunk": 2,
        "primary": 3,
        "secondary": 4,
        "tertiary": 5,
        "unclassified": 6,
        "residential": 7,
        "service": 8,
        "living_street": 9,
        "motorway_link": 10,
        "trunk_link": 11,
        "primary_link": 12,
        "secondary_link": 13,
        "tertiary_link": 14,
    }

    return mapping.get(highway)


def ensure_multilinestring(geom):
    if geom is None:
        return None
    if isinstance(geom, MultiLineString):
        return geom
    if isinstance(geom, LineString):
        return MultiLineString([geom])
    return geom


def assign_commune_id_by_spatial_join(
    roads_gdf: gpd.GeoDataFrame,
    communes_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    roads = roads_gdf.copy()

    join_points = roads.copy()
    join_points["join_geom"] = join_points.geometry.representative_point()
    join_points = join_points.set_geometry("join_geom")

    joined = gpd.sjoin(
        join_points,
        communes_gdf[["commune_id", "geometry"]],
        how="left",
        predicate="within",
    )

    roads["commune_id"] = joined["commune_id"].values
    return roads


def prepare_roads_for_db(
    roads: gpd.GeoDataFrame,
    communes_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    roads = roads.copy()

    wanted_columns = ["osmid", "highway", "oneway", "geometry"]
    for col in wanted_columns:
        if col not in roads.columns:
            roads[col] = None

    roads["osm_id"] = roads["osmid"].apply(parse_osm_id)
    roads["osm_type"] = 1  # way
    roads["highway_type_id"] = roads["highway"].apply(map_highway_to_type_id)
    roads["oneway"] = roads["oneway"].apply(parse_oneway)
    roads["is_active"] = True
    roads["geom"] = roads["geometry"].apply(ensure_multilinestring)

    roads = gpd.GeoDataFrame(roads, geometry="geom", crs="EPSG:4326")

    roads = assign_commune_id_by_spatial_join(roads, communes_gdf)

    # długość w metrach jak coś wiec trzeba w 2180
    roads_metric = roads.to_crs(epsg=2180)
    roads["length_m"] = roads_metric.geometry.length

    roads["commune_id"] = roads["commune_id"].astype(str).str.strip()

    roads_db = roads[
        [
            "osm_id",
            "osm_type",
            "highway_type_id",
            "commune_id",
            "oneway",
            "length_m",
            "geom",
            "is_active",
        ]
    ].copy()

    roads_db["imported_at"] = None
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
    db_url,
    table_name: str = "roads_osm",
    schema: str = "public",
) -> None:
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

    DB_HOST = "localhost"
    DB_PORT = 5432
    DB_NAME = "Drogi_OSM"
    DB_USER = "drogi_user"
    DB_PASSWORD = "drogi"

    # SHP z gmianmi jednak w 3857 a nie w 4326
    COMMUNES_PATH = r"data\admin\Gminy_polski.shp"
    TERYT_COLUMN = "ID"

    ee_init(params.gee_project)

    area = get_area_geometry(params)
    area_gdf = ee_geometry_to_gdf(area)

    communes_gdf = load_communes_gdf(
        communes_path=COMMUNES_PATH,
        teryt_column=TERYT_COLUMN,
    )

    roads_raw = fetch_osm_roads_for_area(
        area_gdf=area_gdf,
        network_type="drive_service",
        simplify=True,
    )
    # ===== DEBUG =====

    print("\n===== DEBUG CRS =====")
    print("ROADS CRS:", roads_raw.crs)
    print("COMMUNES CRS:", communes_gdf.crs)

    print("\n===== DEBUG BOUNDS =====")
    print("ROADS BOUNDS:", roads_raw.total_bounds)
    print("COMMUNES BOUNDS:", communes_gdf.total_bounds)

    print("\n===== DEBUG COUNT =====")
    print("ROADS COUNT:", len(roads_raw))
    print("COMMUNES COUNT:", len(communes_gdf))

    print("\n===== DEBUG COMMUNES SAMPLE =====")
    print(communes_gdf[["commune_id"]].head())

    # test spatial join
    test_points = roads_raw.copy()
    test_points["geometry"] = test_points.geometry.representative_point()
    test_points = test_points.set_geometry("geometry")

    joined_test = gpd.sjoin(
        test_points,
        communes_gdf[["commune_id", "geometry"]],
        how="left",
        predicate="intersects",
    )

    print("\n===== DEBUG JOIN RESULT =====")
    print("JOINED NON-NULL:", joined_test["commune_id"].notna().sum())
    print(joined_test[["commune_id"]].head())

    # ===== END DEBUG =====

    roads_db = prepare_roads_for_db(
        roads=roads_raw,
        communes_gdf=communes_gdf
    )

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

    print("OK ✅ Zaimportowano drogi OSM do PostGIS.")
    print(f"Liczba rekordów: {len(roads_db)}")
    print("Przykładowe commune_id:", roads_db["commune_id"].dropna().head().tolist())


if __name__ == "__main__":
    import_nysa_roads_to_db()