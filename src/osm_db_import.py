from __future__ import annotations

from pathlib import Path

import os
import ee
import geemap
import geopandas as gpd
import osmnx as ox
import pandas as pd
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
    os.environ["SHAPE_RESTORE_SHX"] = "YES"

    communes = gpd.read_file(communes_path)

    if communes.crs is None:
        raise ValueError("Warstwa gmin nie ma zdefiniowanego CRS.")

    # U Ciebie SHP jest faktycznie w 3857, więc nadpisujemy i konwertujemy do 4326
    communes = communes.set_crs(epsg=3857, allow_override=True)
    communes = communes.to_crs(epsg=4326)

    if teryt_column not in communes.columns:
        raise ValueError(
            f"Nie znaleziono kolumny '{teryt_column}'. "
            f"Dostępne kolumny: {list(communes.columns)}"
        )

    communes = communes[[teryt_column, "geometry"]].copy()
    communes = communes.rename(columns={teryt_column: "commune_id"})

    # TERYT musi zostać tekstem, żeby nie ucinało zer z przodu
    communes["commune_id"] = communes["commune_id"].astype(str).str.strip()

    return communes


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
        try:
            return int(str(value).split(",")[0])
        except (TypeError, ValueError):
            return None


def parse_oneway(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False

    value_str = str(value).strip().lower()
    return value_str in {"true", "yes", "1", "-1"}


def normalize_highway_value(highway_value) -> str | None:
    if highway_value is None:
        return None

    if isinstance(highway_value, list):
        highway_value = highway_value[0] if highway_value else None

    if highway_value is None:
        return None

    return str(highway_value).strip().lower()


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


def load_highway_type_mapping(db_url) -> dict[str, int]:
    """
    Pobiera mapowanie osm_highway_tag -> highway_type_id bezpośrednio z bazy.
    Dzięki temu nie ma ryzyka, że hardcoded ID w Pythonie rozjadą się z tabelą highway_type.
    """
    engine = create_engine(db_url)

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT highway_type_id, osm_highway_tag
                FROM public.highway_type
                """
            )
        ).fetchall()

    mapping: dict[str, int] = {}
    for row in rows:
        highway_tag = str(row.osm_highway_tag).strip().lower()
        mapping[highway_tag] = int(row.highway_type_id)

    return mapping


def generate_fallback_road_ids(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    road_id jest PRIMARY KEY i nie może być NULL ani się powtarzać.
    Domyślnie bierzemy osm_id. Jeśli osm_id brak albo są duplikaty,
    nadajemy bezpieczne fallbacki.
    """
    roads = roads.copy()

    roads["road_id"] = roads["osm_id"]

    missing_mask = roads["road_id"].isna()
    if missing_mask.any():
        fallback_start = 10_000_000_000
        fallback_ids = range(fallback_start, fallback_start + int(missing_mask.sum()))
        roads.loc[missing_mask, "road_id"] = list(fallback_ids)

    duplicated_mask = roads["road_id"].duplicated(keep=False)
    if duplicated_mask.any():
        duplicate_indices = roads[duplicated_mask].index.tolist()
        dedup_start = 20_000_000_000

        for i, idx in enumerate(duplicate_indices):
            roads.at[idx, "road_id"] = dedup_start + i

    roads["road_id"] = roads["road_id"].astype("int64")

    return roads


def prepare_roads_for_db(
    roads: gpd.GeoDataFrame,
    communes_gdf: gpd.GeoDataFrame,
    highway_mapping: dict[str, int],
) -> gpd.GeoDataFrame:
    roads = roads.copy()

    wanted_columns = ["osmid", "highway", "oneway", "geometry"]
    for col in wanted_columns:
        if col not in roads.columns:
            roads[col] = None

    roads["osm_id"] = roads["osmid"].apply(parse_osm_id)
    roads["highway_tag"] = roads["highway"].apply(normalize_highway_value)
    roads["highway_type_id"] = roads["highway_tag"].map(highway_mapping)
    roads["oneway"] = roads["oneway"].apply(parse_oneway)
    roads["is_active"] = True
    roads["geom"] = roads["geometry"].apply(ensure_multilinestring)

    roads = gpd.GeoDataFrame(roads, geometry="geom", crs="EPSG:4326")

    roads = assign_commune_id_by_spatial_join(roads, communes_gdf)

    # Długość liczymy w metrach, więc przejście do EPSG:2180
    roads_metric = roads.to_crs(epsg=2180)
    roads["length_m"] = roads_metric.geometry.length

    roads["commune_id"] = roads["commune_id"].where(
        roads["commune_id"].notna(),
        None,
    )

    if roads["commune_id"].notna().any():
        roads.loc[roads["commune_id"].notna(), "commune_id"] = (
            roads.loc[roads["commune_id"].notna(), "commune_id"]
            .astype(str)
            .str.strip()
        )

    roads = generate_fallback_road_ids(roads)

    roads["imported_at"] = pd.Timestamp.now()

    roads_db = roads[
        [
            "road_id",
            "osm_id",
            "oneway",
            "imported_at",
            "geom",
            "highway_type_id",
            "commune_id",
            "length_m",
            "is_active",
        ]
    ].copy()

    roads_db = gpd.GeoDataFrame(roads_db, geometry="geom", crs="EPSG:4326")

    return roads_db


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

    COMMUNES_PATH = r"data\admin\Gminy_polski.shp"
    TERYT_COLUMN = "ID"

    ee_init(params.gee_project)

    area = get_area_geometry(params)
    area_gdf = ee_geometry_to_gdf(area)

    communes_gdf = load_communes_gdf(
        communes_path=COMMUNES_PATH,
        teryt_column=TERYT_COLUMN,
    )

    db_url = build_db_url(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )

    highway_mapping = load_highway_type_mapping(db_url)

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

    print("\n===== DEBUG HIGHWAY TYPE MAPPING =====")
    print(highway_mapping)

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
        communes_gdf=communes_gdf,
        highway_mapping=highway_mapping,
    )

    print("\n===== DEBUG PREPARED ROADS =====")
    print(roads_db.head())
    print("\nBRAKUJĄCE highway_type_id:", roads_db["highway_type_id"].isna().sum())
    print("BRAKUJĄCE commune_id:", roads_db["commune_id"].isna().sum())
    print("DUPLIKATY road_id:", roads_db["road_id"].duplicated().sum())

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