from dataclasses import dataclass
from pathlib import Path


@dataclass
class UserParams:
    center_lat: float = 50.4700
    center_lon: float = 17.3340
    # koło wpisane w kwadrat - kwadrat to zasieg importu osm
    radius_km: float = 4.0

    event_date_str: str = "2024-09-13"
    days_before: int = 12
    days_after: int = 6

    flood_ratio_threshold: float = 1.05

    gee_project: str = "ee-kubek114"
    nysa_asset: str = "projects/ee-kubek114/assets/Nysa_gmina"

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "Drogi_OSM"
    db_user: str = "drogi_user"
    db_password: str = "drogi"

    db_schema: str = "public"
    db_roads_table: str = "roads_osm"


@dataclass
class Paths:
    project_root: Path = Path(__file__).resolve().parents[1]
    outputs_dir: Path = None  # ustawimy w __post_init__

    def __post_init__(self):
        if self.outputs_dir is None:
            self.outputs_dir = self.project_root / "outputs" / "1-05"

    def ensure(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)