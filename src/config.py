from dataclasses import dataclass
from pathlib import Path

@dataclass
class UserParams:
    center_lat: float = 50.4700
    center_lon: float = 17.3340
    radius_km: float = 1.0

    event_date_str: str = "2024-09-13"
    days_before: int = 12
    days_after: int = 6

    gee_project: str = "ee-kubek114"
    nysa_asset: str = "projects/ee-kubek114/assets/Nysa_gmina"

@dataclass
class Paths:
    outputs_dir: Path = Path("outputs")

    def ensure(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

project_root: Path = Path(__file__).resolve().parents[1]
outputs_dir: Path = project_root / "outputs"