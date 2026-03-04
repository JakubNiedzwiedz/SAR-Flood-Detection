from pathlib import Path

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def shp_path(output_dir: Path, name: str) -> Path:
    if not name.lower().endswith(".shp"):
        name += ".shp"
    return output_dir / name