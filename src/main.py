from .config import UserParams, Paths
from .pipeline import run_pipeline

def main() -> None:
    params = UserParams()
    paths = Paths()

    res = run_pipeline(params, paths)
    print("OK ✅ Pipeline done.")
    print("Outputs:", paths.outputs_dir.resolve())
    print("Flooded roads (km):", res.osm.flooded_length_m / 1000)
    print("Dry roads (km):", res.osm.dry_length_m / 1000)

if __name__ == "__main__":
    main()