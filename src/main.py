from .config import UserParams, Paths
from .pipeline import bootstrap

def main() -> None:
    params = UserParams()
    paths = Paths()
    bootstrap(params, paths)
    print(f"GEE initialized with project={params.gee_project}")
    print(f"Outputs dir: {paths.outputs_dir.resolve()}")

if __name__ == "__main__":
    main()