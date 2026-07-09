from .config import UserParams, Paths
from .pipeline import run_pipeline


def _print_used_s1_images(used_images: list[dict] | None) -> None:
    """
    Wypisuje wszystkie zobrazowania Sentinel-1 wykorzystane w detekcji powodzi:
    - ID,
    - nazwę,
    - datę,
    - satelitę,
    - orbitę,
    - informację, czy zobrazowanie było porównawcze przed powodzią,
      czy w trakcie powodzi.
    """
    print("\nWykorzystane zobrazowania Sentinel-1:")

    if not used_images:
        print("Brak informacji o wykorzystanych zobrazowaniach.")
        return

    for i, img in enumerate(used_images, start=1):
        usage_type = img.get("usage_type") or "brak informacji"
        image_id = img.get("id") or "brak danych"
        name = img.get("name") or "brak danych"
        date = img.get("date") or "brak danych"
        satellite = img.get("satellite") or "brak danych"

        orbit = img.get("orbit") or "brak danych"
        relative_orbit = img.get("relative_orbit_number")
        orbit_number = img.get("orbit_number")
        slice_number = img.get("slice_number")
        total_slices = img.get("total_slices")

        orbit_parts = [str(orbit)]

        if relative_orbit is not None:
            orbit_parts.append(f"relativeOrbitNumber={relative_orbit}")

        if orbit_number is not None:
            orbit_parts.append(f"orbitNumber={orbit_number}")

        slice_text = ""
        if slice_number is not None and total_slices is not None:
            slice_text = f" | slice: {slice_number}/{total_slices}"

        print(f"\n{i}. {usage_type}")
        print(f"   ID: {image_id}")
        print(f"   Nazwa: {name}")
        print(f"   Data: {date} UTC")
        print(f"   Satelita: {satellite}")
        print(f"   Orbita: {', '.join(orbit_parts)}{slice_text}")


def main() -> None:
    params = UserParams()
    paths = Paths()

    res = run_pipeline(params, paths)

    print("OK ✅ Pipeline done.")
    print("Outputs:", paths.outputs_dir.resolve())
    print("Flooded roads (km):", res.osm.flooded_length_m / 1000)
    print("Dry roads (km):", res.osm.dry_length_m / 1000)

    _print_used_s1_images(res.flood.used_images)


if __name__ == "__main__":
    main()