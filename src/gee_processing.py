from __future__ import annotations

import datetime
from dataclasses import dataclass

import ee


@dataclass
class FloodResult:
    """
    Wynik detekcji powodzi z Sentinel-1.

    Klasa zachowuje stare pola używane w dotychczasowym pipeline.py:
    - floodedD_raw, floodedD_10m, flood_vectorsD, flood_vectors_filteredD, diffD,
    - analogiczne pola A pozostawione dla kompatybilności.

    W nowym algorytmie nie są już finalnie liczone dwie osobne maski ASC/DESC.
    Najpierw wybierana jest jedna najlepsza orbita, a wynik tej orbity jest
    podstawiany pod stare pola D/A, żeby nie zmieniać struktury pipeline’u,
    eksportów ani analizy dróg. Dodatkowe pola selected_orbit/before_date/after_date
    pozwalają sprawdzić, jaka orbita i jakie sceny zostały faktycznie użyte.
    """

    # Stare pola wymagane przez pipeline.py i starsze eksporty
    floodedD_raw: ee.Image
    floodedA_raw: ee.Image
    floodedD_10m: ee.Image
    floodedA_10m: ee.Image

    flood_vectorsD: ee.FeatureCollection
    flood_vectorsA: ee.FeatureCollection
    flood_vectors_filteredD: ee.FeatureCollection

    diffD: ee.Image
    diffA: ee.Image
    permanent_water_bin: ee.Image
    slope: ee.Image

    # Nowe metadane algorytmu
    selected_orbit: str | None = None
    selected_relative_orbit: int | None = None
    before_date: str | None = None
    after_date: str | None = None
    before_delta_days: float | None = None
    after_delta_days: float | None = None

    # Lista scen Sentinel-1 faktycznie wykorzystanych w detekcji.
    used_images: list[dict] | None = None

    # Nowe aliasy zgodne z pipeline_sar_new.py, gdybyś kiedyś chciał go użyć.
    @property
    def flooded_raw(self) -> ee.Image:
        return self.floodedD_raw

    @property
    def flooded_10m(self) -> ee.Image:
        return self.floodedD_10m

    @property
    def flood_vectors(self) -> ee.FeatureCollection:
        return self.flood_vectorsD

    @property
    def flood_vectors_filtered(self) -> ee.FeatureCollection:
        return self.flood_vectors_filteredD

    @property
    def diff(self) -> ee.Image:
        return self.diffD


# ============================================================
# Konwersja dB <-> skala liniowa
# ============================================================

def to_natural(img_db: ee.Image) -> ee.Image:
    """
    Konwersja z dB do skali liniowej.
    """
    return ee.Image(10.0).pow(img_db.divide(10.0))


def to_db(img_lin: ee.Image) -> ee.Image:
    """
    Konwersja ze skali liniowej do dB.
    """
    return img_lin.log10().multiply(10.0)


# ============================================================
# Pełny Refined Lee jak w notebooku
# ============================================================

def refined_lee_function(img_lin: ee.Image) -> ee.Image:
    """
    Pełny kierunkowy Refined Lee filter.

    Input:
        pojedynczy band w skali liniowej.

    Output:
        przefiltrowany pojedynczy band w skali liniowej.

    To jest wersja zgodna z notebookiem:
    - sampling 7x7,
    - 8 kierunków,
    - kierunkowe kernele 9x9,
    - sigmaV z 5 najmniejszych wartości lokalnej statystyki.
    """
    weights3 = ee.List.repeat(ee.List.repeat(1, 3), 3)
    kernel3 = ee.Kernel.fixed(3, 3, weights3, 1, 1, False)

    mean3 = img_lin.reduceNeighborhood(ee.Reducer.mean(), kernel3)
    var3 = img_lin.reduceNeighborhood(ee.Reducer.variance(), kernel3)

    sample_weights = ee.List([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    sample_kernel = ee.Kernel.fixed(7, 7, sample_weights, 3, 3, False)

    sample_mean = mean3.neighborhoodToBands(sample_kernel)
    sample_var = var3.neighborhoodToBands(sample_kernel)

    gradients = (
        sample_mean.select(1).subtract(sample_mean.select(7)).abs()
        .addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs())
        .addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs())
        .addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs())
    )

    max_gradient = gradients.reduce(ee.Reducer.max())
    gradmask = gradients.eq(max_gradient)
    gradmask = gradmask.addBands(gradmask)

    directions = (
        sample_mean.select(1).subtract(sample_mean.select(4))
        .gt(sample_mean.select(4).subtract(sample_mean.select(7)))
        .multiply(1)
        .addBands(
            sample_mean.select(6).subtract(sample_mean.select(4))
            .gt(sample_mean.select(4).subtract(sample_mean.select(2)))
            .multiply(2)
        )
        .addBands(
            sample_mean.select(3).subtract(sample_mean.select(4))
            .gt(sample_mean.select(4).subtract(sample_mean.select(5)))
            .multiply(3)
        )
        .addBands(
            sample_mean.select(0).subtract(sample_mean.select(4))
            .gt(sample_mean.select(4).subtract(sample_mean.select(8)))
            .multiply(4)
        )
    )

    directions = (
        directions
        .addBands(directions.select(0).Not().multiply(5))
        .addBands(directions.select(1).Not().multiply(6))
        .addBands(directions.select(2).Not().multiply(7))
        .addBands(directions.select(3).Not().multiply(8))
    )

    directions = directions.updateMask(gradmask)
    directions = directions.reduce(ee.Reducer.sum())

    sample_stats = sample_var.divide(sample_mean.multiply(sample_mean))
    sigma_v = (
        sample_stats
        .toArray()
        .arraySort()
        .arraySlice(0, 0, 5)
        .arrayReduce(ee.Reducer.mean(), [0])
    )

    rect_weights = (
        ee.List.repeat(ee.List.repeat(0, 9), 4)
        .cat(ee.List.repeat(ee.List.repeat(1, 9), 5))
    )

    diag_weights = ee.List([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])

    rect_kernel = ee.Kernel.fixed(9, 9, rect_weights, 4, 4, False)
    diag_kernel = ee.Kernel.fixed(9, 9, diag_weights, 4, 4, False)

    dir_mean = img_lin.reduceNeighborhood(
        ee.Reducer.mean(),
        rect_kernel,
    ).updateMask(directions.eq(1))

    dir_var = img_lin.reduceNeighborhood(
        ee.Reducer.variance(),
        rect_kernel,
    ).updateMask(directions.eq(1))

    dir_mean = dir_mean.addBands(
        img_lin.reduceNeighborhood(
            ee.Reducer.mean(),
            diag_kernel,
        ).updateMask(directions.eq(2))
    )

    dir_var = dir_var.addBands(
        img_lin.reduceNeighborhood(
            ee.Reducer.variance(),
            diag_kernel,
        ).updateMask(directions.eq(2))
    )

    for i in range(1, 4):
        dir_mean = dir_mean.addBands(
            img_lin.reduceNeighborhood(
                ee.Reducer.mean(),
                rect_kernel.rotate(i),
            ).updateMask(directions.eq(2 * i + 1))
        )

        dir_var = dir_var.addBands(
            img_lin.reduceNeighborhood(
                ee.Reducer.variance(),
                rect_kernel.rotate(i),
            ).updateMask(directions.eq(2 * i + 1))
        )

        dir_mean = dir_mean.addBands(
            img_lin.reduceNeighborhood(
                ee.Reducer.mean(),
                diag_kernel.rotate(i),
            ).updateMask(directions.eq(2 * i + 2))
        )

        dir_var = dir_var.addBands(
            img_lin.reduceNeighborhood(
                ee.Reducer.variance(),
                diag_kernel.rotate(i),
            ).updateMask(directions.eq(2 * i + 2))
        )

    dir_mean = dir_mean.reduce(ee.Reducer.sum())
    dir_var = dir_var.reduce(ee.Reducer.sum())

    var_x = (
        dir_var
        .subtract(dir_mean.multiply(dir_mean).multiply(sigma_v))
        .divide(sigma_v.add(1.0))
    )

    b = var_x.divide(dir_var)

    result = dir_mean.add(
        b.multiply(img_lin.subtract(dir_mean))
    )

    return result.arrayFlatten([["sum"]])


def refined_lee(image: ee.Image) -> ee.Image:
    """
    Stosuje pełny Refined Lee do każdego bandu obrazu.
    Wejście i wyjście są w dB.
    """
    props = image.toDictionary(image.propertyNames())
    bands = image.bandNames()

    def _filter_band(band_name):
        band_name = ee.String(band_name)
        band = image.select([band_name])

        filtered = to_db(
            refined_lee_function(
                to_natural(band)
            )
        )

        return filtered.rename(band_name)

    filtered_ic = ee.ImageCollection(bands.map(_filter_band))
    filtered = filtered_ic.toBands().rename(bands)

    return filtered.set(props)


def refined_lee_db(img_db: ee.Image) -> ee.Image:
    """
    Wrapper zgodny ze starą wersją kodu.
    Wejście: pojedynczy band w dB. Wyjście: pojedynczy band w dB.
    """
    return refined_lee(img_db)


# ============================================================
# Warstwy pomocnicze
# ============================================================

def build_permanent_water_bin(area: ee.Geometry) -> ee.Image:
    """
    Maska wód stałych na podstawie JRC Global Surface Water.

    1 = woda stała / często występująca,
    0 = pozostały obszar.
    """
    water_surface = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(area)
    seasonality = water_surface.select("seasonality")

    permanent = seasonality.gte(6)

    return permanent.unmask(0).rename("permanent_water")


def build_slope(area: ee.Geometry) -> ee.Image:
    """
    Raster nachylenia terenu.
    """
    dem = ee.Image("WWF/HydroSHEDS/03VFDEM").clip(area)
    terrain = ee.Algorithms.Terrain(dem)

    return terrain.select("slope").rename("slope")


# ============================================================
# Daty, kolekcje, wybór orbity
# ============================================================

def _date_window(
    event_date_str: str,
    days_before: int,
    days_after: int,
) -> tuple[str, str, str, str]:
    event_date = datetime.date.fromisoformat(event_date_str)

    before_start = (event_date - datetime.timedelta(days=days_before)).isoformat()
    before_end = (event_date - datetime.timedelta(days=1)).isoformat()

    after_start = (event_date + datetime.timedelta(days=1)).isoformat()
    after_end = (event_date + datetime.timedelta(days=days_after)).isoformat()

    return before_start, before_end, after_start, after_end


def _event_datetime_utc(event_date_str: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(event_date_str).replace(
        tzinfo=datetime.timezone.utc
    )


def _millis_to_date_str(ms: int) -> str:
    dt = datetime.datetime.fromtimestamp(
        ms / 1000,
        tz=datetime.timezone.utc,
    )
    return dt.date().isoformat()


def _days_between_millis(ms: int, event_date_str: str) -> float:
    event_dt = _event_datetime_utc(event_date_str)
    event_ms = int(event_dt.timestamp() * 1000)

    return abs(ms - event_ms) / (1000 * 60 * 60 * 24)


def _s1_base_collection(
    area: ee.Geometry,
    polarizations: list[str] | None = None,
) -> ee.ImageCollection:
    pols = polarizations or ["VV", "VH"]

    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filterBounds(area)
        .select(pols)
    )


def _add_abs_delta_from_event(
    img: ee.Image,
    event_date_str: str,
) -> ee.Image:
    event_ms = ee.Date(event_date_str).millis()
    delta = img.date().millis().subtract(event_ms).abs()

    return img.set("delta_from_event_ms", delta)


def _closest_image_to_event(
    coll: ee.ImageCollection,
    event_date_str: str,
    area: ee.Geometry,
) -> ee.Image:
    """
    Zachowane dla kompatybilności, ale w głównym algorytmie lepiej używać
    _mosaic_for_selected_date_and_track(). Pojedyncza scena Sentinel-1 może
    nie pokrywać całego AOI i powodować ostre odcięcie wyniku po footprintcie.
    """
    closest = (
        coll
        .map(lambda img: _add_abs_delta_from_event(img, event_date_str))
        .sort("delta_from_event_ms")
        .first()
    )

    return ee.Image(closest).clip(area)


def _selected_s1_collection_for_date_and_track(
    coll: ee.ImageCollection,
    date_str: str,
    selected_orbit: str,
    relative_orbit_number: int,
) -> ee.ImageCollection:
    """
    Zwraca dokładnie te sceny Sentinel-1, które mają wejść do mozaiki:
    wybrana data, kierunek przelotu i numer ścieżki względnej.
    """
    start = ee.Date(date_str)
    end = start.advance(1, "day")

    return (
        coll
        .filterDate(start, end)
        .filter(ee.Filter.eq("orbitProperties_pass", selected_orbit))
        .filter(ee.Filter.eq("relativeOrbitNumber_start", relative_orbit_number))
        .sort("system:time_start")
    )


def _mosaic_for_selected_date_and_track(
    coll: ee.ImageCollection,
    date_str: str,
    selected_orbit: str,
    relative_orbit_number: int,
    area: ee.Geometry,
) -> ee.Image:
    """
    Tworzy mozaikę ze wszystkich scen Sentinel-1 z wybranej daty, kierunku
    przelotu i numeru ścieżki względnej.

    To usuwa problem prostego odcięcia na granicy pojedynczego footprintu GRD.
    Dla większego AOI jedna data może składać się z kilku sąsiadujących scen /
    slices, które trzeba zmozaikować przed filtrowaniem i liczeniem ratio.
    """
    selected = _selected_s1_collection_for_date_and_track(
        coll=coll,
        date_str=date_str,
        selected_orbit=selected_orbit,
        relative_orbit_number=relative_orbit_number,
    )

    return selected.mosaic().clip(area).set({
        "selected_date": date_str,
        "orbitProperties_pass": selected_orbit,
        "relativeOrbitNumber_start": relative_orbit_number,
    })


def _used_s1_images_metadata(
    coll: ee.ImageCollection,
    usage_type: str,
) -> list[dict]:
    """
    Pobiera metadane scen Sentinel-1 faktycznie wykorzystanych w detekcji.

    usage_type:
        - "porównawcze przed powodzią"
        - "w trakcie powodzi"
    """
    count = int(coll.size().getInfo())

    if count == 0:
        return []

    images_list = coll.toList(count)
    images: list[dict] = []

    properties_to_read = [
        "system:index",
        "system:time_start",
        "platform_number",
        "orbitProperties_pass",
        "relativeOrbitNumber_start",
        "orbitNumber_start",
        "sliceNumber",
        "totalSlices",
    ]

    for i in range(count):
        img = ee.Image(images_list.get(i))

        image_id = img.id().getInfo()
        props = img.toDictionary(properties_to_read).getInfo()
        date = img.date().format("YYYY-MM-dd HH:mm:ss").getInfo()

        platform_number = props.get("platform_number")
        if platform_number:
            satellite = f"Sentinel-1{platform_number}"
        else:
            satellite = "Sentinel-1"

        name = props.get("system:index") or (
            image_id.split("/")[-1] if image_id else None
        )

        images.append(
            {
                "usage_type": usage_type,
                "id": image_id,
                "name": name,
                "date": date,
                "satellite": satellite,
                "orbit": props.get("orbitProperties_pass"),
                "relative_orbit_number": props.get("relativeOrbitNumber_start"),
                "orbit_number": props.get("orbitNumber_start"),
                "slice_number": props.get("sliceNumber"),
                "total_slices": props.get("totalSlices"),
            }
        )

    return images


def _closest_image_metadata(
    coll: ee.ImageCollection,
    event_date_str: str,
) -> tuple[int, str, float]:
    """
    Zwraca:
    - time_start_ms,
    - date_str,
    - delta_days względem event_date.
    """
    closest = (
        coll
        .map(lambda img: _add_abs_delta_from_event(img, event_date_str))
        .sort("delta_from_event_ms")
        .first()
    )

    time_start_ms = ee.Image(closest).get("system:time_start").getInfo()

    if time_start_ms is None:
        raise ValueError("Nie udało się odczytać daty najbliższego obrazu Sentinel-1.")

    time_start_ms = int(time_start_ms)

    return (
        time_start_ms,
        _millis_to_date_str(time_start_ms),
        _days_between_millis(time_start_ms, event_date_str),
    )


def _select_best_orbit(
    before: ee.ImageCollection,
    after: ee.ImageCollection,
    event_date_str: str,
) -> dict:
    """
    Wybiera najlepszą parę Sentinel-1 before/after.

    Poprawiona logika:
    - osobno sprawdza ASCENDING i DESCENDING,
    - w ramach kierunku przelotu wymaga tego samego relativeOrbitNumber_start
      dla sceny before i after,
    - wybiera parę z najmniejszą sumą odległości czasowej od daty zdarzenia.

    Dzięki temu porównywane są sceny z tej samej geometrii obrazowania, a nie
    przypadkowe obrazy z tym samym kierunkiem przelotu, ale inną ścieżką.
    """
    candidates = []

    for orbit in ["DESCENDING", "ASCENDING"]:
        before_orbit = before.filter(
            ee.Filter.eq("orbitProperties_pass", orbit)
        )
        after_orbit = after.filter(
            ee.Filter.eq("orbitProperties_pass", orbit)
        )

        before_count = int(before_orbit.size().getInfo())
        after_count = int(after_orbit.size().getInfo())

        if before_count == 0 or after_count == 0:
            continue

        before_rel = before_orbit.aggregate_array(
            "relativeOrbitNumber_start"
        ).distinct().getInfo()

        after_rel = after_orbit.aggregate_array(
            "relativeOrbitNumber_start"
        ).distinct().getInfo()

        common_rel = sorted(
            set(int(x) for x in before_rel if x is not None)
            .intersection(set(int(x) for x in after_rel if x is not None))
        )

        for rel_orbit in common_rel:
            before_track = before_orbit.filter(
                ee.Filter.eq("relativeOrbitNumber_start", rel_orbit)
            )
            after_track = after_orbit.filter(
                ee.Filter.eq("relativeOrbitNumber_start", rel_orbit)
            )

            before_track_count = int(before_track.size().getInfo())
            after_track_count = int(after_track.size().getInfo())

            if before_track_count == 0 or after_track_count == 0:
                continue

            before_ms, before_date, before_delta = _closest_image_metadata(
                before_track,
                event_date_str,
            )

            after_ms, after_date, after_delta = _closest_image_metadata(
                after_track,
                event_date_str,
            )

            candidates.append(
                {
                    "orbit": orbit,
                    "relative_orbit_number": rel_orbit,
                    "before_count": before_track_count,
                    "after_count": after_track_count,
                    "before_ms": before_ms,
                    "after_ms": after_ms,
                    "before_date": before_date,
                    "after_date": after_date,
                    "before_delta_days": before_delta,
                    "after_delta_days": after_delta,
                    "score": before_delta + after_delta,
                }
            )

    if not candidates:
        raise ValueError(
            "Brak kompletnej pary obrazów Sentinel-1 before/after "
            "dla orbit ASCENDING lub DESCENDING oraz wspólnego relativeOrbitNumber_start "
            "w zadanym oknie czasowym."
        )

    candidates.sort(key=lambda x: x["score"])

    return candidates[0]


# ============================================================
# Morfologia i wektoryzacja
# ============================================================

def _clean_flood_mask(
    flooded_mask: ee.Image,
    combined_mask: ee.Image,
    radius_m: float = 20.0,
) -> ee.Image:
    """
    Operacje morfologiczne zgodnie ze screenem:

    1. Otwarcie:
       erozja -> dylatacja
       focal_min -> focal_max

    2. Zamknięcie:
       dylatacja -> erozja
       focal_max -> focal_min
    """
    flooded_masked = flooded_mask.updateMask(combined_mask)

    base = flooded_masked.unmask(0)

    # Otwarcie: erozja -> dylatacja
    flooded_m = (
        base
        .focal_min(radius=radius_m, units="meters")
        .focal_max(radius=radius_m, units="meters")
    )

    # Zamknięcie: dylatacja -> erozja
    flooded_m = (
        flooded_m
        .focal_max(radius=radius_m, units="meters")
        .focal_min(radius=radius_m, units="meters")
    )

    return flooded_m.updateMask(flooded_m)


def _to_10m(img: ee.Image) -> ee.Image:
    return (
        img.setDefaultProjection("EPSG:4326", None, 10)
        .reduceResolution(ee.Reducer.max(), maxPixels=1024)
        .reproject(crs="EPSG:4326", scale=10)
    )


def _vectorize_flood_mask(
    flood_mask_10m: ee.Image,
    area: ee.Geometry,
) -> ee.FeatureCollection:
    return flood_mask_10m.reduceToVectors(
        geometry=area,
        scale=10,
        crs="EPSG:4326",
        geometryType="polygon",
        eightConnected=True,
        labelProperty="water",
        maxPixels=1e10,
        tileScale=4,
    )


def _filter_flood_vectors_by_area(
    flood_vectors: ee.FeatureCollection,
    min_area_m2: float,
) -> ee.FeatureCollection:
    flood_vectors_area = flood_vectors.map(
        lambda f: f.set("area_m2", f.geometry().area(1))
    )

    return flood_vectors_area.filter(
        ee.Filter.gte("area_m2", min_area_m2)
    )


# ============================================================
# Główna detekcja powodzi
# ============================================================

def detect_flood_from_s1(
    area: ee.Geometry,
    event_date_str: str,
    days_before: int,
    days_after: int,
    flood_ratio_threshold: float = 1.35,
    max_slope: float = 5.0,
    min_area_m2: float = 800.0,
) -> FloodResult:
    """
    Detekcja powodzi z Sentinel-1.

    Nowa logika:
    1. Buduje kolekcję Sentinel-1.
    2. Dzieli obrazy na before/after.
    3. Przed obliczeniami wybiera jedną orbitę:
       ASCENDING albo DESCENDING.
    4. Dla wybranej orbity i tej samej ścieżki relativeOrbitNumber wybiera najbliższe daty before/after.
    5. Dla wybranych dat tworzy mozaikę wszystkich scen pokrywających AOI, żeby uniknąć odcięcia po footprintcie pojedynczej sceny.
    6. Stosuje pełny Refined Lee jak w notebooku.
    7. Oblicza ratio before / after.
    8. Stosuje maski permanent water i slope.
    9. Stosuje morfologię: opening -> closing.
    10. Wektoryzuje wynik.
    """
    before_start, before_end, after_start, after_end = _date_window(
        event_date_str=event_date_str,
        days_before=days_before,
        days_after=days_after,
    )

    collection = _s1_base_collection(
        area=area,
        polarizations=["VV", "VH"],
    )

    before = collection.filterDate(before_start, before_end)
    after = collection.filterDate(after_start, after_end)

    selected = _select_best_orbit(
        before=before,
        after=after,
        event_date_str=event_date_str,
    )

    selected_orbit = selected["orbit"]
    relative_orbit_number = int(selected["relative_orbit_number"])

    before_selected_coll = before.filter(
        ee.Filter.eq("orbitProperties_pass", selected_orbit)
    )

    after_selected_coll = after.filter(
        ee.Filter.eq("orbitProperties_pass", selected_orbit)
    )

    before_used_coll = _selected_s1_collection_for_date_and_track(
        coll=before_selected_coll,
        date_str=selected["before_date"],
        selected_orbit=selected_orbit,
        relative_orbit_number=relative_orbit_number,
    )

    after_used_coll = _selected_s1_collection_for_date_and_track(
        coll=after_selected_coll,
        date_str=selected["after_date"],
        selected_orbit=selected_orbit,
        relative_orbit_number=relative_orbit_number,
    )

    used_images = (
        _used_s1_images_metadata(
            before_used_coll,
            usage_type="porównawcze przed powodzią",
        )
        + _used_s1_images_metadata(
            after_used_coll,
            usage_type="w trakcie powodzi",
        )
    )

    before_img = before_used_coll.mosaic().clip(area).set({
        "selected_date": selected["before_date"],
        "orbitProperties_pass": selected_orbit,
        "relativeOrbitNumber_start": relative_orbit_number,
    })

    after_img = after_used_coll.mosaic().clip(area).set({
        "selected_date": selected["after_date"],
        "orbitProperties_pass": selected_orbit,
        "relativeOrbitNumber_start": relative_orbit_number,
    })

    # Pełny Refined Lee na całym obrazie, potem wybór VH.
    before_filtered = refined_lee(before_img)
    after_filtered = refined_lee(after_img)

    before_vh = before_filtered.select("VH")
    after_vh = after_filtered.select("VH")

    # Ratio before / after
    diff = to_natural(before_vh).divide(
        to_natural(after_vh)
    ).rename("ratio")

    flooded_raw = diff.gt(
        flood_ratio_threshold
    ).rename("water").selfMask()

    # Maski pomocnicze
    permanent_water_bin = build_permanent_water_bin(area)
    non_permanent_water = permanent_water_bin.Not()

    slope = build_slope(area)
    low_slope = slope.lt(max_slope)

    combined_mask = non_permanent_water.And(low_slope)

    # Morfologia: opening -> closing
    flooded_clean = _clean_flood_mask(
        flooded_mask=flooded_raw,
        combined_mask=combined_mask,
        radius_m=20.0,
    )

    flooded_10m = _to_10m(flooded_clean)

    flood_vectors = _vectorize_flood_mask(
        flood_mask_10m=flooded_10m,
        area=area,
    )

    flood_vectors_filtered = _filter_flood_vectors_by_area(
        flood_vectors=flood_vectors,
        min_area_m2=min_area_m2,
    )

    return FloodResult(
        # Stare pola / kompatybilność ze starym pipeline.py.
        # D oznacza tutaj wynik finalnie wybranej orbity,
        # bo nowy algorytm wybiera jedną najlepszą orbitę przed detekcją.
        floodedD_raw=flooded_raw,
        floodedA_raw=flooded_raw,
        floodedD_10m=flooded_10m,
        floodedA_10m=flooded_10m,
        flood_vectorsD=flood_vectors,
        flood_vectorsA=flood_vectors,
        flood_vectors_filteredD=flood_vectors_filtered,
        diffD=diff,
        diffA=diff,
        permanent_water_bin=permanent_water_bin,
        slope=slope,

        # Nowe metadane z algorytmu wyboru scen/orbity.
        selected_orbit=selected["orbit"],
        selected_relative_orbit=relative_orbit_number,
        before_date=selected["before_date"],
        after_date=selected["after_date"],
        before_delta_days=round(float(selected["before_delta_days"]), 3),
        after_delta_days=round(float(selected["after_delta_days"]), 3),

        # Lista wszystkich zobrazowań wykorzystanych w detekcji.
        used_images=used_images,
    )