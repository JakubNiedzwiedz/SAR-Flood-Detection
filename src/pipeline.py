import ee
from .config import UserParams, Paths
from .ee_utils import ee_init

def bootstrap(params: UserParams, paths: Paths) -> tuple[UserParams, Paths]:
    paths.ensure()
    ee_init(params.gee_project)
    return params, paths

def get_area_geometry(params: UserParams) -> ee.Geometry:
    nysa_gmina = ee.FeatureCollection(params.nysa_asset)
    return nysa_gmina.geometry()