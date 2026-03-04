import ee

def ee_init(project: str) -> None:
    """
    Initializes Google Earth Engine for local execution (VS Code).
    Assumes you have already done ee.Authenticate() once on this machine.
    """
    ee.Initialize(project=project)

def ee_point(lon: float, lat: float) -> ee.Geometry:
    return ee.Geometry.Point([lon, lat])