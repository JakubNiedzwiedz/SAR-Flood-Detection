# SAR-Flood-Detection
Sentinel-1 SAR flood detection pipeline in Google Earth Engine with Refined Lee speckle filtering, change-based flood mask, vectorization, and OSM road impact analysis (flooded roads, intersections, buffer zones) with shapefile exports.

## Features
- Sentinel-1 GRD collection build for **pre-event** and **post-event** time windows
- Speckle reduction with **Refined Lee** filter
- Change-based flood detection (thresholding on SAR backscatter differences)
- Post-processing:
  - masking **permanent water**
  - masking areas with higher **slope**
  - morphological cleanup
- Vectorization of flood mask to polygons
- OSM road network analysis:
  - flooded vs non-flooded road segments
  - intersection points (roads × flood polygons)
  - 5 m buffers around intersections

## Repository content
- `SAR_GEE_flood+OSM.ipynb` — main notebook (end-to-end workflow + exports)
