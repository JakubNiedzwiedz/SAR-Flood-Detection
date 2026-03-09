# SAR Flood Detection

This repository contains a workflow for detecting flood extent using **Sentinel-1 SAR imagery** and analyzing its impact on **road infrastructure from OpenStreetMap (OSM)**.

Satellite data processing is performed in **Google Earth Engine**, while the post-processing and infrastructure analysis are implemented in **Python**.

---

## Workflow

The pipeline performs the following steps:

1. Sentinel-1 preprocessing and speckle reduction (Refined Lee filter)
2. Flood detection using SAR backscatter change
3. Removal of permanent water (JRC Global Surface Water)
4. Terrain filtering using slope data
5. Flood mask vectorization
6. Road network extraction from OpenStreetMap
7. Detection of flooded road segments and intersections
8. Export of results as shapefiles

---

## Repository Structure

```
SAR-Flood-Detection
│
├── notebooks
│   ├── 00_colab_original.ipynb
│   └── 10_VSC_working_notebook.ipynb
│
├── src
│   ├── config.py
│   ├── gee_processing.py
│   ├── osm_processing.py
│   ├── export_layers.py
│   ├── pipeline.py
│   └── main.py
│
├── outputs
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/JakubNiedzwiedz/SAR-Flood-Detection.git
cd SAR-Flood-Detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it (Windows):

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Authenticate Google Earth Engine (first run only):

```bash
python -c "import ee; ee.Authenticate()"
```

---

## Run the Pipeline

```bash
python -m src.main
```

Results will be saved in:

```
outputs/
```

---

## Outputs

The pipeline generates:

- flood extent derived from Sentinel-1
- permanent water mask
- flooded road segments
- non-flooded road segments
- intersection points between roads and flooded areas

All outputs are exported as **Shapefiles**.

---

## Author

Jakub Niedźwiedź  
Engineering thesis project – flood detection using SAR satellite data.
