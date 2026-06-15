# Accurate DEM Fusion Pipeline

An advanced, automated architecture for the generation of high-precision Digital Elevation Models (DEMs) through multi-source data fusion, machine learning refinement, and hydrological conditioning.

## Project Overview

This pipeline addresses the limitations of individual spaceborne DEMs (like Copernicus GLO-30 or FABDEM) by fusing them using geomorphological weighting and refining the result against NASA ICESat-2 LiDAR ground-truth data. 

**Phase 2 is now COMPLETE**, introducing "Deep Context" intelligence. The pipeline no longer just sees the shape of the terrain; it now understands the **Biological** (vegetation) and **Structural** (buildings/radar texture) layers that sit on top of it. This allows the ML model to explicitly correct for canopy and urban elevation biases.

## Core Architecture

The project is built on a modular, stage-based architecture orchestrated by a `PipelineRegistry`. It enforces strict Area of Interest (AOI) bounding, ensuring all processing is spatially consistent.

### 1. Data Ingestion (Download Stage)
The pipeline autonomously fetches data from multiple global providers:
- **Copernicus GLO-30**: Retrieved via the Microsoft Planetary Computer (STAC API).
- **FABDEM V1.2**: Fetched via the `fabdem` API, providing a forest-and-building-removed version of Copernicus.
- **NASA ICESat-2 (ATL08)**: High-accuracy laser altimetry points fetched via `earthaccess`.
- **Sentinel-2 L2A**: Optical imagery (Red/NIR) for NDVI calculation.
- **Sentinel-1 RTC**: Radar backscatter (VV/VH) providing structural texture intelligence.
- **Building Footprints**: Fetched via the Overpass API (OSM) for precise structural masking.

### 2. Preprocessing & Spatial Alignment
- **Nuth & Kääb 3D Co-registration**: Performs sub-pixel 3D alignment of FABDEM to the Copernicus baseline using `xdem`.
- **Satellite Processing**: Sentinel-1 and Sentinel-2 bands are reprojected and resampled exactly to the DEM grid using `rasterio.warp`.
- **Urban Rasterization**: Vector building footprints are rasterized into a pixel-aligned binary mask.

### 3. The 13-Channel Feature Stack
To help the ML model understand terrain and environment characteristics, the `FeatureEngine` generates a 13-channel normalized tensor:
1.  **Elevation**: Baseline terrain height.
2.  **Slope / Aspect / Hillshade**: Primary geomorphology.
3.  **Roughness / Curvature / Relief**: Secondary geomorphology.
4.  **TRI / TPI**: Ruggedness and local positioning.
5.  **NDVI (10th)**: Biological intelligence (Vegetation density).
6.  **SAR VV (11th)**: Radar backscatter (Surface texture).
7.  **SAR VH (12th)**: Radar cross-polarization (Volume/Structure).
8.  **Building Mask (13th)**: Structural intelligence (Binary structural presence).

### 4. Adaptive Fusion Engine
Uses **Adaptive Slope Weighting** to blend Copernicus and FABDEM:
- **Low Slopes (< 5°)**: Prioritizes FABDEM (best for flat/urban areas).
- **High Slopes (> 30°)**: Prioritizes Copernicus GLO-30 (best for extreme terrain).

### 5. Machine Learning Refinement
A **Spatial Random Forest Regressor** is trained on the residual error between the Fused DEM and ICESat-2 LiDAR. By utilizing the 13-channel stack, the model learns to identify and subtract complex errors caused by slope, canopy, and buildings.

### 6. Hydrological Conditioning
Uses `WhiteboxTools` for **Fast Depression Breaching**, ensuring continuous water flow paths—a critical step for flood and drainage modeling.

---

## Execution Guide

### Prerequisites
- Python 3.10+
- [WhiteboxTools](https://www.whiteboxtools.com/) binary.
- NASA Earthdata Account.

### Secure Authentication
Create a `.env` file in the root directory (using `.env.example` as a template) and add your NASA credentials:
```env
EARTHDATA_USERNAME=your_username
EARTHDATA_PASSWORD=your_password
```

### Installation
```bash
pip install -r requirements.txt
```

### Running the Pipeline
```bash
python run.py
```
*Note: `run.py` automatically clears the processed cache before each run to ensure the 13-channel stack is rebuilt from fresh data.*

## Directory Structure
- `src/download/`: Multi-source data acquisition.
- `src/preprocessing/`: Alignment, NDVI, SAR, and Urban processing.
- `src/features/`: Terrain analysis and 13-channel stacking.
- `src/ml/`: Spatial Random Forest and uncertainty mapping.
- `output/`: `final_dem.tif`, `confidence.tif`, and `quality_report.pdf`.
