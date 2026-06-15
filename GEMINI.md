# Project Overview: Accurate DEM Fusion Pipeline
This project is an advanced, automated Digital Elevation Model (DEM) fusion architecture. It ingests multiple spaceborne datasets (Copernicus GLO-30, FABDEM, NASA ICESat-2 LiDAR), aligns them, extracts geomorphological features, and uses Machine Learning (Random Forest) to predict and correct elevation errors. Finally, it hydrologically conditions the terrain for water flow analysis.

## Current Project State
**Phase 1 is 100% COMPLETE. Phase 2 (Optical Intelligence) is COMPLETE.** The pipeline successfully executes the following sequence via the `PipelineRegistry`:
1. **Download:** Fetches Copernicus, FABDEM, NASA ICESat-2 ATL08 LiDAR, and Sentinel-2 L2A optical imagery (Red/NIR bands) via Planetary Computer.
2. **Preprocess:** Merges, crops, reprojects to UTM, aligns DEMs via `xdem`, and calculates NDVI while rigorously resampling to match the DEM grid exactly.
3. **Features:** Generates a 10-channel normalized tensor stack (Slope, Aspect, Hillshade, Roughness, Curvature, Relief, TRI, TPI, NDVI).
4. **Fusion:** Adaptively blends Copernicus and FABDEM using localized slope weightings.
5. **Refinement:** Trains a spatial Random Forest model on the NASA LiDAR ground-truth using all 10 channels, allowing it to correlate elevation errors with vegetation density. Generates a spatial Uncertainty/Confidence map.
6. **Hydrology:** Uses `WhiteboxTools` (Rust binary) for Fast Depression Breaching to ensure continuous water flow.
7. **Export:** Packages the final DEMs, metadata, and a visually scaled `quality_report.pdf`.

## Architecture & Coding Rules
When writing or editing code for this project, you MUST adhere to the following architectural rules:

1. **AOI-Centric:** The `AreaOfInterest` class is the absolute source of truth. Never write global download or processing logic; everything must be cropped or queried strictly to the AOI bounding box.
2. **Strict Caching:** Never bypass the `CacheManager`. Every intermediate artifact must be saved to `cache/raw/` or `cache/processed/`. Always check if a file `.exists()` before running expensive processing.
3. **Pipeline Registry:** Do not write procedural scripts. All new functionality must be wrapped in an Engine/Manager class and registered as a `PipelineStage` inside `src/core/pipeline.py`.
4. **Absolute Paths for Binaries:** When using wrappers around C/Rust binaries (like `WhiteboxTools`), always use `.resolve()` on `pathlib.Path` objects to pass absolute paths.
5. **Pylance/Type Safety:** When working with `h5py` (HDF5 files), always use `isinstance(obj, h5py.Group)`. When using `rasterio.transform`, import `rowcol` directly.

## Immediate Next Goal: Phase 2 (Radar & Urban Intelligence)
The ML model can now "see" vegetation via NDVI. However, NDVI struggles with urban structures and dense canopy volume. The immediate next step is to integrate **Sentinel-1 SAR (Synthetic Aperture Radar) RTC backscatter** and/or **Urban Building Footprints**. Radar provides structural texture, allowing the ML model to identify buildings and correct artificial urban elevation spikes.