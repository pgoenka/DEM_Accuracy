# Project Overview: Accurate DEM Fusion Pipeline
This project is an advanced, automated Digital Elevation Model (DEM) fusion architecture utilizing a 13-channel geomorphological and environmental tensor stack (Slope, NDVI, SAR, Urban Masks) to correct spaceborne elevation data using a Random Forest model.

## Current Project State
**Phase 1 & 2 Implementation is complete, but we are currently in a Diagnostic & Calibration Phase.** The transition to the 13-channel stack introduced a systematic positive bias (making the final DEM higher than the original). This is highly indicative of NoData (`-9999.0`) leakage during feature normalization and an absence of physical constraints in the ML predictions.

## Architecture & Coding Rules
1. **AOI-Centric:** All processing must be strictly bounded by the `AreaOfInterest`.
2. **Strict Caching:** Always check `.exists()` before running expensive operations to utilize the `CacheManager`.
3. **Pipeline Registry:** All new functionality must be registered as a `PipelineStage` in `src/core/pipeline.py`.
4. **Absolute Paths:** When using `WhiteboxTools`, always use `.resolve()` on `pathlib.Path` objects.
5. **Masking Hygiene (CRITICAL):** Never normalize or train on NoData values. Always explicitly mask `-9999.0` and `0.0` (where appropriate) before applying `np.nanmean`, `np.nanstd`, or passing data to `sklearn`.

## Immediate Next Goal: Model Calibration & Physics Constraints
Before attempting Deep Learning (Phase 3), we must stabilize the Random Forest.
1. **Fix the Normalizer:** Update `src/features/normalizer.py` to aggressively replace all NoData variants with `np.nan` BEFORE calculating statistics.
2. **Feature Importance:** The ML Engine must calculate and export `rf.feature_importances_` to `metadata.json` so we can audit what the model is learning.
3. **Physics-Informed Clamping:** Enforce a hard physical rule in `predictor.py`: If a pixel has a Building Mask == 1 or high NDVI, the predicted error *cannot* be positive (trees and buildings do not dig holes; they only increase DSM height).