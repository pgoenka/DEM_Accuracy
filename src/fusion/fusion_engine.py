import rasterio
import numpy as np
from pathlib import Path

class FusionEngine:

    def fuse(self, context, cache):
        
        print("\n=== ADAPTIVE FUSION ENGINE ===")

        # 1. Retrieve the required datasets from the pipeline context
        base_dem_path = context.raw_dems.get("utm")
        fabdem_path = context.aligned_dems.get("fabdem")
        slope_path = context.features.get("slope")

        if not base_dem_path or not fabdem_path:
            raise FileNotFoundError("Missing base Copernicus or aligned FABDEM. Check Preprocess stage.")

        # 2. Define output path
        fused_path = cache.path(context.aoi, "fused_dem.tif")

        if fused_path.exists():
            print("Fused DEM already exists.")
            context.outputs["fused_dem"] = fused_path
            return

        print("Loading datasets into memory for spatial blending...")
        
        # Load baseline Copernicus DEM
        with rasterio.open(base_dem_path) as src_base:
            cop_data = src_base.read(1).astype(np.float32)
            profile = src_base.profile
            nodata = profile.get('nodata', -9999.0)

        # Load aligned FABDEM
        with rasterio.open(fabdem_path) as src_fab:
            fab_data = src_fab.read(1).astype(np.float32)

        # 3. Apply Adaptive Weighting using Slope
        if slope_path and slope_path.exists():
            with rasterio.open(slope_path) as src_slope:
                slope_data = src_slope.read(1).astype(np.float32)
            
            print("Applying geomorphological adaptive weighting...")
            print("  -> High slope: Prioritizing Copernicus")
            print("  -> Low slope : Prioritizing FABDEM")
            
            # Normalize slope from 0 to 30 degrees to represent weights (0.0 to 1.0)
            # Anything above 30 degrees becomes 1.0 (100% Copernicus)
            cop_weight = np.clip(slope_data / 30.0, 0.0, 1.0)
            fab_weight = 1.0 - cop_weight
            
            # Mask out nodata values before doing math
            valid_mask = (cop_data != nodata) & (fab_data != nodata)
            
            # Initialize array with baseline
            fused_data = np.copy(cop_data)
            
            # Blend only where we have valid data
            fused_data[valid_mask] = (
                (cop_data[valid_mask] * cop_weight[valid_mask]) + 
                (fab_data[valid_mask] * fab_weight[valid_mask])
            )
            
        else:
            print("[!] Slope feature not found. Applying flat 50/50 fallback blend...")
            valid_mask = (cop_data != nodata) & (fab_data != nodata)
            fused_data = np.copy(cop_data)
            fused_data[valid_mask] = (cop_data[valid_mask] * 0.5) + (fab_data[valid_mask] * 0.5)

        # 4. Save the dynamically fused result
        print(f"Saving final terrain model to {fused_path.name}...")
        with rasterio.open(fused_path, 'w', **profile) as dst:
            dst.write(fused_data, 1)

        # Register the final output
        context.outputs["fused_dem"] = fused_path
        print("  ✓ Successfully generated Adaptive Fused DEM.")