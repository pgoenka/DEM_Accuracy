import rasterio
from rasterio.transform import rowcol
import numpy as np
import pandas as pd
from pyproj import Transformer
import shutil
from pathlib import Path

class MLEngine:
    def refine_dem(self, context, cache):
        print("\n=== ML REFINEMENT ENGINE ===")
        
        fused_path = context.outputs.get("fused_dem")
        stack_path = cache.path(context.aoi, "feature_stack_normalized.npy")
        final_path = cache.path(context.aoi, "final_dem.tif")
        conf_path = cache.path(context.aoi, "confidence.tif")
        gt_path = cache.path(context.aoi, "ground_truth.csv")

        if final_path.exists() and conf_path.exists():
            print("Final Refined DEM and Confidence Map already exist.")
            context.outputs["final_dem"] = final_path
            context.outputs["confidence_map"] = conf_path
            return

        if not fused_path or not stack_path.exists() or not gt_path.exists():
            print("[!] Missing required data (Fused DEM, Stack, or Ground Truth CSV).")
            return

        from sklearn.ensemble import RandomForestRegressor
        
        print("Loading normalized feature stack into memory...")
        stack = np.load(stack_path) 
        
        with rasterio.open(fused_path) as src:
            fused_dem = src.read(1)
            profile = src.profile
            transform = src.transform
            nodata = profile.get('nodata', -9999.0)
            
        print("Loading real ground-truth API points...")
        df = pd.read_csv(gt_path)
        
        if df.empty:
            print("Ground truth is empty. Skipping ML.")
            return
            
        print("Reprojecting Lat/Lon to UTM for spatial alignment...")
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{context.aoi.utm_epsg}", always_xy=True)
        df['utm_x'], df['utm_y'] = transformer.transform(df['lon'].values, df['lat'].values)
        
        print("Sampling feature stack at precise geographic coordinates...")
        rows, cols = rowcol(transform, df['utm_x'].values, df['utm_y'].values)
        
        h, w, c = stack.shape
        rows = np.array(rows)
        cols = np.array(cols)
        true_elev = df['true_elevation'].values
        
        in_bounds = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        rows, cols, true_elev = rows[in_bounds], cols[in_bounds], true_elev[in_bounds]
        
        fused_elev = fused_dem[rows, cols]
        
        valid_data = fused_elev != nodata
        rows, cols = rows[valid_data], cols[valid_data]
        true_elev, fused_elev = true_elev[valid_data], fused_elev[valid_data]
        
        X_train = stack[rows, cols, :]
        y_train = true_elev - fused_elev
        
        print(f"Training Random Forest on {len(X_train)} valid spatial points...")
        rf = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        print("Applying learned ICESat-2 corrections to the entire fused DEM grid...")
        fused_1d = fused_dem.flatten()
        stack_2d = stack.reshape(h * w, c)
        
        valid_mask = fused_1d != nodata
        X_valid = stack_2d[valid_mask]
        
        predicted_error = rf.predict(X_valid)
        
        # --- NEW: UNCERTAINTY ESTIMATION ---
        print("Calculating spatial uncertainty (confidence map)...")
        # Extract predictions from every individual tree to find the standard deviation (uncertainty)
        all_tree_preds = np.stack([tree.predict(X_valid) for tree in rf.estimators_])
        uncertainty = np.std(all_tree_preds, axis=0)
        
        final_1d = np.copy(fused_1d)
        final_1d[valid_mask] = fused_1d[valid_mask] + predicted_error
        final_dem = final_1d.reshape(h, w)
        
        conf_1d = np.copy(fused_1d)
        conf_1d[valid_mask] = uncertainty
        conf_1d[~valid_mask] = nodata
        confidence_map = conf_1d.reshape(h, w)
        
        print(f"Saving machine-learned DTM to {final_path.name}...")
        with rasterio.open(final_path, 'w', **profile) as dst:
            dst.write(final_dem.astype(np.float32), 1)
            
        print(f"Saving uncertainty map to {conf_path.name}...")
        with rasterio.open(conf_path, 'w', **profile) as dst:
            dst.write(confidence_map.astype(np.float32), 1)
            
        print("  ✓ Successfully generated ML-Refined Final DEM and Confidence Map.")
        context.outputs["final_dem"] = final_path
        context.outputs["confidence_map"] = conf_path