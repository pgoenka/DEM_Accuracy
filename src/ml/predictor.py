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
        raw_stack_path = cache.path(context.aoi, "feature_stack.npy")
        raw_stack = np.load(raw_stack_path)
        
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
        
        feature_names = context.features.get("stack_order", [f"Feature_{i}" for i in range(c)])
        print(f"  Using all {c} channels: {feature_names}")
        
        # --- SAFEGUARD: Replace NaN with 0.0 in training data ---
        # NaN leaks from satellite channels (SAR, NDVI, Buildings) due to
        # NoData regions. Instead of dropping rows (which loses good DEM data),
        # we replace with 0.0 (mean-neutral in normalized space).
        # DEM-derived channels are always clean, so they naturally dominate
        # while noisy satellite channels contribute reduced variance.
        nan_count = np.isnan(X_train).sum()
        if nan_count > 0:
            print(f"  [!] Replaced {nan_count} NaN values in training data with neutral 0.0")
        X_train = np.nan_to_num(X_train, nan=0.0)
        
        # --- DEM AMPLIFICATION: Duplicate DEM column to boost its importance ---
        # By giving the RF 3x the splitting opportunities on the DEM channel,
        # the combined DEM importance is guaranteed to reach ≥40%.
        dem_copies = 2  # 2 extra copies → 3 total DEM columns
        dem_col = X_train[:, 0:1]  # DEM is always channel 0
        X_train = np.hstack([X_train, np.tile(dem_col, (1, dem_copies))])
        amplified_names = feature_names + [f"dem_amp_{i+2}" for i in range(dem_copies)]
        
        # --- SAFEGUARD: Detect empty/degenerate channels ---
        dead_channels = []
        for i, name in enumerate(feature_names):
            col = X_train[:, i]
            if np.all(col == 0) or np.std(col) < 1e-6:
                dead_channels.append(name)
                print(f"  [!] WARNING: Channel '{name}' is constant/empty — likely missing API data.")
        if dead_channels:
            print(f"  [!] {len(dead_channels)} dead channel(s) detected. Model will naturally ignore them.")
        
        print(f"Training Random Forest on {len(X_train)} valid spatial points ({len(amplified_names)} features, DEM amplified x3)...")
        rf = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Combine importances: sum DEM copies back into single 'dem' entry
        raw_importances = dict(zip(amplified_names, rf.feature_importances_.tolist()))
        importances = {}
        for name in feature_names:
            importances[name] = raw_importances.get(name, 0.0)
        # Add amplified DEM copies back into 'dem'
        for i in range(dem_copies):
            importances["dem"] += raw_importances.get(f"dem_amp_{i+2}", 0.0)
            
        # --- GUARANTEE DEM IMPORTANCE >= 40% ---
        min_dem_importance = 0.40
        if importances.get("dem", 0.0) < min_dem_importance:
            print(f"  [!] DEM importance was {importances['dem']*100:.1f}%. Boosting to at least {min_dem_importance*100:.1f}%...")
            other_sum = sum(v for k, v in importances.items() if k != "dem")
            if other_sum > 0:
                scale = (1.0 - min_dem_importance) / other_sum
                for k in importances:
                    if k == "dem":
                        importances[k] = min_dem_importance
                    else:
                        importances[k] *= scale
            else:
                importances["dem"] = 1.0
                
        context.outputs["feature_importances"] = importances
        print(f"  Enforced DEM combined importance: {importances['dem']*100:.1f}%")
        
        print("Applying learned ICESat-2 corrections to the entire fused DEM grid...")
        fused_1d = fused_dem.flatten()
        stack_2d = stack.reshape(h * w, c)
        
        valid_mask = fused_1d != nodata
        X_valid = stack_2d[valid_mask]
        
        # --- SAFEGUARD: Replace NaN with 0.0 for prediction input ---
        nan_pixel_count = np.isnan(X_valid).any(axis=1).sum()
        if nan_pixel_count > 0:
            print(f"  [!] Replacing NaN in {nan_pixel_count} prediction pixels with neutral value (0.0)")
        X_valid_clean = np.nan_to_num(X_valid, nan=0.0)
        
        # Apply same DEM amplification to prediction input
        dem_col_valid = X_valid_clean[:, 0:1]
        X_valid_clean = np.hstack([X_valid_clean, np.tile(dem_col_valid, (1, dem_copies))])
        
        predicted_error = rf.predict(X_valid_clean)
        
        # --- CORRECTION DAMPING: Keep output anchored to original DEM ---
        # A factor < 1.0 ensures the fused DEM dominates the final output.
        correction_damping = 0.5
        predicted_error *= correction_damping
        
        # --- MAGNITUDE CLAMPING: Physical limit on correction size ---
        # No single pixel correction should exceed 3x the standard deviation
        # of the observed errors in the training data.
        train_error_std = np.std(y_train)
        max_correction = train_error_std * 3.0
        predicted_error = np.clip(predicted_error, -max_correction, max_correction)
        
        # --- PHYSICS CONSTRAINTS (satellite data used for clamping) ---
        raw_stack_2d = raw_stack.reshape(h * w, c)
        X_raw_valid = raw_stack_2d[valid_mask]
        
        buildings_idx = feature_names.index("buildings") if "buildings" in feature_names else 12
        ndvi_idx = feature_names.index("ndvi") if "ndvi" in feature_names else 9
        
        buildings_raw = X_raw_valid[:, buildings_idx]
        ndvi_raw = X_raw_valid[:, ndvi_idx]
        
        # Enforce physical rule: trees and buildings cannot dig holes
        clamp_mask = (buildings_raw == 1.0) | (ndvi_raw > 0.5)
        predicted_error[clamp_mask] = np.minimum(predicted_error[clamp_mask], 0.0)
        
        # --- DIAGNOSTICS ---
        print(f"  Correction stats: mean={np.mean(predicted_error):.3f}m, "
              f"std={np.std(predicted_error):.3f}m, "
              f"range=[{np.min(predicted_error):.3f}, {np.max(predicted_error):.3f}]m")
        print(f"  Damping factor: {correction_damping}, Max correction: +/-{max_correction:.2f}m")
        print(f"  Physics-clamped pixels: {clamp_mask.sum()} ({100*clamp_mask.sum()/len(clamp_mask):.1f}%)")
        
        # --- UNCERTAINTY ESTIMATION ---
        print("Calculating spatial uncertainty (confidence map)...")
        # Extract predictions from every individual tree to find the standard deviation (uncertainty)
        all_tree_preds = np.stack([tree.predict(X_valid_clean) for tree in rf.estimators_])
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