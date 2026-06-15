import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
from pathlib import Path
from tqdm import tqdm

class PatchGenerator:
    """Generates spatial patches for Deep Learning (U-Net) training."""

    def generate_patches(self, context, cache, patch_size=256):
        print(f"\n=== CNN PATCH GENERATOR (Phase 3) ===")
        
        stack_path = cache.path(context.aoi, "feature_stack_normalized.npy")
        fused_path = context.outputs.get("fused_dem")
        gt_path = cache.path(context.aoi, "ground_truth.csv")
        
        out_x_path = cache.path(context.aoi, "cnn_patches_X.npy")
        out_y_path = cache.path(context.aoi, "cnn_patches_y.npy")

        if out_x_path.exists() and out_y_path.exists():
            print("CNN patches already exist in cache.")
            return out_x_path, out_y_path

        if not stack_path.exists() or not fused_path or not gt_path.exists():
            print("[!] Missing required data for patch generation.")
            return None, None

        # 1. Load Data
        print("Loading feature stack and ground truth...")
        stack = np.load(stack_path)
        df = pd.read_csv(gt_path)
        
        with rasterio.open(fused_path) as src:
            fused_dem = src.read(1)
            transform = src.transform
            nodata = src.nodata

        # 2. Map GT points to Pixel Coordinates
        print("Mapping geographic coordinates to pixel indices...")
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{context.aoi.utm_epsg}", always_xy=True)
        df['utm_x'], df['utm_y'] = transformer.transform(df['lon'].values, df['lat'].values)
        rows, cols = rowcol(transform, df['utm_x'].values, df['utm_y'].values)
        
        df['row'] = rows
        df['col'] = cols
        
        h, w, c = stack.shape
        half_size = patch_size // 2
        
        # 3. Extract Patches
        X_list = []
        y_list = []
        
        print(f"Extracting {patch_size}x{patch_size} patches centered around {len(df)} points...")
        
        # Pad the stack to handle points near edges
        # Padding with zeros for normalized features (usually mean-centered)
        padded_stack = np.pad(
            stack, 
            ((half_size, half_size), (half_size, half_size), (0, 0)), 
            mode='constant', 
            constant_values=0
        )
        
        # Fused DEM elevation for target calculation
        padded_fused = np.pad(
            fused_dem,
            ((half_size, half_size), (half_size, half_size)),
            mode='constant',
            constant_values=nodata if nodata is not None else 0
        )

        for _, row_data in tqdm(df.iterrows(), total=len(df), desc="Patch Extraction"):
            r, c_idx = int(row_data['row']), int(row_data['col'])
            
            # Check if center pixel is valid
            if r < 0 or r >= h or c_idx < 0 or c_idx >= w:
                continue
                
            base_elev = fused_dem[r, c_idx]
            if nodata is not None and base_elev == nodata:
                continue
                
            # Extract from padded arrays
            # Center (r, c_idx) in padded corresponds to (r + half_size, c_idx + half_size)
            pr, pc = r + half_size, c_idx + half_size
            
            patch_x = padded_stack[pr - half_size : pr + half_size, pc - half_size : pc + half_size, :]
            
            # Calculate target error: True - Fused
            target_error = row_data['true_elevation'] - base_elev
            
            X_list.append(patch_x.astype(np.float32))
            y_list.append(float(target_error))

        if not X_list:
            print("[!] No valid patches extracted.")
            return None, None

        # 4. Save to Cache
        X_train = np.array(X_list)
        y_train = np.array(y_list)
        
        print(f"Saving {len(X_train)} patches to cache...")
        np.save(out_x_path, X_train)
        np.save(out_y_path, y_train)
        
        print(f"  ✓ X_train shape: {X_train.shape}")
        print(f"  ✓ y_train shape: {y_train.shape}")
        
        return out_x_path, out_y_path
