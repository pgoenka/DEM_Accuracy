import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter
from pathlib import Path

class SuperResUpscaler:
    """Upscales 30m DEMs to 10m using Sentinel optical and radar bands as structural guides."""

    def upscale(self, context, cache):
        print("\n=== SUPER-RESOLUTION UPSCALER (10m) ===")
        
        input_path = context.outputs.get("final_dem")
        if not input_path:
            print("[!] No ML-Refined DEM found. Skipping Super-Resolution.")
            return None

        output_path = cache.path(context.aoi, "super_res_dem_10m.tif")
        if output_path.exists():
            print("Super-Resolution DEM already exists.")
            context.outputs["super_res_dem"] = output_path
            return output_path

        # 1. Setup 10m Grid Profile
        with rasterio.open(input_path) as src:
            src_data = src.read(1)
            src_profile = src.profile.copy()
            src_transform = src.transform
            src_crs = src.crs

        # Calculate 10m dimensions (3x resolution)
        dst_width = src.width * 3
        dst_height = src.height * 3
        
        # New transform for 10m (1/3 of the 30m pixel size)
        dst_transform = src_transform * src_transform.scale(
            (src.width / dst_width),
            (src.height / dst_height)
        )

        dst_profile = src_profile.copy()
        dst_profile.update({
            'width': dst_width,
            'height': dst_height,
            'transform': dst_transform
        })

        # 2. Cubic Spline Upsampling (Baseline)
        print(f"Upsampling 30m DEM to 10m grid ({dst_width}x{dst_height})...")
        upsampled_dem = np.empty((dst_height, dst_width), dtype=np.float32)
        
        reproject(
            source=src_data,
            destination=upsampled_dem,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=src_crs,
            resampling=Resampling.cubic
        )

        # 3. Extract 10m Structural Guides
        print("Extracting 10m structural guides (NDVI & SAR)...")
        guide_detail = self._get_structural_detail(context, cache, dst_profile)

        # 4. Inject High-Frequency Details
        # We add the high-frequency structural noise to the smooth cubic-spline baseline
        # Detail intensity is moderated to avoid artifacting
        detail_weight = 0.15 
        final_10m = upsampled_dem + (guide_detail * detail_weight)

        # 5. Save Output
        with rasterio.open(output_path, "w", **dst_profile) as dst:
            dst.write(final_10m.astype(np.float32), 1)

        print(f"✓ Super-Resolution DEM saved to {output_path.name}")
        context.outputs["super_res_dem"] = output_path
        return output_path

    def _get_structural_detail(self, context, cache, dst_profile):
        """Generates a high-pass filtered guide from 10m Sentinel bands."""
        h, w = dst_profile['height'], dst_profile['width']
        combined_guide = np.zeros((h, w), dtype=np.float32)
        count = 0

        # Helper to align raw bands to our 10m grid
        def align_to_10m(file_path):
            if not file_path or not Path(file_path).exists():
                return None
            aligned = np.empty((h, w), dtype=np.float32)
            with rasterio.open(file_path) as src:
                reproject(
                    source=src.read(1),
                    destination=aligned,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_profile['transform'],
                    dst_crs=dst_profile['crs'],
                    resampling=Resampling.bilinear,
                    src_nodata=0
                )
            return aligned

        # Fetch native 10m bands
        s2_files = context.satellite.get("sentinel2", [])
        s1_files = context.satellite.get("sentinel1", [])

        # NDVI Guide
        b04 = next((f for f in s2_files if "B04" in Path(f).name), None)
        b08 = next((f for f in s2_files if "B08" in Path(f).name), None)
        
        if b04 and b08:
            red = align_to_10m(b04)
            nir = align_to_10m(b08)
            denom = nir + red
            ndvi = np.where(denom == 0, 0, (nir - red) / denom)
            combined_guide += (ndvi - np.mean(ndvi)) / (np.std(ndvi) + 1e-6)
            count += 1

        # SAR Guide (VH often shows more structural texture)
        vh_path = next((f for f in s1_files if "vh" in Path(f).name.lower()), None)
        if vh_path:
            vh = align_to_10m(vh_path)
            # Log transform SAR for better feature extraction
            vh_log = np.log1p(np.maximum(vh, 0))
            combined_guide += (vh_log - np.mean(vh_log)) / (np.std(vh_log) + 1e-6)
            count += 1

        if count == 0:
            return np.zeros((h, w), dtype=np.float32)

        # Average the guides
        combined_guide /= count

        # High-pass filter: Image - Blurred Image = High-Frequency Details
        # sigma=1 captures the sharpest transitions (roads, riverbanks)
        low_freq = gaussian_filter(combined_guide, sigma=1.0)
        high_freq = combined_guide - low_freq

        return high_freq
