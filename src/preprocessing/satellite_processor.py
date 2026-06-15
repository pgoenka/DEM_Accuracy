import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path

class NDVIProcessor:
    """Processor to calculate NDVI from Sentinel-2 Red and NIR bands and align it to the DEM grid."""

    def compute_and_align(self, context, cache, target_dem_path):
        """
        Calculates NDVI from raw Sentinel-2 bands, reprojects and aligns it
        exactly to the grid of the target UTM DEM, and saves it.
        """
        ndvi_utm_path = cache.path(context.aoi, "ndvi_utm.tif")

        if ndvi_utm_path.exists():
            print("Aligned NDVI already exists.")
            return ndvi_utm_path

        s2_files = context.satellite.get("sentinel2", [])
        if not s2_files:
            # Look in cache directly as fallback
            raw_dir = cache.raw_dir(context.aoi) / "sentinel2"
            s2_files = list(raw_dir.glob("*.tif"))

        b04_path = None
        b08_path = None

        for f in s2_files:
            p = Path(f)
            if "B04" in p.name:
                b04_path = p
            elif "B08" in p.name:
                b08_path = p

        if not b04_path or not b08_path:
            raise ValueError("Sentinel-2 B04 (Red) or B08 (NIR) band missing from context or cache.")

        print(f"Calculating NDVI using:\n  Red: {b04_path.name}\n  NIR: {b08_path.name}")

        # Read the target DEM's profile to align exactly with its grid
        with rasterio.open(target_dem_path) as dem_src:
            target_profile = dem_src.profile.copy()
            target_crs = dem_src.crs
            target_transform = dem_src.transform
            target_width = dem_src.width
            target_height = dem_src.height

        # Process NDVI in memory from raw bands (which are typically in the same UTM zone/CRS)
        with rasterio.open(b04_path) as red_src, rasterio.open(b08_path) as nir_src:
            red = red_src.read(1).astype(np.float32)
            nir = nir_src.read(1).astype(np.float32)

            # Prevent division by zero
            denom = nir + red
            ndvi = np.where(denom == 0, 0, (nir - red) / denom)
            
            # Bound NDVI to [-1, 1]
            ndvi = np.clip(ndvi, -1.0, 1.0)

            # Store source profile for reprojection
            src_crs = red_src.crs
            src_transform = red_src.transform

        print(f"Reprojecting and aligning NDVI to match DEM grid ({target_width}x{target_height})...")

        # Set up output profile matching the target DEM exactly, except for data type
        target_profile.update(
            dtype="float32",
            count=1,
            nodata=-9999.0
        )

        # Reproject NDVI to match DEM grid
        aligned_ndvi = np.full((target_height, target_width), -9999.0, dtype=np.float32)

        with rasterio.open(ndvi_utm_path, "w", **target_profile) as dst:
            reproject(
                source=ndvi,
                destination=aligned_ndvi,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
                src_nodata=0,  # Sentinel-2 uses 0 as nodata
                dst_nodata=-9999.0
            )
            dst.write(aligned_ndvi, 1)

        print(f"✓ Aligned NDVI saved to {ndvi_utm_path.name}")
        return ndvi_utm_path

class SARProcessor:
    """Processor to reproject and align Sentinel-1 RTC SAR bands to the DEM grid."""

    def align_sar(self, context, cache, target_dem_path):
        """
        Reprojects and aligns Sentinel-1 RTC bands (VV, VH)
        exactly to the grid of the target UTM DEM and saves them.
        """
        sar_vv_path = cache.path(context.aoi, "sar_vv_utm.tif")
        sar_vh_path = cache.path(context.aoi, "sar_vh_utm.tif")

        if sar_vv_path.exists() and sar_vh_path.exists():
            print("Aligned SAR bands already exist.")
            return sar_vv_path, sar_vh_path

        s1_files = context.satellite.get("sentinel1", [])
        if not s1_files:
            # Look in cache directly as fallback
            raw_dir = cache.raw_dir(context.aoi) / "sentinel1"
            s1_files = list(raw_dir.glob("*.tif"))

        vv_src_path = None
        vh_src_path = None

        for f in s1_files:
            p = Path(f)
            if "_vv" in p.name.lower():
                vv_src_path = p
            elif "_vh" in p.name.lower():
                vh_src_path = p

        if not vv_src_path or not vh_src_path:
            raise ValueError("Sentinel-1 VV or VH band missing from context or cache.")

        # Read the target DEM's profile to align exactly with its grid
        with rasterio.open(target_dem_path) as dem_src:
            target_profile = dem_src.profile.copy()
            target_crs = dem_src.crs
            target_transform = dem_src.transform
            target_width = dem_src.width
            target_height = dem_src.height

        target_profile.update(
            dtype="float32",
            count=1,
            nodata=-9999.0
        )

        for src_path, out_path, label in [(vv_src_path, sar_vv_path, "VV"), (vh_src_path, sar_vh_path, "VH")]:
            print(f"Reprojecting and aligning SAR {label} band to DEM grid...")
            
            with rasterio.open(src_path) as src:
                data = src.read(1).astype(np.float32)
                src_crs = src.crs
                src_transform = src.transform
                src_nodata = src.nodata or 0

            aligned_data = np.full((target_height, target_width), -9999.0, dtype=np.float32)

            with rasterio.open(out_path, "w", **target_profile) as dst:
                reproject(
                    source=data,
                    destination=aligned_data,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src_nodata,
                    dst_nodata=-9999.0
                )
                dst.write(aligned_data, 1)
            
            print(f"✓ Aligned SAR {label} saved to {out_path.name}")

        return sar_vv_path, sar_vh_path
