import json
import numpy as np
import rasterio

from src.features.terrain import TerrainFeatures
from src.features.normalizer import FeatureNormalizer

class FeatureEngine:

    def __init__(self):

        self.terrain = TerrainFeatures()
        self.normalizer = FeatureNormalizer()

    def build(self, context, cache):

        print("\n=== FEATURE ENGINE ===")

        dem_path = context.raw_dems["utm"]

        # ------------------------------------------------------------------
        # Output paths
        # ------------------------------------------------------------------

        slope_path = cache.path(context.aoi, "slope.tif")
        aspect_path = cache.path(context.aoi, "aspect.tif")
        hillshade_path = cache.path(context.aoi, "hillshade.tif")
        roughness_path = cache.path(context.aoi, "roughness.tif")
        curvature_path = cache.path(context.aoi, "curvature.tif")
        relief_path = cache.path(context.aoi, "relief.tif")
        tri_path = cache.path(context.aoi, "tri.tif")
        tpi_path = cache.path(context.aoi, "tpi.tif")
        ndvi_path = cache.path(context.aoi, "ndvi_utm.tif")
        sar_vv_path = cache.path(context.aoi, "sar_vv_utm.tif")
        sar_vh_path = cache.path(context.aoi, "sar_vh_utm.tif")
        building_mask_path = cache.path(context.aoi, "building_mask.tif")

        stack_path = cache.path(
            context.aoi,
            "feature_stack.npy"
        )

        normalized_path = cache.path(
            context.aoi,
            "feature_stack_normalized.npy"
        )

        stats_path = cache.path(
            context.aoi,
            "feature_stats.json"
        )

        feature_paths = {

            "dem": dem_path,

            "slope": slope_path,

            "aspect": aspect_path,

            "hillshade": hillshade_path,

            "roughness": roughness_path,

            "curvature": curvature_path,

            "relief": relief_path,

            "tri": tri_path,

            "tpi": tpi_path,
            
            "ndvi": ndvi_path,

            "sar_vv": sar_vv_path,

            "sar_vh": sar_vh_path,

            "buildings": building_mask_path,

        }

        required = [

            slope_path,

            aspect_path,

            hillshade_path,

            roughness_path,

            curvature_path,

            relief_path,

            stack_path,

            tri_path,

            tpi_path,
            
            ndvi_path,

            sar_vv_path,

            sar_vh_path,

            building_mask_path,

            normalized_path,

            stats_path,

        ]

        # ------------------------------------------------------------------
        # Cache hit
        # ------------------------------------------------------------------

        if all(path.exists() for path in required):

            print("Features already exist.")

            context.features.update(feature_paths)

            context.features["stack"] = stack_path

            context.features["normalized_stack"] = normalized_path

            context.features["stats"] = stats_path

            context.features["stack_order"] = [

                "dem",

                "slope",

                "aspect",

                "hillshade",

                "roughness",

                "curvature",

                "relief",

                "tri",

                "tpi",
                
                "ndvi",

                "sar_vv",

                "sar_vh",

                "buildings",

            ]

            return

        # ------------------------------------------------------------------
        # Read Data
        # ------------------------------------------------------------------

        with rasterio.open(dem_path) as src:

            dem = src.read(1)

            profile = src.profile
            
        def read_layer(path, label):
            if not path.exists():
                print(f"Warning: {label} path not found. Using zeros.")
                return np.zeros_like(dem, dtype=np.float32)
            with rasterio.open(path) as src:
                return src.read(1).astype(np.float32)

        ndvi = read_layer(ndvi_path, "NDVI")
        sar_vv = read_layer(sar_vv_path, "SAR VV")
        sar_vh = read_layer(sar_vh_path, "SAR VH")
        buildings = read_layer(building_mask_path, "Building Mask")

        # ------------------------------------------------------------------
        # Generate terrain features
        # ------------------------------------------------------------------

        generated = {

            "slope":
                self.terrain.compute_slope(dem),

            "aspect":
                self.terrain.compute_aspect(dem),

            "hillshade":
                self.terrain.compute_hillshade(dem),

            "roughness":
                self.terrain.compute_roughness(dem),

            "curvature":
                self.terrain.compute_curvature(dem),

            "relief":
                self.terrain.compute_local_relief(dem),

            "tri":
                self.terrain.compute_tri(dem),

            "tpi":
                self.terrain.compute_tpi(dem),

        }

        # ------------------------------------------------------------------
        # Save GeoTIFF features
        # ------------------------------------------------------------------

        for name, array in generated.items():

            self.terrain.save(

                array,

                profile,

                feature_paths[name]

            )

        context.features.update(feature_paths)

        # ------------------------------------------------------------------
        # Build feature stack
        # ------------------------------------------------------------------

        stack_layers = [

            dem.astype(np.float32)

        ]

        for name in generated:

            stack_layers.append(

                generated[name].astype(np.float32)

            )
            
        # Add optical, radar, and urban channels
        stack_layers.extend([ndvi, sar_vv, sar_vh, buildings])

        stack = np.stack(

            stack_layers,

            axis=-1

        )

        np.save(

            stack_path,

            stack

        )

        context.features["stack"] = stack_path

        context.features["stack_order"] = [

            "dem",

            *generated.keys(),
            
            "ndvi",

            "sar_vv",

            "sar_vh",

            "buildings"

        ]

        # ------------------------------------------------------------------
        # Normalize
        # ------------------------------------------------------------------

        normalized_stack, stats = self.normalizer.normalize(

            stack

        )

        np.save(

            normalized_path,

            normalized_stack

        )

        context.features["normalized_stack"] = normalized_path

        context.features["normalization"] = stats

        # ------------------------------------------------------------------
        # Save statistics
        # ------------------------------------------------------------------

        with open(

            stats_path,

            "w"

        ) as f:

            json.dump(

                stats,

                f,

                indent=4

            )

        context.features["stats"] = stats_path

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------

        print("\nCreated features:\n")

        for name in generated:

            print(f"  ✓ {name}")
            
        print("  ✓ ndvi")
        print("  ✓ sar_vv")
        print("  ✓ sar_vh")
        print("  ✓ buildings")

        print("\nCreated feature artifacts:\n")

        print("  ✓ feature_stack.npy")

        print("  ✓ feature_stack_normalized.npy")

        print("  ✓ feature_stats.json")