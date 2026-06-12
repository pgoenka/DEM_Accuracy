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

        }

        required = [

            slope_path,

            aspect_path,

            hillshade_path,

            roughness_path,

            curvature_path,

            relief_path,

            stack_path,

            normalized_path,

            stats_path,

        ]

        # ------------------------------------------------------------------
        # Cache hit
        # ------------------------------------------------------------------

        if all(path.exists() for path in required):

            print("Terrain features already exist.")

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

            ]

            return

        # ------------------------------------------------------------------
        # Read DEM
        # ------------------------------------------------------------------

        with rasterio.open(dem_path) as src:

            dem = src.read(1)

            profile = src.profile

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

            *generated.keys()

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

        print("\nCreated terrain features:\n")

        for name in generated:

            print(f"  ✓ {name}")

        print("\nCreated feature artifacts:\n")

        print("  ✓ feature_stack.npy")

        print("  ✓ feature_stack_normalized.npy")

        print("  ✓ feature_stats.json")