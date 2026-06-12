from src.cache.cache_manager import CacheManager
from src.core.context import PipelineContext
from src.download.download_manager import DownloadManager
from src.preprocessing.copernicus_processor import CopernicusProcessor
from src.preprocessing.reproject import Reprojector
from pathlib import Path
from src.features.terrain import TerrainFeatures
import rasterio
from src.features.feature_engine import FeatureEngine
from src.core.pipeline_registry import PipelineRegistry
from src.core.stage import PipelineStage

class Pipeline:

    def __init__(self, aoi):

        self.context = PipelineContext(aoi)
        self.cache = CacheManager()
        self.download_manager = DownloadManager(self.cache)
        self.cop_processor = CopernicusProcessor()
        self.reprojector = Reprojector()
        self.terrain = TerrainFeatures()
        self.feature_engine = FeatureEngine()
        self.registry = PipelineRegistry()
        self.registry.add(
            PipelineStage(
                "Download",
                self.download,
            )
        )

        self.registry.add(
            PipelineStage(
                "Preprocess",
                self.preprocess,
            )
        )

        self.registry.add(
            PipelineStage(
                "Features",
                self.features,
            )
        )

        self.registry.add(
            PipelineStage(
                "Fusion",
                self.fuse,
            )
        )

        self.registry.add(
            PipelineStage(
                "Export",
                self.export,
            )
        )

    def run(self):

        print("Pipeline started")

        self.registry.run()

    def download(self):

        print("\n=== DOWNLOAD ===")

        self.download_manager.download_all(
            self.context.aoi
        )

    def preprocess(self):
        print("\n=== PREPROCESS ===")

        # ----------------------------
        # Step 1: Merge raw Copernicus tiles
        # ----------------------------

        raw_dir = self.cache.raw_dir(
            self.context.aoi
        )

        input_files = sorted(
            raw_dir.glob("*.tif")
        )

        cop_dem = self.cache.path(
            self.context.aoi,
            "cop_dem_aoi.tif"
        )

        if cop_dem.exists():

            print("AOI DEM already exists.")

        else:

            self.cop_processor.merge_and_crop(
                input_files,
                self.context.aoi,
                cop_dem
            )

        # Save into context
        self.context.raw_dems["copernicus"] = cop_dem

        # ----------------------------
        # Step 2: Reproject
        # ----------------------------

        utm_dem = self.cache.path(
            self.context.aoi,
            "cop_dem_utm.tif"
        )

        if utm_dem.exists():

            print("Projected DEM already exists.")

        else:

            self.reprojector.reproject(
                cop_dem,
                utm_dem,
                self.context.aoi.utm_epsg,
            )

            print("Created projected DEM.")

        self.context.raw_dems["utm"] = utm_dem
        
    def features(self):

        self.feature_engine.build(
            self.context,
            self.cache
        )

    def fuse(self):
        print("Fusion stage")

    def export(self):
        print("Export stage")