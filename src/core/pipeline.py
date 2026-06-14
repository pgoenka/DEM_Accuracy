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
from src.fusion.fusion_engine import FusionEngine
from src.preprocessing.coregistration import Coregistrator
from src.ml.predictor import MLEngine
from src.export.exporter import Exporter
from src.preprocessing.hydro_conditioning import HydroConditioner

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
        self.fusion_engine = FusionEngine()
        self.coregistrator = Coregistrator()
        self.ml_engine = MLEngine()
        self.exporter = Exporter()
        self.hydro_conditioner = HydroConditioner()
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
                "Refinement",
                self.refine,
            )
        )

        self.registry.add(
            PipelineStage(
                "Hydrology", 
                self.condition,
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

        # ----------------------------
        # Step 3: Prepare FABDEM
        # ----------------------------
        print("\n--- Processing FABDEM ---")
        fabdem_raw = self.cache.path(self.context.aoi, "fabdem_raw_aoi.tif")
        fabdem_utm = self.cache.path(self.context.aoi, "fabdem_utm.tif")

        if fabdem_utm.exists():
            print("Projected FABDEM already exists.")
        else:
            if fabdem_raw.exists():
                print("Reprojecting raw FABDEM...")
                self.reprojector.reproject(
                    fabdem_raw,
                    fabdem_utm,
                    self.context.aoi.utm_epsg,
                )
                print("Created projected FABDEM.")
            else:
                import shutil
                print("Raw FABDEM not found. Using Copernicus as a placeholder to allow pipeline to continue.")
                shutil.copy2(utm_dem, fabdem_utm)
                print("Created placeholder FABDEM.")

        # Register the UTM FABDEM in the context
        self.context.raw_dems["fabdem_utm"] = fabdem_utm

        # ----------------------------
        # Step 4: DEM Co-registration
        # ----------------------------
        print("\n--- Co-registration ---")
        
        fabdem_aligned = self.cache.path(self.context.aoi, "fabdem_aligned.tif")
        
        self.coregistrator.align_nuth_kaab(
            ref_path=self.context.raw_dems["utm"],        # Copernicus is the reference
            tba_path=self.context.raw_dems["fabdem_utm"], # FABDEM is to be aligned
            out_path=fabdem_aligned
        )
        
        # Register the aligned DEM in the context for the Fusion engine
        self.context.aligned_dems["fabdem"] = fabdem_aligned
        
    def features(self):

        self.feature_engine.build(
            self.context,
            self.cache
        )

    def fuse(self):
        self.fusion_engine.fuse(self.context, self.cache)

    def condition(self):
        self.hydro_conditioner.condition(self.context, self.cache)

    def refine(self):
        self.ml_engine.refine_dem(self.context, self.cache)

    def export(self):
        self.exporter.export(self.context, self.cache)