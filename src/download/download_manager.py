from src.download.planetary_loader import PlanetaryLoader
from src.download.fabdem_loader import FabdemLoader
from src.download.ground_truth_api import GroundTruthAPI
from src.download.lidar_loader import LidarLoader

class DownloadManager:

    def __init__(self, cache_manager):

        self.cache = cache_manager

        self.planetary = PlanetaryLoader()
        self.fabdem = FabdemLoader()
        self.gt_api = GroundTruthAPI()
        self.lidar = LidarLoader()

    def download_all(self, aoi):

        print("\n=== DOWNLOAD MANAGER ===")

        # Get the raw cache directory
        raw_dir = self.cache.raw_dir(aoi)

        # Download the Copernicus DEM files
        cop_files = self.planetary.download_cop_dem(
            aoi,
            raw_dir
        )

        # Print summary
        print(f"\n✓ Downloaded {len(cop_files)} Copernicus tile(s)")

        # Download Sentinel-2 L2A Bands (Red & NIR) for Phase 2
        print("\n--- SENTINEL-2 (Phase 2) ---")
        s2_files = self.planetary.download_sentinel2_bands(
            aoi,
            raw_dir,
            bands=["B04", "B08"]
        )
        print(f"✓ Downloaded {len(s2_files)} Sentinel-2 band(s)")

        # Download Sentinel-1 RTC Bands (VV & VH) for Phase 2 Radar Intelligence
        print("\n--- SENTINEL-1 SAR (Phase 2) ---")
        s1_files = self.planetary.download_sentinel1_bands(
            aoi,
            raw_dir,
            bands=["vv", "vh"]
        )
        print(f"✓ Downloaded {len(s1_files)} Sentinel-1 band(s)")

        # Download Building Footprints for Phase 2 Urban Intelligence
        print("\n--- BUILDINGS (Phase 2) ---")
        building_files = self.planetary.download_buildings(
            aoi,
            raw_dir
        )
        print(f"✓ Downloaded {len(building_files)} building footprint file(s)")

        print("\n--- FABDEM ---")
        fabdem_raw_path = self.cache.path(aoi, "fabdem_raw_aoi.tif")
        self.fabdem.download_aoi(aoi, fabdem_raw_path)

        # LiDAR Ground Truth
        print("\n--- LiDAR (ICESat-2) ---")
        gt_path = self.cache.path(aoi, "ground_truth.csv")
        self.lidar.fetch(aoi, gt_path)

        # Return structured results
        return {
            "copernicus": cop_files,
            "sentinel2": s2_files,
            "sentinel1": s1_files,
            "buildings": building_files,
            "fabdem": [fabdem_raw_path] if fabdem_raw_path.exists() else [],
            "lidar": [gt_path] if gt_path.exists() else []
        }