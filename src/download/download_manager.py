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
        files = self.planetary.download_cop_dem(
            aoi,
            raw_dir
        )

        # Print summary
        print(f"\n✓ Downloaded {len(files)} Copernicus tile(s)\n")

        print("Downloaded files:")

        for f in files:
            print(f)

        print("\n--- FABDEM ---")
        fabdem_raw_path = self.cache.path(aoi, "fabdem_raw_aoi.tif")
        self.fabdem.download_aoi(aoi, fabdem_raw_path)

        # Replace the old gt_api line with this:
        gt_path = self.cache.path(aoi, "ground_truth.csv")
        self.lidar.fetch(aoi, gt_path)

        # Return the list so Pipeline can use it
        return files