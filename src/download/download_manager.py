from src.download.planetary_loader import PlanetaryLoader


class DownloadManager:

    def __init__(self, cache_manager):

        self.cache = cache_manager

        self.planetary = PlanetaryLoader()

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

        # Return the list so Pipeline can use it
        return files