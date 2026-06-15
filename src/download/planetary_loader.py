import requests
from pathlib import Path
from tqdm import tqdm
import planetary_computer
import pystac_client


class PlanetaryLoader:

    def __init__(self):

        self.catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

    def search_cop_dem(self, aoi):

        search = self.catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=aoi.bbox,
        )

        return list(search.items())

    def search_sentinel2(self, aoi, datetime="2023-01-01/2023-12-31"):
        """Search for Sentinel-2 L2A items with low cloud cover."""
        search = self.catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=aoi.bbox,
            datetime=datetime,
            query={"eo:cloud_cover": {"lt": 10}},
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}]
        )

        return list(search.items())

    def search_sentinel1_rtc(self, aoi, datetime="2023-01-01/2023-12-31"):
        """Search for Sentinel-1 RTC items."""
        search = self.catalog.search(
            collections=["sentinel-1-rtc"],
            bbox=aoi.bbox,
            datetime=datetime,
        )

        return list(search.items())

    def search_ms_buildings(self, aoi):
        """Search for Microsoft Building Footprints."""
        search = self.catalog.search(
            collections=["ms-buildings"],
            bbox=aoi.bbox,
        )
        return list(search.items())

    def _download_url(self, url, filename):
        """Helper to download a file with a progress bar."""
        if filename.exists():
            print(f"Already exists: {filename.name}")
            return True

        # Check if URL is Azure Blob Storage (abfs://)
        if url.startswith("abfs://"):
            # We handle this via specialized loaders like geopandas for buildings
            return False

        print(f"Downloading {filename.name}")
        r = requests.get(url, stream=True)
        r.raise_for_status()

        total_size = int(r.headers.get("content-length", 0))
        chunk_size = 1024 * 1024  # 1 MB

        with open(filename, "wb") as f, tqdm(
            desc=filename.name,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
        return True

    def download_cop_dem(self, aoi, output_dir):

        items = self.search_cop_dem(aoi)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded = []

        for item in items:
            asset = item.assets["data"]
            url = asset.href
            filename = output_dir / f"{item.id}.tif"
            
            if self._download_url(url, filename):
                downloaded.append(filename)

        return downloaded

    def download_sentinel2_bands(self, aoi, output_dir, bands=["B04", "B08"]):
        """Download specific bands for the best Sentinel-2 item found."""
        items = self.search_sentinel2(aoi)
        
        if not items:
            print("No Sentinel-2 items found matching criteria.")
            return []

        # Select the item with lowest cloud cover (first in sorted list)
        item = items[0]
        print(f"Selected Sentinel-2 item: {item.id} (Cloud Cover: {item.properties['eo:cloud_cover']}%)")

        output_dir = Path(output_dir) / "sentinel2"
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded = []

        for band in bands:
            if band not in item.assets:
                print(f"Warning: Band {band} not found in item {item.id}")
                continue
            
            asset = item.assets[band]
            url = asset.href
            filename = output_dir / f"{item.id}_{band}.tif"

            if self._download_url(url, filename):
                downloaded.append(filename)

        return downloaded

    def download_sentinel1_bands(self, aoi, output_dir, bands=["vv", "vh"]):
        """Download specific bands for the first Sentinel-1 item found."""
        items = self.search_sentinel1_rtc(aoi)

        if not items:
            print("No Sentinel-1 RTC items found.")
            return []

        # Select the first available item
        item = items[0]
        print(f"Selected Sentinel-1 item: {item.id}")

        output_dir = Path(output_dir) / "sentinel1"
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded = []

        for band in bands:
            if band not in item.assets:
                print(f"Warning: Band {band} not found in item {item.id}")
                continue

            asset = item.assets[band]
            url = asset.href
            filename = output_dir / f"{item.id}_{band}.tif"

            if self._download_url(url, filename):
                downloaded.append(filename)

        return downloaded

    def download_buildings(self, aoi, output_dir):
        """
        Download building footprints for the AOI.
        Uses a lightweight query to OpenStreetMap (Overpass API) as a robust,
        fast alternative to multi-gigabyte country-wide datasets.
        """
        output_dir = Path(output_dir) / "buildings"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / "buildings_osm.geojson"

        if filename.exists():
            print(f"Already exists: {filename.name}")
            return [filename]

        print(f"Querying Overpass API for buildings in AOI...")
        import requests
        import json

        # Overpass QL query for buildings in the bbox
        # bbox format: south, west, north, east
        overpass_url = "https://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json][timeout:25];
        (
          way["building"]({aoi.min_lat},{aoi.min_lon},{aoi.max_lat},{aoi.max_lon});
          relation["building"]({aoi.min_lat},{aoi.min_lon},{aoi.max_lat},{aoi.max_lon});
        );
        out body;
        >;
        out skel qt;
        """

        try:
            response = requests.post(overpass_url, data={'data': overpass_query})
            response.raise_for_status()
            data = response.json()

            if not data.get('elements'):
                print("No buildings found in this AOI.")
                return []

            # Convert Overpass JSON to a simple GeoJSON-like structure that GeoPandas can read
            # Note: For simplicity and speed, we just save the JSON and let GeoPandas handle it.
            with open(filename, 'w') as f:
                json.dump(data, f)

            print(f"✓ Found and saved building data to {filename.name}")
            return [filename]

        except Exception as e:
            print(f"Error fetching buildings from Overpass: {e}")
            return []