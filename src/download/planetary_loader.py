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

    def download_cop_dem(self, aoi, output_dir):

        items = self.search_cop_dem(aoi)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded = []

        for item in items:

            asset = item.assets["data"]

            url = asset.href

            filename = output_dir / f"{item.id}.tif"

            if filename.exists():

                print(f"Already exists: {filename.name}")

                downloaded.append(filename)

                continue

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

            downloaded.append(filename)

        return downloaded