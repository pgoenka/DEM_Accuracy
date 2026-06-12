from pathlib import Path

import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds


class CopernicusProcessor:

    def merge_and_crop(self, input_files, aoi, output_file):

        print("\n=== COPERNICUS PROCESSOR ===")

        datasets = [rasterio.open(f) for f in input_files]

        print(f"Merging {len(datasets)} tile(s)...")

        mosaic, transform = merge(datasets)

        profile = datasets[0].profile.copy()

        profile.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
            }
        )

        # Write merged raster temporarily
        temp_file = Path(output_file).with_suffix(".tmp.tif")

        with rasterio.open(temp_file, "w", **profile) as dst:
            dst.write(mosaic)

        # Re-open merged raster
        with rasterio.open(temp_file) as src:

            window = from_bounds(
                *aoi.bbox,
                transform=src.transform
            )

            data = src.read(window=window)

            transform = src.window_transform(window)

            profile = src.profile.copy()

            profile.update(
                {
                    "height": data.shape[1],
                    "width": data.shape[2],
                    "transform": transform,
                }
            )

        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(data)

        temp_file.unlink()

        for ds in datasets:
            ds.close()

        print("AOI DEM created.")

        return output_file