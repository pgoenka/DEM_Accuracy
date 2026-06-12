import rasterio

from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)


class Reprojector:

    def reproject(
        self,
        input_file,
        output_file,
        dst_epsg,
    ):

        with rasterio.open(input_file) as src:

            transform, width, height = calculate_default_transform(
                src.crs,
                f"EPSG:{dst_epsg}",
                src.width,
                src.height,
                *src.bounds,
            )

            kwargs = src.meta.copy()

            kwargs.update(
                {
                    "crs": f"EPSG:{dst_epsg}",
                    "transform": transform,
                    "width": width,
                    "height": height,
                }
            )

            with rasterio.open(output_file, "w", **kwargs) as dst:

                for i in range(1, src.count + 1):

                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=f"EPSG:{dst_epsg}",
                        resampling=Resampling.bilinear,
                    )