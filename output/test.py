import rasterio
import numpy as np

threshold = 130

with rasterio.open("final_dem.tif") as src:
    dem = src.read(1)
    profile = src.profile

    nodata = -9999
    profile.update(nodata=nodata)

    out = np.where(dem >= threshold, dem, nodata)

    with rasterio.open("output_dem.tif", "w", **profile) as dst:
        dst.write(out, 1)