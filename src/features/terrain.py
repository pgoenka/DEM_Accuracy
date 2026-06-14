import numpy as np
import rasterio

from scipy.ndimage import generic_gradient_magnitude
from scipy.ndimage import sobel
from scipy.ndimage import maximum_filter
from scipy.ndimage import minimum_filter
from scipy.ndimage import uniform_filter
from scipy.ndimage import laplace

class TerrainFeatures:

    def compute_slope(self, dem):

        dx = sobel(dem, axis=1)

        dy = sobel(dem, axis=0)

        slope = np.degrees(
            np.arctan(
                np.sqrt(dx**2 + dy**2)
            )
        )

        return slope

    def compute_aspect(self, dem):

        dx = sobel(dem, axis=1)

        dy = sobel(dem, axis=0)

        aspect = np.degrees(
            np.arctan2(
                -dx,
                dy
            )
        )

        return aspect

    def save(self, array, profile, output):

        profile = profile.copy()

        profile.update(dtype="float32")

        with rasterio.open(output, "w", **profile) as dst:

            dst.write(
                array.astype("float32"),
                1
            )

    def compute_curvature(self, dem):

        return laplace(dem)
    
    def compute_hillshade(self, dem):

        dx = sobel(dem, axis=1)

        dy = sobel(dem, axis=0)

        slope = np.pi / 2 - np.arctan(
            np.sqrt(dx * dx + dy * dy)
        )

        aspect = np.arctan2(
            -dx,
            dy
        )

        azimuth = np.radians(315)

        altitude = np.radians(45)

        shaded = (
            np.sin(altitude) * np.sin(slope)
            + np.cos(altitude)
            * np.cos(slope)
            * np.cos(azimuth - aspect)
        )

        return 255 * shaded

    def compute_roughness(self, dem):

        return (
            maximum_filter(dem, size=3)
            - minimum_filter(dem, size=3)
        )
    
    def compute_local_relief(self, dem):

        mean = uniform_filter(
            dem,
            size=15
        )

        return dem - mean

    def compute_tri(self, dem):
        # We use local standard deviation as a high-performance proxy for TRI.
        # This highlights highly rugged, oscillating terrain using variance.
        mean_sq = uniform_filter(dem**2, size=3)
        sq_mean = uniform_filter(dem, size=3)**2
        return np.sqrt(np.maximum(mean_sq - sq_mean, 0))

    def compute_tpi(self, dem):
        # Topographic Position Index (TPI)
        # Positive values = ridges/hills. Negative values = valleys/pits.
        return dem - uniform_filter(dem, size=3)