from dataclasses import dataclass
import hashlib
from shapely.geometry import box


@dataclass(frozen=True)
class AreaOfInterest:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @property
    def bbox(self):
        return (
            self.min_lon,
            self.min_lat,
            self.max_lon,
            self.max_lat,
        )

    @property
    def polygon(self):
        return box(
            self.min_lon,
            self.min_lat,
            self.max_lon,
            self.max_lat,
        )

    @property
    def hash(self):
        s = (
            f"{self.min_lon},"
            f"{self.min_lat},"
            f"{self.max_lon},"
            f"{self.max_lat}"
        )
        return hashlib.md5(s.encode()).hexdigest()
    
    @property
    def utm_epsg(self):

        center_lon = (self.min_lon + self.max_lon) / 2
        center_lat = (self.min_lat + self.max_lat) / 2

        zone = int((center_lon + 180) / 6) + 1

        if center_lat >= 0:
            return 32600 + zone
        else:
            return 32700 + zone

    def __repr__(self):
        return (
            f"AOI("
            f"{self.min_lon}, {self.min_lat}, "
            f"{self.max_lon}, {self.max_lat})"
        )