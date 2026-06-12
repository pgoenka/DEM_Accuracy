from src.core.aoi import AreaOfInterest
from src.download.planetary_loader import PlanetaryLoader

aoi = AreaOfInterest(
    77.0,
    28.0,
    77.1,
    28.1,
)

loader = PlanetaryLoader()

items = loader.search_cop_dem(aoi)

print(f"Found {len(items)} items")

for item in items:
    print(item.id)