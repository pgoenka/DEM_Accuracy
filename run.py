from src.core.aoi import AreaOfInterest
from src.core.pipeline import Pipeline

aoi = AreaOfInterest(
    77.0,
    28.0,
    77.1,
    28.1,
)

pipeline = Pipeline(aoi)

pipeline.run()