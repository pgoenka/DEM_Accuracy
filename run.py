from src.core.aoi import AreaOfInterest
from src.core.pipeline import Pipeline

def main():
    print("============================================================")
    print("STARTING ACCURATE DEM PIPELINE")
    print("============================================================")

    aoi = AreaOfInterest(
        77.0,
        28.0,
        77.1,
        28.1,
    )

    pipeline = Pipeline(aoi)
    
    # Let the registry handle the execution automatically!
    pipeline.run()

if __name__ == "__main__":
    main()