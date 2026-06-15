import shutil
from pathlib import Path
from src.core.aoi import AreaOfInterest
from src.core.pipeline import Pipeline

def main():
    print("============================================================")
    print("STARTING ACCURATE DEM PIPELINE")
    print("============================================================")

    # --- PRE-RUN CLEANUP ---
    processed_cache = Path("cache/processed")
    if processed_cache.exists():
        print(f"Clearing processed cache at {processed_cache}...")
        try:
            # We delete the contents but keep the directory structure if possible, 
            # or just wipe and let the pipeline recreate.
            shutil.rmtree(processed_cache)
            processed_cache.mkdir(parents=True, exist_ok=True)
            print("✓ Cache cleared.")
        except Exception as e:
            print(f"Warning: Failed to clear cache: {e}")

    output_dir = Path("output")
    if output_dir.exists():
        print(f"Clearing output directory at {output_dir}...")
        try:
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print("✓ Output directory cleared.")
        except Exception as e:
            print(f"Warning: Failed to clear output directory: {e}")

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