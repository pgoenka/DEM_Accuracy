import sys
import io
# Force UTF-8 output on Windows to avoid crashes from Unicode characters (e.g. ✓)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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
        85.83,
        27.65,
        86.55,
        28.1,
    )


    pipeline = Pipeline(aoi)
    
    # Let the registry handle the execution automatically!
    pipeline.run()

if __name__ == "__main__":
    main()