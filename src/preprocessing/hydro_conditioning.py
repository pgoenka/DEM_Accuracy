import whitebox
import shutil
from pathlib import Path

class HydroConditioner:
    def condition(self, context, cache):
        print("\n=== HYDROLOGICAL CONDITIONING ===")
        
        input_dem = context.outputs.get("final_dem") or context.outputs.get("fused_dem")
        
        if not input_dem or not input_dem.exists():
            print("[!] No DEM found to condition. Skipping.")
            return
            
        hydro_path = cache.path(context.aoi, "hydro_dem.tif")
        
        if hydro_path.exists():
            print("Hydrologically conditioned DEM already exists.")
            context.outputs["hydro_dem"] = hydro_path
            return
            
        try:
            print(f"Applying Fast Depression Breaching to {input_dem.name}...")
            print("This ensures continuous water flow for hydrological modeling.")
            
            wbt = whitebox.WhiteboxTools()
            wbt.set_verbose_mode(False)
            
            # FIX: Convert relative paths to absolute paths so the Rust binary doesn't get lost
            wbt.breach_depressions(
                dem=str(input_dem.resolve()),
                output=str(hydro_path.resolve())
            )
            
            # Verify the Rust binary actually wrote the file
            if not hydro_path.exists():
                raise FileNotFoundError("WhiteboxTools executed but failed to write the output file.")
            
            print("  ✓ Successfully generated Hydrologically Conditioned DEM.")
            context.outputs["hydro_dem"] = hydro_path
            
        except Exception as e:
            print(f"\n[!] WARNING: Hydrology module failed: {e}")
            print("[!] Skipping hydrological conditioning. Exporter will fallback to ML DEM.")
            # We don't set context.outputs["hydro_dem"] so the exporter safely ignores it