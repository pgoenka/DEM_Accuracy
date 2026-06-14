import shutil
from pathlib import Path

class Coregistrator:

    def align_nuth_kaab(self, ref_path, tba_path, out_path):
        """
        Aligns a DEM (tba_path) to a reference DEM (ref_path) 
        using the Nuth & Kääb algorithm.
        """
        if Path(out_path).exists():
            print(f"Aligned DEM already exists: {Path(out_path).name}")
            return out_path

        try:
            import xdem
            import geoutils as gu
            
            print(f"Applying Nuth & Kääb 3D alignment to {Path(tba_path).name}...")
            
            # Load the reference DEM (Copernicus) and the DEM To Be Aligned (FABDEM)
            ref_dem = xdem.DEM(str(ref_path))
            tba_dem = xdem.DEM(str(tba_path))
            
            # Initialize and fit the Nuth and Kääb algorithm
            nk = xdem.coreg.NuthKaab()
            nk.fit(ref_dem, tba_dem)
            
            # Apply the transformation to the TBA DEM
            aligned = nk.apply(tba_dem)
            
            # Handle Pylance type-hinting / tuple returns
            # If xdem returns a tuple (e.g., (aligned_dem, transform)), extract the DEM
            if isinstance(aligned, tuple):
                aligned_dem = aligned[0]
            else:
                aligned_dem = aligned
                
            # Save the aligned DEM
            aligned_dem.save(str(out_path))
            
            print("✓ Co-registration complete.")
            return out_path
            
        except ImportError:
            print("\n[!] WARNING: 'xdem' or 'geoutils' package not found.")
            print("[!] Skipping spatial alignment. Using unaligned DEM as fallback.")
            shutil.copy2(tba_path, out_path)
            return out_path