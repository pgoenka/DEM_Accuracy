import earthaccess
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv

class LidarLoader:
    def fetch(self, aoi, out_file):
        out_path = Path(out_file)
        if out_path.exists():
            print(f"Already exists: {out_path.name}")
            return out_path

        print("\n--- NASA LiDAR (ICESat-2) ---")
        
        # Load environment variables from .env file
        load_dotenv()
        
        print("Authenticating with NASA Earthdata...")
        try:
            if os.getenv("EARTHDATA_USERNAME") and os.getenv("EARTHDATA_PASSWORD"):
                print("Using credentials from environment variables.")
                earthaccess.login(strategy="environment")
            else:
                # This will prompt for credentials in the terminal on the first run
                # and automatically save them to ~/.netrc for future runs.
                print("No credentials found in .env file. Falling back to interactive login.")
                earthaccess.login(strategy="interactive") 
        except Exception as e:
            print(f"[!] Earthdata Login Failed: {e}")
            print("[!] Please create a free account at https://urs.earthdata.nasa.gov/")
            return None

        print(f"Searching for ICESat-2 ATL08 (Land/Vegetation) granules over AOI...")
        results = earthaccess.search_data(
            short_name="ATL08",
            bounding_box=(aoi.min_lon, aoi.min_lat, aoi.max_lon, aoi.max_lat),
            count=2 # We pull the 2 most recent intersecting orbital tracks
        )

        if not results:
            print("[!] No ICESat-2 orbital tracks found for this specific AOI.")
            return None

        print(f"Found {len(results)} orbital passes. Downloading HDF5 files...")
        download_dir = Path("cache/raw/lidar")
        download_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the files via Earthdata API
        downloaded_files = earthaccess.download(results, local_path=str(download_dir))
        
        print("Parsing HDF5 laser tracks for true bare-earth elevation...")
        dfs = []
        # ICESat-2 fires 6 laser beams simultaneously
        beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
        
        for file_path in downloaded_files:
            try:
                with h5py.File(file_path, 'r') as h5:
                    for beam in beams:
                        if beam in h5:
                            beam_group = h5[beam]
                            
                            # Safely assure Pylance that this object is a Group, not a Dataset
                            if isinstance(beam_group, h5py.Group) and 'land_segments' in beam_group:
                                land_segments = beam_group['land_segments']
                                
                                if isinstance(land_segments, h5py.Group) and 'terrain' in land_segments:
                                    terrain = land_segments['terrain']
                                    
                                    if isinstance(terrain, h5py.Group):
                                        lats = np.array(land_segments['latitude'])
                                        lons = np.array(land_segments['longitude'])
                                        elevs = np.array(terrain['h_te_best_fit'])
                                        
                                        # Filter out invalid measurements/cloud hits
                                        valid_mask = elevs < 10000 
                                        
                                        df_beam = pd.DataFrame({
                                            "lat": lats[valid_mask],
                                            "lon": lons[valid_mask],
                                            "true_elevation": elevs[valid_mask]
                                        })
                                        
                                        # Clip the laser tracks strictly to our AOI bounding box
                                        df_beam = df_beam[
                                            (df_beam['lat'] >= aoi.min_lat) & (df_beam['lat'] <= aoi.max_lat) &
                                            (df_beam['lon'] >= aoi.min_lon) & (df_beam['lon'] <= aoi.max_lon)
                                        ]
                                        
                                        if not df_beam.empty:
                                            dfs.append(df_beam)
            except Exception as e:
                print(f"  [!] Error parsing {Path(file_path).name}: {e}")
                
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            # Remove any exact duplicate spatial points
            final_df = final_df.drop_duplicates(subset=['lat', 'lon'])
            final_df.to_csv(out_path, index=False)
            print(f"  ✓ Extracted {len(final_df)} highly-accurate ICESat-2 points to {out_path.name}")
            return out_path
        else:
            print("[!] Laser tracks missed the AOI or hit heavy cloud cover. No valid points extracted.")
            return None