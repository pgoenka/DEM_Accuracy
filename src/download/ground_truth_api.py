import requests
import pandas as pd
import numpy as np
from pathlib import Path

class GroundTruthAPI:
    def fetch(self, aoi, out_file, num_points=100):
        out_path = Path(out_file)
        if out_path.exists():
            print(f"Already exists: {out_path.name}")
            return out_path

        print("\n--- GROUND TRUTH API ---")
        print("Generating geographic coordinate samples within the AOI...")
        
        # Generate random spatial points
        lats = np.random.uniform(aoi.min_lat, aoi.max_lat, num_points)
        lons = np.random.uniform(aoi.min_lon, aoi.max_lon, num_points)

        print(f"Fetching {num_points} real elevation points via REST API...")
        url = "https://api.open-meteo.com/v1/elevation"
        
        params = {
            "latitude": ",".join(map(lambda x: str(round(x, 5)), lats)),
            "longitude": ",".join(map(lambda x: str(round(x, 5)), lons))
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            elevations = response.json().get("elevation")
            df = pd.DataFrame({
                "lat": lats,
                "lon": lons,
                "true_elevation": elevations
            })
            # Drop any points the API couldn't process
            df = df.dropna()
            df.to_csv(out_path, index=False)
            print(f"✓ Saved {len(df)} real API ground truth points to {out_path.name}")
            return out_path
        else:
            print(f"[!] API Request failed: {response.status_code}")
            return None