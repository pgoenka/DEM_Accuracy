import numpy as np
import rasterio
from rasterio.features import rasterize
from pathlib import Path
import geopandas as gpd

class UrbanProcessor:
    """Processor to rasterize building footprints into a binary mask aligned with the DEM grid."""

    def process_buildings(self, context, cache, target_dem_path):
        """
        Loads building footprint vector data, reprojects it to match the DEM,
        and rasterizes it into a binary mask.
        """
        building_mask_path = cache.path(context.aoi, "building_mask.tif")

        if building_mask_path.exists():
            print("Building mask already exists.")
            return building_mask_path

        building_files = context.satellite.get("buildings", [])
        if not building_files:
            # Look in cache directly as fallback
            raw_dir = cache.raw_dir(context.aoi) / "buildings"
            # Support both parquet and geojson depending on what was downloaded
            building_files = list(raw_dir.glob("*.parquet")) + list(raw_dir.glob("*.geojson"))

        if not building_files:
            print("Warning: No building footprint files found. Creating empty mask.")
            self._create_empty_mask(target_dem_path, building_mask_path)
            return building_mask_path

        print(f"Rasterizing building footprints from {len(building_files)} file(s)...")

        # Read the target DEM's profile to align exactly with its grid
        with rasterio.open(target_dem_path) as dem_src:
            target_profile = dem_src.profile.copy()
            target_crs = dem_src.crs
            target_transform = dem_src.transform
            target_width = dem_src.width
            target_height = dem_src.height

        # Load all building footprints
        gdfs = []
        for f in building_files:
            p = Path(f)
            try:
                if p.suffix == '.parquet':
                    gdf = gpd.read_parquet(p)
                elif p.suffix == '.geojson':
                    # Parse the custom Overpass JSON structure we saved
                    import json
                    with open(p, 'r') as fp:
                        data = json.load(fp)
                    
                    # Convert nodes to a lookup dict
                    nodes = {node['id']: (node['lon'], node['lat']) for node in data.get('elements', []) if node['type'] == 'node'}
                    
                    from shapely.geometry import Polygon
                    polygons = []
                    for el in data.get('elements', []):
                        if el['type'] == 'way' and 'nodes' in el:
                            coords = [nodes[nid] for nid in el['nodes'] if nid in nodes]
                            if len(coords) >= 3:
                                polygons.append(Polygon(coords))
                    
                    if polygons:
                        gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
                    else:
                        gdf = gpd.GeoDataFrame()
                else:
                    continue

                if not gdf.empty:
                    # Clip to AOI bbox (in 4326 usually) just to be sure
                    if gdf.crs == "EPSG:4326":
                        gdf = gdf.cx[context.aoi.min_lon:context.aoi.max_lon, context.aoi.min_lat:context.aoi.max_lat]
                    
                    if not gdf.empty:
                        gdfs.append(gdf)
            except Exception as e:
                print(f"Error reading building file {p.name}: {e}")

        if not gdfs:
            print("No buildings found within AOI. Creating empty mask.")
            self._create_empty_mask(target_dem_path, building_mask_path)
            return building_mask_path

        combined_gdf = gpd.GeoDataFrame(np.concatenate([gdf.geometry.values for gdf in gdfs]), columns=['geometry'], crs=gdfs[0].crs)
        
        # Reproject to target UTM CRS
        print(f"Reprojecting buildings to {target_crs}...")
        combined_gdf = combined_gdf.to_crs(target_crs)

        # Rasterize
        print("Creating binary building mask...")
        shapes = [(geom, 1) for geom in combined_gdf.geometry if geom.is_valid]
        
        if shapes:
            mask = rasterize(
                shapes=shapes,
                out_shape=(target_height, target_width),
                transform=target_transform,
                fill=0,
                all_touched=True,
                dtype='uint8'
            )
        else:
             mask = np.zeros((target_height, target_width), dtype='uint8')

        # Save mask
        target_profile.update(
            dtype='uint8',
            count=1,
            nodata=0,
            compress='lzw'
        )

        with rasterio.open(building_mask_path, "w", **target_profile) as dst:
            dst.write(mask, 1)

        print(f"✓ Building mask saved to {building_mask_path.name}")
        return building_mask_path

    def _create_empty_mask(self, target_dem_path, output_path):
        with rasterio.open(target_dem_path) as src:
            profile = src.profile.copy()
            shape = (src.height, src.width)
        
        profile.update(dtype='uint8', count=1, nodata=0, compress='lzw')
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(np.zeros(shape, dtype='uint8'), 1)
