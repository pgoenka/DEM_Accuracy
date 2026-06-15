import json
import shutil
import datetime
from pathlib import Path

class Exporter:
   def export(self, context, cache):
        print("\n=== EXPORT ENGINE ===")

        import datetime
        import rasterio
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path

        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        final_dem_path = context.outputs.get("hydro_dem") or context.outputs.get("final_dem") or context.outputs.get("fused_dem")
        conf_map_path = context.outputs.get("confidence_map")

        if not final_dem_path or not final_dem_path.exists():
            print("[!] No final DEM found to export. Did the pipeline complete?")
            return

        out_dem = output_dir / "final_dem.tif"
        import shutil
        shutil.copy2(final_dem_path, out_dem)
        print(f"  ✓ Exported final DEM to: {out_dem}")
        
        if conf_map_path and conf_map_path.exists():
            out_conf = output_dir / "confidence.tif"
            shutil.copy2(conf_map_path, out_conf)
            print(f"  ✓ Exported Confidence Map to: {out_conf}")

        metadata = {
            "project": "Accurate DEM Pipeline",
            "timestamp": datetime.datetime.now().isoformat(),
            "aoi": {"bbox": context.aoi.bbox, "utm_zone": context.aoi.utm_epsg},
            "sources": ["Copernicus DSM 30m", "FABDEM V1.2"],
            "processing_steps": [
                "UTM Reprojection",
                "Nuth & Kääb 3D Co-registration",
                "Geomorphological Feature Extraction (9 channels)",
                "Slope-based Adaptive Fusion"
            ]
        }
        
        if context.outputs.get("final_dem") and context.outputs.get("final_dem").exists():
            metadata["processing_steps"].append("Random Forest LiDAR Refinement (NASA ICESat-2)")
            metadata["sources"].append("NASA ICESat-2 ATL08")
            
        if context.outputs.get("hydro_dem") and context.outputs.get("hydro_dem").exists():
            metadata["processing_steps"].append("Hydrological Conditioning (Fast Depression Breaching)")

        if "feature_importances" in context.outputs:
            metadata["feature_importances"] = context.outputs["feature_importances"]

        out_meta = output_dir / "metadata.json"
        import json
        with open(out_meta, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        print(f"  ✓ Exported Metadata to: {out_meta}")

        print("Generating visual Quality Report PDF...")
        try:
            with rasterio.open(out_dem) as src:
                dem_data = src.read(1)
                # FIX: Aggressively mask out all NoData and ocean-floor artifacts
                dem_data = np.where(dem_data < 0, np.nan, dem_data)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle("DEM Quality & Generation Report", fontsize=16, fontweight='bold')

            # Dynamic color scaling based on real terrain bounds
            vmin_dem = np.nanpercentile(dem_data, 2)
            vmax_dem = np.nanpercentile(dem_data, 98)
            
            im1 = axes[0].imshow(dem_data, cmap='terrain', vmin=vmin_dem, vmax=vmax_dem)
            axes[0].set_title("Final Hydrologically Conditioned DEM")
            axes[0].axis('off')
            fig.colorbar(im1, ax=axes[0], label="Elevation (meters)", fraction=0.046, pad=0.04)

            if conf_map_path and conf_map_path.exists():
                with rasterio.open(conf_map_path) as src_conf:
                    conf_data = src_conf.read(1)
                    conf_data = np.where(conf_data < -500, np.nan, conf_data)
                
                vmin_conf = np.nanpercentile(conf_data, 2)
                vmax_conf = np.nanpercentile(conf_data, 98)
                
                im2 = axes[1].imshow(conf_data, cmap='magma', vmin=vmin_conf, vmax=vmax_conf)
                axes[1].set_title("ML Uncertainty (Confidence Map)")
                axes[1].axis('off')
                fig.colorbar(im2, ax=axes[1], label="Uncertainty (Std Dev in meters)", fraction=0.046, pad=0.04)
            else:
                axes[1].hist(dem_data[~np.isnan(dem_data)], bins=50, color='blue', alpha=0.7)
                axes[1].set_title("Elevation Distribution")
                axes[1].set_xlabel("Elevation (m)")
                axes[1].set_ylabel("Pixel Count")

            plt.tight_layout()
            out_pdf = output_dir / "quality_report.pdf"
            plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
            plt.close()
            print(f"  ✓ Exported Quality Report to: {out_pdf}")

        except Exception as e:
            print(f"  [!] Failed to generate PDF report: {e}")

        print("\n============================================================")
        print("🎉 PHASE 1 COMPLETE! Check the 'output/' folder. 🎉")
        print("============================================================")