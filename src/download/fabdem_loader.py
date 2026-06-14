from pathlib import Path

class FabdemLoader:

    def download_aoi(self, aoi, output_file):
        output_path = Path(output_file)

        if output_path.exists():
            print(f"Already exists: {output_path.name}")
            return output_path

        try:
            import fabdem
            print("Downloading FABDEM via 'fabdem' API...")
            
            # FIXED: Changed 'dest' to 'output_path' to resolve the Pylance parameter error
            fabdem.download(bounds=aoi.bbox, output_path=str(output_path))
            
            print("✓ FABDEM downloaded.")
            return output_path
            
        except ImportError:
            print("\n[!] WARNING: 'fabdem' package not found.")
            print("[!] Please install the package to download real FABDEM data.")
            return None