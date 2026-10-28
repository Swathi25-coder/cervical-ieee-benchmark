import kagglehub
import shutil
from pathlib import Path

def download_data():
    print("--- DOWNLOADING HERLEV DATASET ---")
    try:
        herlev_path = kagglehub.dataset_download("yuvrajchandravanshi/herlev-dataset")
        print(f"Herlev downloaded to: {herlev_path}")
        
        # Link or copy to data/herlev
        src = Path(herlev_path)
        dest = Path("data/herlev")
        dest.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            target = dest / item.name
            if not target.exists():
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
        print("Herlev successfully moved into data/herlev/")
    except Exception as e:
        print(f"Error downloading Herlev: {e}")

if __name__ == "__main__":
    download_data()
