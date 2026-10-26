import os
from pathlib import Path
from PIL import Image
import numpy as np

def create_samples():
    print("Generating synthetic sample images for pipeline smoke testing...")
    
    # SIPaKMeD categories
    sipak_classes = ["dyskeratotic", "koilocytotic", "metaplastic", "parabasal", "superficial"]
    for cls in sipak_classes:
        dir_path = Path(f"data/sipakmed/Cropped/{cls}")
        dir_path.mkdir(parents=True, exist_ok=True)
        for slide_id in range(1, 4):  # 3 slides
            for cell_id in range(1, 3):  # 2 crops per slide
                fname = dir_path / f"{slide_id:03d}_{cell_id:02d}.bmp"
                if not fname.exists():
                    img = Image.fromarray(np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8))
                    img.save(fname)
                    
    # Herlev categories
    herlev_classes = ["normal_superficiel", "carcinoma_in_situ", "light_dysplastic"]
    for cls in herlev_classes:
        dir_path = Path(f"data/herlev/{cls}")
        dir_path.mkdir(parents=True, exist_ok=True)
        for i in range(1, 6):
            fname = dir_path / f"cell_{i}.bmp"
            if not fname.exists():
                img = Image.fromarray(np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8))
                img.save(fname)
                
    print("Sample images created in data/sipakmed/ and data/herlev/")

if __name__ == "__main__":
    create_samples()
