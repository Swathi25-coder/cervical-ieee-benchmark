import shutil
from pathlib import Path

def setup_herlev():
    src_base = Path("d:/CervicalCancerResearch/temp_herlev/projects/Pap-Smear/PapSmearImages")
    dest_base = Path("d:/CervicalCancerResearch/cervical-ieee-benchmark/data/herlev")
    dest_base.mkdir(parents=True, exist_ok=True)
    
    classes = [
        "carcinoma_in_situ", "light_dysplastic", "moderate_dysplastic",
        "normal_columnar", "normal_intermediate", "normal_superficiel", "severe_dysplastic"
    ]
    
    total_copied = 0
    for split in ["train", "test"]:
        split_dir = src_base / split
        if not split_dir.exists():
            continue
        for cls in classes:
            cls_src = split_dir / cls
            if not cls_src.exists():
                continue
            cls_dest = dest_base / cls
            cls_dest.mkdir(parents=True, exist_ok=True)
            for f in cls_src.glob("*"):
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".bmp", ".png"]:
                    # Ensure unique filename if overlaps exist
                    target_file = cls_dest / f"{split}_{f.name}"
                    shutil.copy2(f, target_file)
                    total_copied += 1
                    
    print(f"Successfully copied {total_copied} Herlev images into {dest_base}")

if __name__ == "__main__":
    setup_herlev()
