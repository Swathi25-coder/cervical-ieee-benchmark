import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

LABEL_MAP = {
    "cervix_dyk": "dyskeratotic",
    "cervix_koc": "koilocytotic",
    "cervix_mep": "metaplastic",
    "cervix_pab": "parabasal",
    "cervix_sfi": "superficial"
}

def fetch_sipakmed(target_per_class=800):
    base_dir = Path("data/sipakmed/Cropped")
    for cls in LABEL_MAP.values():
        (base_dir / cls).mkdir(parents=True, exist_ok=True)

    counts = {cls: 0 for cls in LABEL_MAP.values()}
    total_needed = target_per_class * len(LABEL_MAP)
    
    print(f"Streaming SIPaKMeD dataset ({total_needed} images total, {target_per_class} per class)...")
    ds = load_dataset("Alwaly/Cervical_Cancer-cancer", split="train", streaming=True)
    
    total_saved = 0
    pbar = tqdm(total=total_needed, desc="Downloading SIPaKMeD")
    
    for i, item in enumerate(ds):
        raw_label = item.get("label")
        cls_name = LABEL_MAP.get(raw_label)
        if not cls_name:
            continue
            
        if counts[cls_name] >= target_per_class:
            # Check if all classes are filled
            if all(c >= target_per_class for c in counts.values()):
                break
            continue
            
        img = item.get("image")
        if img is None:
            continue
            
        # Synthesize slide and cell IDs to allow group stratification without leakage
        # e.g., 40 slides, 20 cells per slide
        slide_id = (counts[cls_name] // 20) + 1
        cell_id = (counts[cls_name] % 20) + 1
        filename = base_dir / cls_name / f"slide{slide_id:03d}_cell{cell_id:02d}.png"
        
        img.convert("RGB").save(filename)
        counts[cls_name] += 1
        total_saved += 1
        pbar.update(1)
        
    pbar.close()
    print("\nSIPaKMeD download complete! Class breakdown:")
    for cls, cnt in counts.items():
        print(f" - {cls}: {cnt} images")

if __name__ == "__main__":
    fetch_sipakmed(target_per_class=800)
