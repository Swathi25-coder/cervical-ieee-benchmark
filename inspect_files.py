from pathlib import Path
from collections import Counter

EXT = {".bmp", ".png", ".jpg", ".jpeg"}

def inspect_dataset(name, root_dir):
    root = Path(root_dir)
    print(f"\n{'='*20} Inspecting: {name} ({root}) {'='*20}")
    if not root.exists():
        print(f"Directory not found: {root}")
        return

    all_files = list(root.rglob("*"))
    files_only = [f for f in all_files if f.is_file()]
    img_files = [f for f in files_only if f.suffix.lower() in EXT]

    ext_counts = Counter(f.suffix.lower() for f in files_only)
    subdirs = sorted([d.relative_to(root) for d in all_files if d.is_dir() and len(d.relative_to(root).parts) <= 2])

    print(f"Total files: {len(files_only)}")
    print(f"Total image files: {len(img_files)}")
    print(f"File extensions: {dict(ext_counts)}")
    print(f"Top-level / second-level folders ({len(subdirs)} found):")
    for d in subdirs[:15]:
        print(f"  - {d}")
    if len(subdirs) > 15:
        print(f"  ... and {len(subdirs) - 15} more subdirectories")

    if name == "sipakmed":
        cropped_count = sum(1 for f in img_files if "cropped" in str(f).lower())
        print(f"Cropped SIPaKMeD images found: {cropped_count} (Expected ~4,049)")
    elif name == "herlev":
        non_mask_count = sum(1 for f in img_files if not f.stem.lower().endswith(("-d", "_d")) and "mask" not in str(f).lower())
        print(f"Non-mask Herlev images found: {non_mask_count} (Expected ~917)")

if __name__ == "__main__":
    inspect_dataset("sipakmed", "data/sipakmed")
    inspect_dataset("herlev", "data/herlev")
