import re, sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

EXT = {".bmp", ".png", ".jpg", ".jpeg"}
SIP_CLS = r"dyskeratotic|koilocytotic|metaplastic|parabasal|superficial"

def build(name, root):
    root_path = Path(root)
    if not root_path.exists():
        print(f"Directory {root} does not exist yet. Please create it and add dataset images.")
        return

    rows = []
    for p in sorted(root_path.rglob("*")):
        if p.suffix.lower() not in EXT:
            continue
        rel = str(p.relative_to(root_path)).lower()
        if name == "sipakmed":
            if "cropped" not in rel:
                continue
            m = re.search(SIP_CLS, rel)
            if not m:
                continue
            y = 1 if m.group() in ("dyskeratotic", "koilocytotic") else 0
            g = f"{m.group()}_{p.stem.split('_')[0]}"   # source image id
        else:
            if p.stem.lower().endswith(("-d", "_d")) or "mask" in rel:
                continue
            if "normal" in rel: y = 0
            elif "dysplastic" in rel or "carcinoma" in rel: y = 1
            else: continue
            g = p.stem                                   # Herlev has no slide IDs
        rows.append((str(p), y, g))

    if not rows:
        print(f"[{name}] No valid images found in {root}. Please verify files are placed correctly.")
        return

    df = pd.DataFrame(rows, columns=["path", "label", "group"])
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for f, (_, te) in enumerate(sgkf.split(df, df.label, df.group)):
        df.loc[df.index[te], "fold"] = f

    Path("splits").mkdir(exist_ok=True)
    out_csv = Path(f"splits/{name}_folds.csv")
    df.to_csv(out_csv, index=False)
    print(f"{name} images: {len(df)} | groups: {df.group.nunique()}")
    print(df.groupby(["fold", "label"]).size().unstack(), "\n")

if __name__ == "__main__":
    build("sipakmed", "data/sipakmed")
    build("herlev", "data/herlev")
