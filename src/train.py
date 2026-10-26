import argparse, copy, time, os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
import torchvision as tv, torchvision.transforms as T
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

dev = "cuda" if torch.cuda.is_available() else "cpu"
M, S = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TR = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(0.1, 0.1, 0.1),
    T.ToTensor(),
    T.Normalize(M, S)
])
TE = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(M, S)
])
OUT = Path("results/results.csv")

class DS(torch.utils.data.Dataset):
    def __init__(self, df, tf):
        self.p = df.path.tolist()
        self.y = df.label.tolist()
        self.tf = tf
    def __len__(self):
        return len(self.p)
    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), self.y[i]

def build_model(n):
    if n == "mobilenet_v3_large":
        m = tv.models.mobilenet_v3_large(weights="DEFAULT")
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    elif n == "efficientnet_b0":
        m = tv.models.efficientnet_b0(weights="DEFAULT")
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    elif n == "resnet50":
        m = tv.models.resnet50(weights="DEFAULT")
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif n == "densenet121":
        m = tv.models.densenet121(weights="DEFAULT")
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    elif n == "swin_t":
        m = tv.models.swin_t(weights="DEFAULT")
        m.head = nn.Linear(m.head.in_features, 2)
    else:
        raise ValueError(f"Unknown model: {n}")
    return m.to(dev)

@torch.no_grad()
def predict(m, loader):
    m.eval()
    P, Y = [], []
    for x, y in loader:
        P.append(torch.softmax(m(x.to(dev)), 1)[:, 1].cpu())
        Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()

def compute_metrics(y, p):
    pr = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
    conf, ok = np.maximum(p, 1 - p), (pr == y).astype(float)
    b, ece = np.linspace(0.5, 1, 11), 0.0
    for lo, hi in zip(b[:-1], b[1:]):
        k = (conf > lo) & (conf <= hi)
        if k.any():
            ece += k.mean() * abs(ok[k].mean() - conf[k].mean())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else np.nan
    return dict(
        acc=ok.mean(),
        f1=2 * prec * sens / max(prec + sens, 1e-9),
        sens=sens,
        spec=spec,
        auc=auc,
        ece=ece
    )

def run(tr_df, te_df, name, seed, epochs, bs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    a, v = next(GroupShuffleSplit(1, test_size=0.1, random_state=seed).split(tr_df, groups=tr_df.group))
    trd, vad = tr_df.iloc[a], tr_df.iloc[v]

    num_workers = 2 if dev == "cuda" else 0
    kw = dict(batch_size=bs, num_workers=num_workers, pin_memory=dev == "cuda")
    tl = torch.utils.data.DataLoader(DS(trd, TR), shuffle=True, **kw)
    vl = torch.utils.data.DataLoader(DS(vad, TE), **kw)
    el = torch.utils.data.DataLoader(DS(te_df, TE), **kw)

    m = build_model(name)
    cnt = np.bincount(trd.label, minlength=2)
    w = torch.tensor(len(trd) / (2 * np.maximum(cnt, 1)), dtype=torch.float32).to(dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
    sc = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    best, bst = -1, None
    for ep in range(epochs):
        m.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.autocast(device_type=dev, enabled=dev == "cuda"):
                l = lossf(m(x), y)
            sc.scale(l).backward()
            sc.step(opt)
            sc.update()
        p, y = predict(m, vl)
        mv = compute_metrics(y, p)
        score = mv["auc"] if not np.isnan(mv["auc"]) else mv["acc"]
        if score > best:
            best, bst = score, copy.deepcopy(m.state_dict())

    if bst is not None:
        m.load_state_dict(bst)
    p, y = predict(m, el)
    return compute_metrics(y, p), sum(q.numel() for q in m.parameters()) / 1e6

def summarize():
    if not OUT.exists():
        print(f"No results found at {OUT}")
        return
    d = pd.read_csv(OUT)
    metrics_cols = ["acc", "f1", "sens", "spec", "auc", "ece"]
    g = d.groupby(["model", "setting", "train", "test"])[metrics_cols]
    m = g.mean().round(4)
    s_std = g.std(ddof=1).fillna(0.0).round(4)
    s = m.astype(str) + " ± " + s_std.astype(str)
    s["runs"] = g.size()
    s.to_csv("results/summary.csv")
    print(s.to_string())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenet_v3_large")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()

    if a.summarize:
        summarize()
        raise SystemExit

    Path("results").mkdir(exist_ok=True)
    for n in ("sipakmed", "herlev"):
        split_p = Path(f"splits/{n}_folds.csv")
        if not split_p.exists():
            print(f"Error: {split_p} not found. Please run 'python src/prep.py' first.")
            raise SystemExit(1)

    D = {n: pd.read_csv(f"splits/{n}_folds.csv") for n in ("sipakmed", "herlev")}
    done = set(map(tuple, pd.read_csv(OUT)[["model", "setting", "train", "test", "seed", "fold"]].values)) if OUT.exists() else set()
    jobs = []
    for sd in a.seeds:
        for n, df in D.items():
            for f in range(a.folds):
                jobs.append(("within", n, n, sd, f, df[df.fold != f], df[df.fold == f]))
        for s, t in (("sipakmed", "herlev"), ("herlev", "sipakmed")):
            jobs.append(("cross", s, t, sd, -1, D[s], D[t]))

    for setting, s, t, sd, f, trd, ted in jobs:
        if (a.model, setting, s, t, sd, f) in done:
            continue
        t0 = time.time()
        r, par = run(trd, ted, a.model, sd, a.epochs, a.bs)
        row = dict(
            model=a.model,
            setting=setting,
            train=s,
            test=t,
            seed=sd,
            fold=f,
            params_M=round(par, 2),
            secs=round(time.time() - t0),
            **{k: round(float(v), 4) for k, v in r.items()}
        )
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(row, flush=True)
