# ===== RMS-Kontrast (binäre Maske am Dateiende) – gezielte Vergleiche =====
# Bits (links → rechts): ["meme","ort","person","politik","text"]
# Beispiel: P01_IMG004_10100.jpg → "10100" → label "meme_person"

import os, glob, re, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from skimage import io, color
from scipy.stats import kruskal, mannwhitneyu

# -------------------------
# Pfade & Settings
# -------------------------
from pathlib import Path
import argparse, os

def parse_args():
    p = argparse.ArgumentParser(description="RMS / binäre Labels")
    p.add_argument("--input","-i", default=os.environ.get("EWS_INPUT_DIR","EWS/data/img/img_bin"),
                   help="Eingabeordner (repo-relativ oder absolut).")
    p.add_argument("--out","-o", default=os.environ.get("EWS_OUT_DIR","EWS/code/kontraste/rms_binary_custom_sets"),
                   help="Output-Ordner (repo-relativ oder absolut).")
    return p.parse_args()

ARGS = parse_args()

SCRIPT_DIR = Path(__file__).resolve().parent
# .../EWS/code/kontraste/ -> parents[2] == .../EWS
REPO_ROOT = SCRIPT_DIR.parents[2] if len(SCRIPT_DIR.parents) >= 2 else SCRIPT_DIR

IN_DIR  = (Path(ARGS.input) if Path(ARGS.input).is_absolute() else (REPO_ROOT / ARGS.input)).resolve()
OUT_DIR = (Path(ARGS.out)   if Path(ARGS.out).is_absolute()   else (REPO_ROOT / ARGS.out)).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted({str(p) for ext in ("*.jpg","*.jpeg","*.png") for p in IN_DIR.rglob(ext)})
if not image_paths:
    raise SystemExit(f"Keine Bilder gefunden in: {IN_DIR}")

print(f"[Info] RepoRoot: {REPO_ROOT}")
print(f"[Info] Input   : {IN_DIR}")
print(f"[Info] Output  : {OUT_DIR}")
print(f"[Info] Bilder  : {len(image_paths)} gefunden")

os.makedirs(OUT_DIR, exist_ok=True)

BIT_ORDER = ["meme", "ort", "person", "politik", "text"]  # 10000=meme, 00001=text
AGG_STAT = "mean"   # "mean" oder "median" für Heatmaps
DO_STATS = True     # KW + paarweise MWU (FDR)
ROUND = 4
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 120})

# -------------------------
# Binärmaske am Dateiende
# -------------------------
BIN_RE = re.compile(r"_(?P<mask>[01]{5})(?=\.[A-Za-z0-9]+$)")

def extract_mask_and_label(path: str):
    base = os.path.basename(path)
    m = BIN_RE.search(base)
    if not m:
        return None, None
    mask = m.group("mask")
    bits = list(mask)
    cats = [BIT_ORDER[i] for i, b in enumerate(bits) if b == "1"]
    label = "_".join(cats) if cats else "none"
    return mask, label

def canon_label(components):
    """Kanonischer Label-String in BIT_ORDER-Reihenfolge."""
    comp_set = set(components)
    ordered = [c for c in BIT_ORDER if c in comp_set]
    return "_".join(ordered) if ordered else "none"

# -------------------------
# Bild laden & RMS
# -------------------------
def load_gray01(path: str) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 2:
        g = img.astype(np.float32)
        if g.max() > 1.0: g /= 255.0
        return g
    if img.ndim == 3:
        if img.shape[2] >= 3:
            img = img[..., :3]
        img = img.astype(np.float32)
        if img.max() > 1.0: img /= 255.0
        return color.rgb2gray(img).astype(np.float32)
    raise ValueError(f"Unsupported image shape: {img.shape}")

def rms_contrast(gray01: np.ndarray) -> float:
    return float(np.std(gray01, ddof=0))

# -------------------------
# Plot-Hilfen (dezente Blau-Töne)
# -------------------------
def _truncate_cmap(cmap, minval=0.25, maxval=0.95, n=256):
    new_colors = cmap(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(f"trunc_{cmap.name}", new_colors)

def plot_value_heatmap(values: pd.Series, counts: pd.Series, title: str, out_png: str,
                       stat_name: str = "mean", digits: int = 4):
    s_vals = values.dropna()
    cats = list(s_vals.index)
    if not cats:
        print("[Info] Werte-Heatmap: keine Daten."); return
    vals = s_vals.to_numpy(float)
    ns   = counts.reindex(cats).fillna(0).astype(int).to_numpy()

    cmap_soft = _truncate_cmap(cm.get_cmap("Blues"), 0.35, 0.95)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmax <= vmin: vmin -= 1e-6; vmax += 1e-6

    fig, ax = plt.subplots(figsize=(max(7, 1.0*len(cats)+3), 3.0))
    im = ax.imshow(vals.reshape(1,-1), aspect='auto', cmap=cmap_soft, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label(f"RMS ({stat_name})")
    ax.set_yticks([0], [f"RMS ({stat_name})"])
    ax.set_xticks(range(len(cats)), cats, rotation=0)
    ax.set_title(title)
    for j, v in enumerate(vals):
        ax.text(j, 0, f"{v:.{digits}f}\n(n={ns[j]})", ha="center", va="center", color="black")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    print(f"[OK] Werte-Heatmap: {out_png}")

def plot_diff_heatmap(values: pd.Series, counts: pd.Series, title: str, out_png: str,
                      stat_name: str = "mean", digits: int = 4):
    s_vals = values.dropna()
    cats = list(s_vals.index)
    if len(cats) < 2:
        print("[Info] Differenz-Heatmap: <2 Kategorien — skip."); return
    vals = s_vals.to_numpy(float)
    ns   = counts.reindex(cats).fillna(0).astype(int).to_numpy()

    X = vals.reshape(-1,1)
    M = np.abs(X - X.T)
    n = len(cats)

    cmap_soft = _truncate_cmap(cm.get_cmap("Blues"), 0.25, 0.85)
    fig_w = max(7, min(18, 0.9*n + 4))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w))
    im = ax.imshow(M, aspect='equal', cmap=cmap_soft)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(f"|Δ RMS ({stat_name})|")
    ax.set_xticks(range(n), cats, rotation=0); ax.set_yticks(range(n), cats)
    ax.set_title(title + f" (|Δ {stat_name}|)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i,j]:.{digits}f}\n(n={ns[i]}/{ns[j]})",
                    ha="center", va="center", color="black")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    print(f"[OK] Differenz-Heatmap: {out_png}")

def run_stats_subset(df_img: pd.DataFrame, labels: list[str], out_dir: str, feature: str = "rms_contrast"):
    present = [l for l in labels if l in set(df_img["label"])]
    if len(present) < 2:
        print("[Info] Signifikanz: <2 Kategorien — skip."); return
    samples = [df_img.loc[df_img["label"]==l, feature].dropna().to_numpy() for l in present]

    # Omnibus
    H, p_kw = kruskal(*samples)
    with open(os.path.join(out_dir, "significance_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Kruskal–Wallis (RMS): H={H:.3f}, p={p_kw:.3g}\n")
        f.write("Kategorien: " + ", ".join(present) + "\n")

    # Paarweise + BH-FDR
    pairs = list(itertools.combinations(range(len(present)), 2))
    raw_p, rbc, pair_names = [], [], []
    for i,j in pairs:
        a, b = present[i], present[j]
        x = df_img.loc[df_img["label"]==a, feature].to_numpy()
        y = df_img.loc[df_img["label"]==b, feature].to_numpy()
        pval = mannwhitneyu(x, y, alternative="two-sided").pvalue
        raw_p.append(pval); pair_names.append(f"{a} vs {b}")
        nx, ny = len(x), len(y)
        U_greater = mannwhitneyu(x, y, alternative="greater").statistic
        rbc.append(2*U_greater/(nx*ny) - 1)
    raw_p = np.array(raw_p, float)
    order = np.argsort(raw_p); m = len(raw_p); adj = np.empty_like(raw_p); prev = 1.0
    for idx in range(m-1, -1, -1):
        rank = idx+1; val = raw_p[order[idx]] * m / rank; prev = min(prev, val); adj[order[idx]] = min(prev, 1.0)

    pd.DataFrame({"pair": pair_names, "p_raw": raw_p, "p_fdr": adj, "effect_rbc": rbc})\
      .sort_values("p_fdr").to_csv(os.path.join(out_dir, "pairwise_mwu_rms.csv"), index=False)
    print(f"[OK] Stats gespeichert: {os.path.join(out_dir, 'pairwise_mwu_rms.csv')}")

# -------------------------
# 1) Bilder laden & RMS pro Datei
# -------------------------
# Bildsuche: rekursiv unter IN_DIR für jpg/jpeg/png
image_paths = sorted({
    str(p) for ext in ("*.jpg", "*.jpeg", "*.png")
    for p in IN_DIR.rglob(ext)
})
if not image_paths:
    raise SystemExit(f"Keine Bilder gefunden in: {IN_DIR}")


rows, skipped = [], 0
for p in image_paths:
    try:
        mask, label = extract_mask_and_label(p)
        if not mask or label == "none":
            skipped += 1
        else:
            g = load_gray01(p)
            rms = rms_contrast(g)
            rows.append({"file": p, "mask": mask, "label": label, "rms_contrast": rms})
    except Exception as e:
        print(f"[Warnung] {p}: {e}")

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("Keine auswertbaren Bilder mit gültiger Maske.")
print(f"[Info] {len(df)} Bilder verwendet, {skipped} ohne gültige Maske übersprungen.")
print("[Info] Verfügbare Labels:", sorted(df["label"].unique()))

# -------------------------
# 2) Deine vier Vergleichs-Sets
# -------------------------
GROUPS = {
    "01_singletons_meme_ort_person_politik_text": [
        canon_label(["meme"]),
        canon_label(["ort"]),
        canon_label(["person"]),
        canon_label(["politik"]),
        canon_label(["text"]),
    ],
    "02_text_pairs_and_singletons": [
        canon_label(["meme"]),
        canon_label(["ort"]),
        canon_label(["person"]),
        canon_label(["text"]),
        canon_label(["meme","text"]),
        canon_label(["ort","text"]),
        canon_label(["politik","text"]),
        canon_label(["person","text"]),
    ],
    "03_person_chain": [
        canon_label(["person"]),
        canon_label(["person","text"]),
        canon_label(["person","politik"]),
        canon_label(["person","text","politik"]),
    ],
    "04_text_politik_meme_vs_text_politik_person": [
        canon_label(["text","politik","meme"]),
        canon_label(["text","politik","person"]),
    ],
}

# -------------------------
# 3) Rendern pro Gruppe (kleine Plots + Stats)
# -------------------------
for group_name, wanted_labels in GROUPS.items():
    present = [l for l in wanted_labels if l in set(df["label"])]
    if len(present) < 1:
        print(f"[Info] {group_name}: keine der gewünschten Labels vorhanden – skip.")
        continue

    sub = df[df["label"].isin(present)].copy()
    agg = (sub.groupby("label")["rms_contrast"]
              .agg(["count","mean","median","std","min","max"])
              .reindex(present))

    group_dir = os.path.join(OUT_DIR, group_name)
    os.makedirs(group_dir, exist_ok=True)
    sub.to_csv(os.path.join(group_dir, "rms_per_file.csv"), index=False)
    agg.to_csv(os.path.join(group_dir, "rms_agg.csv"))

    # Plots
    vals, ns = agg[AGG_STAT], agg["count"]
    plot_value_heatmap(vals, ns,
        title=f"RMS ({AGG_STAT})",
        out_png=os.path.join(group_dir, f"heat_values_{AGG_STAT}.png"),
        stat_name=AGG_STAT, digits=ROUND
    )
    plot_diff_heatmap(vals, ns,
        title=f"Differenz",
        out_png=os.path.join(group_dir, f"heat_diff_{AGG_STAT}.png"),
        stat_name=AGG_STAT, digits=ROUND
    )

    if DO_STATS and len(present) >= 2:
        run_stats_subset(sub, present, group_dir, "rms_contrast")

print(f"\nFertig. Ergebnisse liegen in : {OUT_DIR}")
