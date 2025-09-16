# ===== Alle Bilder + CSVs: Laplace (normalisiert), Fixationen, Panels speichern, Gruppen-Heatmaps =====
# Bits (links→rechts): ["meme","ort","person","politik","text"]
# Dateiname: ..._<5bit>.jpg|.png|.csv  (z. B. P01_IMG004_10100.jpg / .csv)

from pathlib import Path
import argparse, os, re, itertools, ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from skimage import io, color, filters, exposure
import scipy.ndimage as ndimage
from scipy.stats import gaussian_kde, kruskal, mannwhitneyu

# --------------------------------------------------------------------------------------
# CLI / Pfade
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Alle Bilder+CSVs auswerten (Laplace, Fixationen, Gruppen-Heatmaps)")
    # Eingaben (repo-relativ ODER absolut)
    p.add_argument("--input-img", default=os.environ.get("EWS_IMG_DIR","EWS/data/img/img_bin"),
                   help="Bild-Ordner (jpg/png; repo-relativ oder absolut).")
    p.add_argument("--input-csv", default=os.environ.get("EWS_CSV_DIR", "EWS/data/raw/raw_bin"),
                   help="CSV-Ordner (falls leer, wird --input-img verwendet).")
    p.add_argument("--require-csv", action="store_true",
                   help="Nur Bilder verarbeiten, wenn passende CSV (gleicher Dateistamm) existiert.")
    # Output
    p.add_argument("--out", "-o", default=os.environ.get("EWS_OUT_DIR", "EWS/code/kontraste/laplace_all_results"),
                   help="Output-Ordner (repo-relativ oder absolut).")
    # Metrik/Visuals
    p.add_argument("--sigma", type=float, default=1.0, help="Gauss-Glättung vor Laplace (σ). 0 = aus.")
    p.add_argument("--kde-bw", type=float, default=0.5, help="KDE Bandwidth-Skalierung (bw_method).")
    p.add_argument("--save-panels", action="store_true",
                   help="Für jedes Bild 4er-Panel (Original/Laplace/ColorDiff/KDE) als PNG speichern.")
    p.add_argument("--agg", choices=["mean","median"], default="mean", help="Kennzahl in Werte-Heatmaps.")
    p.add_argument("--no-stats", action="store_true", help="Signifikanztests deaktivieren.")
    # Ausgabedateien (Platzhalter)
    p.add_argument("--per-file-csv", default="laplace_per_file.csv",
                   help="Dateiname (relativ zu --out) für per-File-CSV.")
    p.add_argument("--agg-csv", default="laplace_agg_by_label.csv",
                   help="Dateiname (relativ zu --out) für Aggregat-CSV.")
    return p.parse_args()

ARGS = parse_args()

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[2] if len(SCRIPT_DIR.parents) >= 2 else SCRIPT_DIR

def resolve_path(maybe_path: str, default_root: Path) -> Path:
    return (Path(maybe_path) if maybe_path and Path(maybe_path).is_absolute()
            else (default_root / (maybe_path or "") )).resolve()

IMG_DIR = resolve_path(ARGS.input_img, REPO_ROOT)
CSV_DIR = resolve_path(ARGS.input_csv, REPO_ROOT) if ARGS.input_csv else IMG_DIR
OUT_DIR = resolve_path(ARGS.out, REPO_ROOT)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Dateien einsammeln
image_paths = sorted({str(p) for ext in ("*.jpg","*.jpeg","*.png") for p in IMG_DIR.rglob(ext)})
csv_paths   = sorted({str(p) for p in CSV_DIR.rglob("*.csv")})
if not image_paths:
    raise SystemExit(f"Keine Bilder gefunden in: {IMG_DIR}")

print(f"[Info] IMG_DIR : {IMG_DIR}  ({len(image_paths)} Bilder)")
print(f"[Info] CSV_DIR : {CSV_DIR}  ({len(csv_paths)} CSVs)")
print(f"[Info] OUT_DIR : {OUT_DIR}")

# --------------------------------------------------------------------------------------
# Konfiguration
# --------------------------------------------------------------------------------------
BIT_ORDER = ["meme", "ort", "person", "politik", "text"]   # 10000=meme, 00001=text
AGG_STAT  = ARGS.agg
DO_STATS  = not ARGS.no_stats
ROUND     = 4
SIGMA     = max(0.0, ARGS.sigma)
KDE_BW    = ARGS.kde_bw

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 120})

BIN_RE = re.compile(r"_(?P<mask>[01]{5})(?=\.[A-Za-z0-9]+$)")

OUT_CSV_PER_FILE = (OUT_DIR / ARGS.per_file_csv).resolve()
OUT_CSV_AGG      = (OUT_DIR / ARGS.agg_csv).resolve()
PANELS_DIR       = OUT_DIR / "per_image_panels"
if ARGS.save_panels:
    PANELS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Utils (Label, Bilder, Laplace, Fixationen, Plots)
# --------------------------------------------------------------------------------------
def extract_mask_and_label(path: str):
    base = Path(path).name
    m = BIN_RE.search(base)
    if not m:
        return None, None
    mask = m.group("mask")
    bits = list(mask)
    cats = [BIT_ORDER[i] for i, b in enumerate(bits) if b == "1"]
    label = "_".join(cats) if cats else "none"
    return mask, label

def canon_label(components):
    comp_set = set(components)
    ordered = [c for c in BIT_ORDER if c in comp_set]
    return "_".join(ordered) if ordered else "none"

def load_rgb_and_gray01(path: str):
    img = io.imread(path)
    if img.ndim == 2:
        rgb = color.gray2rgb(img)
        g   = img.astype(np.float32)
        if g.max() > 1.0: g /= 255.0
        return rgb, g
    if img.ndim == 3:
        if img.shape[2] >= 3:
            rgb = img[..., :3]
        else:
            rgb = color.gray2rgb(img[...,0])
        rgb = rgb.astype(np.float32)
        if rgb.max() > 1.0: rgb /= 255.0
        g = color.rgb2gray(rgb).astype(np.float32)
        return (rgb*255).astype(np.uint8), g
    raise ValueError(f"Unsupported image shape: {img.shape}")

def laplacian_var_norm(gray01: np.ndarray, sigma: float = 1.0) -> float:
    I = gray01
    if sigma > 0:
        I = filters.gaussian(I, sigma=sigma, preserve_range=True)
    L = filters.laplace(I, ksize=3)
    vL = float(np.var(L, ddof=0))
    vI = float(np.var(I, ddof=0))
    return float(vL / (vI + 1e-12)), L, I  # zusätzlich L und evtl. geglättetes I zurück

def color_diff_map(rgb_img_uint8: np.ndarray):
    lab_img = color.rgb2lab(rgb_img_uint8)
    a_channel = lab_img[:, :, 1]
    b_channel = lab_img[:, :, 2]
    def local_std(arr): return np.std(arr)
    std_a = ndimage.generic_filter(a_channel, local_std, size=5)
    std_b = ndimage.generic_filter(b_channel, local_std, size=5)
    # sqrt(std_a^2 + std_b^2)
    cdm = np.sqrt(std_a*std_a + std_b*std_b)
    cdm_rescaled = exposure.rescale_intensity(cdm, in_range='image', out_range=(0, 1))
    return cdm_rescaled

def detect_fixations(gaze_data, min_duration=50, max_dispersion=25):
    fixations = []
    if len(gaze_data) < 2:
        return pd.DataFrame()
    s_idx = 0
    while s_idx < len(gaze_data):
        anchor = gaze_data.iloc[s_idx]
        e_idx = s_idx
        while True:
            e_idx += 1
            if e_idx >= len(gaze_data): break
            nxt = gaze_data.iloc[e_idx]
            dist = np.hypot(nxt['gaze_point_x'] - anchor['gaze_point_x'],
                            nxt['gaze_point_y'] - anchor['gaze_point_y'])
            if dist > max_dispersion: break
        duration = gaze_data['device_time_stamp'].iloc[e_idx-1] - gaze_data['device_time_stamp'].iloc[s_idx]
        if duration >= min_duration:
            chunk = gaze_data.iloc[s_idx:e_idx]
            fixations.append({
                'x': chunk['gaze_point_x'].mean(),
                'y': chunk['gaze_point_y'].mean(),
                'duration': duration
            })
        s_idx = e_idx
    return pd.DataFrame(fixations)

def binocular_average(df, img_w, img_h):
    valid = df[(df['left_gaze_point_validity'] == 1) & (df['right_gaze_point_validity'] == 1)].copy()
    if valid.empty:
        return pd.DataFrame()
    valid['left_gaze_point_on_display_area']  = valid['left_gaze_point_on_display_area'].apply(ast.literal_eval)
    valid['right_gaze_point_on_display_area'] = valid['right_gaze_point_on_display_area'].apply(ast.literal_eval)
    valid['gaze_point_x'] = valid.apply(lambda r: (r['left_gaze_point_on_display_area'][0] + r['right_gaze_point_on_display_area'][0]) / 2 * img_w, axis=1)
    valid['gaze_point_y'] = valid.apply(lambda r: (r['left_gaze_point_on_display_area'][1] + r['right_gaze_point_on_display_area'][1]) / 2 * img_h, axis=1)
    return valid

def plot_panel(rgb_img, lap_map_rescaled, colordiff_rescaled, heatmap, out_png: Path):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(rgb_img); axes[0].set_title('Original Image'); axes[0].axis('off')
    axes[1].imshow(lap_map_rescaled, cmap='gray'); axes[1].set_title('Contrast Map (Laplace)'); axes[1].axis('off')
    axes[2].imshow(colordiff_rescaled, cmap='gray'); axes[2].set_title('Color Difference Map'); axes[2].axis('off')
    axes[3].imshow(heatmap, cmap='hot', origin='lower'); axes[3].set_title('Fixation Heat Map (KDE)'); axes[3].axis('off')
    plt.tight_layout(); fig.savefig(out_png, dpi=180); plt.close(fig)

def _truncate_cmap(cmap, minval=0.25, maxval=0.95, n=256):
    new_colors = cmap(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(f"trunc_{cmap.name}", new_colors)

def plot_value_heatmap(values: pd.Series, counts: pd.Series, title: str, out_png: Path,
                       stat_name: str = "mean", digits: int = 4):
    s_vals = values.dropna(); cats = list(s_vals.index)
    if not cats: return
    vals = s_vals.to_numpy(float); ns = counts.reindex(cats).fillna(0).astype(int).to_numpy()
    cmap_soft = _truncate_cmap(cm.get_cmap("Blues"), 0.35, 0.95)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmax <= vmin: vmin -= 1e-6; vmax += 1e-6
    fig, ax = plt.subplots(figsize=(max(7, 1.0*len(cats)+3), 3.0))
    im = ax.imshow(vals.reshape(1,-1), aspect='auto', cmap=cmap_soft, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label(f"Lap Var Norm ({stat_name})")
    ax.set_yticks([0], [f"Lap Var Norm ({stat_name})"]); ax.set_xticks(range(len(cats)), cats)
    ax.set_title(title)
    for j, v in enumerate(vals):
        ax.text(j, 0, f"{v:.{digits}f}\n(n={ns[j]})", ha="center", va="center", color="black")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def plot_diff_heatmap(values: pd.Series, counts: pd.Series, title: str, out_png: Path,
                      stat_name: str = "mean", digits: int = 4):
    s_vals = values.dropna(); cats = list(s_vals.index)
    if len(cats) < 2: return
    vals = s_vals.to_numpy(float); ns = counts.reindex(cats).fillna(0).astype(int).to_numpy()
    X = vals.reshape(-1,1); M = np.abs(X - X.T); n = len(cats)
    cmap_soft = _truncate_cmap(cm.get_cmap("Blues"), 0.25, 0.85)
    fig_w = max(7, min(18, 0.9*n + 4))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w))
    im = ax.imshow(M, aspect='equal', cmap=cmap_soft)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(f"|Δ Lap Var Norm ({stat_name})|")
    ax.set_xticks(range(n), cats); ax.set_yticks(range(n), cats)
    ax.set_title(title + f" (|Δ {stat_name}|)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i,j]:.{digits}f}\n(n={ns[i]}/{ns[j]})", ha="center", va="center", color="black")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def run_stats_subset(df_img: pd.DataFrame, labels: list[str], out_dir: Path, feature: str = "lap_var_norm"):
    present = [l for l in labels if l in set(df_img["label"])]
    if len(present) < 2: return
    samples = [df_img.loc[df_img["label"]==l, feature].dropna().to_numpy() for l in present]
    H, p_kw = kruskal(*samples)
    with open(out_dir / "significance_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Kruskal–Wallis (Lap Var Norm): H={H:.3f}, p={p_kw:.3g}\n")
        f.write("Kategorien: " + ", ".join(present) + "\n")
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
      .sort_values("p_fdr").to_csv(out_dir / "pairwise_mwu_lap.csv", index=False)

# --------------------------------------------------------------------------------------
# CSV-Paarung vorbereiten (stem->csv)
# --------------------------------------------------------------------------------------
csv_by_stem = {}
for c in csv_paths:
    stem = Path(c).stem
    # optional: nur CSVs mit gültiger Maske beibehalten
    mask, _ = extract_mask_and_label(c)
    if mask is None:
        continue
    csv_by_stem[stem] = c

# --------------------------------------------------------------------------------------
# 1) Schleife: alle Bilder + CSVs → Metrik + optional Panel speichern
# --------------------------------------------------------------------------------------
rows, skipped_no_csv = [], 0
for img_path in image_paths:
    stem = Path(img_path).stem
    gaze_csv = csv_by_stem.get(stem)

    if ARGS.require_csv and not gaze_csv:
        skipped_no_csv += 1
        continue

    mask, label = extract_mask_and_label(img_path)
    if not mask or label == "none":
        continue

    try:
        rgb_img, gray_img = load_rgb_and_gray01(img_path)
        height, width = gray_img.shape

        # Laplace (norm.) + Map
        lap_norm, L_map, I_smooth = laplacian_var_norm(gray_img, sigma=SIGMA)
        lap_map_rescaled = exposure.rescale_intensity(np.abs(L_map), in_range='image', out_range=(0, 1))

        # Farb-Differenz
        colordiff_rescaled = color_diff_map(rgb_img)

        # Fixationen + KDE (nur wenn CSV vorhanden)
        heatmap = np.zeros_like(gray_img)
        if gaze_csv:
            df = pd.read_csv(gaze_csv)
            bino = binocular_average(df, width, height)
            fix_df = detect_fixations(bino, min_duration=50, max_dispersion=25)
            if len(fix_df) >= 2:
                pts = np.vstack([fix_df['x'], fix_df['y']])
                kde = gaussian_kde(pts, bw_method=KDE_BW)
                xg, yg = np.meshgrid(np.arange(width), np.arange(height))
                heatmap = kde.evaluate(np.vstack([xg.ravel(), yg.ravel()])).reshape(height, width)

        if ARGS.save_panels:
            out_png = PANELS_DIR / f"{stem}.png"
            plot_panel(rgb_img, lap_map_rescaled, colordiff_rescaled, heatmap, out_png)

        rows.append({
            "file_image": img_path,
            "file_csv": gaze_csv or "",
            "stem": stem,
            "mask": mask,
            "label": label,
            "lap_var_norm": lap_norm
        })

    except Exception as e:
        print(f"[Warnung] Fehler bei {img_path}: {e}")

if not rows:
    msg = "Keine auswertbaren Bilder"
    if ARGS.require_csv: msg += " (require_csv aktiv, keine passenden CSVs gefunden?)"
    raise SystemExit(msg + ".")

df = pd.DataFrame(rows).sort_values(["label","stem","file_image"])
df.to_csv(OUT_CSV_PER_FILE, index=False)
print(f"[OK] per-File CSV: {OUT_CSV_PER_FILE}  | ohne CSV verworfen: {skipped_no_csv}")

# --------------------------------------------------------------------------------------
# 2) Aggregation nach Label + Gesamt-Heatmaps
# --------------------------------------------------------------------------------------
agg = (df.groupby("label")["lap_var_norm"]
         .agg(["count","mean","median","std","min","max"])
         .sort_index())
agg.to_csv(OUT_CSV_AGG)
print(f"[OK] Aggregat-CSV: {OUT_CSV_AGG}\n{agg}\n")

vals_all, ns_all = agg[AGG_STAT], agg["count"]
plot_value_heatmap(vals_all, ns_all,
    title=f"Werte-Heatmap: Laplace-Varianz (norm.) — alle Labels [{AGG_STAT}]",
    out_png=OUT_DIR / f"all_labels_heat_values_{AGG_STAT}.png",
    stat_name=AGG_STAT, digits=ROUND
)
plot_diff_heatmap(vals_all, ns_all,
    title=f"Differenz-Heatmap: Laplace-Varianz (norm.) — alle Labels",
    out_png=OUT_DIR / f"all_labels_heat_diff_{AGG_STAT}.png",
    stat_name=AGG_STAT, digits=ROUND
)

# --------------------------------------------------------------------------------------
# 3) Gruppen wie bei RMS
# --------------------------------------------------------------------------------------
GROUPS = {
    "01_singletons_meme_ort_person_politik_text": [
        canon_label(["meme"]), canon_label(["ort"]), canon_label(["person"]),
        canon_label(["politik"]), canon_label(["text"]),
    ],
    "02_text_pairs_and_singletons": [
        canon_label(["meme"]), canon_label(["ort"]), canon_label(["person"]), canon_label(["text"]),
        canon_label(["meme","text"]), canon_label(["ort","text"]),
        canon_label(["politik","text"]), canon_label(["person","text"]),
    ],
    "03_person_chain": [
        canon_label(["person"]), canon_label(["person","text"]),
        canon_label(["person","politik"]), canon_label(["person","text","politik"]),
    ],
    "04_text_politik_meme_vs_text_politik_person": [
        canon_label(["text","politik","meme"]), canon_label(["text","politik","person"]),
    ],
}

for group_name, wanted in GROUPS.items():
    present = [l for l in wanted if l in set(df["label"])]
    if not present:
        print(f"[Info] Gruppe {group_name}: keine passenden Labels – skip.")
        continue
    sub = df[df["label"].isin(present)].copy()
    agg_g = (sub.groupby("label")["lap_var_norm"]
                .agg(["count","mean","median","std","min","max"])
                .reindex(present))
    group_dir = OUT_DIR / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    sub.to_csv(group_dir / "laplace_per_file.csv", index=False)
    agg_g.to_csv(group_dir / "laplace_agg.csv")

    v, n = agg_g[AGG_STAT], agg_g["count"]
    plot_value_heatmap(v, n,
        title=f"Laplace-Varianz (norm.) — {group_name.replace('_',' ')} [{AGG_STAT}]",
        out_png=group_dir / f"heat_values_{AGG_STAT}.png",
        stat_name=AGG_STAT, digits=ROUND
    )
    plot_diff_heatmap(v, n,
        title=f"Differenz — {group_name.replace('_',' ')}",
        out_png=group_dir / f"heat_diff_{AGG_STAT}.png",
        stat_name=AGG_STAT, digits=ROUND
    )
    if DO_STATS and len(present) >= 2:
        run_stats_subset(sub, present, group_dir, feature="lap_var_norm")

print("\nFertig! CSVs & Plots liegen in:", OUT_DIR)
