# ===== Laplace: Mittelwerte pro Kategorie (wie bei RMS) + optional Anchors =====
# Erwartet eine CSV im Format der vorherigen Auswertung (z. B. laplace_per_file.csv)
# mit Spalten: file_image, stem, mask, label, lap_var_norm

from pathlib import Path
import argparse, os
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------------------
# CLI / Pfade
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Laplace-Aggregation: Mittelwerte pro Kategorie (wie bei RMS)")
    p.add_argument("--csv", default=os.environ.get("EWS_LAP_PERFILE",
                                                  "EWS/code/kontraste/laplace_per_image/laplace_per_file.csv"),
                   help="Pfad zur per-File-CSV (repo-relativ oder absolut).")
    p.add_argument("--out", "-o", default=os.environ.get("EWS_OUT_DIR",
                                                         "EWS/code/kontraste/laplace_agg"),
                   help="Output-Ordner (repo-relativ oder absolut).")
    p.add_argument("--do-anchors", action="store_true",
                   help="Zusätzlich Anchor-Aggregation (meme/ort/person/politik/text) ausgeben.")
    return p.parse_args()

ARGS = parse_args()

SCRIPT_DIR = Path(__file__).resolve().parent
# Repo-Root annehmen wie beim RMS (zwei Ebenen hoch, falls Skript unter .../code/... liegt)
REPO_ROOT  = SCRIPT_DIR.parents[2] if len(SCRIPT_DIR.parents) >= 2 else SCRIPT_DIR

def resolve_path(maybe_path: str, base: Path) -> Path:
    p = Path(maybe_path)
    return (p if p.is_absolute() else (base / p)).resolve()

CSV_PATH = resolve_path(ARGS.csv, REPO_ROOT)
OUT_DIR  = resolve_path(ARGS.out, REPO_ROOT)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[Info] per-file CSV : {CSV_PATH}")
print(f"[Info] Output-Ordner: {OUT_DIR}")

# --------------------------------------------------------------------------------------
# CSV laden & prüfen
# --------------------------------------------------------------------------------------
if not CSV_PATH.exists():
    raise SystemExit(f"CSV nicht gefunden: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
needed_cols = {"label", "lap_var_norm"}
missing = needed_cols - set(df.columns)
if missing:
    raise SystemExit(f"Fehlende Spalten in CSV: {missing}. Erwartet: {needed_cols}")

# Nur gültige Werte
df = df.dropna(subset=["label", "lap_var_norm"]).copy()
if df.empty:
    raise SystemExit("Keine gültigen Zeilen nach Filter (label/lap_var_norm).")

# --------------------------------------------------------------------------------------
# 1) Aggregation nach exakt gleichem Label (z. B. 'meme', 'meme_text', 'text_politik_person', ...)
# --------------------------------------------------------------------------------------
agg_label = (df.groupby("label")["lap_var_norm"]
               .agg(["count","mean","median","std","min","max"])
               .sort_values("mean", ascending=False))

agg_label_path = OUT_DIR / "laplace_agg_by_label.csv"
agg_label.to_csv(agg_label_path)
print(f"[OK] Aggregat pro Label gespeichert: {agg_label_path}")

# Leaderboards (optional, wie beim RMS)
(agg_label["mean"].sort_values(ascending=False)
          .to_csv(OUT_DIR / "leaderboard_mean.csv", header=["mean"]))
(agg_label["median"].sort_values(ascending=False)
          .to_csv(OUT_DIR / "leaderboard_median.csv", header=["median"]))

print("\n=== Top-Kategorien nach Mittelwert (Lap Var Norm) ===")
print(agg_label.head(10))

# --------------------------------------------------------------------------------------
# 2) (Optional) Anchor-Aggregation:
#    Für jeden Anker (meme/ort/person/politik/text) werden ALLE Zeilen gezählt,
#    deren Label den Anker enthält – unabhängig von weiteren Kombis.
# --------------------------------------------------------------------------------------
if ARGS.do_anchors:
    anchors = ["meme","ort","person","politik","text"]

    def has_anchor(label: str, a: str) -> bool:
        parts = label.split("_") if isinstance(label, str) else []
        return a in parts

    rows = []
    for a in anchors:
        sub = df[df["label"].apply(lambda s: has_anchor(s, a))]
        if sub.empty:
            rows.append({"anchor": a, "count": 0, "mean": np.nan,
                         "median": np.nan, "std": np.nan, "min": np.nan, "max": np.nan})
        else:
            rows.append({
                "anchor": a,
                "count": int(sub.shape[0]),
                "mean": float(sub["lap_var_norm"].mean()),
                "median": float(sub["lap_var_norm"].median()),
                "std": float(sub["lap_var_norm"].std(ddof=0)),
                "min": float(sub["lap_var_norm"].min()),
                "max": float(sub["lap_var_norm"].max()),
            })

    agg_anchor = (pd.DataFrame(rows)
                    .set_index("anchor")
                    .sort_values("mean", ascending=False))

    agg_anchor_path = OUT_DIR / "laplace_agg_by_anchor.csv"
    agg_anchor.to_csv(agg_anchor_path)
    print(f"\n[OK] Aggregat pro Anchor gespeichert: {agg_anchor_path}")
    print("\n=== Anchors nach Mittelwert (Lap Var Norm) ===")
    print(agg_anchor)
