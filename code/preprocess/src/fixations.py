from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class IDTConfig:
    min_fix_dur_ms: int
    dispersion_thresh_norm: float
    merge_fix_gap_ms: int
    merge_fix_dist_norm: float


def _dispersion(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0:
        return np.inf
    return (np.nanmax(x) - np.nanmin(x)) + (np.nanmax(y) - np.nanmin(y))


def idt_fixations(samples: pd.DataFrame, cfg: IDTConfig) -> pd.DataFrame:
    """Identify fixations with I-DT on normalized x,y.

    samples: DataFrame with columns ['timestamp_ms','x','y'] and valid rows only.
    Returns a DataFrame of fixations with columns: ['start_ms','end_ms','duration_ms','x','y','n_samples']
    """
    df = samples.dropna(subset=["x", "y"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["start_ms", "end_ms", "duration_ms", "x", "y", "n_samples"]).astype({
            "start_ms": float, "end_ms": float, "duration_ms": float, "x": float, "y": float, "n_samples": int
        })

    t = df["timestamp_ms"].to_numpy()
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()

    fixations: List[Tuple[float, float, float, float, int]] = []  # start_ms, end_ms, cx, cy, n
    i = 0
    min_dur = cfg.min_fix_dur_ms
    while i < len(df):
        # Grow window until min duration is reached
        j = i
        while j < len(df) and (t[j] - t[i]) < min_dur:
            j += 1
        if j >= len(df):
            break

        # Check dispersion within window
        while True:
            d = _dispersion(xs[i:j+1], ys[i:j+1])
            if d <= cfg.dispersion_thresh_norm:
                # Try to include more points while staying within threshold
                k = j + 1
                while k < len(df) and _dispersion(xs[i:k+1], ys[i:k+1]) <= cfg.dispersion_thresh_norm:
                    k += 1
                # Register fixation from i..k-1
                x_mean = float(np.nanmean(xs[i:k]))
                y_mean = float(np.nanmean(ys[i:k]))
                start_ms = float(t[i])
                end_ms = float(t[k-1])
                fixations.append((start_ms, end_ms, x_mean, y_mean, k - i))
                i = k  # advance past this fixation
                break
            else:
                # Slide window forward
                i += 1
                # Re-establish j for new i
                j = i
                while j < len(df) and (t[j] - t[i]) < min_dur:
                    j += 1
                if j >= len(df):
                    i = len(df)
                    break

    if not fixations:
        return pd.DataFrame(columns=["start_ms", "end_ms", "duration_ms", "x", "y", "n_samples"]).astype({
            "start_ms": float, "end_ms": float, "duration_ms": float, "x": float, "y": float, "n_samples": int
        })

    fx = pd.DataFrame(fixations, columns=["start_ms", "end_ms", "x", "y", "n_samples"])
    fx["duration_ms"] = fx["end_ms"] - fx["start_ms"]
    fx = fx[["start_ms", "end_ms", "duration_ms", "x", "y", "n_samples"]]
    return _merge_fixations(fx, cfg)


def _merge_fixations(fx: pd.DataFrame, cfg: IDTConfig) -> pd.DataFrame:
    if fx.empty:
        return fx
    merged = []
    cur = fx.iloc[0].to_dict()
    for _, row in fx.iloc[1:].iterrows():
        gap = row["start_ms"] - cur["end_ms"]
        dist = float(np.hypot(row["x"] - cur["x"], row["y"] - cur["y"]))
        if gap <= cfg.merge_fix_gap_ms and dist <= cfg.merge_fix_dist_norm:
            # Merge into current
            total_dur = (cur["end_ms"] - cur["start_ms"]) + (row["end_ms"] - row["start_ms"]) + gap
            # Weighted center by duration of each fixation
            w1 = cur["end_ms"] - cur["start_ms"]
            w2 = row["end_ms"] - row["start_ms"]
            if (w1 + w2) > 0:
                cur["x"] = float((cur["x"] * w1 + row["x"] * w2) / (w1 + w2))
                cur["y"] = float((cur["y"] * w1 + row["y"] * w2) / (w1 + w2))
            cur["end_ms"] = float(row["end_ms"])  # extend end
            cur["n_samples"] = int(cur.get("n_samples", 0) + row["n_samples"])
            cur["duration_ms"] = float(cur["end_ms"] - cur["start_ms"])  # recompute
        else:
            merged.append(cur)
            cur = row.to_dict()
    merged.append(cur)
    return pd.DataFrame(merged)[["start_ms", "end_ms", "duration_ms", "x", "y", "n_samples"]]


