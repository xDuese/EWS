from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def compute_trial_summary(
    trial_id: str,
    participant_id: Optional[str],
    image_id: Optional[str],
    cleaned: pd.DataFrame,
    fixations: pd.DataFrame,
    saccades: pd.DataFrame,
) -> Dict[str, object]:
    duration_s = float((cleaned["timestamp_ms"].iloc[-1] - cleaned["timestamp_ms"].iloc[0]) / 1000.0) if len(cleaned) > 1 else 0.0
    valid_pct = float(cleaned[["x", "y"]].notna().all(axis=1).mean()) if len(cleaned) else 0.0
    n_fix = int(len(fixations)) if fixations is not None else 0
    mean_fix_dur = float(fixations["duration_ms"].mean()) if n_fix > 0 else np.nan
    first_fix_latency = float(fixations["start_ms"].iloc[0] - cleaned["timestamp_ms"].iloc[0]) if n_fix > 0 else np.nan
    mean_sacc_amp = float(saccades["amp"].mean()) if saccades is not None and len(saccades) > 0 else np.nan

    # Pupil mean 0.5..2.0 s window
    t0 = cleaned["timestamp_ms"].iloc[0] if len(cleaned) else 0.0
    m = (cleaned["timestamp_ms"] - t0).between(500.0, 2000.0)
    pupil_mean_0p5_2s = float(cleaned.loc[m, "pupil"].mean()) if "pupil" in cleaned and m.any() else np.nan

    return {
        "trial_id": str(trial_id),
        "participant_id": None if participant_id is None else str(participant_id),
        "image_id": None if image_id is None else str(image_id),
        "duration_s": duration_s,
        "valid_pct": valid_pct,
        "n_fix": n_fix,
        "mean_fix_dur": mean_fix_dur,
        "first_fix_latency": first_fix_latency,
        "mean_sacc_amp": mean_sacc_amp,
        "pupil_mean_0p5_2s": pupil_mean_0p5_2s,
    }


def aggregate_image_summary(trial_summary: pd.DataFrame) -> pd.DataFrame:
    # Aggregate by image_id with means; keep counts
    agg_cols = ["duration_s", "valid_pct", "n_fix", "mean_fix_dur", "first_fix_latency", "mean_sacc_amp", "pupil_mean_0p5_2s"]
    g = trial_summary.groupby("image_id", dropna=False)
    out = g[agg_cols].mean().reset_index()
    out["n_trials"] = g.size().values
    return out


def aggregate_participant_summary(trial_summary: pd.DataFrame, per_trial_valid_counts: pd.DataFrame) -> pd.DataFrame:
    # overall valid_pct: fraction of valid samples across all trials for participant
    # dropped_trials: number of trials with zero fixations
    g = trial_summary.groupby("participant_id", dropna=False)
    dropped = g.apply(lambda df: int((df["n_fix"] == 0).sum())).rename("dropped_trials").reset_index()

    # Merge per-trial valid counts to compute weighted valid percent across all samples
    merged = trial_summary[["participant_id", "trial_id"]].merge(per_trial_valid_counts, on=["trial_id"], how="left")
    g2 = merged.groupby("participant_id", dropna=False)
    valid_pct_overall = (g2.apply(lambda df: float((df["n_valid"].sum() / max(1, df["n_total"].sum()))))
                         .rename("valid_pct").reset_index())

    out = dropped.merge(valid_pct_overall, on="participant_id", how="outer")
    return out


