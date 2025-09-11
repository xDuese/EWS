from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .io_utils import ensure_dirs, load_params, Params, get_time_series, find_column_case_insensitive
from .cleaning import CleaningConfig, fuse_gaze_samples, interpolate_and_smooth, preprocess_pupil
from .fixations import IDTConfig, idt_fixations
from .saccades import derive_saccades
from .visuals import save_heatmap, save_scanpath
from .summaries import compute_trial_summary, aggregate_image_summary, aggregate_participant_summary


# Write outputs under code/preprocess to combine code and artifacts
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # code/preprocess
OUT_BASE = BASE_DIR
DIRS = {
    "cleaned": os.path.join(OUT_BASE, "cleaned"),
    "fixations": os.path.join(OUT_BASE, "events", "fixations"),
    "saccades": os.path.join(OUT_BASE, "events", "saccades"),
    "heatmaps": os.path.join(OUT_BASE, "heatmaps"),
    "scanpaths": os.path.join(OUT_BASE, "scanpaths"),
    "summaries": os.path.join(OUT_BASE, "summaries"),
}


def _to_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # Fallback to CSV if parquet engine missing; still save data
        alt = os.path.splitext(path)[0] + ".csv"
        df.to_csv(alt, index=False)


def load_input(csv_path: str, params: Params) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure time column exists for downstream processing
    df["timestamp_ms"] = get_time_series(df, params.sampling_rate_hz)
    return df


def extract_trial_keys(df: pd.DataFrame) -> Dict[str, Dict[str, Optional[str]]]:
    """Return mapping trial_id -> {participant_id, image_id} using best-effort column search."""
    trial_col = find_column_case_insensitive(df, ["trial_id", "trial", "trialid", "recording_session_id"])
    if trial_col is None:
        # Create a single trial if missing
        return {"trial_0": {"participant_id": find_column_case_insensitive(df, ["participant_id", "participant", "pid"]) or None,
                              "image_id": find_column_case_insensitive(df, ["image_id", "image", "stimulus", "stim_id"]) or None}}

    part_col = find_column_case_insensitive(df, ["participant_id", "participant", "pid"]) \
               or find_column_case_insensitive(df, ["user", "subject_id", "subject"]) \
               or None
    img_col = find_column_case_insensitive(df, ["image_id", "image", "stimulus", "stim_id", "stimulus_name"]) or None

    keys = {}
    for tid, sub in df.groupby(trial_col, sort=False):
        participant_id = str(sub[part_col].iloc[0]) if part_col else None
        image_id = str(sub[img_col].iloc[0]) if img_col else None
        keys[str(tid)] = {"participant_id": participant_id, "image_id": image_id}
    return keys


def process_trial(trial_id: str, trial_df: pd.DataFrame, params: Params) -> Dict[str, str]:
    # Cleaning configuration
    c_cfg = CleaningConfig(
        sampling_rate_hz=float(params.sampling_rate_hz),
        interp_max_gap_ms=int(params.interp_max_gap_ms),
        pupil_interp_max_gap_ms=int(params.pupil_interp_max_gap_ms),
        pupil_pad_invalid_ms=int(params.pupil_pad_invalid_ms),
        validity=str(params.validity),
    )

    # Gaze fusion and smoothing
    fused = fuse_gaze_samples(trial_df, c_cfg)
    fused = interpolate_and_smooth(fused, c_cfg)

    # Pupil processing aligned to fused index/timebase
    df_pupil = trial_df.loc[fused.index].copy()
    df_pupil["timestamp_ms"] = fused["timestamp_ms"].values
    pupil = preprocess_pupil(df_pupil, c_cfg)
    pupil = pupil.reindex(fused.index)
    fused["pupil"] = pupil.values

    # Save cleaned samples
    cleaned_path = os.path.join(DIRS["cleaned"], f"{trial_id}.parquet")
    _to_parquet(fused, cleaned_path)

    # Fixations (I-DT)
    f_cfg = IDTConfig(
        min_fix_dur_ms=int(params.min_fix_dur_ms),
        dispersion_thresh_norm=float(params.dispersion_thresh_norm),
        merge_fix_gap_ms=int(params.merge_fix_gap_ms),
        merge_fix_dist_norm=float(params.merge_fix_dist_norm),
    )
    fixations = idt_fixations(fused[["timestamp_ms", "x", "y"]], f_cfg)
    fx_path = os.path.join(DIRS["fixations"], f"{trial_id}.parquet")
    _to_parquet(fixations, fx_path)

    # Saccades
    saccades = derive_saccades(fixations)
    sc_path = os.path.join(DIRS["saccades"], f"{trial_id}.parquet")
    _to_parquet(saccades, sc_path)

    # Visuals
    heatmap_path = os.path.join(DIRS["heatmaps"], f"{trial_id}.png")
    save_heatmap(fixations, heatmap_path, sigma_norm=0.03)
    scanpath_path = os.path.join(DIRS["scanpaths"], f"{trial_id}.png")
    save_scanpath(fixations, scanpath_path)

    return {
        "cleaned": cleaned_path,
        "fixations": fx_path,
        "saccades": sc_path,
        "heatmap": heatmap_path,
        "scanpath": scanpath_path,
    }


def run_pipeline(input_csv: str, params_path: str) -> None:
    ensure_dirs(DIRS.values())

    params = load_params(params_path)
    df = load_input(input_csv, params)

    trial_info = extract_trial_keys(df)

    trial_summ_rows = []
    valid_counts_rows = []  # for participant-level overall valid pct

    trial_col = find_column_case_insensitive(df, ["trial_id", "trial", "trialid", "recording_session_id"])
    if trial_col is None:
        groups = [("trial_0", df)]
    else:
        groups = list(df.groupby(trial_col, sort=False))

    for tid, sub in groups:
        tid = str(tid)
        outputs = process_trial(tid, sub.copy(), params)

        # Load back saved cleaned/fixations/saccades for summary metrics
        try:
            cleaned = pd.read_parquet(outputs["cleaned"]) if outputs["cleaned"].endswith(".parquet") else pd.read_csv(outputs["cleaned"])  # type: ignore
        except Exception:
            cleaned = pd.read_csv(outputs["cleaned"].replace(".parquet", ".csv"))
        try:
            fx = pd.read_parquet(outputs["fixations"]) if outputs["fixations"].endswith(".parquet") else pd.read_csv(outputs["fixations"])  # type: ignore
        except Exception:
            fx = pd.read_csv(outputs["fixations"].replace(".parquet", ".csv"))
        try:
            sc = pd.read_parquet(outputs["saccades"]) if outputs["saccades"].endswith(".parquet") else pd.read_csv(outputs["saccades"])  # type: ignore
        except Exception:
            sc = pd.read_csv(outputs["saccades"].replace(".parquet", ".csv"))

        info = trial_info.get(tid, {"participant_id": None, "image_id": None})
        trial_summ_rows.append(compute_trial_summary(tid, info.get("participant_id"), info.get("image_id"), cleaned, fx, sc))

        # Sample counts for participant-level overall valid pct
        n_total = int(len(sub))
        n_valid = int(cleaned[["x", "y"]].notna().all(axis=1).sum())
        valid_counts_rows.append({"trial_id": tid, "n_total": n_total, "n_valid": n_valid})

    trial_summary = pd.DataFrame(trial_summ_rows)
    trial_summary_path = os.path.join(DIRS["summaries"], "trial_summary.csv")
    trial_summary.to_csv(trial_summary_path, index=False)

    per_trial_valid_counts = pd.DataFrame(valid_counts_rows)
    image_summary = aggregate_image_summary(trial_summary)
    image_summary_path = os.path.join(DIRS["summaries"], "image_summary.csv")
    image_summary.to_csv(image_summary_path, index=False)

    participant_summary = aggregate_participant_summary(trial_summary, per_trial_valid_counts)
    participant_summary_path = os.path.join(DIRS["summaries"], "participant_summary.csv")
    participant_summary.to_csv(participant_summary_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Eye-tracking preprocessing pipeline")
    parser.add_argument("--input-csv", required=True, help="Path to input CSV with all samples")
    default_params = os.path.join(os.path.dirname(__file__), "params.yaml")
    parser.add_argument("--params", default=default_params, help="Path to params.yaml")
    args = parser.parse_args()

    run_pipeline(args.input_csv, args.params)


if __name__ == "__main__":
    main()
