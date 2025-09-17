from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .io_utils import find_column_case_insensitive, parse_point_column, get_time_series


@dataclass
class CleaningConfig:
    sampling_rate_hz: float
    interp_max_gap_ms: int
    pupil_interp_max_gap_ms: int
    pupil_pad_invalid_ms: int
    stim_x0_norm: float = 0.0
    stim_x1_norm: float = 1.0
    stim_y0_norm: float = 0.0
    stim_y1_norm: float = 1.0
    validity: str = "at_least_one_eye"


def moving_average(series: pd.Series, window: int = 3) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


def _interp_short_gaps(ts_ms: pd.Series, values: pd.Series, max_gap_ms: int) -> pd.Series:
    v = values.copy()
    isnan = v.isna().values
    if not isnan.any():
        return v

    # Identify contiguous NaN runs
    idx = np.arange(len(v))
    run_starts = []
    run_ends = []
    in_run = False
    for i, nan in enumerate(isnan):
        if nan and not in_run:
            run_starts.append(i)
            in_run = True
        elif not nan and in_run:
            run_ends.append(i - 1)
            in_run = False
    if in_run:
        run_ends.append(len(v) - 1)

    for s, e in zip(run_starts, run_ends):
        # Determine temporal gap length
        # Need valid neighbors on both sides to interpolate
        left = s - 1
        right = e + 1
        if left < 0 or right >= len(v):
            continue
        if pd.isna(v.iloc[left]) or pd.isna(v.iloc[right]):
            continue
        gap_ms = ts_ms.iloc[right] - ts_ms.iloc[left]
        if gap_ms <= max_gap_ms:
            # Linear interpolation across the gap (including interior NaNs)
            v.iloc[left:right + 1] = np.interp(
                ts_ms.iloc[left:right + 1].values,
                [ts_ms.iloc[left], ts_ms.iloc[right]],
                [v.iloc[left], v.iloc[right]],
            )
    return v


def _combine_two_channels(a: pd.Series, a_valid: pd.Series, b: pd.Series, b_valid: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Combine two channels: mean if both valid, else the valid one.
    Returns (combined_value, any_valid) where any_valid is boolean mask.
    """
    a_val = pd.to_numeric(a, errors="coerce")
    b_val = pd.to_numeric(b, errors="coerce")
    a_ok = a_valid.fillna(False) & a_val.notna()
    b_ok = b_valid.fillna(False) & b_val.notna()
    any_ok = a_ok | b_ok
    both_ok = a_ok & b_ok
    out = pd.Series(np.nan, index=a.index, dtype=float)
    out[both_ok] = (a_val[both_ok].values + b_val[both_ok].values) / 2.0
    out[~both_ok & a_ok] = a_val[~both_ok & a_ok]
    out[~both_ok & b_ok] = b_val[~both_ok & b_ok]
    return out, any_ok


def _infer_validity(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(False, index=pd.RangeIndex(0))
    s = series.copy()
    if s.dtype == bool:
        return s
    # Normalize typical encodings
    s = s.map(lambda x: str(x).strip().lower() if pd.notna(x) else "")
    valid_values = {"1", "true", "valid", "yes", "ok"}
    return s.isin(valid_values)


def fuse_gaze_samples(df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """Parse left/right gaze to lx,ly, rx,ry, fuse into x,y, drop rows with neither eye valid.

    Expected columns (best-effort, case-insensitive):
    - left/right gaze point (single column with "x,y" strings) OR separate lx,ly and rx,ry
    - left/right validity (boolean/int/string). If absent, validity inferred from coordinate presence in [0,1].
    """
    out = df.copy()

    # Find or parse left/right gaze coordinates
    l_col = find_column_case_insensitive(out, ["left_gaze_point_on_display_area", "left_point", "left_gaze_point"])
    r_col = find_column_case_insensitive(out, ["right_gaze_point_on_display_area", "right_point", "right_gaze_point"])

    if l_col is not None:
        out["lx"], out["ly"] = parse_point_column(out[l_col])
    else:
        lx = find_column_case_insensitive(out, ["lx", "left_x", "left_gaze_x"]) or "lx"
        ly = find_column_case_insensitive(out, ["ly", "left_y", "left_gaze_y"]) or "ly"
        # If missing, fill NaNs
        if lx not in out:
            out[lx] = np.nan
        if ly not in out:
            out[ly] = np.nan
        out.rename(columns={lx: "lx", ly: "ly"}, inplace=True)

    if r_col is not None:
        out["rx"], out["ry"] = parse_point_column(out[r_col])
    else:
        rx = find_column_case_insensitive(out, ["rx", "right_x", "right_gaze_x"]) or "rx"
        ry = find_column_case_insensitive(out, ["ry", "right_y", "right_gaze_y"]) or "ry"
        if rx not in out:
            out[rx] = np.nan
        if ry not in out:
            out[ry] = np.nan
        out.rename(columns={rx: "rx", ry: "ry"}, inplace=True)

    # Validities
    l_val_col = find_column_case_insensitive(out, ["left_gaze_point_validity", "left_valid", "left_eye_valid"])
    r_val_col = find_column_case_insensitive(out, ["right_gaze_point_validity", "right_valid", "right_eye_valid"])
    if l_val_col is not None:
        l_valid = _infer_validity(out[l_val_col])
    else:
        l_valid = out[["lx", "ly"]].notna().all(axis=1)
        l_valid &= out["lx"].between(0.0, 1.0) & out["ly"].between(0.0, 1.0)

    if r_val_col is not None:
        r_valid = _infer_validity(out[r_val_col])
    else:
        r_valid = out[["rx", "ry"]].notna().all(axis=1)
        r_valid &= out["rx"].between(0.0, 1.0) & out["ry"].between(0.0, 1.0)

    # Combine into x,y according to policy
    if cfg.validity == "at_least_one_eye":
        x_comb, any_ok_x = _combine_two_channels(out["lx"], l_valid, out["rx"], r_valid)
        y_comb, any_ok_y = _combine_two_channels(out["ly"], l_valid, out["ry"], r_valid)
        valid = any_ok_x & any_ok_y
    else:
        # Default to both-eyes-valid policy if unknown
        both_ok = l_valid & r_valid
        x_comb = (out["lx"] + out["rx"]) / 2.0
        y_comb = (out["ly"] + out["ry"]) / 2.0
        valid = both_ok

    out["x_display"] = x_comb
    out["y_display"] = y_comb
    x0 = float(cfg.stim_x0_norm)
    x1 = float(cfg.stim_x1_norm)
    y0 = float(cfg.stim_y0_norm)
    y1 = float(cfg.stim_y1_norm)
    width = max(1e-9, x1 - x0)
    height = max(1e-9, y1 - y0)
    x_img = (x_comb - x0) / width
    y_img = (y_comb - y0) / height
    on_stimulus = x_img.between(0.0, 1.0) & y_img.between(0.0, 1.0)
    x_img = x_img.clip(0.0, 1.0)
    out["x"] = x_img
    y_img = y_img.clip(0.0, 1.0)
    out["y"] = y_img
    out["on_stimulus"] = on_stimulus
    valid = valid & on_stimulus

    # Timebase
    out["timestamp_ms"] = get_time_series(out, cfg.sampling_rate_hz)

    # Drop rows with neither eye valid
    out["valid"] = valid
    out = out[valid].copy()

    return out


def interpolate_and_smooth(clean: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """Interpolate short gaps (x,y) and apply light smoothing (3-sample MA)."""
    out = clean.copy()
    ts = out["timestamp_ms"]

    out["x"] = _interp_short_gaps(ts, out["x"], cfg.interp_max_gap_ms)
    out["y"] = _interp_short_gaps(ts, out["y"], cfg.interp_max_gap_ms)

    out["x"] = moving_average(out["x"], 3)
    out["y"] = moving_average(out["y"], 3)

    # Re-flag validity after interpolation (values still within bounds)
    out["valid"] = out["x"].between(0.0, 1.0) & out["y"].between(0.0, 1.0)

    return out


def preprocess_pupil(df: pd.DataFrame, cfg: CleaningConfig) -> pd.Series:
    """Combine left/right pupil diameter, pad around invalid stretches, interpolate short gaps, baseline-correct.

    Returns a Series aligned to df index with name 'pupil'.
    """
    left_p_col = find_column_case_insensitive(df, ["left_pupil_diameter", "pupil_left", "lpupil", "left_pupil"])
    right_p_col = find_column_case_insensitive(df, ["right_pupil_diameter", "pupil_right", "rpupil", "right_pupil"])
    if left_p_col is None and right_p_col is None:
        # No pupil data
        return pd.Series(np.nan, index=df.index, name="pupil")

    # Validity columns for pupil if present
    l_val_col = find_column_case_insensitive(df, ["left_pupil_validity", "left_valid", "left_eye_valid"])
    r_val_col = find_column_case_insensitive(df, ["right_pupil_validity", "right_valid", "right_eye_valid"])

    l_valid = _infer_validity(df[l_val_col]) if l_val_col else df[left_p_col].notna() if left_p_col else pd.Series(False, index=df.index)
    r_valid = _infer_validity(df[r_val_col]) if r_val_col else df[right_p_col].notna() if right_p_col else pd.Series(False, index=df.index)

    left = pd.to_numeric(df[left_p_col], errors="coerce") if left_p_col else pd.Series(np.nan, index=df.index)
    right = pd.to_numeric(df[right_p_col], errors="coerce") if right_p_col else pd.Series(np.nan, index=df.index)

    pupil, any_ok = _combine_two_channels(left, l_valid, right, r_valid)
    pupil = pupil.astype(float)

    # Pad invalid stretches by ±pad ms
    ts = df["timestamp_ms"] if "timestamp_ms" in df else get_time_series(df, cfg.sampling_rate_hz)
    pad = int(np.ceil(cfg.pupil_pad_invalid_ms / max(1e-9, np.median(np.diff(ts.dropna().values))))) if len(ts.dropna()) > 1 else 0
    invalid = ~any_ok
    if invalid.any() and pad > 0:
        invalid_idx = invalid.to_numpy().astype(np.int8)
        # Dilate the invalid mask by pad on both sides
        kernel = np.ones(2 * pad + 1, dtype=np.int8)
        dil = np.convolve(invalid_idx, kernel, mode="same") > 0
        pupil[dil] = np.nan

    # Interpolate short gaps
    ts_series = ts if isinstance(ts, pd.Series) else pd.Series(ts)
    pupil = _interp_short_gaps(ts_series, pupil, cfg.pupil_interp_max_gap_ms)

    # Baseline correct: subtract first 200 ms median
    t0 = ts_series.iloc[0] if len(ts_series) else 0.0
    baseline_window = (ts_series - t0) <= 200.0
    baseline = np.nanmedian(pupil[baseline_window]) if baseline_window.any() else np.nanmedian(pupil)
    if np.isfinite(baseline):
        pupil = pupil - float(baseline)

    pupil.name = "pupil"
    return pupil


