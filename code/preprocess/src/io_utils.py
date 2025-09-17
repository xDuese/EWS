from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Params:
    sampling_rate_hz: float = 60.0
    validity: str = "at_least_one_eye"
    interp_max_gap_ms: int = 50
    min_fix_dur_ms: int = 60
    dispersion_thresh_norm: float = 0.05
    merge_fix_gap_ms: int = 50
    merge_fix_dist_norm: float = 0.03
    pupil_interp_max_gap_ms: int = 150
    pupil_pad_invalid_ms: int = 50
    stimulus_x0_norm: float = 0.0
    stimulus_x1_norm: float = 1.0
    stimulus_y0_norm: float = 0.0
    stimulus_y1_norm: float = 1.0


def _parse_simple_yaml(path: str) -> Dict[str, str]:
    """Minimal YAML parser for simple key: value pairs.

    - Ignores comments (# ...)
    - Strips quotes and whitespace
    - Supports int / float / str values
    """
    kv: Dict[str, str] = {}
    if not os.path.exists(path):
        return kv
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Remove comments
            line = re.split(r"\s+#", line, maxsplit=1)[0]
            if not line.strip():
                continue
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if not key:
                continue
            if not val:
                continue
            # Coerce types
            if re.fullmatch(r"[-+]?\d+", val):
                kv[key] = int(val)
            elif re.fullmatch(r"[-+]?\d*\.\d+(e[-+]?\d+)?", val, flags=re.I):
                kv[key] = float(val)
            else:
                kv[key] = val
    return kv


def load_params(params_path: str) -> Params:
    raw = _parse_simple_yaml(params_path)
    p = Params()
    for k, v in raw.items():
        if hasattr(p, k):
            setattr(p, k, v)
    return p


def ensure_dirs(paths: Iterable[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def find_column_case_insensitive(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def parse_point_column(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Parse a column that contains points encoded as strings "x,y" or "(x, y)".

    If the series is already numeric or tuple-like, this tries best-effort parsing.
    Returns (x, y) as float Series (may contain NaNs).
    """
    # If already split into two numeric subcolumns (rare), just return
    if series.dtype.kind in {"f", "i"}:
        # Numeric but single column → cannot split; return NaNs
        return pd.to_numeric(series, errors="coerce"), pd.Series(np.nan, index=series.index)

    def parse_one(val):
        if pd.isna(val):
            return (np.nan, np.nan)
        if isinstance(val, (tuple, list, np.ndarray)) and len(val) >= 2:
            try:
                return float(val[0]), float(val[1])
            except Exception:
                return (np.nan, np.nan)
        s = str(val).strip()
        s = s.strip("()[]<>")
        s = s.replace(";", ",")
        parts = [p for p in re.split(r"[,\s]+", s) if p]
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1])
            except Exception:
                return (np.nan, np.nan)
        return (np.nan, np.nan)

    arr = series.apply(parse_one)
    xs = arr.apply(lambda t: t[0])
    ys = arr.apply(lambda t: t[1])
    return pd.to_numeric(xs, errors="coerce"), pd.to_numeric(ys, errors="coerce")


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def get_time_series(df: pd.DataFrame, sampling_rate_hz: float) -> pd.Series:
    ts_col = find_column_case_insensitive(df, [
        "timestamp_ms", "time_ms", "timestamp", "recording_timestamp", "time"
    ])
    if ts_col is not None:
        ts = pd.to_numeric(df[ts_col], errors="coerce")
        # Heuristic: if appears to be in seconds, convert to ms
        if ts.dropna().between(0, 1000).mean() > 0.9 and ts.max() < 100000:  # mostly < 1000
            ts = (ts * 1000.0).round()
        return ts
    # Fallback: derive from index and sampling rate
    dt = 1000.0 / float(sampling_rate_hz)
    return pd.Series(np.arange(len(df)) * dt, index=df.index)


