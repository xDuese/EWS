from __future__ import annotations

import numpy as np
import pandas as pd


def derive_saccades(fx: pd.DataFrame) -> pd.DataFrame:
    """Compute saccades between consecutive fixations.

    Returns columns: ['start_ms','end_ms','duration_ms','amp','dir_deg']
    where direction is atan2(dy, dx) in degrees, 0° = +x, counter-clockwise positive.
    """
    if fx is None or fx.empty or len(fx) < 2:
        return pd.DataFrame(columns=["start_ms", "end_ms", "duration_ms", "amp", "dir_deg"]).astype({
            "start_ms": float, "end_ms": float, "duration_ms": float, "amp": float, "dir_deg": float
        })
    x1 = fx["x"].to_numpy()[:-1]
    y1 = fx["y"].to_numpy()[:-1]
    x2 = fx["x"].to_numpy()[1:]
    y2 = fx["y"].to_numpy()[1:]
    dx = x2 - x1
    dy = y2 - y1
    amp = np.hypot(dx, dy)
    dir_deg = np.degrees(np.arctan2(dy, dx))
    start_ms = fx["end_ms"].to_numpy()[:-1]
    end_ms = fx["start_ms"].to_numpy()[1:]
    dur = end_ms - start_ms
    sc = pd.DataFrame({
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": dur,
        "amp": amp,
        "dir_deg": dir_deg,
    })
    return sc


