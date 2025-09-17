from __future__ import annotations

import glob
import os
import sys
from typing import List

import pandas as pd


def derive_ids_from_stem(stem: str) -> tuple[str | None, str | None, str | None]:
    # Filenames like P01_IMG001_10100 -> participant_id=P01, image_id=IMG001, bincode=10100
    toks = stem.split("_")
    participant_id = toks[0] if len(toks) > 0 else None
    image_id = toks[1] if len(toks) > 1 else None
    bincode = toks[2] if len(toks) > 2 else None
    return participant_id, image_id, bincode


def build_combined(raw_dir: str = "data/raw/raw_bin",
                   out_csv: str = "code/preprocess/raw_bin_combined.csv") -> str:
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found under {raw_dir}")

    parts: List[pd.DataFrame] = []
    seen_trial_ids: set[str] = set()
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        participant_id, image_id, bincode = derive_ids_from_stem(stem)
        # Keep the entire name for trial_id (participant + image + bincode)
        trial_id = stem
        if trial_id in seen_trial_ids:
            print(f"Warning: duplicate trial_id {trial_id} from file {f}")
        seen_trial_ids.add(trial_id)
        df = pd.read_csv(f)
        # Remove any conflicting trial-like columns coming from raw exports
        drop_candidates = [c for c in df.columns if c.strip().lower() in ("trial_id", "trial", "trialid", "recording_session_id")]
        if drop_candidates:
            df = df.drop(columns=drop_candidates)
        # Our canonical keys
        df["trial_id"] = trial_id
        df["participant_id"] = participant_id
        df["image_id"] = image_id
        df["bincode"] = bincode
        parts.append(df)

    big = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    big.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(big)} rows from {len(parts)} files")
    return out_csv


def main():
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/raw_bin"
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "code/preprocess/raw_bin_combined.csv"
    build_combined(raw_dir, out_csv)


if __name__ == "__main__":
    main()
