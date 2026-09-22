"""Selection from released training-dynamics coordinates."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FRACTION = 0.3319  # original train_dy_filtering.py
RANDOM_FRACTION = 0.33  # original random_filtering.py


def read_coordinates(path: str | Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8")]
    frame = pd.DataFrame(rows).rename(columns={"guid": "example_id"})
    required = {"example_id", "confidence", "variability", "correctness"}
    if not required.issubset(frame):
        raise ValueError(f"Missing coordinate fields: {required - set(frame)}")
    if frame.example_id.duplicated().any():
        raise ValueError("Duplicate coordinate ID")
    return frame


def align(train: pd.DataFrame, coordinates: pd.DataFrame) -> pd.DataFrame:
    """Require a bijection; the released file has 549,367 rows."""
    if len(train) != len(coordinates) or set(train.example_id) != set(coordinates.example_id):
        missing = set(train.example_id) - set(coordinates.example_id)
        extra = set(coordinates.example_id) - set(train.example_id)
        raise ValueError(f"SNLI-coordinate mismatch: {len(missing)} missing, {len(extra)} extra")
    return train[["example_id"]].merge(coordinates[["example_id", "confidence", "variability", "correctness"]], on="example_id", validate="one_to_one")


def select(aligned: pd.DataFrame, subset: str, seed: int, fraction: float = FRACTION) -> pd.DataFrame:
    """Match original sort direction and its head(n=int(f*N)+1) behavior."""
    if subset == "full":
        chosen = aligned.copy()
    else:
        n = int((RANDOM_FRACTION if subset == "random33" else fraction) * len(aligned))
        if subset == "random33":
            chosen = aligned.sample(n=n, random_state=seed).copy()
        elif subset in ("ambiguous33", "hard33"):
            metric = "variability" if subset == "ambiguous33" else "confidence"
            chosen = aligned.sort_values(metric, ascending=subset == "hard33", kind="stable").head(n + 1).copy()
        else:
            raise ValueError(f"Unknown subset: {subset}")
        expected = n if subset == "random33" else n + 1
        if len(chosen) != expected:
            raise ValueError("Unexpected selected size")
    chosen.insert(1, "subset", subset)
    return chosen[["example_id", "subset", "confidence", "variability", "correctness"]]
