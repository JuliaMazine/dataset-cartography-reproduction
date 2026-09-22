"""Training-dynamics metrics and deterministic label corruption."""
from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Population standard deviation and count of correct epochs, as in AllenAI code."""
    required = {"example_id", "epoch", "gold_probability", "correct"}
    if not required.issubset(frame):
        raise ValueError(f"Missing dynamics columns: {required - set(frame)}")
    if frame.duplicated(["example_id", "epoch"]).any():
        raise ValueError("Duplicate example/epoch")
    counts = frame.groupby("example_id").size()
    if counts.nunique() != 1:
        raise ValueError("Incomplete dynamics epochs")
    result = frame.groupby("example_id", as_index=False).agg(
        confidence=("gold_probability", "mean"),
        variability=("gold_probability", lambda x: float(np.std(x, ddof=0))),
        correctness=("correct", "sum"),
    )
    return result


def corrupt_labels(labels: np.ndarray, rate: float, seed: int) -> pd.DataFrame:
    if not 0 <= rate <= 1 or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Expected three valid NLI labels and rate in [0,1]")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(labels), int(rate * len(labels)), replace=False)
    new = labels.copy()
    new[chosen] = (new[chosen] + rng.integers(1, 3, size=len(chosen))) % 3
    return pd.DataFrame({"original_label": labels, "corrupted_label": new, "is_corrupted": labels != new})
