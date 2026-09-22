"""Simple accuracy and paper-reference reports."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = {"full": (92.0, 61.8), "random33": (91.3, 60.4), "hard33": (91.8, 62.0), "ambiguous33": (92.2, 63.5)}


def accuracy(gold: np.ndarray, predicted: np.ndarray) -> float:
    if len(gold) == 0 or len(gold) != len(predicted):
        raise ValueError("Accuracy requires nonempty, equally sized arrays")
    return float(100 * np.mean(np.asarray(gold) == np.asarray(predicted)))


def paper_comparison(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset, (paper_id, paper_ood) in PAPER.items():
        own = results.loc[results.subset == subset]
        best = own.sort_values("validation_accuracy", ascending=False).iloc[0] if len(own) else None
        our_id = float(best.id_accuracy) if best is not None else np.nan
        our_ood = float(best.ood_accuracy) if best is not None else np.nan
        rows.append((subset, paper_id, our_id, abs(paper_id - our_id), paper_ood, our_ood, abs(paper_ood - our_ood)))
    return pd.DataFrame(rows, columns=["subset", "paper_snli", "ours_snli", "abs_id_difference", "paper_ood", "ours_ood", "abs_ood_difference"])
