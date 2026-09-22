"""Load the original Stanford SNLI release, retaining pair IDs."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

LABELS = ("entailment", "neutral", "contradiction")
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
SNLI_COUNTS = {"train": 549367, "dev": 9842, "test": 9824}


def numeric_guid(guid: str) -> int:
    """Reproduce the AllenAI SNLI GUID encoding in data_utils_glue.py."""
    prefix = "555" if guid.startswith("vg_len") else "444" if guid.startswith("vg_verb") else "000"
    digits = re.sub(r"\D", "", guid)
    if not digits:
        raise ValueError(f"Cannot encode SNLI pairID: {guid!r}")
    return int(prefix + digits + {"e": "0", "c": "1", "n": "2"}.get(guid[-1], "3"))


def read_snli(path: str | Path, *, expected_count: int | None = None) -> pd.DataFrame:
    """Match the original processor: sentence1/2, pairID, and valid gold labels."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        needed = {"pairID", "sentence1", "sentence2", "gold_label"}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing SNLI columns: {needed - set(reader.fieldnames or [])}")
        for row in reader:
            label = row["gold_label"]
            if label in ("-", ""):
                continue
            if label not in LABEL_TO_ID:
                raise ValueError(f"Unexpected SNLI label {label!r}")
            rows.append((row["pairID"], numeric_guid(row["pairID"]), row["sentence1"], row["sentence2"], label, LABEL_TO_ID[label]))
    frame = pd.DataFrame(rows, columns=["pair_id", "example_id", "premise", "hypothesis", "label", "label_id"])
    if frame.example_id.duplicated().any() or frame.pair_id.duplicated().any():
        raise ValueError("Duplicate SNLI ID")
    if expected_count is not None and len(frame) != expected_count:
        raise ValueError(f"Expected {expected_count} valid examples, found {len(frame)}")
    return frame


def read_diagnostics(path: str | Path) -> pd.DataFrame:
    """Read GLUE diagnostic-full.tsv; score each of its 1,104 examples once."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        for i, fields in enumerate(reader):
            if len(fields) < 4:
                raise ValueError(f"Malformed diagnostic row {i}")
            label = fields[-1]
            if label not in LABEL_TO_ID:
                raise ValueError(f"Unexpected diagnostic label {label!r}")
            rows.append((i, fields[-3], fields[-2], label, LABEL_TO_ID[label]))
    frame = pd.DataFrame(rows, columns=["example_id", "premise", "hypothesis", "label", "label_id"])
    if len(frame) != 1104:
        raise ValueError(f"Expected 1104 diagnostic examples, found {len(frame)}")
    return frame
