"""Validate and combine training-time dynamics saved by train_snli.py."""
import argparse
import json
from pathlib import Path

import pandas as pd

from cartography_repro.data import read_snli


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="outputs/checkpoints/full_seed93078")
    parser.add_argument("--train", default="data/snli/train.tsv")
    parser.add_argument("--output", default="outputs/cartography/training_dynamics.csv")
    args = parser.parse_args()
    train = read_snli(args.train, expected_count=549367)
    files = sorted((Path(args.run) / "training_dynamics").glob("epoch_*.csv"), key=lambda path: int(path.stem.split("_")[-1]))
    if not files:
        raise ValueError("No training-time dynamics found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    expected_ids = set(train.example_id)
    for epoch, path in enumerate(files):
        frame = pd.read_csv(path)
        if frame.epoch.nunique() != 1 or frame.epoch.iloc[0] != epoch:
            raise ValueError(f"Unexpected epoch number in {path}")
        if not set(frame.example_id).issubset(expected_ids):
            raise ValueError(f"Unknown example IDs in epoch {epoch}")
        # Gradient accumulation can cross a dataloader epoch boundary, repeating
        # the final microbatch and shifting a few examples into the next epoch.
        # Keep one observation per ID and use only IDs observed in every epoch.
        frames.append(frame.drop_duplicates("example_id", keep="first"))
    common_ids = set.intersection(*(set(frame.example_id) for frame in frames))
    if len(common_ids) < 0.99 * len(train):
        raise ValueError(f"Dynamics coverage too low: {len(common_ids)}/{len(train)}")
    for epoch, frame in enumerate(frames):
        frame = frame[frame.example_id.isin(common_ids)].sort_values("example_id")
        frame.to_csv(output, mode="w" if epoch == 0 else "a", index=False, header=epoch == 0)
    coverage = {
        "epochs": len(frames),
        "expected_examples": len(train),
        "mapped_examples": len(common_ids),
        "coverage": len(common_ids) / len(train),
        "excluded_examples": len(train) - len(common_ids),
    }
    output.with_name("training_dynamics_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")


if __name__ == "__main__":
    main()
