"""Validate and combine training-time dynamics saved by train_snli.py."""
import argparse
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
    for epoch, path in enumerate(files):
        frame = pd.read_csv(path)
        if frame.epoch.nunique() != 1 or frame.epoch.iloc[0] != epoch:
            raise ValueError(f"Unexpected epoch number in {path}")
        if len(frame) != len(train) or frame.example_id.duplicated().any() or set(frame.example_id) != set(train.example_id):
            raise ValueError(f"Incomplete or duplicate dynamics for epoch {epoch}")
        frame.to_csv(output, mode="w" if epoch == 0 else "a", index=False, header=epoch == 0)


if __name__ == "__main__":
    main()
