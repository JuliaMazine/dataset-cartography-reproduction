"""Evaluate a saved classifier on SNLI dev and test."""
import argparse
import json
from pathlib import Path

from cartography_repro.data import read_snli
from cartography_repro.evaluation import accuracy
from cartography_repro.inference import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/snli")
    parser.add_argument("--output", default="outputs/results/snli_evaluation.json")
    args = parser.parse_args()
    results = {}
    for split in ("dev", "test"):
        frame = read_snli(Path(args.data_dir) / f"{split}.tsv")
        pred = predict(args.checkpoint, frame)
        results[f"{split}_accuracy"] = accuracy(pred.gold_label.to_numpy(), pred.predicted_label.to_numpy())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
