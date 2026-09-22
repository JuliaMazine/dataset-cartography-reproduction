"""Evaluate aggregate three-class accuracy on original NLI Diagnostics."""
import argparse
import json
from pathlib import Path

from cartography_repro.data import read_diagnostics
from cartography_repro.evaluation import accuracy
from cartography_repro.inference import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--diagnostics", default="data/diagnostics/diagnostic-full.tsv")
    parser.add_argument("--output", default="outputs/results/diagnostics_evaluation.json")
    args = parser.parse_args()
    frame = read_diagnostics(args.diagnostics)
    pred = predict(args.checkpoint, frame)
    result = {"num_examples": len(frame), "accuracy": accuracy(pred.gold_label.to_numpy(), pred.predicted_label.to_numpy())}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
