"""Aggregate training dynamics and plot the data map."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cartography_repro.data import read_snli
from cartography_repro.dynamics import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamics", default="outputs/cartography/training_dynamics.csv")
    parser.add_argument("--train", default="data/snli/train.tsv")
    parser.add_argument("--output-dir", default="outputs/cartography")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    coords = summarize(pd.read_csv(args.dynamics))
    coords.to_csv(output / "our_snli_coordinates.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    sample = coords.sample(n=min(55000, len(coords)), random_state=93078)
    sc = ax.scatter(sample.variability, sample.confidence, c=sample.correctness, s=2, alpha=0.35, rasterized=True)
    ax.set(xlabel="Variability", ylabel="Confidence", title="SNLI training dynamics")
    fig.colorbar(sc, ax=ax, label="Correct epochs")
    fig.tight_layout()
    fig.savefig(output / "snli_data_map.png")
    fig.savefig(output / "snli_data_map.pdf")
    plt.close(fig)
    train = read_snli(args.train)
    examples = coords.merge(train[["example_id", "premise", "hypothesis", "label"]], on="example_id", validate="one_to_one")
    for name, metric, ascending in (("easy", "confidence", False), ("ambiguous", "variability", False), ("hard", "confidence", True)):
        examples.sort_values(metric, ascending=ascending).head(20).to_csv(output / f"{name}_examples.csv", index=False)


if __name__ == "__main__":
    main()
