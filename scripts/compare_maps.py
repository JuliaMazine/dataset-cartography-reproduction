"""Compare locally measured maps with the released coordinates."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from cartography_repro.subsets import read_coordinates, select


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", default="outputs/cartography/our_snli_coordinates.csv")
    parser.add_argument("--paper", default="data/original/snli_coordinates.jsonl")
    parser.add_argument("--output", default="outputs/results/map_comparison.json")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ours = pd.read_csv(args.ours)
    paper = read_coordinates(args.paper)
    joined = paper.merge(ours, on="example_id", suffixes=("_paper", "_ours"), validate="one_to_one")
    if len(joined) != len(paper):
        raise ValueError("Our map lacks released coordinate IDs")
    result = {"num_examples": len(joined)}
    for metric in ("confidence", "variability"):
        result[metric] = {
            "pearson": float(pearsonr(joined[f"{metric}_paper"], joined[f"{metric}_ours"]).statistic),
            "spearman": float(spearmanr(joined[f"{metric}_paper"], joined[f"{metric}_ours"]).statistic),
        }
        fig, ax = plt.subplots(figsize=(6, 6))
        sample = joined.sample(n=min(20000, len(joined)), random_state=93078)
        ax.scatter(sample[f"{metric}_paper"], sample[f"{metric}_ours"], s=2, alpha=0.25, rasterized=True)
        ax.set(xlabel=f"Paper {metric}", ylabel=f"Our {metric}")
        fig.tight_layout()
        fig.savefig(Path(args.output).with_name(f"{metric}_comparison.png"), dpi=180)
        plt.close(fig)
    count = int(0.3319 * len(joined)) + 1
    for name, metric, ascending in (("ambiguous", "variability", False), ("hard", "confidence", True)):
        p = set(joined.sort_values(f"{metric}_paper", ascending=ascending).head(count).example_id)
        o = set(joined.sort_values(f"{metric}_ours", ascending=ascending).head(count).example_id)
        result[name] = {"overlap": len(p & o) / len(p), "jaccard": len(p & o) / len(p | o)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
