"""Produce a paper-vs-measured table; missing runs remain missing."""
from pathlib import Path

import pandas as pd

from cartography_repro.evaluation import paper_comparison


def main():
    records = [pd.read_json(path, typ="series") for path in Path("outputs/checkpoints").glob("*/metrics.json")]
    results = pd.DataFrame(records) if records else pd.DataFrame(columns=["subset", "id_accuracy", "ood_accuracy"])
    output = Path("outputs/tables")
    output.mkdir(parents=True, exist_ok=True)
    table = paper_comparison(results)
    table.to_csv(output / "paper_vs_reproduction.csv", index=False)
    headers = list(table.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |")
    (output / "paper_vs_reproduction.md").write_text("\n".join(lines) + "\n")
    if len(results):
        Path("outputs/results").mkdir(parents=True, exist_ok=True)
        results.to_csv(Path("outputs/results/snli_reproduction.csv"), index=False)
        aggregate = results.groupby("subset").agg(
            runs=("seed", "count"),
            id_mean=("id_accuracy", "mean"), id_std=("id_accuracy", "std"),
            ood_mean=("ood_accuracy", "mean"), ood_std=("ood_accuracy", "std"),
        ).reset_index()
        aggregate.to_csv(Path("outputs/results/snli_aggregate.csv"), index=False)


if __name__ == "__main__":
    main()
