"""Create ID-only manifests after complete coordinate alignment."""
import argparse
import logging
from pathlib import Path

from cartography_repro.data import read_snli
from cartography_repro.subsets import align, read_coordinates, select


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/snli/train.tsv")
    parser.add_argument("--coordinates", default="data/original/snli_coordinates.jsonl")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--seed", type=int, default=725862)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    train = read_snli(args.train, expected_count=549367)
    joined = align(train, read_coordinates(args.coordinates))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("full", "random33", "ambiguous33", "hard33"):
        manifest = select(joined, name, args.seed)
        manifest.to_csv(output / f"{name}.csv", index=False)
        logging.info("%s: %d IDs", name, len(manifest))


if __name__ == "__main__":
    main()
