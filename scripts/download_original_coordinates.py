"""Fetch the authors' released SNLI coordinates from a pinned commit."""
import argparse
import logging
import urllib.request
from pathlib import Path

from cartography_repro.subsets import read_coordinates

COMMIT = "3df3438fcc7324e706dee2e787426389bbd8fb1a"
URL = f"https://raw.githubusercontent.com/allenai/cartography/{COMMIT}/data/data_map_coordinates/snli_roberta_0_6_data_map_coordinates.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/original/snli_coordinates.jsonl")
    parser.add_argument("--source", help="Existing copy of the same released file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        if args.source:
            import shutil
            shutil.copyfile(args.source, dest)
        else:
            urllib.request.urlretrieve(URL, dest)
    frame = read_coordinates(dest)
    if len(frame) != 549367:
        raise ValueError(f"Expected 549367 coordinates, found {len(frame)}")
    logging.info("Verified %d unique released SNLI coordinates", len(frame))


if __name__ == "__main__":
    main()
