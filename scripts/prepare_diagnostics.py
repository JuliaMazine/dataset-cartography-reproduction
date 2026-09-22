"""Download the original labeled three-class GLUE diagnostic-full.tsv."""
import argparse
import logging
import urllib.request
from pathlib import Path

from cartography_repro.data import read_diagnostics

URL = "https://dl.fbaipublicfiles.com/glue/data/diagnostic-full.tsv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/diagnostics/diagnostic-full.tsv")
    parser.add_argument("--source", help="Already downloaded original diagnostic-full.tsv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        if args.source:
            import shutil
            shutil.copyfile(args.source, output)
        else:
            urllib.request.urlretrieve(URL, output)
    frame = read_diagnostics(output)
    logging.info("Verified %d labeled NLI Diagnostics examples", len(frame))


if __name__ == "__main__":
    main()
