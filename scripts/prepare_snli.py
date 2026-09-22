"""Download the official Stanford SNLI 1.0 release and extract source TSVs."""
import argparse
import logging
import urllib.request
import zipfile
from pathlib import Path

from cartography_repro.data import SNLI_COUNTS, read_snli

URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/snli")
    parser.add_argument("--archive", help="Use an already downloaded official archive")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    folder = Path(args.data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive) if args.archive else folder / "snli_1.0.zip"
    if not archive.exists():
        logging.info("Downloading %s", URL)
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        for split, count in SNLI_COUNTS.items():
            source = f"snli_1.0/snli_1.0_{split}.txt"
            destination = folder / f"{split}.tsv"
            if not destination.exists():
                with z.open(source) as input_file, destination.open("wb") as output:
                    import shutil
                    shutil.copyfileobj(input_file, output)
            frame = read_snli(destination, expected_count=count)
            logging.info("%s: %d valid examples", split, len(frame))


if __name__ == "__main__":
    main()
