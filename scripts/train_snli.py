"""Train one configured SNLI condition."""
import argparse
import logging
from pathlib import Path

import yaml

from cartography_repro.training import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/snli_full.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-checkpoint")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = yaml.safe_load(Path(args.config).read_text())
    for name in ("max_steps", "output_dir", "resume_checkpoint"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    run(config, max_train_examples=args.max_train_examples, smoke=args.smoke)


if __name__ == "__main__":
    main()
