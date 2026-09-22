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
    parser.add_argument("--physical-batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--dynamic-padding", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = yaml.safe_load(Path(args.config).read_text())
    for name in ("max_steps", "output_dir", "resume_checkpoint", "physical_batch_size", "gradient_accumulation"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.dynamic_padding:
        config["dynamic_padding"] = True
    if args.no_gradient_checkpointing:
        config["gradient_checkpointing"] = False
    run(config, max_train_examples=args.max_train_examples, smoke=args.smoke)


if __name__ == "__main__":
    main()
