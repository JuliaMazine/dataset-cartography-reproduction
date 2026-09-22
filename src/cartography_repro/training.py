"""RoBERTa-large fine-tuning with resumable Hugging Face checkpoints."""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, disable_progress_bar
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    EarlyStoppingCallback, Trainer, TrainingArguments, set_seed,
)

from .data import LABELS, read_diagnostics, read_snli
from .evaluation import accuracy

LOG = logging.getLogger(__name__)


def effective_batch(physical: int, accumulation: int, devices: int = 1) -> int:
    if min(physical, accumulation, devices) < 1:
        raise ValueError("Batch factors must be positive")
    return physical * accumulation * devices


def tokenized(frame: pd.DataFrame, tokenizer, max_length: int, *, include_id: bool = False) -> Dataset:
    columns = ["premise", "hypothesis", "label_id"] + (["example_id"] if include_id else [])
    data = Dataset.from_pandas(frame[columns].rename(columns={"label_id": "labels"}), preserve_index=False)
    return data.map(lambda batch: tokenizer(batch["premise"], batch["hypothesis"], truncation=True, padding="max_length", max_length=max_length), batched=True, remove_columns=["premise", "hypothesis"])


class IdCollator:
    def __init__(self, tokenizer):
        self.inner = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        ids = [feature.pop("example_id", None) for feature in features]
        batch = self.inner(features)
        if ids[0] is not None:
            batch["example_id"] = torch.tensor(ids, dtype=torch.long)
        return batch


class CartographyTrainer(Trainer):
    """Capture gold probabilities from the actual training forward pass."""

    def __init__(self, *args, dynamics_dir: Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamics_dir = dynamics_dir
        self._dynamics_files = {}
        self._dynamics_writers = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        example_ids = inputs.pop("example_id", None)
        labels = inputs["labels"]
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        if model.training and self.dynamics_dir is not None:
            if example_ids is None:
                raise ValueError("Training dynamics requires example IDs")
            epoch = int(self.state.epoch or 0)
            if epoch not in self._dynamics_writers:
                self.dynamics_dir.mkdir(parents=True, exist_ok=True)
                handle = (self.dynamics_dir / f"epoch_{epoch}.csv").open("w", newline="")
                writer = csv.writer(handle)
                writer.writerow(["example_id", "epoch", "gold_label", "predicted_label", "gold_probability", "correct"])
                self._dynamics_files[epoch] = handle
                self._dynamics_writers[epoch] = writer
            logits = outputs.logits.detach().float().cpu()
            gold = labels.detach().cpu()
            probs = logits.softmax(-1)
            predicted = logits.argmax(-1)
            for i, example_id in enumerate(example_ids.tolist()):
                self._dynamics_writers[epoch].writerow((example_id, epoch, int(gold[i]), int(predicted[i]), float(probs[i, gold[i]]), int(predicted[i] == gold[i])))
        return (loss, outputs) if return_outputs else loss

    def close_dynamics(self):
        for handle in self._dynamics_files.values():
            handle.close()


def make_model(checkpoint: str):
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=3, id2label={i: x for i, x in enumerate(LABELS)},
        label2id={x: i for i, x in enumerate(LABELS)}, ignore_mismatched_sizes=False,
    )
    if model.config.model_type != "roberta" or model.config.num_labels != 3:
        raise ValueError("Expected three-label RoBERTa model")
    return model


def run(config: dict, *, max_train_examples: int | None = None, smoke: bool = False) -> dict:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    disable_progress_bar()
    seed = int(config["seed"])
    set_seed(seed)
    if not smoke and config.get("diagnostics_file") and not Path(config["diagnostics_file"]).is_file():
        raise FileNotFoundError(f"NLI Diagnostics source is required before training: {config['diagnostics_file']}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train = read_snli(config["train_file"])
    dev = read_snli(config["dev_file"])
    test = read_snli(config["test_file"])
    if config.get("manifest"):
        selected = pd.read_csv(config["manifest"])
        if selected.example_id.duplicated().any():
            raise ValueError("Duplicate subset ID")
        train = train.merge(selected[["example_id"]], on="example_id", validate="one_to_one")
        if len(train) != len(selected):
            raise ValueError("Subset IDs missing from SNLI")
    if max_train_examples:
        train = train.sample(n=min(max_train_examples, len(train)), random_state=seed)
    checkpoint = config.get("model", "roberta-large")
    if checkpoint != "roberta-large" and not config.get("allow_other_model", False):
        raise ValueError("Reproduction requires roberta-large")
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = make_model(checkpoint)
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    physical = int(config.get("physical_batch_size", 2))
    accumulation = int(config.get("gradient_accumulation", 48))
    if not smoke and effective_batch(physical, accumulation) != 96:
        raise ValueError("Full reproduction requires effective batch size 96")
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    args = TrainingArguments(
        output_dir=str(out), overwrite_output_dir=False,
        per_device_train_batch_size=physical,
        per_device_eval_batch_size=int(config.get("eval_batch_size", 8)),
        gradient_accumulation_steps=accumulation,
        learning_rate=float(config.get("learning_rate", 1.0708609960508476e-5)),
        num_train_epochs=float(config.get("epochs", 23)),
        max_steps=int(config.get("max_steps", -1)),
        warmup_steps=int(config.get("warmup_steps", 0)),
        weight_decay=float(config.get("weight_decay", 0)),
        max_grad_norm=1.0,
        lr_scheduler_type="linear", optim="adamw_torch",
        bf16=use_bf16, fp16=bool(torch.cuda.is_available() and not use_bf16),
        eval_strategy="epoch" if not smoke else "no",
        save_strategy="epoch" if not smoke else "steps",
        save_steps=10 if smoke else 500,
        save_total_limit=2,
        load_best_model_at_end=not smoke,
        metric_for_best_model="accuracy" if not smoke else None,
        greater_is_better=True if not smoke else None,
        logging_steps=max(1, int(config.get("logging_steps", 100))),
        seed=seed, data_seed=seed, report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        disable_tqdm=True,
        gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
    )
    trainer = CartographyTrainer(
        model=model, args=args, train_dataset=tokenized(train, tokenizer, int(config.get("max_length", 128)), include_id=bool(config.get("record_dynamics", False)) and not smoke),
        eval_dataset=tokenized(dev, tokenizer, int(config.get("max_length", 128))) if not smoke else None,
        processing_class=tokenizer, data_collator=IdCollator(tokenizer),
        dynamics_dir=out / "training_dynamics" if config.get("record_dynamics", False) and not smoke else None,
        compute_metrics=lambda p: {"accuracy": accuracy(p.label_ids, np.argmax(p.predictions, axis=-1)) / 100},
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(config.get("patience", 3)))] if not smoke else [],
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    resume = config.get("resume_checkpoint")
    trainer.train(resume_from_checkpoint=resume if resume else None)
    trainer.close_dynamics()
    runtime = (time.monotonic() - started) / 60
    trainer.save_model(str(out / "final"))
    results = {
        "subset": config.get("subset", "full"), "seed": seed,
        "num_train_examples": len(train), "physical_batch_size": physical,
        "gradient_accumulation": accumulation,
        "effective_batch_size": effective_batch(physical, accumulation),
        "runtime_minutes": runtime,
        "peak_vram_mb": torch.cuda.max_memory_reserved() / 2**20 if torch.cuda.is_available() else None,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
        "id_accuracy": accuracy(test.label_id.to_numpy(), np.argmax(trainer.predict(tokenized(test, tokenizer, int(config.get("max_length", 128)))).predictions, axis=-1)),
        "validation_accuracy": accuracy(dev.label_id.to_numpy(), np.argmax(trainer.predict(tokenized(dev, tokenizer, int(config.get("max_length", 128)))).predictions, axis=-1)),
        "ood_accuracy": None,
        "model": checkpoint, "learning_rate": args.learning_rate,
        "epochs": trainer.state.epoch, "max_epochs": args.num_train_epochs,
        "optimizer_steps": trainer.state.global_step,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "max_length": int(config.get("max_length", 128)), "mixed_precision": "bf16" if use_bf16 else "fp16" if args.fp16 else "fp32",
    }
    if not smoke and config.get("diagnostics_file"):
        diag = read_diagnostics(config["diagnostics_file"])
        pred = trainer.predict(tokenized(diag, tokenizer, int(config.get("max_length", 128))))
        results["ood_accuracy"] = accuracy(diag.label_id.to_numpy(), np.argmax(pred.predictions, axis=-1))
    results["runtime_minutes"] = (time.monotonic() - started) / 60
    if torch.cuda.is_available():
        results["peak_vram_mb"] = torch.cuda.max_memory_reserved() / 2**20
        results["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
    (out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    LOG.info("Results: %s", results)
    return results
