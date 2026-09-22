"""Batched evaluation of saved classifiers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from .data import LABELS
from .training import tokenized


def predict(checkpoint: str, frame: pd.DataFrame, *, batch_size: int = 8, max_length: int = 128) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    if tuple(model.config.id2label[i].lower() for i in range(3)) != LABELS:
        raise ValueError("Checkpoint label order differs from SNLI reproduction")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    loader = DataLoader(tokenized(frame, tokenizer, max_length), batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer))
    predicted, probabilities = [], []
    with torch.inference_mode():
        for batch in loader:
            gold = batch.pop("labels").numpy()
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits.float().cpu()
            probs = logits.softmax(-1).numpy()
            predicted.extend(probs.argmax(-1).tolist())
            probabilities.extend(probs[np.arange(len(gold)), gold].tolist())
    return pd.DataFrame({"example_id": frame.example_id.to_numpy(), "gold_label": frame.label_id.to_numpy(), "predicted_label": predicted, "gold_probability": probabilities, "correct": frame.label_id.to_numpy() == np.asarray(predicted)})
