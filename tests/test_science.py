import numpy as np
import pandas as pd
import pytest

from cartography_repro.data import LABELS, LABEL_TO_ID, numeric_guid
from cartography_repro.dynamics import corrupt_labels, summarize
from cartography_repro.evaluation import accuracy, paper_comparison
from cartography_repro.subsets import align, select
from cartography_repro.training import effective_batch


def test_label_order_matches_original_processor():
    assert LABELS == ("entailment", "neutral", "contradiction")
    assert [LABEL_TO_ID[x] for x in LABELS] == [0, 1, 2]


def test_guid_encoding_matches_original_hack():
    assert numeric_guid("123e") == 1230
    assert numeric_guid("vg_verb_42c") == 444421
    assert numeric_guid("vg_len_42n") == 555422


def test_population_variability_and_correctness_count():
    frame = pd.DataFrame({"example_id": [1, 1, 2, 2], "epoch": [0, 1, 0, 1], "gold_probability": [0.2, 0.6, 0.5, 0.5], "correct": [0, 1, 1, 1]})
    result = summarize(frame).set_index("example_id")
    assert result.loc[1, "confidence"] == pytest.approx(0.4)
    assert result.loc[1, "variability"] == pytest.approx(0.2)
    assert result.loc[1, "correctness"] == 1
    assert result.loc[2, "variability"] == 0


def test_subset_ranking_and_determinism():
    aligned = pd.DataFrame({"example_id": range(100), "confidence": np.linspace(0, 1, 100), "variability": np.linspace(1, 0, 100), "correctness": 0})
    assert len(select(aligned, "ambiguous33", 1)) == int(0.3319 * 100) + 1
    assert len(select(aligned, "random33", 1)) == 33
    assert select(aligned, "ambiguous33", 1).iloc[0].example_id == 0
    assert select(aligned, "hard33", 1).iloc[0].example_id == 0
    assert select(aligned, "random33", 1).example_id.tolist() == select(aligned, "random33", 1).example_id.tolist()


def test_join_rejects_missing_coordinate():
    train = pd.DataFrame({"example_id": [1, 2]})
    coords = pd.DataFrame({"example_id": [1, 3], "confidence": [0, 0], "variability": [0, 0], "correctness": [0, 0]})
    with pytest.raises(ValueError, match="mismatch"):
        align(train, coords)


def test_noise_changes_to_other_valid_label_exactly():
    labels = np.tile(np.arange(3), 100)
    a = corrupt_labels(labels, 0.1, 17)
    b = corrupt_labels(labels, 0.1, 17)
    pd.testing.assert_frame_equal(a, b)
    assert a.is_corrupted.sum() == 30
    assert (a.loc[a.is_corrupted, "original_label"] != a.loc[a.is_corrupted, "corrupted_label"]).all()


def test_accuracy_and_batch():
    assert accuracy(np.array([0, 1, 2]), np.array([0, 2, 2])) == pytest.approx(200 / 3)
    assert effective_batch(2, 48) == 96
    with pytest.raises(ValueError):
        accuracy(np.array([]), np.array([]))


def test_paper_values_never_become_our_results():
    empty = pd.DataFrame(columns=["subset", "validation_accuracy", "id_accuracy", "ood_accuracy"])
    comparison = paper_comparison(empty)
    assert comparison.ours_snli.isna().all()
    assert comparison.ours_ood.isna().all()


def test_training_dynamics_capture_actual_forward_pass(tmp_path):
    import torch
    from transformers import TrainingArguments
    from transformers.modeling_outputs import SequenceClassifierOutput
    from cartography_repro.training import CartographyTrainer

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(3, 3))

        def forward(self, input_ids, labels):
            logits = input_ids.float() @ self.weight
            return SequenceClassifierOutput(loss=torch.nn.functional.cross_entropy(logits, labels), logits=logits)

    args = TrainingArguments(output_dir=str(tmp_path / "checkpoints"), report_to="none", disable_tqdm=True)
    trainer = CartographyTrainer(model=Tiny(), args=args, dynamics_dir=tmp_path / "dynamics")
    trainer.model.train()
    trainer.state.epoch = 0
    batch = {"input_ids": torch.eye(3)[:2], "labels": torch.tensor([0, 1]), "example_id": torch.tensor([11, 12])}
    trainer.compute_loss(trainer.model, batch)
    trainer.close_dynamics()
    recorded = pd.read_csv(tmp_path / "dynamics" / "epoch_0.csv")
    assert recorded.example_id.tolist() == [11, 12]
    assert recorded.gold_probability.tolist() == pytest.approx([1 / 3, 1 / 3])
