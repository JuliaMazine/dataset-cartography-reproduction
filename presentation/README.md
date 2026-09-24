# Academic presentation

`dataset_cartography_reproduction.pptx` is a complete 20–25 minute research presentation covering both Swayamdipta et al. (EMNLP 2020) and this repository's SNLI reproduction.

## Contents

- 20 main slides: motivation, SNLI, training dynamics, formulas, data maps, original findings, reproduction protocol, ID/OOD results, map comparison, and conclusion;
- 4 appendix slides: exact hyperparameters, run metrics, subset construction, and full comparison;
- embedded presenter notes, also available as `speaker_notes.md`;
- real SNLI examples and actual trajectories from the completed run;
- exact measured results loaded from `results/reproduction_metrics.csv` and `results/run_metrics/`;
- original paper claims checked against the official ACL Anthology PDF in `assets/`.

The visual convention is consistent throughout:

- muted slate = original paper;
- rust = our reproduction;
- green, amber, and red = easy-to-learn, ambiguous, and hard-to-learn behavior where relevant.

## Open the deck

Open:

```text
presentation/dataset_cartography_reproduction.pptx
```

A rendered PDF and visual contact sheet are kept under `presentation/rendered/` for review.

## Regenerate

From the repository root:

```bash
uv sync --extra presentation
MPLCONFIGDIR=/tmp/cartography-presentation-mpl \
  uv run --extra presentation python presentation/generate_presentation.py
```

The generator uses committed compact results and presentation assets. Model checkpoints and the full dataset are not required.

Render with LibreOffice:

```bash
mkdir -p presentation/rendered
libreoffice --headless \
  --convert-to pdf \
  --outdir presentation/rendered \
  presentation/dataset_cartography_reproduction.pptx
```

## Scientific scope

The deck distinguishes published paper values from this project's one-seed measurements. It explicitly reports the hardware and software deviations, early stopping, 99.91% map coverage, and the difference between ID and OOD evaluation. Synthetic label-noise training was not run in this reproduction; the deck discusses label noise only as an original-paper finding.
