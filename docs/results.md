# Reproduction results

All values below are measured outputs from the local run with `roberta-large`, seed 93078, effective batch size 96, and the authors' released SNLI coordinates for subset selection. Accuracy is reported in percentage points.

## Main comparison

| Training data | Examples | Paper SNLI | Our SNLI | Paper Diagnostics | Our Diagnostics |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 549,367 | 92.0 | **92.53** | 61.8 | **62.77** |
| Random 33% | 181,291 | 91.3 | **91.82** | 60.4 | **62.05** |
| Hard-to-learn 33% | 182,335 | 91.8 | **90.42** | 62.0 | **62.14** |
| Ambiguous 33% | 182,335 | 92.2 | **92.19** | 63.5 | **64.31** |

The central result reproduced: the ambiguous third reaches 92.19% SNLI accuracy, essentially matching the paper's 92.2%, and outperforms the equally sized random subset by 0.38 points. It comes within 0.34 points of our full-data run while using roughly one third of the examples.

![Paper and reproduction accuracy](../results/figures/accuracy_comparison.png)

## Data-map comparison

The locally reconstructed map contains 548,869 of 549,367 training examples (99.91% coverage). Across shared examples:

| Measure | Pearson | Spearman |
| --- | ---: | ---: |
| Confidence | 0.944 | 0.888 |
| Variability | 0.708 | 0.772 |

The locally selected ambiguous set overlaps 75.2% of the authors' ambiguous set; the hard-to-learn set overlaps 82.2%.

## Interpretation and limitations

- The paper reports the best of three seeds. This reproduction reports one available seed, 93078, so small score differences are expected.
- Early stopping ended the full and random runs after five epochs, hard-to-learn after four, and ambiguous after six. Every reported score uses the best validation checkpoint.
- The map uses five logged epochs. Gradient accumulation crossed epoch boundaries in the training logger, so 498 examples without one observation in every epoch were excluded rather than imputed. This affects map coverage, not the evaluation scores.
- BF16, dynamic padding, and a physical batch of 32 with three accumulation steps were used to fit a 16 GB GPU. The effective batch size remains 96.
- Current `transformers` and PyTorch versions differ from the original 2020 software stack.

## Reproduce the presentation figures

```bash
uv run python scripts/make_presentation_figures.py
```

The script reads `results/reproduction_metrics.csv` and writes slide-ready PNG files to `results/figures/`. Machine-readable run metrics and map statistics are included in `results/`.
