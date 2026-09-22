# Recovered SNLI experiment

Sources: [paper](https://aclanthology.org/2020.emnlp-main.746.pdf), [original SNLI config](https://github.com/allenai/cartography/blob/3df3438fcc7324e706dee2e787426389bbd8fb1a/configs/snli.jsonnet), [parameter defaults](https://github.com/allenai/cartography/blob/3df3438fcc7324e706dee2e787426389bbd8fb1a/cartography/classification/params.py), [training code](https://github.com/allenai/cartography/blob/3df3438fcc7324e706dee2e787426389bbd8fb1a/cartography/classification/run_glue.py), [SNLI processor](https://github.com/allenai/cartography/blob/3df3438fcc7324e706dee2e787426389bbd8fb1a/cartography/classification/snli_utils.py), [selection code](https://github.com/allenai/cartography/blob/3df3438fcc7324e706dee2e787426389bbd8fb1a/cartography/selection/train_dy_filtering.py).

| Setting | Recovered value | Status/source |
| --- | --- | --- |
| Model | `roberta-large` | Exact: config and paper Appendix A.3 |
| Learning rate | `1.0708609960508476e-05` | Exact: config; tuned with AllenTune per Appendix A.3 |
| Batch size | 96 on one GPU | Exact: config and Appendix A.3 |
| Optimizer | AdamW, epsilon `1e-8`, weight decay `0`, max gradient norm `1` | Exact: training code and parameter defaults |
| Scheduler | Linear decay, zero warmup steps | Exact: training code and parameter defaults |
| Max token length | 128 | Exact: parameter default, absent from SNLI config |
| Map epochs | Six, from epochs 0–5 | Exact: Appendix A.3 and released coordinate filename |
| Fine-tuning epochs | Config cap 23, patience 3; paper says six | **Conflicting evidence.** Paper Appendix A.3 says six; config uses cap 23 and early stopping. This implementation follows the released config and selects the best validation checkpoint. It may stop after a different number of epochs. |
| Seed | 93078 for released SNLI config | Exact: config. Other two reporting seeds not published in repository config; uncertain. |
| Selection fraction | 0.3319, `int(fraction*N)+1` for coordinate subsets | Exact: selection code. Random baseline uses 0.33 and `int(fraction*N)` in separate script. |
| Ambiguous | Highest variability | Exact: README and selection code |
| Hard to learn | Lowest confidence | Exact: README and selection code |
| Original SNLI data | Official SNLI 1.0 sentences and `pairID`, invalid `-` labels skipped | Exact: processor; original uses reformatted GLUE-style TSV |
| Label order | entailment, neutral, contradiction | Exact: SNLI processor |
| ID test | SNLI test split | Exact: Table 3; validation results separately in Table 7 |
| Diagnostics | Three-label NLI Diagnostics, total accuracy across 1,104 valid records | Original public file and evaluator; Table 5 says 1,105, apparently counting the header |
| Table 3 aggregation | Best of three random seeds | Exact: Table 3 caption |

The caption does not specify how the best seed was chosen. The report script selects the run with the highest SNLI validation accuracy, consistent with the original checkpoint selection code; this seed-selection detail is **inferred**. It also publishes mean and sample standard deviation across available seeds.

The paper's Table 5 lists 549,368 SNLI training examples and 9,843/9,825 dev/test examples. Direct inspection of the official archive yields **549,367 / 9,842 / 9,824 examples with valid gold labels**, matching the 549,367 released coordinates. Thus Table 5 counts differ by one in each split from the valid-label data. We require the valid-label counts and a complete one-to-one join. The original `data_utils_glue.py` converts string `pairID` into an integer GUID by combining a special-prefix code, all digits, and a final label-character code; this project implements the same rule and fails if it cannot encode an ID.

The released SNLI coordinates are from one map run. Its selected IDs are used before any locally generated map. The original selection script defines metric sort direction but does not define tie order; tie behavior may differ here. The released coordinates have correctness counts from six epochs (0–5), verified by their maximum value of six.

## Necessary deviations on a 16 GB GPU

The original experiment used a Quadro RTX 8000 and physical batch 96. Here physical batch 2 plus 48 accumulation steps matches effective batch 96. BF16 (or FP16) and gradient checkpointing reduce memory use. Optimizer steps and numerical rounding can still differ from the original. Modern `transformers` also differs from the 2020 implementation. These are methodological deviations and should be considered when interpreting accuracy differences.

The original diagnostics source path is a private absolute path (`diagnostic-full.tsv`). The public original is [GLUE diagnostic-full.tsv](https://dl.fbaipublicfiles.com/glue/data/diagnostic-full.tsv). Direct inspection found 1,105 physical lines: one header and 1,104 labeled examples. Table 5's 1,105 likely includes the header. The separate GLUE AX file is unlabeled and must not be substituted.
