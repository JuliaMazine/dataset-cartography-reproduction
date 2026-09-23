# Speaker notes

Target duration: 22–25 minutes for slides 1–20; appendix is backup.

## 1. Title

This talk explains the Dataset Cartography paper first, then a focused reproduction of its SNLI data-selection result. The question running through the talk is whether the path a model takes while learning an example can reveal useful structure in the dataset. I will spend roughly the first half on the paper and the second half on what we reproduced. The structure is motivation, method, original evidence, reproduction protocol, results, and interpretation. Appendix slides hold configuration details so the main argument stays readable.

## 2. Aggregate metrics hide example-level structure

We normally describe a training run with aggregate loss and evaluation accuracy. Those are necessary, but they collapse thousands of different example histories into one number. Some examples may become correct almost immediately, some may fluctuate, and some may resist the provided label. Dataset Cartography starts from the idea that this variation is useful information about the dataset itself. Two runs with identical final accuracy can contain very different datasets: one may have many consistently easy cases, while another may depend on unstable borderline cases. Aggregate evaluation therefore cannot tell us which examples drive learning, which are redundant, or which deserve annotation review.

## 3. SNLI is a three-way natural-language inference task

SNLI is central to our reproduction. Each example contains a premise and a hypothesis, and the task is to decide whether the hypothesis follows from the premise, could be true without following, or conflicts with it. These three hypotheses share one real SNLI premise. In-distribution evaluation uses held-out SNLI examples; later we will distinguish that from the hand-built NLI Diagnostics set. Entailment means the hypothesis must follow from the premise, contradiction means it cannot be true given the premise, and neutral covers cases where neither conclusion follows. The task is simple to state but exposes lexical, logical, and commonsense reasoning failures.

## 4. Training dynamics record the path, not only the endpoint

The conceptual shift is simple. Instead of keeping only the final prediction, record the probability assigned to the gold label at the end of every epoch. The sequence of probabilities is an example’s training trajectory. The illustrated example rises smoothly from 0.42 to 0.91. Dataset Cartography summarizes each such trajectory and plots every training example. Recording once per epoch is computationally cheap because the model already performs these forward passes during training. The original work used all epochs, including the first, and reports that early approximations can still correlate well with converged maps.

## 5. Three examples produce three distinct trajectories

These are not stylized curves; they come from our recorded SNLI training dynamics. One example is assigned essentially full gold probability in every epoch. The tennis pair switches abruptly from near zero to near one, showing high instability. The girl-versus-boy example stays near zero because its provided entailment label conflicts with the sentences. These trajectories motivate confidence, variability, and correctness. The unstable example also shows why a final prediction alone is insufficient: at the last epoch it looks confidently correct, but its earlier epochs tell us the model crossed a sharp decision boundary. Confidence and variability retain that history.

## 6. Confidence summarizes how strongly an example is learned

Confidence is the mean gold-label probability over epochs. High confidence means the learner usually assigns strong probability to the annotation; low confidence means it repeatedly does not. This is not the model’s maximum class probability and it is not an objective property of the sentence pair. It depends on the gold label, the model family, and training procedure. A high mean can coexist with one unusual epoch, and a low mean can arise from a bad annotation, a genuinely difficult case, or a model limitation. Confidence is therefore a diagnostic coordinate, not a verdict about data quality.

## 7. Variability captures instability; correctness counts decisions

Variability is the standard deviation of those probabilities. An example can have low variability for two opposite reasons: it is confidently correct every epoch, or confidently inconsistent with its annotation every epoch. Correctness provides a coarser count of how often the predicted class matches the label. Together the measures distinguish stable-high, unstable, and stable-low behavior. The paper uses population standard deviation across epochs. Correctness takes only E plus one possible values, so it loses probability information, but it makes the distinction between stable correct and stable incorrect examples especially intuitive.

## 8. A data map places every example by confidence and variability

The data map uses variability on the horizontal axis and confidence on the vertical axis. Each point is one training example, and color shows the number of epochs predicted correctly. The bell shape gives three characteristic regions. The labels are descriptive and model-relative; they are not fixed semantic categories or permanent properties of an example. The authors observed broadly similar bell-shaped geometry across SNLI, MultiNLI, WinoGrande, and QNLI with RoBERTa-large. They also stress that changing the encoder can move individual coordinates even when the overall structure remains recognizable.

## 9. Map regions become interpretable through real examples

These examples make the geometry concrete. The beach-versus-jungle contradiction is consistently easy. The tennis pair is arguably compatible rather than contradictory, and the model switches sharply during training. The hard example’s entailment label says a girl entails a boy, so its near-zero confidence is unsurprising. These examples illustrate tendencies, not equivalences: ambiguity does not mean annotation error, and hard-to-learn does not guarantee one. The middle example also warns against reading the gold label as unquestionable truth: practicing tennis is compatible with playing tennis, so contradiction is debatable. The map helps prioritize such cases for human inspection without asserting the correction automatically.

## 10. The paper tested whether map regions predict useful behavior

The paper did more than visualize four datasets. It treated the regions as empirical hypotheses. The authors trained new models on selected regions, tested in-distribution and independently collected out-of-distribution sets, varied subset size to study optimization, injected label noise, and compared dynamics with human agreement. This validation program is why the method is interesting. The subset experiments retrain from scratch after ranking examples, controlling the amount of training data. OOD sets were collected independently or designed adversarially, so they ask whether a selection learns something broader than the original dataset distribution.

## 11. Ambiguous subsets often supported stronger generalization

On WinoGrande, the most ambiguous third produced the strongest reported OOD score, 87.6, while retaining much more ID performance than a random third. In SNLI, the ambiguous third was within 0.2 percentage points of full-data ID performance and exceeded full-data Diagnostics accuracy. The paper found related patterns across datasets, but we should state this as empirical evidence for these settings, not a universal rule. On WinoGrande, easy selections underperformed the random subset, while ambiguous and hard selections were competitive or better. For SNLI, the paper reports improvements across the Diagnostics categories, although the aggregate numbers shown here are the comparison we reproduce.

## 12. Easy examples can be necessary for optimization

The optimization experiment adds an important qualification. WinoGrande models trained on 25 percent or more of the most ambiguous data learned successfully, but at 17 percent or less they collapsed to chance despite restarts. Replacing just one tenth of that 17 percent subset with easy examples restored learning and beat the random ID baseline. Easy examples therefore are not useless; they can provide stable optimization signal. This result suggests a curriculum-like balance: difficult variable examples may define useful distinctions, while some stable examples help the optimizer find a workable solution. The paper leaves the optimal mixture as an open question rather than prescribing one fixed ratio.

## 13. Hard-to-learn regions are enriched for annotation problems

The authors injected controlled noise by flipping labels for one percent of easy WinoGrande examples. After retraining, the corrupted examples shifted toward lower confidence. A simple confidence-based classifier then identified candidate issues. In human review, 67 percent versus 13 percent of WinoGrande’s predicted noisy and clean groups were mislabeled or ambiguous; on SNLI the contrast was 76 versus 4 percent. This shows enrichment, not perfect diagnosis. The simple noise classifier used confidence as its single feature and achieved a sanity-check score of 100 percent F1 on similarly constructed artificial data. Real-data human review was weaker, which is expected because naturally difficult and mislabeled examples are not the same population.

## 14. The selected SNLI experiment tests data efficiency and generalization

This is the concrete experiment we chose. ID means the ordinary SNLI test distribution. OOD means the NLI Diagnostics set, a small hand-crafted challenge set targeting multiple reasoning categories. The paper’s ambiguous third slightly exceeded the full-data ID score and improved Diagnostics by 1.7 points. The numerical ID difference is small; the interesting claim is data efficiency plus OOD behavior. Diagnostics contains categories for lexical semantics, predicate–argument structure, logic, and knowledge. We report one aggregate accuracy because that is what our evaluator produces. This keeps ID and OOD claims distinct and avoids implying that a higher SNLI score guarantees broader reasoning.

## 15. Reproduction protocol follows the published SNLI selection

We retained every official SNLI example with a valid label and joined all 549,367 training IDs to the authors’ released coordinates. The random subset follows their published sampling seed and procedure. Ambiguous selects highest variability and hard-to-learn selects lowest confidence. Every model is RoBERTa-large with the recovered learning rate, effective batch 96, and best validation checkpoint. Using released coordinates matters: it reproduces the paper's selection directly before asking whether our locally reconstructed map agrees. Otherwise a difference in final accuracy could combine selection error with training error and would be harder to interpret.

## 16. Hardware constraints changed implementation details

The original physical batch of 96 does not fit a 16 GB card, so we used 32 examples with three accumulation steps, preserving the effective batch. BF16 and dynamic padding reduce memory. The software stack is modern. Most importantly, the paper reports the best of three seeds, while the released configuration exposes one seed. Our comparisons therefore test whether the reported pattern appears, not whether every decimal is identical. Gradient accumulation preserves optimizer batch frequency but not every floating-point operation, and BF16 changes numerical rounding. Dynamic padding preserves attention masks and the 128-token truncation limit while reducing wasted computation. These are practical, documented deviations.

## 17. In-distribution accuracy closely matches the paper

On SNLI test, full and random are about half a point above the paper. The ambiguous subset is essentially exact: 92.19 versus 92.2. It also beats our random third by 0.38 points and comes within 0.34 of our full-data run. Hard-to-learn is the largest ID disagreement, 1.38 points below the paper. The plotted axis is zoomed and explicitly labeled; the table keeps the size of each difference visible. The ordering is not identical because our full run exceeds ambiguous by 0.34 points, whereas the published ambiguous value exceeds full by 0.2. Given best-of-three versus one seed, the defensible conclusion is near-equivalent performance with one third of the data, not that ambiguous selection always wins ID.

## 18. The ambiguous subset also leads on NLI Diagnostics

On Diagnostics, every reproduction score is within 1.65 points of the published value. Ambiguous remains best at 64.31, 1.54 points above our full run and 2.26 above our random third. The ranking supports the paper’s main OOD pattern. The random run is our largest positive difference. Diagnostics is only 1,104 examples, so a small number of predictions and seed differences can move the percentage. Here the qualitative ordering is especially informative: ambiguous is best in both paper and reproduction, and random is lowest. Hard-to-learn roughly matches the paper's OOD result even though its ID result is lower, reinforcing that ID and OOD answer different questions.

## 19. Reproduced maps agree more on confidence than variability

We also reconstructed the map from our own training dynamics. Coverage is 99.91 percent because 498 examples lacked exactly one observation in every logged epoch and were excluded rather than imputed. Confidence correlates strongly with the released map: Pearson 0.944. Variability is less stable but still substantial at 0.708. Seventy-five percent of ambiguous selections overlap. Qualitatively, the exact girl-versus-boy example highlighted as hard in the paper also appears among our lowest-confidence examples. Set overlap is a stricter criterion than correlation because ranking changes near the one-third cutoff can swap membership. A 75.2 percent ambiguous overlap therefore coexists naturally with a Spearman variability correlation of 0.772. The excluded 0.09 percent is transparently reported.

## 20. The selection result reproduces; map details remain sensitive

The evidence supports the paper’s central SNLI selection result: the ambiguous third nearly matches full-data ID accuracy and leads on Diagnostics. The map’s confidence structure is robust, while variability and the hard-subset score are more sensitive. The broader lesson is that per-example training histories can diagnose usefulness, uncertainty, and possible annotation issues, but all region names remain conditional on the model and training process. That is the conclusion I would defend from this reproduction. Practical uses include dataset auditing, selecting compact training sets, and studying uncertainty, but computing the initial map still requires training on the full dataset. Our evidence is one dataset, one model family, and one seed, so broader deployment should validate the map again under the intended setup.

## A1. Exact reproduction hyperparameters

Backup slide with the full settings that materially affect comparison. The recovered settings are documented in the repository and the raw run metrics.

## A2. Individual run metrics

Backup slide containing every individual measured run rather than rounded headline values. There is no hidden aggregation across seeds.

## A3. Subset construction

Backup slide describing exact subset construction. The slight count difference between random and coordinate-based subsets follows the separate procedures in the original code.

## A4. Complete paper-versus-reproduction table

Backup slide containing the full numerical comparison and signed differences in percentage points.

## A5. Reproduction environment and artifact locations

Backup slide for reproducibility questions and artifact navigation.
