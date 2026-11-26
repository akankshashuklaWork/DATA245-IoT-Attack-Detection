# Stage 4 methods: five-versus-seven feature ablation

## Study status

Stage 4 is a controlled, classical-machine-learning comparison of four feature configurations with the existing Decision Tree and k-nearest neighbors (kNN) classifiers. It is **exploratory/comparative**, not a confirmatory external validation: the same fixed test partition was already examined during Stage 3.

No neural model, embedding, pretrained feature extractor, ensemble, or additional dataset is used.

## Research question

Can `proto` (network protocol) and/or `conn_state` (connection state) improve the detection of MITM, ransomware, and XSS traffic without an unjustified increase in runtime or model complexity?

The prespecified configurations are:

| ID | Predictors |
|---|---|
| `F5` | `duration`, `src_bytes`, `dst_bytes`, `src_ip_bytes`, `dst_ip_bytes` |
| `F5_PROTO` | F5 plus `proto` |
| `F5_STATE` | F5 plus `conn_state` |
| `F7` | F5 plus both categorical fields |

The target is the ten-class `type` field. The binary `label` field, addresses, ports, timestamps, identifiers, and all other columns are excluded by an explicit whitelist.

## Data and fixed partitions

`src/common.py` remains the sole source for reading and cleaning `data/train_test_network.csv`. Exact full-row duplicates are removed in the same order as the baseline, leaving 190,474 rows from 211,043 original observations. Stage 4 reuses `results/fixed_split_assignments.csv` unchanged:

| Partition | Rows |
|---|---:|
| Train | 140,812 |
| Validation | 24,831 |
| Test | 24,831 |

The code verifies all of the following before fitting:

- saved row indexes cover the deduplicated data exactly once and in the original order;
- saved labels equal the current deduplicated `type` column;
- no exact F5 pattern crosses train, validation, and test;
- no exact F7 pattern crosses the partitions;
- the raw dataset and fixed-split checksums are recorded.

F5 group separation implies F7 group separation because matching F7 rows also match on all five F5 fields. This control does **not** establish independence by device, attack session, time period, or network environment.

## Train-only preprocessing

Every learned transformation is inside a scikit-learn `Pipeline` and is fitted on training rows only.

For Decision Tree:

- numeric fields use median imputation;
- numeric scaling is not added, matching Stage 3;
- categorical fields use most-frequent imputation and dense one-hot encoding;
- unknown validation/test categories are ignored safely.

For kNN:

- numeric fields use median imputation, `log1p`, and standard scaling in that order;
- nonnegative inputs are required before `log1p`;
- categorical fields use the same train-only imputation and dense one-hot encoding;
- F5 retains `algorithm='kd_tree'` for baseline parity;
- augmented configurations use `algorithm='auto'` with dense inputs, `k=5`, uniform weights, and Euclidean Minkowski distance (`p=2`).

One-hot columns change kNN geometry: a categorical mismatch contributes distance in the encoded feature space alongside the standardized numeric dimensions. No categorical weight was tuned. Any future weighting experiment must be selected on validation only and reported separately.

The supplied data has no missing `proto` or `conn_state` values. The training partition contains all 3 observed protocol categories and all 13 observed connection-state categories. Missing/unseen-category tests are retained for future data.

## Fixed-split experiment protocol

`src/07_feature_ablation.py` fits all eight model/configuration combinations on the fixed training rows. The workflow is deliberately two-phase:

1. Fit and save all eight pipelines, then predict validation.
2. Write `frozen_settings.json`, then predict the fixed test once per frozen combination.

No winner is selected or tuned using test metrics. Each test prediction includes the deduplicated `row_index`, actual label, and predicted label. Summary metrics and class-wise metrics are derived from those records.

Fitted models are written only to `models/stage4/`; Stage 3 model files are never overwritten. Results are written only to `results/stage4/`; existing baseline CSVs, diagnostics, figures, and the midterm deck remain unchanged.

## Outcomes and error accounting

Primary outcomes are MITM, ransomware, and XSS precision/recall/F1, macro-F1, and missed-actual counts. Accuracy and weighted F1 are context metrics.

The security alarm statistic is explicitly:

```text
normal-to-attack false-positive rate =
actual normal rows predicted as any attack / actual normal rows
```

It is not the same as a multiclass one-versus-rest false-positive rate. Raw numerator and denominator counts are saved with the rate.

`src/08_comparative_evaluation.py` independently recomputes every summary and class-wise score from `ablation_predictions.csv` and verifies fixed row membership before creating comparison tables.

## Latency and size protocol

Latency uses the identical 24,831-row fixed-test batch for every saved pipeline. Each model receives one warm-up run followed by five measured runs using `time.perf_counter`. The report saves every repetition plus median, mean, sample standard deviation, minimum, maximum, and interquartile range in milliseconds per 1,000 rows. Model loading time is measured separately, and serialized file size is recorded.

These are same-machine comparative measurements, not deployment-ready throughput claims.

## Baseline parity finding

The refitted F5 Decision Tree reproduces every saved Stage 3 prediction. The current scikit-learn 1.8.0 F5 kNN refit differs on two validation rows and one test row out of 24,831 per partition. The largest summary-metric difference is 0.0107 percentage points (balanced accuracy).

Inspection found scaler/standardized training-array differences at approximately machine precision (`10^-15`), which can change a near-tied neighbor ordering. `experiment_manifest.json` records the exact row indexes and both predictions. The discrepancy is not hidden by overwriting the Stage 3 result or claiming exact parity.

## Fixed-test results

The main test results are mixed rather than a universal seven-feature win:

| Model / configuration | Macro-F1 | MITM recall | Ransomware recall | XSS recall | Normal-to-attack FPR |
|---|---:|---:|---:|---:|---:|
| Decision Tree / F5 | 90.34% | 69.80% | 72.64% | 84.02% | 8.06% |
| Decision Tree / F5_PROTO | 93.60% | **71.81%** | 98.24% | 83.90% | 0.90% |
| Decision Tree / F5_STATE | 92.23% | 65.77% | 97.72% | 85.07% | 7.14% |
| Decision Tree / F7 | **93.85%** | 68.46% | **100.00%** | **85.36%** | **0.78%** |
| kNN / F5 | 84.88% | 65.77% | 19.29% | 84.54% | 6.08% |
| kNN / F5_PROTO | 88.32% | 67.11% | 38.15% | 84.78% | 0.90% |
| kNN / F5_STATE | 87.87% | **68.46%** | 36.53% | 86.65% | 4.55% |
| kNN / F7 | **88.79%** | **68.46%** | **38.72%** | **86.83%** | **0.73%** |

For the Decision Tree, protocol alone gives the highest MITM recall (+2.01 percentage points versus F5), while F7 reduces MITM recall by 1.34 points but substantially improves ransomware, XSS, macro-F1, and the normal-to-attack false-alarm rate. For kNN, F7 improves MITM by 2.68 points, ransomware by 19.43 points, and XSS by 2.28 points, but ransomware recall remains only 38.72%.

Median inference latency rises from 0.056 to 0.277 ms per 1,000 rows for Decision Tree F5 versus F7 and from 12.795 to 37.373 ms per 1,000 rows for kNN. Absolute timings are environment-specific.

## Optional matched split-sensitivity study

`src/10_stage4_split_stability.py` repeats the comparison for seeds `[42, 7, 17, 23, 29, 41, 73, 101, 202, 313]`. Within a seed, all eight combinations use identical train/validation/test rows. The study saves per-seed metrics, class-wise scores, paired deltas versus matched F5, and mean/std/min/max summaries.

The ten runs overlap in observations. They measure split sensitivity, not ten independent blind evaluations. Ransomware and XSS deltas vary substantially by seed, reinforcing that the fixed-test gains should not be generalized to a new network without independent evidence.

## Reproducible run order

```bash
python tests/check_saved_outputs.py
python src/06_feature_augmentation.py
python src/07_feature_ablation.py --split fixed --output-dir results/stage4
python src/08_comparative_evaluation.py
python src/10_stage4_split_stability.py --seeds 42 --output-dir /tmp/stage4_one_seed
python src/10_stage4_split_stability.py --seeds 42,7,17,23,29,41,73,101,202,313
python src/09_final_visualizations.py
python -m pytest -q tests/test_stage4.py
python tests/check_saved_outputs.py
```

## Limitations

- The fixed test was already inspected during Stage 3 and is not a blind final holdout.
- The dataset does not provide a demonstrated unseen-device, unseen-session, temporal, or cross-environment test here.
- Exact feature-vector grouping is a leakage control, not proof of real-world independence.
- kNN one-hot geometry is a deliberate prespecified choice, not a tuned optimum.
- Latency comes from one software/hardware environment.
- A confirmatory claim requires a separately collected independent dataset, temporal holdout, or environment holdout that is not used for model selection.
