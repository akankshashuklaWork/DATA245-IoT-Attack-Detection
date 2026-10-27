# DATA245 — Stage 1–3 Results (Original Five Features Only)

## Scope
Stage 1: verify ToN-IoT; Stage 2: clean, perform EDA and create held-out splits; Stage 3: fit baseline Decision Tree and kNN classifiers for ten-class classification.

No protocol, connection-state, IP address, port, deep learning, embeddings, or new engineered predictors were used.

## Dataset observations
- Original rows: **211,043**, columns: **44**.
- Exact duplicate full rows removed: **20,569**.
- Rows remaining: **190,474**.
- MITM rows after deduplication: **1,041**.
- Unique 5-feature patterns: **92,347**.
- Patterns with multiple different class labels: **224** (covering **37,997** rows). This limits what any deterministic classifier can infer from only these five inputs.

## Data partitioning
Fixed random seed 42; 7-fold StratifiedGroupKFold. Fold 5 = validation, fold 6 = test, five other folds = train. All rows sharing exactly the same five-feature vector remain in the same partition. Approximate target is 71/14/14; actual class and row counts are in `results/split_class_counts.csv`.

| Split | Rows | MITM rows |
|---|---:|---:|
| train | 140,812 | 743 |
| validation | 24,831 | 149 |
| test | 24,831 | 149 |

## Baseline models
- **Decision Tree**: Gini splitting, random state 42, default complexity; median imputation fitted on train.
- **kNN**: k=5, uniform neighbors, Euclidean distance; median imputation, log(1+x), and standardization fitted exclusively on train. Transformations do not add predictors.

## Untouched test results
| Model | Accuracy | Macro-F1 | MITM precision | MITM recall | MITM F1 |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.932 | 0.903 | 0.658 | 0.698 | 0.678 |
| kNN | 0.891 | 0.849 | 0.645 | 0.658 | 0.651 |

## What these results mean
The paper reports high classification accuracy on a different data selection and evaluation split. **These are our actual observed baseline values**, not a numerical reproduction of the paper. High overall accuracy does not guarantee MITM detection; use MITM recall, precision, macro-F1 and the confusion matrices.
The dataset has label ambiguity: multiple classes sometimes share identical five-feature vectors. A model cannot distinguish such rows using only these inputs.

## Files to review
- `results/class_distribution_original.csv`: original class sizes.
- `results/five_feature_summary.csv`: statistics including extreme values.
- `results/baseline_metrics.csv`: validation and test metrics and runtime.
- `results/classwise_metrics.csv`: per-class precision, recall, F1 and support.
- `results/confusion_*.csv`, `figures/04_confusion_*.png`: classification errors.
- `models/*joblib`: trained baseline pipelines (load only from trusted files).

## Stage 4 (not included)
Protocol or connection-state features, improvements, tuning, feature ablation, and claims of improved performance are intentionally excluded.

## Primary reference
Dharini, N., Janani, V. S., & Katiravan, J. (2026). Efficient detection of intrusions in TON-IoT dataset using hybrid feature selection approach. Scientific Reports, 16, 7763. https://doi.org/10.1038/s41598-026-37834-y
