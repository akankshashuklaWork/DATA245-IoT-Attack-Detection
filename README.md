# DATA245 IoT Intrusion Detection — Complete Modular Python Project

**Topic:** Feature-Efficient Detection of Rare and Hard-to-Classify IoT Attacks Using Classical Machine Learning  
**Data:** ToN-IoT network CSV, 211,043 data records (211,044 physical CSV lines including its header), 44 columns, 10 classes.  
**Models:** Decision Tree and k-Nearest Neighbors (kNN) using **only five numerical predictors**.

This package combines the full Stage 1–3 baseline and the additional diagnostics used in the 12-slide midterm presentation. **The seven-feature protocol extension has not been implemented**; it is planned future work.

## 1. Six separate, beginner-friendly scripts

| Run order | Script | What it does | Main outputs |
|---|---|---|---|
| 1 | `src/01_eda_preprocessing.py` | Reads original CSV; checks class counts and missingness; removes exact duplicates **before** splitting; counts distinct 5-feature patterns; creates data split; plots EDA | `results/data_audit.json`, `results/fixed_split_assignments.csv`, `diagnostics/class_pattern_diversity.csv`, `figures/01_*.png` etc. |
| 2 | `src/02_train_models.py` | Fits **only** Decision Tree and kNN on the training split and saves them. No test evaluation. | `models/*.joblib`, `results/training_times.csv` |
| 3 | `src/03_model_evaluation.py` | Loads trained models; predicts validation and test records; calculates accuracy, macro-F1, per-class precision/recall/F1; saves confusion matrices. **Does not fit models.** | `results/baseline_metrics.csv`, `results/classwise_metrics.csv`, `results/predictions_val_test.csv`, `results/confusion_*.csv` |
| 4 | `src/04_classification_errors.py` | Reads saved actual/predicted labels, counts MITM mistakes and all-class confusions. **Does not fit models.** | `diagnostics/original_stage3_mitm_prediction_breakdown.csv`, `diagnostics/all_class_error_breakdown.csv`, `figures/07_mitm_error_breakdown.png` |
| 5 | `src/05_repeated_split_experiments.py` | Separately reruns BOTH classifiers across 10 new group-disjoint splits; reports variability. **This fits 20 models.** | `diagnostics/repeated_group_safe_splits.csv`, `repeated_per_class_scores.csv`, `repeat_summary.csv`, stability plot |
| 6 | `src/make_presentation_charts.py` | Reads saved results and creates the numerical presentation charts. **No training or new experiments.** | `presentation_charts/slide07_*.png`, `slide08_*.png`, `slide10_*.png`, `slide11_*.png` |

All scripts use `src/common.py` for the same five predictor columns, dataset paths, model definitions, and validation rules.

## 2. Quick start on Mac (Terminal)

1. Extract the ZIP archive.
2. Open Terminal. Type `cd ` (including the trailing space) and drag the extracted project folder into Terminal, then press Return.
3. Create a Python virtual environment and install packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

4. Run scripts one by one (recommended if learning):

```bash
python src/01_eda_preprocessing.py
python src/02_train_models.py
python src/03_model_evaluation.py
python src/04_classification_errors.py
python src/05_repeated_split_experiments.py
python src/make_presentation_charts.py
```

**Important:** Step 5 is optional and slower because it retrains two models 10 times. The ZIP already includes all 10-run results. To regenerate charts without retraining, run only:

```bash
python src/make_presentation_charts.py
```

Or run the core stages automatically:

```bash
python src/run_all.py
```

For the complete run including repeated splits:

```bash
python src/run_all.py --include-repeated
```

## 3. Three different evaluation activities — do not confuse them

**Model evaluation** (`03_model_evaluation.py`) scores the two saved baseline models on their fixed validation/test partitions. This supplies Slide 10.

**Classification error analysis** (`04_classification_errors.py`) starts from the saved predictions from the previous step and asks *which wrong labels did the model assign?* This supplies Slide 11 (left).

**Repeated-split evaluation** (`05_repeated_split_experiments.py`) creates new group-disjoint partitions, trains the two models anew for each seed, and summarizes variation in per-class recall. This supplies Slide 11 (right). **These are ten different partitions, not ten independent external test datasets.**

## 4. How the fixed split works

- Raw dataset: **211,043 observations**, 44 attributes.
- Exact full-row duplicates removed: **20,569**.
- Deduplicated observations: **190,474**.
- MITM observations after deduplication: **1,041**.
- Fixed split: **140,812 train**, **24,831 validation**, **24,831 test**.
- 7-fold `StratifiedGroupKFold` was used originally with random seed 42, with five folds train, one validation, one test.
- All rows sharing an exact five-feature pattern remain in the same partition so a pattern cannot leak between train/test.
- `results/fixed_split_assignments.csv` contains the original split, and Stage 1 **validates and reuses it** by default to preserve the previous presentation results.
- To rebuild the fixed split from scratch, use `python src/01_eda_preprocessing.py --rebuild-split` (slower). This overwrites split assignments; retrain and reevaluate afterward.
- The baseline models do not use `proto`, `conn_state`, IPs, ports, timestamps, target labels, embeddings, or neural networks as input features.

## 5. Original five input features

1. `duration`
2. `src_bytes`
3. `dst_bytes`
4. `src_ip_bytes`
5. `dst_ip_bytes`

Target = `type` (normal + nine attacks); `label` is not included as a predictor.

## 6. Saved baseline findings for the midterm

| Held-out TEST metric | Decision Tree | kNN |
|---|---:|---:|
| Accuracy | 93.2% | 89.1% |
| Macro-F1 | 90.3% | 84.9% |
| MITM precision | 65.8% | 64.5% |
| MITM recall | 69.8% | 65.8% |
| MITM cases detected | 104/149 | 98/149 |

Separate 10-run diagnostics show MITM mean Decision Tree recall ~65.2%, ransomware ~32.5%, and XSS ~67.3%. Repeated splits are NOT the single fixed holdout split.

## 7. Presentation charts and provenance

| PPT slide | Graph | Code that computes underlying data | Code that plots final figure |
|---|---|---|---|
| 7 | Attack class distribution | `01_eda_preprocessing.py` | `make_presentation_charts.py` |
| 8 | Five-feature pattern diversity | `01_eda_preprocessing.py` | `make_presentation_charts.py` |
| 10 | Decision Tree vs kNN performance | `03_model_evaluation.py` | `make_presentation_charts.py` |
| 11 left | Wrong predicted labels for MITM | `04_classification_errors.py` | `make_presentation_charts.py` |
| 11 right | Class recall averaged across 10 splits | `05_repeated_split_experiments.py` | `make_presentation_charts.py` |

Charts use the **saved calculated data**, not invented numbers. PowerPoint itself contains editable shapes so the same measurements may have different colors/layouts than the exported PNGs. The presentation deck is at `presentation/DATA245_IoT_Midterm_12_Slides.pptx`.

## 8. Verify the project

Run after installing requirements:

```bash
python tests/check_saved_outputs.py
```

This checks data dimensions, deduplication, split leakage, score counts, and the reproducibility of the reported MITM recall.

A quick check of repeated-split code without overwriting the existing 10-run results:

```bash
python src/05_repeated_split_experiments.py --seeds 42 --output-dir tests/one_seed
```

## 9. Research and reporting limitations

- Original research article reports a **461,043-record** data selection, which differs from this **211,043-record** CSV. This package is an independent empirical extension, **not** an exact reproduction.
- A unique five-feature combination is **not necessarily** a distinct attack session. Rounding `duration` can reduce apparent pattern diversity.
- Group-disjoint splits guard against identical five-feature vector leakage, but do not establish cross-device, cross-network, or chronological generalization.
- The repeated-split experiments reuse observations across runs, so interpret ranges as **split sensitivity**, not independent confidence intervals.
- The fixed test set has already been used for exploratory error analysis. Avoid claiming it is an untouched, final confirmatory evaluation for future feature development; plan a separate final held-out test protocol if doing Stage 4.
- Jobs stored with `joblib` are pickle-based; load only trusted model files.

## References

- Dharini, N., Janani, V. S., & Katiravan, J. (2026). *Efficient detection of intrusions in TON-IoT dataset using hybrid feature selection approach*. Scientific Reports. https://doi.org/10.1038/s41598-026-37834-y
- UNSW Canberra, *ToN-IoT Datasets*: https://research.unsw.edu.au/projects/toniot-datasets

## Archived scripts

`archive_original_scripts/` preserves the **original combined code** for audit. Use the new numbered scripts as the primary implementation: they are easier to understand and test.
