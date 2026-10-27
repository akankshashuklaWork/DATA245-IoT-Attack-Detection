# DATA245 | Five-Feature ToN-IoT Attack-Diversity Diagnostic

**Scope:** Five checks requested: MITM pattern diversity; cross-label pattern conflicts; actual Stage 3 prediction errors; repeated leakage-aware validation; and recommendation on the research project's scope. This is a diagnostic analysis, **not** Stage 4 feature augmentation.

## Dataset, scope and preprocessing

- User-provided `train_test_network.csv`: **211,043 rows, 44 columns, 10 categories**, including normal.
- **20,569 exact full-row duplicates** removed using `drop_duplicates()` on all columns. **190,474 rows** remain.
- MITM: 1,043 original rows; **1,041** after full-row deduplication.
- Five numerical input features ONLY: `duration`, `src_bytes`, `dst_bytes`, `src_ip_bytes`, `dst_ip_bytes`. Target: `type`. No `proto`, `conn_state`, IDs, IPs, ports, neural features, or target-derived predictors.
- The CSV is a **211,043-row variant**, not the paper's **461,043-row** selection. These results cannot be claimed as an exact numerical reproduction of Dharini et al. (2026).

## Check 1: Unique MITM patterns

| Definition | MITM patterns |
|---|---:|
| Exact five-feature combinations | 1,007 |
| Duration rounded to 0.01 seconds | 727 |
| Duration rounded to whole seconds | 572 |
| Four byte-related attributes only (duration removed) | 557 |

Although **96.7%** of deduplicated MITM rows have distinct exact five-feature vectors, precise duration values inflate this figure. Nevertheless, the non-duration inputs retain 557 distinct combinations. Unique features do **not** prove 557 independent attack scenarios; the dataset offers no reliable session/group identifiers or independent testbed provenance information.

## Check 2: MITM collisions with other traffic

- **7** distinct MITM five-feature patterns are also observed with a different class label.
- **7** of 1,041 MITM records sit on these conflicting patterns.
- All records of all classes falling on *one of the MITM patterns* include: ransomware 1,353; MITM 1,041; DDoS 29; XSS 23; normal 16; injection 2; password 2; scanning 1. These other-class records all correspond to the seven conflicting MITM patterns.
- These patterns cannot be unambiguously labeled using just the same five input attributes. However, the bulk of MITM records (1,034 of 1,041) have no exact cross-label pattern collision in this dataset.

**Contrast:** ransomware has 14,735 deduplicated rows but only **251** distinct five-feature combinations (1.7% ratio). XSS has 15,137 rows and 8,589 distinct patterns. Therefore large record count does not guarantee pattern diversity.

## Check 3: What did saved Stage 3 baselines predict for MITM?

**Original held-out Stage 3 split, 149 MITM records in the test partition:**

| Predicted class | Decision Tree | kNN |
|---|---:|---:|
| MITM (correct) | 104 | 98 |
| normal | 12 | 17 |
| dos | 10 | 16 |
| password | 11 | 8 |
| injection | 5 | 4 |
| ddos | 3 | 1 |
| scanning | 3 | 3 |
| xss | 1 | 2 |
| Total | 149 | 149 |

Decision Tree MITM recall: **69.8%**. kNN MITM recall: **65.8%**. Source: saved `results/predictions_val_test.csv` (not recomputed in this section).

## Check 4: Repeated leakage-aware splits

To measure split sensitivity, **ten independently seeded group-disjoint partitions** were created. Unlike the original Stage 3 fixed seven-fold `StratifiedGroupKFold`, the diagnostic uses `train_test_split` *on the 92,347 distinct five-feature groups*, stratifying groups by their majority class label, then expands entire groups to rows. Approximate split is 70%/15%/15% of **groups**, not rows. Identical five-feature inputs NEVER cross train/validation/test boundaries. Different model runs overlap in their samples, so these are sensitivity tests, not ten independent deployment evaluations.

- Seeds: 42, 7, 17, 23, 29, 41, 73, 101, 202, 313.
- Actual test row count varied from 23,140 to 40,083 due to uneven group sizes.
- Each test set held 151 to 159 MITM rows.
- Pipeline specifications match Stage 3: Decision Tree with Gini criterion (untuned, no max depth); kNN k=5, uniform weights, log1p plus train-only standardization. Median imputation, fit on training only.

| Repeated test metric (10 seeds) | Decision Tree | kNN |
|---|---:|---:|
| Mean MITM recall | 65.2% | 63.1% |
| MITM recall SD | 2.7 pp | 4.5 pp |
| MITM recall min-max | 61.3%-68.6% | 55.6%-71.2% |
| Mean MITM precision | 59.7% | 66.6% |
| Mean macro-F1 | 80.4% | 80.6% |
| Macro-F1 min-max | 67.6%-92.1% | 73.0%-93.5% |

**Key finding:** MITM recall is moderately split-sensitive, especially for kNN. **Overall multiclass model performance is much less stable**, particularly because some other classes contain large or conflicting repeated-pattern groups.

### Per-class recall under repeated partitions

| Class | Decision Tree mean recall | Decision Tree range | kNN mean recall | kNN range |
|---|---:|---:|---:|---:|
| MITM | 65.2% | 61.3%-68.6% | 63.1% | 55.6%-71.2% |
| Ransomware | 32.5% | 4.1%-88.5% | 29.1% | 5.4%-93.1% |
| XSS | 67.3% | 17.6%-91.2% | 68.1% | 17.8%-92.5% |
| Injection | 96.2% | 94.8%-96.9% | 96.0% | 94.8%-96.7% |

**Important caution:** This is not a fully grouped *session-level* or chronological test: we lack known temporal/session/scenario identifiers. A feature-vector-disjoint test is more conservative than random row splitting but does not guarantee deployment-level generalization.

## Check 5: Are there sufficient distinct MITM patterns?

**Yes for a carefully qualified DATA245 class experiment; not enough to claim production-grade MITM detection.** Evidence:

1. At least 557 distinct byte-only combinations remain, even ignoring connection duration.
2. Only 7 exact MITM patterns overlap other class labels.
3. MITM recall is consistently well below 75% with the same five predictors, so there is room to test a precise improvement hypothesis.
4. Repeated tests have roughly 150 MITM cases per split, so uncertainty remains material; small claimed gains need confidence intervals and replicate checks.
5. Multiple flows may come from the same attack scenario; distinct numerical vectors do not certify independently sampled attacks or generalization to new sites.

## Recommendation: broaden the *framing*, not the existing codebase

**Recommended course project name:**

**Feature-Efficient Detection of Rare and Hard-to-Classify IoT Attacks Using Classical Machine Learning**

**Primary research question:** When restricting intrusion detection to five inexpensive numerical flow features, which attack categories fail reliably under leakage-aware evaluation, and why?

**Secondary question (Stage 4):** Can a small addition of network protocol/state information (`proto`, `conn_state`, optionally one carefully justified attribute) improve MITM and ransomware recall **without materially decreasing macro-F1 or increasing inference cost?** This is a hypothesis, *not* a demonstrated result.

Keep the previous Stages 1-3 baseline code and data. Add a defensible diagnostic component that compares **MITM (rare, fairly diverse)**, **ransomware (more rows, very low distinct-pattern diversity)** and **XSS (variable recall)**. Use *normal* and *injection* as performance context. Test the Stage 4 feature extensions on these fixed outcomes, not just the overall accuracy. Limit experiments to Decision Tree and kNN, feature ablations, per-class confusion matrices and a leakage-aware, repeated-split comparison.

**Why broader is justified:** The data uncover a more general problem than MITM rarity: record count and pattern diversity diverge, and a small feature budget does not represent all attack classes equally. A narrow MITM-only project remains valid, but the broader framing is supported by these empirical failures and offers stronger explanatory storytelling.

## Limitations and next steps

- Original 211,043-row data differs from paper's 461,043-row variant; do not claim exact reproduction.
- No measured protocol-aware Stage 4 improvement yet.
- Coarse and fine pattern diversity may be artifacts of traffic generation or timestamp precision.
- Ten splits reuse data; mean and standard deviation reflect sensitivity, not uncertainty on a new network domain.
- Proposed primary metric: **MITM recall** and **ransomware recall**, with per-class precision, macro-F1 and false positives. Keep XSS as secondary stress test. Report each run's test support.
- Use a reserved final test set when tuning feature additions; the repeated baseline test scores here are diagnostics only and should not become the selection criterion for the final model.

## Files in this package

- `diagnose_mitm.py` - reproducible audit and repeated baseline experiments.
- `summary.json` - core counts and summary.
- `class_pattern_diversity.csv` - counts for all ten classes.
- `original_stage3_mitm_prediction_breakdown.csv` - saved Stage 3 MITM confusion breakdown.
- `repeated_group_safe_splits.csv` - ten seeds, both models, actual scores.
- `repeated_per_class_scores.csv` - class-specific statistics for each run.
- `repeat_summary.csv` - aggregate statistics.
- `repeated_split_stability.png` - visualization of MITM recall and macro-F1.

## Dataset and research attribution

Dharini, N., Janani, V. S., & Katiravan, J. (2026). "Efficient detection of intrusions in TON-IoT dataset using hybrid feature selection approach." *Scientific Reports* 16, 7763. https://doi.org/10.1038/s41598-026-37834-y

The diagnostics here are independently implemented and are not the original authors' released code.
