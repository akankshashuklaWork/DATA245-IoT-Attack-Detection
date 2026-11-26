# Stage 4 execution report

**Run date:** 25 November 2026

**Environment:** Python 3.12.15, pandas 2.1.4, NumPy 1.26.4, scikit-learn 1.8.0, joblib 1.6.0

**Status:** Core fixed-split study, five figures, tests, and optional ten-seed split-sensitivity study completed.

## Commands executed

From the repository root, the completed workflow used:

```bash
MPLCONFIGDIR=/tmp/data245-mpl .venv/bin/python tests/check_saved_outputs.py
.venv/bin/python src/06_feature_augmentation.py --output-dir /tmp/data245-stage4-audit
MPLCONFIGDIR=/tmp/data245-mpl .venv/bin/python src/07_feature_ablation.py \
  --smoke --output-dir /tmp/data245-stage4-smoke \
  --model-dir /tmp/data245-stage4-smoke-models --force
MPLCONFIGDIR=/tmp/data245-mpl .venv/bin/python -m pytest -q tests/test_stage4.py \
  -k 'not full_ablation and not all_eight and not manifest_and_latency and not every_final'
MPLCONFIGDIR=/tmp/data245-mpl LOKY_MAX_CPU_COUNT=8 \
  .venv/bin/python src/07_feature_ablation.py --split fixed \
  --output-dir results/stage4 --model-dir models/stage4
MPLCONFIGDIR=/tmp/data245-mpl LOKY_MAX_CPU_COUNT=8 \
  .venv/bin/python src/08_comparative_evaluation.py --warmups 1 --timing-repeats 5
MPLCONFIGDIR=/tmp/data245-mpl LOKY_MAX_CPU_COUNT=8 \
  .venv/bin/python src/10_stage4_split_stability.py --seeds 42 \
  --output-dir /tmp/data245-stage4-one-seed
MPLCONFIGDIR=/tmp/data245-mpl LOKY_MAX_CPU_COUNT=8 \
  .venv/bin/python src/10_stage4_split_stability.py \
  --seeds 42,7,17,23,29,41,73,101,202,313 --output-dir results/stage4
MPLCONFIGDIR=/tmp/data245-mpl .venv/bin/python src/09_final_visualizations.py
MPLCONFIGDIR=/tmp/data245-mpl .venv/bin/python -m pytest -q tests/test_stage4.py
MPLCONFIGDIR=/tmp/data245-mpl .venv/bin/python tests/check_saved_outputs.py
```

A first parity-gated full execution stopped after detecting a one-row kNN test discrepancy versus the saved Stage 3 model. Investigation showed only machine-precision scaler differences and three changed kNN predictions across validation plus test. The discrepancy is fully recorded in `results/stage4/experiment_manifest.json`; the experiment settings were not tuned in response. A comparison-table column collision was also corrected before final outputs were produced. These failed intermediate runs did not alter Stage 1–3 files.

## Main outcomes

- Decision Tree F7 has the highest Decision Tree macro-F1 (93.85%), perfect fixed-test ransomware recall, and a 0.78% normal-to-attack false-positive rate, but its MITM recall (68.46%) is lower than F5 (69.80%).
- Decision Tree F5_PROTO has the highest Decision Tree MITM recall (71.81%) and nearly the same macro-F1 (93.60%) as F7.
- kNN F7 has the highest kNN macro-F1 (88.79%), ransomware recall (38.72%), and XSS recall (86.83%); MITM recall is 68.46% versus 65.77% for F5.
- kNN ransomware recall improves materially with categorical fields but remains weak in absolute terms.
- F7 adds runtime: median inference is 0.277 versus 0.056 ms/1,000 rows for Decision Tree and 37.373 versus 12.795 ms/1,000 rows for kNN on this machine.
- The ten-seed paired study shows positive mean deltas for many augmented configurations, but ransomware and XSS gains are highly split-sensitive.

## Verification outcome

- Original saved-output check: passed before implementation.
- Stage 4 unit/acceptance suite: 17 tests pass, including the completed ten-seed extension and chart-source checks.
- Original saved-output check: rerun after implementation to confirm Stage 1–3 preservation.
- Dataset checksum, split checksum, package versions, model settings, encoded names/categories, split sizes, protected artifact hashes, and commands are recorded in manifests.
- Every final figure has a companion plotted-data CSV and source-file checksums.

## Known limitations

The fixed-test study is exploratory because Stage 3 already exposed the test labels and errors. Repeated seeds reuse observations. There is no independent temporal, environment, session, or device holdout, so the results do not establish deployment generalization. The latency benchmark is comparative and machine-specific.
