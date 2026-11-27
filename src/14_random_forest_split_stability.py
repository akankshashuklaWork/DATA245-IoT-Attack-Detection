"""STAGE 5D (optional): matched Random Forest split-sensitivity study.

Run one seed before the full extension:

    python src/14_random_forest_split_stability.py --seeds 42 \
        --output-dir /tmp/stage5-rf-one-seed

The ten runs reuse observations and are not independent blind test sets.
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from common import SEEDS, TARGET, clean_data

augmentation = importlib.import_module("06_feature_augmentation")
stage4_ablation = importlib.import_module("07_feature_ablation")
stage4_stability = importlib.import_module("10_stage4_split_stability")
rf_ablation = importlib.import_module("11_random_forest_ablation")
FEATURE_CONFIGS = augmentation.FEATURE_CONFIGS


def _paired_deltas(classwise: pd.DataFrame) -> pd.DataFrame:
    baseline = classwise[classwise.feature_set.eq("F5")][
        ["seed", "class", "precision", "recall", "f1", "support"]
    ].rename(
        columns={
            "precision": "precision_F5",
            "recall": "recall_F5",
            "f1": "f1_F5",
            "support": "support_F5",
        }
    )
    paired = classwise.merge(
        baseline, on=["seed", "class"], validate="many_to_one"
    )
    if not paired.support.eq(paired.support_F5).all():
        raise AssertionError("Feature configurations do not share matched class support")
    for metric in ("precision", "recall", "f1"):
        paired[f"{metric}_delta_vs_F5"] = paired[metric] - paired[f"{metric}_F5"]
    return paired


def run(
    seeds: list[int],
    *,
    output_dir: str | Path = "results/stage5_random_forest",
    n_estimators: int = rf_ablation.DEFAULT_ESTIMATORS,
    n_jobs: int = 1,
) -> dict[str, Path]:
    """Evaluate matched feature-set deltas for Random Forest across split seeds."""
    if not seeds:
        raise ValueError("Provide at least one split seed")
    destination = augmentation.resolve_output_dir(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _, clean = clean_data()
    y = clean[TARGET].astype(str)
    labels = sorted(y.unique())
    frames = {
        feature_set: augmentation.feature_frame(clean, feature_set)
        for feature_set in FEATURE_CONFIGS
    }
    metric_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for run_number, seed in enumerate(seeds, start=1):
        positions = stage4_stability.matched_group_split(clean, seed)
        split_rows.extend(
            {
                "seed": seed,
                "split": name,
                "rows": len(values),
                "mitm_rows": int(y.iloc[values].eq("mitm").sum()),
            }
            for name, values in positions.items()
        )
        for feature_set, x in frames.items():
            model = rf_ablation.build_random_forest_pipeline(
                feature_set,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                seed=42,
            )
            started = time.perf_counter()
            model.fit(x.iloc[positions["train"]], y.iloc[positions["train"]])
            training_seconds = time.perf_counter() - started
            started = time.perf_counter()
            predicted = model.predict(x.iloc[positions["test"]])
            prediction_seconds = time.perf_counter() - started
            actual = y.iloc[positions["test"]].to_numpy(dtype=str)
            summary, per_class, _ = stage4_ablation.calculate_metrics(
                actual, predicted, labels
            )
            metric_rows.append(
                {
                    "seed": seed,
                    "model": "Random_Forest",
                    "feature_set": feature_set,
                    "train_rows": len(positions["train"]),
                    "validation_rows": len(positions["validation"]),
                    "test_rows": len(positions["test"]),
                    "training_seconds": training_seconds,
                    "prediction_seconds": prediction_seconds,
                    **summary,
                }
            )
            class_rows.extend(
                {
                    "seed": seed,
                    "model": "Random_Forest",
                    "feature_set": feature_set,
                    **row,
                }
                for row in per_class
            )
        print(f"Completed Random Forest matched split {run_number}/{len(seeds)} (seed {seed})", flush=True)

    metrics = pd.DataFrame(metric_rows)
    classwise = pd.DataFrame(class_rows)
    paired = _paired_deltas(classwise)
    summary = (
        paired.groupby(["feature_set", "class"])
        .agg(
            runs=("seed", "count"),
            support_mean=("support", "mean"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            recall_min=("recall", "min"),
            recall_max=("recall", "max"),
            recall_delta_mean=("recall_delta_vs_F5", "mean"),
            recall_delta_std=("recall_delta_vs_F5", "std"),
            recall_delta_min=("recall_delta_vs_F5", "min"),
            recall_delta_max=("recall_delta_vs_F5", "max"),
            precision_delta_mean=("precision_delta_vs_F5", "mean"),
            f1_delta_mean=("f1_delta_vs_F5", "mean"),
        )
        .reset_index()
    )
    outputs = {
        "metrics": destination / "random_forest_stability_metrics.csv",
        "classwise": destination / "random_forest_stability_classwise_metrics.csv",
        "paired_deltas": destination / "random_forest_stability_paired_deltas.csv",
        "summary": destination / "random_forest_stability_summary.csv",
        "split_sizes": destination / "random_forest_stability_split_sizes.csv",
    }
    metrics.to_csv(outputs["metrics"], index=False, float_format="%.10f")
    classwise.to_csv(outputs["classwise"], index=False, float_format="%.10f")
    paired.to_csv(outputs["paired_deltas"], index=False, float_format="%.10f")
    summary.to_csv(outputs["summary"], index=False, float_format="%.10f")
    pd.DataFrame(split_rows).to_csv(outputs["split_sizes"], index=False)
    manifest = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "n_runs": len(seeds),
        "random_forest_settings": {
            "n_estimators": n_estimators,
            "criterion": "gini",
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": 42,
            "n_jobs": n_jobs,
        },
        "matched_design": (
            "identical train/validation/test rows for F5, F5_PROTO, F5_STATE, and F7 "
            "within each seed"
        ),
        "split_design": "70/15/15 exact-F5-group split with majority-label stratification",
        "interpretation": (
            "split sensitivity on overlapping observations; not independent or blind validation"
        ),
        "dataset_sha256": augmentation.sha256_file(augmentation.DATA),
        "package_versions": augmentation.package_versions(),
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    manifest_path = destination / "random_forest_stability_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    outputs["manifest"] = manifest_path
    print(f"Random Forest split-sensitivity outputs saved in {destination}")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--output-dir", default="results/stage5_random_forest")
    parser.add_argument("--n-estimators", type=int, default=rf_ablation.DEFAULT_ESTIMATORS)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    parsed = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    run(
        parsed,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )
