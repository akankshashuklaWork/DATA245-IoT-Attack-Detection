"""STAGE 5B: verify Random Forest results and compare all three algorithms.

Run after ``11_random_forest_ablation.py``:

    python src/12_random_forest_evaluation.py

The script traces Random Forest metrics to row-level predictions, then times
Decision Tree, kNN, and Random Forest together on the identical fixed-test
batch.  This is a same-machine comparison, not a deployment claim.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from common import MODELS, RESULTS, ROOT, TARGET, clean_data

augmentation = importlib.import_module("06_feature_augmentation")
stage4_ablation = importlib.import_module("07_feature_ablation")
rf_ablation = importlib.import_module("11_random_forest_ablation")
FEATURE_CONFIGS = augmentation.FEATURE_CONFIGS
MODEL_ORDER = ("Decision_Tree", "kNN", "Random_Forest")
FOCUS_CLASSES = ("mitm", "ransomware", "xss")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the preceding experiment first")
    return pd.read_csv(path)


def verify_random_forest_results(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    classwise: pd.DataFrame,
    clean: pd.DataFrame,
    split: dict[str, np.ndarray],
    tolerance: float = 5e-9,
) -> dict[str, Any]:
    """Recompute every Random Forest score and verify fixed row membership."""
    expected_keys = {
        (feature_set, subset)
        for feature_set in FEATURE_CONFIGS
        for subset in ("validation", "test")
    }
    found_keys = set(
        predictions[["feature_set", "split"]].itertuples(index=False, name=None)
    )
    if found_keys != expected_keys:
        raise AssertionError(f"Random Forest output keys differ: {expected_keys - found_keys}")
    if not predictions.model.eq("Random_Forest").all():
        raise AssertionError("Unexpected model name in Random Forest predictions")
    y = clean[TARGET].astype(str)
    labels = sorted(y.unique())
    metric_rows = []
    class_rows = []
    for (feature_set, subset), frame in predictions.groupby(["feature_set", "split"]):
        frame = frame.sort_values("row_index")
        positions = frame.row_index.to_numpy(dtype=int)
        if not np.array_equal(positions, np.sort(split[subset])):
            raise AssertionError(f"{feature_set}/{subset} does not use the unchanged fixed rows")
        actual = y.iloc[positions].to_numpy(dtype=str)
        if not np.array_equal(actual, frame.actual.astype(str).to_numpy()):
            raise AssertionError(f"{feature_set}/{subset} actual labels are misaligned")
        summary, per_class, _ = stage4_ablation.calculate_metrics(
            actual, frame.predicted.astype(str).to_numpy(), labels
        )
        metric_rows.append(
            {
                "model": "Random_Forest",
                "feature_set": feature_set,
                "split": subset,
                **summary,
            }
        )
        class_rows.extend(
            {
                "model": "Random_Forest",
                "feature_set": feature_set,
                "split": subset,
                **row,
            }
            for row in per_class
        )

    recomputed = pd.DataFrame(metric_rows)
    joined = metrics.merge(
        recomputed,
        on=["model", "feature_set", "split"],
        suffixes=("_saved", "_recomputed"),
        validate="one_to_one",
    )
    numeric_metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mitm_precision",
        "mitm_recall",
        "mitm_f1",
        "normal_to_attack_fpr",
    ]
    maximum_differences = {}
    for column in numeric_metrics:
        difference = np.abs(joined[f"{column}_saved"] - joined[f"{column}_recomputed"])
        maximum_differences[column] = float(difference.max())
        if (difference > tolerance).any():
            raise AssertionError(f"Saved Random Forest {column} does not trace to predictions")
    for column in (
        "n_rows",
        "mitm_support",
        "normal_rows",
        "normal_to_attack_false_alarms",
        "actual_attack_rows",
        "missed_actual_attacks",
    ):
        if not np.array_equal(
            joined[f"{column}_saved"].to_numpy(),
            joined[f"{column}_recomputed"].to_numpy(),
        ):
            raise AssertionError(f"Saved Random Forest {column} count does not trace")

    recomputed_classes = pd.DataFrame(class_rows)
    class_joined = classwise.merge(
        recomputed_classes,
        on=["model", "feature_set", "split", "class"],
        suffixes=("_saved", "_recomputed"),
        validate="one_to_one",
    )
    for column in ("precision", "recall", "f1"):
        difference = np.abs(
            class_joined[f"{column}_saved"] - class_joined[f"{column}_recomputed"]
        )
        maximum_differences[f"classwise_{column}"] = float(difference.max())
        if (difference > tolerance).any():
            raise AssertionError(f"Random Forest class-wise {column} does not trace")
    if not np.array_equal(
        class_joined.support_saved.to_numpy(), class_joined.support_recomputed.to_numpy()
    ):
        raise AssertionError("Random Forest class supports do not trace")
    return {
        "status": "passed",
        "groups_verified": len(expected_keys),
        "prediction_rows_verified": int(len(predictions)),
        "tolerance": tolerance,
        "maximum_absolute_differences": maximum_differences,
    }


def _model_path(model: str, feature_set: str, rf_model_dir: Path) -> Path:
    if model == "Random_Forest":
        return rf_model_dir / f"Random_Forest_{feature_set}.joblib"
    return MODELS / "stage4" / f"{model}_{feature_set}.joblib"


def benchmark_all_models(
    *,
    clean: pd.DataFrame,
    test_positions: np.ndarray,
    rf_model_dir: Path,
    warmups: int,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark all twelve pipelines together on one fixed-test batch."""
    if warmups < 1 or repeats < 2:
        raise ValueError("Timing requires at least one warm-up and two measured repeats")
    raw_rows = []
    summaries = []
    for feature_set in FEATURE_CONFIGS:
        x_test = augmentation.feature_frame(clean, feature_set).iloc[test_positions]
        for model_name in MODEL_ORDER:
            path = _model_path(model_name, feature_set, rf_model_dir)
            if not path.exists():
                raise FileNotFoundError(f"Missing fitted model: {path}")
            started = time.perf_counter()
            model = joblib.load(path)  # trusted artifact created by this project
            load_seconds = time.perf_counter() - started
            for _ in range(warmups):
                model.predict(x_test)
            elapsed_values = []
            for repeat in range(1, repeats + 1):
                started = time.perf_counter()
                predicted = model.predict(x_test)
                elapsed = time.perf_counter() - started
                if len(predicted) != len(test_positions):
                    raise RuntimeError("Timing prediction count is incorrect")
                elapsed_values.append(elapsed)
                raw_rows.append(
                    {
                        "model": model_name,
                        "feature_set": feature_set,
                        "repeat": repeat,
                        "n_rows": len(test_positions),
                        "seconds": elapsed,
                        "ms_per_1000_rows": 1_000_000 * elapsed / len(test_positions),
                    }
                )
            milliseconds = 1_000_000 * np.asarray(elapsed_values) / len(test_positions)
            summaries.append(
                {
                    "model": model_name,
                    "feature_set": feature_set,
                    "n_rows": len(test_positions),
                    "warmup_runs": warmups,
                    "measured_repeats": repeats,
                    "load_seconds": load_seconds,
                    "median_ms_per_1000_rows": float(np.median(milliseconds)),
                    "mean_ms_per_1000_rows": float(np.mean(milliseconds)),
                    "std_ms_per_1000_rows": float(np.std(milliseconds, ddof=1)),
                    "min_ms_per_1000_rows": float(np.min(milliseconds)),
                    "max_ms_per_1000_rows": float(np.max(milliseconds)),
                    "iqr_ms_per_1000_rows": float(
                        np.percentile(milliseconds, 75) - np.percentile(milliseconds, 25)
                    ),
                    "model_size_bytes": path.stat().st_size,
                }
            )
            print(
                f"Timed {model_name}/{feature_set}: "
                f"{summaries[-1]['median_ms_per_1000_rows']:.3f} ms/1,000 rows",
                flush=True,
            )
    return pd.DataFrame(raw_rows), pd.DataFrame(summaries)


def build_combined_tables(
    stage4_metrics: pd.DataFrame,
    stage4_classwise: pd.DataFrame,
    rf_metrics: pd.DataFrame,
    rf_classwise: pd.DataFrame,
    rf_predictions: pd.DataFrame,
    latency: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build normalized comparison tables for all models/configurations."""
    metrics = pd.concat(
        [
            stage4_metrics[stage4_metrics.split.eq("test")],
            rf_metrics[rf_metrics.split.eq("test")],
        ],
        ignore_index=True,
        sort=False,
    )
    classwise = pd.concat(
        [
            stage4_classwise[stage4_classwise.split.eq("test")],
            rf_classwise[rf_classwise.split.eq("test")],
        ],
        ignore_index=True,
        sort=False,
    )
    focus = classwise[classwise["class"].isin(FOCUS_CLASSES)].copy()
    focus["missed_actual"] = focus.support - (focus.recall * focus.support).round().astype(int)
    focus_wide = focus.pivot(
        index=["model", "feature_set"],
        columns="class",
        values=["precision", "recall", "f1", "support", "missed_actual"],
    )
    focus_wide.columns = [f"{attack}_{metric}" for metric, attack in focus_wide.columns]
    focus_wide = focus_wide.reset_index().drop(
        columns=["mitm_precision", "mitm_recall", "mitm_f1", "mitm_support"]
    )
    comparison = metrics.merge(
        focus_wide, on=["model", "feature_set"], validate="one_to_one"
    ).merge(
        latency,
        on=["model", "feature_set"],
        validate="one_to_one",
        suffixes=("", "_timing"),
    )
    for metric in (
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "normal_to_attack_fpr",
        "mitm_recall",
        "ransomware_recall",
        "xss_recall",
    ):
        baseline = comparison[comparison.feature_set.eq("F5")].set_index("model")[metric]
        comparison[f"{metric}_delta_pp_vs_same_model_F5"] = comparison.apply(
            lambda row: 100 * (row[metric] - baseline.loc[row.model]), axis=1
        )
    comparison["macro_f1_rank"] = comparison.macro_f1.rank(
        ascending=False, method="min"
    ).astype(int)
    comparison["normal_to_attack_fpr_rank"] = comparison.normal_to_attack_fpr.rank(
        ascending=True, method="min"
    ).astype(int)

    false_alarms = comparison[
        [
            "model",
            "feature_set",
            "normal_rows",
            "normal_to_attack_false_alarms",
            "normal_to_attack_fpr",
            "actual_attack_rows",
            "missed_actual_attacks",
        ]
    ].copy()
    false_alarms["definition"] = (
        "actual normal rows predicted as any attack / actual normal rows"
    )
    mitm = rf_predictions[
        rf_predictions.split.eq("test") & rf_predictions.actual.eq("mitm")
    ]
    mitm_breakdown = (
        mitm.groupby(["feature_set", "predicted"])
        .size()
        .reset_index(name="count")
        .rename(columns={"predicted": "predicted_class"})
        .sort_values(["feature_set", "count"], ascending=[True, False])
    )
    mitm_breakdown["is_true_positive"] = mitm_breakdown.predicted_class.eq("mitm")
    return {
        "combined_three_model_test_comparison": comparison,
        "combined_difficult_class_metrics": focus,
        "combined_false_alarm_comparison": false_alarms,
        "random_forest_mitm_breakdown": mitm_breakdown,
    }


def run(
    *,
    output_dir: str | Path = "results/stage5_random_forest",
    rf_model_dir: str | Path = "models/stage5_random_forest",
    warmups: int = 1,
    timing_repeats: int = 5,
) -> dict[str, Path]:
    """Verify Random Forest outputs and save a fair three-model comparison."""
    output_path = augmentation.resolve_output_dir(output_dir)
    model_path = augmentation.resolve_output_dir(rf_model_dir)
    stage4_metrics_path = RESULTS / "stage4" / "ablation_metrics.csv"
    stage4_classwise_path = RESULTS / "stage4" / "ablation_classwise_metrics.csv"
    rf_metrics_path = output_path / "random_forest_metrics.csv"
    rf_classwise_path = output_path / "random_forest_classwise_metrics.csv"
    rf_predictions_path = output_path / "random_forest_predictions.csv"
    stage4_metrics = _read(stage4_metrics_path)
    stage4_classwise = _read(stage4_classwise_path)
    rf_metrics = _read(rf_metrics_path)
    rf_classwise = _read(rf_classwise_path)
    rf_predictions = _read(rf_predictions_path)
    _, clean = clean_data()
    split = augmentation.validate_stage4_data(clean)
    trace = verify_random_forest_results(
        rf_predictions, rf_metrics, rf_classwise, clean, split
    )
    raw_latency, latency = benchmark_all_models(
        clean=clean,
        test_positions=split["test"],
        rf_model_dir=model_path,
        warmups=warmups,
        repeats=timing_repeats,
    )
    outputs: dict[str, Path] = {
        "latency_measurements": output_path / "all_model_latency_measurements.csv",
        "latency_summary": output_path / "all_model_latency_summary.csv",
    }
    raw_latency.to_csv(outputs["latency_measurements"], index=False, float_format="%.10f")
    latency.to_csv(outputs["latency_summary"], index=False, float_format="%.10f")
    tables = build_combined_tables(
        stage4_metrics,
        stage4_classwise,
        rf_metrics,
        rf_classwise,
        rf_predictions,
        latency,
    )
    for name, frame in tables.items():
        destination = output_path / f"{name}.csv"
        frame.to_csv(destination, index=False, float_format="%.10f")
        outputs[name] = destination

    manifest = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_status": "exploratory/comparative; fixed test was previously examined",
        "random_forest_trace_verification": trace,
        "sources": {
            str(path.relative_to(ROOT)): augmentation.sha256_file(path)
            for path in (
                stage4_metrics_path,
                stage4_classwise_path,
                rf_metrics_path,
                rf_classwise_path,
                rf_predictions_path,
            )
        },
        "latency_protocol": {
            "models": list(MODEL_ORDER),
            "feature_sets": list(FEATURE_CONFIGS),
            "batch": "unchanged fixed test partition",
            "rows_per_batch": int(len(split["test"])),
            "warmup_runs": warmups,
            "measured_repeats": timing_repeats,
            "clock": "time.perf_counter",
            "model_loading_timed_separately": True,
            "claim_limit": "same-machine comparative benchmark, not deployment throughput",
        },
        "package_versions": augmentation.package_versions(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    manifest_path = output_path / "random_forest_comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    outputs["manifest"] = manifest_path
    print(f"Three-model comparison complete: {outputs['combined_three_model_test_comparison']}")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/stage5_random_forest")
    parser.add_argument("--rf-model-dir", default="models/stage5_random_forest")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=5)
    args = parser.parse_args()
    run(
        output_dir=args.output_dir,
        rf_model_dir=args.rf_model_dir,
        warmups=args.warmups,
        timing_repeats=args.timing_repeats,
    )
