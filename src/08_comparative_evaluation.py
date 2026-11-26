"""STAGE 4C: verify saved predictions and compare difficult-class trade-offs.

Run after the full ablation experiment:

    python src/08_comparative_evaluation.py

All score tables are recomputed from row-level predictions.  The latency
benchmark uses identical fixed-test row indexes, one or more warm-ups, and
repeated monotonic-clock measurements; it is a same-machine comparison, not a
deployment throughput claim.
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
ablation = importlib.import_module("07_feature_ablation")
ALGORITHMS = augmentation.ALGORITHMS
FEATURE_CONFIGS = augmentation.FEATURE_CONFIGS
FOCUS_CLASSES = ("mitm", "ransomware", "xss")


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run 07_feature_ablation.py first")
    return pd.read_csv(path)


def verify_saved_results(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    classwise: pd.DataFrame,
    clean: pd.DataFrame,
    split: dict[str, np.ndarray],
    tolerance: float = 5e-9,
) -> dict[str, Any]:
    """Trace every saved metric back to predictions and fixed row indexes."""
    required_prediction_columns = {
        "model",
        "feature_set",
        "split",
        "row_index",
        "actual",
        "predicted",
    }
    if not required_prediction_columns.issubset(predictions.columns):
        raise ValueError(
            f"Prediction schema missing: {sorted(required_prediction_columns - set(predictions.columns))}"
        )
    expected_keys = {
        (model, feature_set, subset)
        for model in ALGORITHMS
        for feature_set in FEATURE_CONFIGS
        for subset in ("validation", "test")
    }
    found_keys = set(
        predictions[["model", "feature_set", "split"]].itertuples(index=False, name=None)
    )
    if found_keys != expected_keys:
        raise ValueError(f"Expected 16 model/config/split keys; missing={expected_keys - found_keys}")

    y = clean[TARGET].astype(str)
    labels = sorted(y.unique())
    recomputed_metrics = []
    recomputed_classwise = []
    for key, frame in predictions.groupby(["model", "feature_set", "split"], sort=False):
        model, feature_set, subset = key
        frame = frame.sort_values("row_index")
        expected_positions = np.sort(split[subset])
        positions = frame.row_index.to_numpy(dtype=int)
        if not np.array_equal(positions, expected_positions):
            raise AssertionError(f"{key}: predictions do not use the unchanged fixed {subset} rows")
        expected_actual = y.iloc[positions].to_numpy(dtype=str)
        if not np.array_equal(frame.actual.astype(str).to_numpy(), expected_actual):
            raise AssertionError(f"{key}: actual labels do not align to the deduplicated dataset")
        summary, rows, _ = ablation.calculate_metrics(
            expected_actual, frame.predicted.astype(str).to_numpy(), labels
        )
        recomputed_metrics.append(
            {"model": model, "feature_set": feature_set, "split": subset, **summary}
        )
        recomputed_classwise.extend(
            {"model": model, "feature_set": feature_set, "split": subset, **row}
            for row in rows
        )

    recomputed = pd.DataFrame(recomputed_metrics)
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mitm_precision",
        "mitm_recall",
        "mitm_f1",
        "normal_to_attack_fpr",
    ]
    joined = metrics.merge(
        recomputed,
        on=["model", "feature_set", "split"],
        suffixes=("_saved", "_recomputed"),
        validate="one_to_one",
    )
    maximum_differences = {}
    for column in metric_columns:
        difference = np.abs(joined[f"{column}_saved"] - joined[f"{column}_recomputed"])
        maximum_differences[column] = float(difference.max())
        if (difference > tolerance).any():
            raise AssertionError(f"Saved {column} does not trace to predictions")
    count_columns = [
        "n_rows",
        "mitm_support",
        "normal_rows",
        "normal_to_attack_false_alarms",
        "actual_attack_rows",
        "missed_actual_attacks",
    ]
    for column in count_columns:
        if not np.array_equal(
            joined[f"{column}_saved"].to_numpy(), joined[f"{column}_recomputed"].to_numpy()
        ):
            raise AssertionError(f"Saved {column} does not trace to predictions")

    recomputed_classes = pd.DataFrame(recomputed_classwise)
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
            raise AssertionError(f"Saved class-wise {column} does not trace to predictions")
    if not np.array_equal(
        class_joined["support_saved"].to_numpy(),
        class_joined["support_recomputed"].to_numpy(),
    ):
        raise AssertionError("Saved class support does not trace to predictions")
    return {
        "status": "passed",
        "groups_verified": len(expected_keys),
        "prediction_rows_verified": int(len(predictions)),
        "tolerance": tolerance,
        "maximum_absolute_differences": maximum_differences,
    }


def benchmark_latency(
    *,
    clean: pd.DataFrame,
    test_positions: np.ndarray,
    model_dir: Path,
    warmups: int,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark identical fixed-test batches and return raw plus summary rows."""
    if warmups < 1 or repeats < 2:
        raise ValueError("Latency protocol requires at least 1 warm-up and 2 measured repeats")
    raw_rows = []
    summaries = []
    for feature_set in FEATURE_CONFIGS:
        x_test = augmentation.feature_frame(clean, feature_set).iloc[test_positions]
        for algorithm in ALGORITHMS:
            path = model_dir / f"{algorithm}_{feature_set}.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Missing fitted pipeline: {path}")
            load_started = time.perf_counter()
            model = joblib.load(path)  # trusted model produced by this project
            load_seconds = time.perf_counter() - load_started
            for _ in range(warmups):
                model.predict(x_test)
            seconds = []
            for repeat in range(1, repeats + 1):
                started = time.perf_counter()
                predicted = model.predict(x_test)
                elapsed = time.perf_counter() - started
                if len(predicted) != len(test_positions):
                    raise RuntimeError("Latency run returned the wrong number of predictions")
                seconds.append(elapsed)
                raw_rows.append(
                    {
                        "model": algorithm,
                        "feature_set": feature_set,
                        "repeat": repeat,
                        "n_rows": len(test_positions),
                        "seconds": elapsed,
                        "ms_per_1000_rows": 1_000_000 * elapsed / len(test_positions),
                    }
                )
            milliseconds = 1_000_000 * np.asarray(seconds) / len(test_positions)
            summaries.append(
                {
                    "model": algorithm,
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
                f"Timed {algorithm}/{feature_set}: median "
                f"{summaries[-1]['median_ms_per_1000_rows']:.3f} ms/1,000 rows",
                flush=True,
            )
    return pd.DataFrame(raw_rows), pd.DataFrame(summaries)


def build_comparison_tables(
    metrics: pd.DataFrame,
    classwise: pd.DataFrame,
    predictions: pd.DataFrame,
    latency: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build fixed-sample score, false-alarm, and difficult-case tables."""
    test_metrics = metrics[metrics.split.eq("test")].copy()
    focus = classwise[
        classwise.split.eq("test") & classwise["class"].isin(FOCUS_CLASSES)
    ].copy()
    focus["missed_actual"] = focus["support"] - (
        focus["recall"] * focus["support"]
    ).round().astype(int)
    focus_wide = focus.pivot(
        index=["model", "feature_set"],
        columns="class",
        values=["precision", "recall", "f1", "support", "missed_actual"],
    )
    focus_wide.columns = [f"{label}_{metric}" for metric, label in focus_wide.columns]
    focus_wide = focus_wide.reset_index()
    # MITM summary columns already exist in ablation_metrics.csv.  Keep that
    # single source and add only the missed-count field from the class-wise view.
    focus_wide = focus_wide.drop(
        columns=["mitm_precision", "mitm_recall", "mitm_f1", "mitm_support"]
    )

    comparison = test_metrics.merge(
        focus_wide, on=["model", "feature_set"], validate="one_to_one"
    ).merge(latency, on=["model", "feature_set"], validate="one_to_one", suffixes=("", "_latency"))
    delta_metrics = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "normal_to_attack_fpr",
        "mitm_recall",
        "ransomware_recall",
        "xss_recall",
    ]
    baseline = comparison[comparison.feature_set.eq("F5")].set_index("model")
    for metric in delta_metrics:
        comparison[f"{metric}_delta_pp_vs_F5"] = comparison.apply(
            lambda row: 100.0 * (row[metric] - baseline.loc[row["model"], metric]), axis=1
        )

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

    mitm_predictions = predictions[
        predictions.split.eq("test") & predictions.actual.eq("mitm")
    ]
    mitm_breakdown = (
        mitm_predictions.groupby(["model", "feature_set", "predicted"])
        .size()
        .reset_index(name="count")
        .rename(columns={"predicted": "predicted_class"})
        .sort_values(["model", "feature_set", "count"], ascending=[True, True, False])
    )
    mitm_breakdown["is_true_positive"] = mitm_breakdown.predicted_class.eq("mitm")

    stage3 = pd.read_csv(RESULTS / "baseline_metrics.csv")
    stage4_f5 = metrics[metrics.feature_set.eq("F5")]
    parity = stage4_f5.merge(
        stage3,
        on=["model", "split"],
        suffixes=("_stage4_F5", "_stage3"),
        validate="one_to_one",
    )
    for metric in ("accuracy", "macro_f1", "weighted_f1", "mitm_recall"):
        parity[f"{metric}_difference"] = (
            parity[f"{metric}_stage4_F5"] - parity[f"{metric}_stage3"]
        )
    return {
        "fixed_test_comparison": comparison,
        "difficult_class_metrics": focus,
        "normal_to_attack_false_alarms": false_alarms,
        "mitm_prediction_breakdown": mitm_breakdown,
        "stage3_stage4_f5_parity": parity,
    }


def run(
    output_dir: str | Path = "results/stage4",
    model_dir: str | Path = "models/stage4",
    warmups: int = 1,
    timing_repeats: int = 5,
) -> dict[str, Path]:
    """Verify the completed ablation and write comparison/latency outputs."""
    output_path = augmentation.resolve_output_dir(output_dir)
    model_path = augmentation.resolve_output_dir(model_dir)
    metrics_path = output_path / "ablation_metrics.csv"
    class_path = output_path / "ablation_classwise_metrics.csv"
    predictions_path = output_path / "ablation_predictions.csv"
    metrics = _read_required(metrics_path)
    classwise = _read_required(class_path)
    predictions = _read_required(predictions_path)
    _, clean = clean_data()
    split = augmentation.validate_stage4_data(clean)
    trace = verify_saved_results(predictions, metrics, classwise, clean, split)
    raw_latency, latency = benchmark_latency(
        clean=clean,
        test_positions=split["test"],
        model_dir=model_path,
        warmups=warmups,
        repeats=timing_repeats,
    )
    raw_latency_path = output_path / "latency_measurements.csv"
    latency_path = output_path / "latency_summary.csv"
    raw_latency.to_csv(raw_latency_path, index=False, float_format="%.10f")
    latency.to_csv(latency_path, index=False, float_format="%.10f")

    tables = build_comparison_tables(metrics, classwise, predictions, latency)
    outputs: dict[str, Path] = {
        "latency_measurements": raw_latency_path,
        "latency_summary": latency_path,
    }
    for name, frame in tables.items():
        destination = output_path / f"{name}.csv"
        frame.to_csv(destination, index=False, float_format="%.10f")
        outputs[name] = destination

    manifest = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_status": "exploratory/comparative; not an independent blind holdout",
        "metric_trace_verification": trace,
        "sources": {
            path.name: augmentation.sha256_file(path)
            for path in (metrics_path, class_path, predictions_path)
        },
        "latency_protocol": {
            "batch": "all rows in the unchanged fixed test partition",
            "rows_per_batch": int(len(split["test"])),
            "warmup_runs": warmups,
            "measured_repeats": timing_repeats,
            "clock": "time.perf_counter (monotonic)",
            "summary": "median, mean, sample standard deviation, min, max, and IQR",
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
    manifest_path = output_path / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    outputs["manifest"] = manifest_path
    print(f"Stage 4 comparison complete: {outputs['fixed_test_comparison']}")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/stage4")
    parser.add_argument("--model-dir", default="models/stage4")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=5)
    args = parser.parse_args()
    run(args.output_dir, args.model_dir, args.warmups, args.timing_repeats)
