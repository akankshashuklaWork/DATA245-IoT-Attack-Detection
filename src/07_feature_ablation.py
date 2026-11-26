"""STAGE 4B: controlled fixed-split feature-ablation experiment.

Full prespecified run (evaluates validation, then the previously studied test):

    python src/07_feature_ablation.py --split fixed --output-dir results/stage4

Fast validation-only smoke run (never reads or predicts the test partition):

    python src/07_feature_ablation.py --smoke --output-dir /tmp/stage4-smoke \
        --model-dir /tmp/stage4-smoke-models

The old fixed test set was examined in Stage 3, so every Stage 4 test result is
labelled exploratory/comparative rather than a blind confirmatory result.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from common import MODELS, RESULTS, ROOT, SEED, TARGET, clean_data

augmentation = importlib.import_module("06_feature_augmentation")
ALGORITHMS = augmentation.ALGORITHMS
FEATURE_CONFIGS = augmentation.FEATURE_CONFIGS


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    labels: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    """Calculate summary, class-wise, and confusion values from predictions."""
    report = classification_report(
        actual,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    normal_mask = actual == "normal"
    normal_false_alarms = int(np.sum(normal_mask & (predicted != "normal")))
    normal_count = int(np.sum(normal_mask))
    attack_mask = actual != "normal"
    missed_attacks = int(np.sum(attack_mask & (predicted == "normal")))
    summary = {
        "n_rows": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(actual, predicted, labels=labels, average="weighted", zero_division=0)
        ),
        "mitm_precision": float(report["mitm"]["precision"]),
        "mitm_recall": float(report["mitm"]["recall"]),
        "mitm_f1": float(report["mitm"]["f1-score"]),
        "mitm_support": int(report["mitm"]["support"]),
        "normal_rows": normal_count,
        "normal_to_attack_false_alarms": normal_false_alarms,
        "normal_to_attack_fpr": float(normal_false_alarms / normal_count) if normal_count else np.nan,
        "actual_attack_rows": int(np.sum(attack_mask)),
        "missed_actual_attacks": missed_attacks,
    }
    classwise = [
        {
            "class": label,
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in labels
    ]
    return summary, classwise, confusion_matrix(actual, predicted, labels=labels)


def _sample_positions(
    positions: np.ndarray,
    y: pd.Series,
    maximum: int,
    seed: int,
) -> np.ndarray:
    """Create a deterministic, roughly class-stratified smoke-test subset."""
    if maximum >= len(positions):
        return positions
    local = pd.DataFrame({"position": positions, "label": y.iloc[positions].to_numpy()})
    fraction = maximum / len(local)
    sampled = (
        local.groupby("label", group_keys=False, sort=True)
        .apply(lambda frame: frame.sample(max(1, round(len(frame) * fraction)), random_state=seed))
        .head(maximum)
    )
    if len(sampled) < maximum:
        remaining = local[~local.position.isin(sampled.position)]
        sampled = pd.concat(
            [sampled, remaining.sample(maximum - len(sampled), random_state=seed)],
            ignore_index=True,
        )
    return np.sort(sampled.position.to_numpy(dtype=int))


def _model_details(model: Any, algorithm: str) -> dict[str, int]:
    classifier = model.named_steps["classifier"]
    if algorithm == "Decision_Tree":
        return {
            "tree_depth": int(classifier.get_depth()),
            "tree_leaves": int(classifier.get_n_leaves()),
        }
    return {"knn_reference_rows": int(classifier.n_samples_fit_)}


def _assert_baseline_parity(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    metric_tolerance: float = 2e-4,
    maximum_prediction_disagreements_per_split: int = 2,
) -> dict[str, Any]:
    """Require F5 results to reproduce Stage 3 up to tiny numeric tie variance.

    The count-based guard is the primary criterion: no F5 model may disagree
    with the saved Stage 3 predictions on more than two of 24,831 rows in a
    partition.  The metric guard prevents those few rows from masking a larger
    class-wise change.  Any nonzero disagreement is reported, never hidden.
    """
    baseline_path = RESULTS / "baseline_metrics.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Cannot check F5 parity; missing {baseline_path}")
    baseline = pd.read_csv(baseline_path)
    current = metrics[metrics.feature_set.eq("F5")]
    keys = ["model", "split"]
    merged = current.merge(baseline, on=keys, suffixes=("_stage4", "_stage3"), validate="one_to_one")
    checked = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mitm_recall"]
    differences: dict[str, float] = {}
    for metric in checked:
        delta = np.abs(merged[f"{metric}_stage4"] - merged[f"{metric}_stage3"])
        differences[metric] = float(delta.max())
        if (delta > metric_tolerance).any():
            failures = merged.loc[
                delta > metric_tolerance,
                keys + [f"{metric}_stage4", f"{metric}_stage3"],
            ]
            raise AssertionError(
                f"F5 parity failure for {metric} (tolerance={metric_tolerance}):\n"
                f"{failures.to_string(index=False)}"
            )
    disagreement_rows: list[dict[str, Any]] = []
    disagreement_counts: dict[str, int] = {}
    if predictions is not None:
        saved_path = RESULTS / "predictions_val_test.csv"
        saved = pd.read_csv(saved_path)
        current = predictions[predictions.feature_set.eq("F5")]
        paired = current.merge(
            saved,
            on=["model", "split", "row_index", "actual"],
            suffixes=("_stage4", "_stage3"),
            validate="one_to_one",
        )
        for (model, subset), frame in paired.groupby(["model", "split"]):
            disagreements = frame[frame.predicted_stage4.ne(frame.predicted_stage3)]
            key = f"{model}/{subset}"
            disagreement_counts[key] = int(len(disagreements))
            if len(disagreements) > maximum_prediction_disagreements_per_split:
                raise AssertionError(
                    f"F5 parity failure: {key} has {len(disagreements)} prediction "
                    f"disagreements (maximum {maximum_prediction_disagreements_per_split})"
                )
            disagreement_rows.extend(
                {
                    "model": model,
                    "split": subset,
                    "row_index": int(row.row_index),
                    "actual": str(row.actual),
                    "stage3_predicted": str(row.predicted_stage3),
                    "stage4_predicted": str(row.predicted_stage4),
                }
                for row in disagreements.itertuples(index=False)
            )
    nonzero = sum(disagreement_counts.values())
    return {
        "status": "passed_exactly" if nonzero == 0 else "passed_with_documented_numeric_variance",
        "metric_tolerance": metric_tolerance,
        "maximum_prediction_disagreements_per_split": maximum_prediction_disagreements_per_split,
        "prediction_disagreement_counts": disagreement_counts,
        "prediction_disagreements": disagreement_rows,
        "maximum_absolute_metric_differences": differences,
        "variance_note": (
            "The refit uses the recorded current package versions. Tiny floating-point "
            "differences in standardized kNN geometry can change near-tied neighbors."
            if nonzero
            else "No saved Stage 3 prediction changed."
        ),
    }


def _protected_artifact_hashes() -> dict[str, str]:
    paths = [
        augmentation.DATA,
        RESULTS / "fixed_split_assignments.csv",
        RESULTS / "baseline_metrics.csv",
        MODELS / "Decision_Tree_five_features.joblib",
        MODELS / "kNN_five_features.joblib",
        ROOT / "presentation" / "DATA245_IoT_Midterm_12_Slides.pptx",
    ]
    return {str(path.relative_to(ROOT)): augmentation.sha256_file(path) for path in paths}


def run(
    *,
    split_name: str = "fixed",
    output_dir: str | Path = "results/stage4",
    model_dir: str | Path = "models/stage4",
    smoke: bool = False,
    smoke_train_rows: int = 5000,
    smoke_validation_rows: int = 1500,
    n_jobs: int = 1,
    force: bool = False,
) -> dict[str, Path]:
    """Fit all eight frozen combinations and save traceable outputs."""
    if split_name != "fixed":
        raise ValueError("Stage 4 core study supports only --split fixed")
    output_path = augmentation.resolve_output_dir(output_dir)
    model_path = augmentation.resolve_output_dir(model_dir)
    final_manifest = output_path / "experiment_manifest.json"
    if final_manifest.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite completed run {final_manifest}. Use --force only for an explicit rerun."
        )
    output_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    protected_before = _protected_artifact_hashes()
    raw, clean = clean_data()
    fixed = augmentation.validate_stage4_data(clean)
    y = clean[TARGET].astype(str)
    labels = sorted(y.unique())
    train_positions = fixed["train"]
    validation_positions = fixed["validation"]
    evaluated_splits = ["validation"]
    if smoke:
        train_positions = _sample_positions(train_positions, y, smoke_train_rows, SEED)
        validation_positions = _sample_positions(
            validation_positions, y, smoke_validation_rows, SEED + 1
        )
    else:
        evaluated_splits.append("test")

    audit = augmentation.build_data_audit(clean, fixed)
    audit["package_versions"] = augmentation.package_versions()
    (output_path / "preprocessing_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    metric_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    fitted: list[dict[str, Any]] = []
    model_metadata: list[dict[str, Any]] = []

    # Phase 1: train and validate every prespecified combination.  No test rows
    # are transformed or predicted until all eight configurations are frozen.
    for feature_set in FEATURE_CONFIGS:
        x = augmentation.feature_frame(clean, feature_set)
        for algorithm in ALGORITHMS:
            print(f"Fitting {algorithm}/{feature_set} on {len(train_positions):,} rows...", flush=True)
            model = augmentation.build_stage4_pipeline(
                feature_set, algorithm, seed=SEED, n_jobs=n_jobs
            )
            started = time.perf_counter()
            model.fit(x.iloc[train_positions], y.iloc[train_positions])
            training_seconds = time.perf_counter() - started
            encoded_names, category_mapping = augmentation.encoded_feature_metadata(
                model, feature_set
            )
            destination = model_path / f"{algorithm}_{feature_set}.joblib"
            joblib.dump(model, destination, compress=3)
            model_bytes = destination.stat().st_size
            details = _model_details(model, algorithm)
            record = {
                "model": algorithm,
                "feature_set": feature_set,
                "input_columns": FEATURE_CONFIGS[feature_set],
                "encoded_feature_names": encoded_names,
                "encoded_feature_count": len(encoded_names),
                "category_mapping": category_mapping,
                "training_rows": int(len(train_positions)),
                "training_seconds": training_seconds,
                "model_path": str(destination.relative_to(ROOT))
                if destination.is_relative_to(ROOT)
                else str(destination),
                "model_size_bytes": model_bytes,
                "classifier_settings": model.named_steps["classifier"].get_params(deep=False),
                **details,
            }
            model_metadata.append(record)
            fitted.append(
                {
                    "model": model,
                    "x": x,
                    "algorithm": algorithm,
                    "feature_set": feature_set,
                    "training_seconds": training_seconds,
                    "model_size_bytes": model_bytes,
                    "encoded_feature_count": len(encoded_names),
                    "details": details,
                }
            )
            _evaluate_one(
                model=model,
                x=x,
                y=y,
                positions=validation_positions,
                split="validation",
                algorithm=algorithm,
                feature_set=feature_set,
                labels=labels,
                output_path=output_path,
                training_seconds=training_seconds,
                model_size_bytes=model_bytes,
                encoded_feature_count=len(encoded_names),
                details=details,
                metric_rows=metric_rows,
                class_rows=class_rows,
                prediction_frames=prediction_frames,
            )

    frozen_at = datetime.now(timezone.utc).isoformat()
    frozen_settings_path = output_path / "frozen_settings.json"
    frozen_settings_path.write_text(
        json.dumps(
            {
                "frozen_before_test_prediction": True,
                "frozen_at_utc": frozen_at,
                "random_seed": SEED,
                "feature_configurations": FEATURE_CONFIGS,
                "algorithms": list(ALGORITHMS),
                "model_metadata": model_metadata,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    # Phase 2: a single exploratory test prediction per frozen full-run model.
    if not smoke:
        for record in fitted:
            _evaluate_one(
                model=record["model"],
                x=record["x"],
                y=y,
                positions=fixed["test"],
                split="test",
                algorithm=record["algorithm"],
                feature_set=record["feature_set"],
                labels=labels,
                output_path=output_path,
                training_seconds=record["training_seconds"],
                model_size_bytes=record["model_size_bytes"],
                encoded_feature_count=record["encoded_feature_count"],
                details=record["details"],
                metric_rows=metric_rows,
                class_rows=class_rows,
                prediction_frames=prediction_frames,
            )

    metrics = pd.DataFrame(metric_rows).sort_values(["model", "feature_set", "split"])
    classwise = pd.DataFrame(class_rows).sort_values(
        ["model", "feature_set", "split", "class"]
    )
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["model", "feature_set", "split", "row_index"]
    )
    parity = None if smoke else _assert_baseline_parity(metrics, predictions)
    metrics_path = output_path / "ablation_metrics.csv"
    class_path = output_path / "ablation_classwise_metrics.csv"
    prediction_path = output_path / "ablation_predictions.csv"
    metrics.to_csv(metrics_path, index=False, float_format="%.10f")
    classwise.to_csv(class_path, index=False, float_format="%.10f")
    predictions.to_csv(prediction_path, index=False)

    protected_after = _protected_artifact_hashes()
    if protected_before != protected_after:
        changed = sorted(
            path for path in protected_before if protected_before[path] != protected_after[path]
        )
        raise RuntimeError(f"Protected Stage 1-3 artifacts changed during Stage 4 run: {changed}")

    manifest = {
        "run_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_status": (
            "smoke test; validation subset only"
            if smoke
            else "exploratory/comparative; fixed test was previously examined in Stage 3"
        ),
        "split": split_name,
        "smoke": smoke,
        "test_predictions_per_frozen_configuration": 0 if smoke else 1,
        "data": {
            "raw_rows": int(len(raw)),
            "deduplicated_rows": int(len(clean)),
            "dataset_sha256": augmentation.sha256_file(augmentation.DATA),
            "split_assignment_sha256": augmentation.sha256_file(
                RESULTS / "fixed_split_assignments.csv"
            ),
            "fixed_split_sizes": {name: int(len(values)) for name, values in fixed.items()},
            "actual_training_rows": int(len(train_positions)),
            "actual_validation_rows": int(len(validation_positions)),
            "actual_test_rows": 0 if smoke else int(len(fixed["test"])),
        },
        "random_seed": SEED,
        "feature_configurations": FEATURE_CONFIGS,
        "algorithms": list(ALGORITHMS),
        "preprocessing_fit_scope": "training rows only",
        "test_policy": "all settings frozen before any Stage 4 test prediction",
        "frozen_at_utc": frozen_at,
        "baseline_parity": parity,
        "model_metadata": model_metadata,
        "package_versions": augmentation.package_versions(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "command": " ".join(sys.argv),
        "protected_artifact_sha256_before_and_after": protected_after,
        "outputs": {
            "metrics": str(metrics_path),
            "classwise_metrics": str(class_path),
            "predictions": str(prediction_path),
            "frozen_settings": str(frozen_settings_path),
        },
    }
    final_manifest.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Stage 4 ablation complete: {metrics_path}")
    if not smoke:
        print("F5 parity:", json.dumps(parity, indent=2))
    return {
        "metrics": metrics_path,
        "classwise": class_path,
        "predictions": prediction_path,
        "manifest": final_manifest,
    }


def _evaluate_one(
    *,
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    positions: np.ndarray,
    split: str,
    algorithm: str,
    feature_set: str,
    labels: list[str],
    output_path: Path,
    training_seconds: float,
    model_size_bytes: int,
    encoded_feature_count: int,
    details: dict[str, int],
    metric_rows: list[dict[str, Any]],
    class_rows: list[dict[str, Any]],
    prediction_frames: list[pd.DataFrame],
) -> None:
    """Predict one partition and append all traceable result records."""
    started = time.perf_counter()
    predicted = np.asarray(model.predict(x.iloc[positions]), dtype=str)
    prediction_seconds = time.perf_counter() - started
    actual = y.iloc[positions].to_numpy(dtype=str)
    summary, classwise, matrix = calculate_metrics(actual, predicted, labels)
    common = {
        "model": algorithm,
        "feature_set": feature_set,
        "split": split,
    }
    metric_rows.append(
        {
            **common,
            **summary,
            "training_seconds": training_seconds,
            "prediction_seconds_single_run": prediction_seconds,
            "ms_per_1000_single_run": 1_000_000 * prediction_seconds / len(positions),
            "input_feature_count": len(FEATURE_CONFIGS[feature_set]),
            "encoded_feature_count": encoded_feature_count,
            "model_size_bytes": model_size_bytes,
            **details,
        }
    )
    class_rows.extend({**common, **row} for row in classwise)
    prediction_frames.append(
        pd.DataFrame(
            {
                **common,
                "row_index": positions,
                "actual": actual,
                "predicted": predicted,
            }
        )
    )
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_path / f"confusion_{algorithm}_{feature_set}_{split}.csv",
        index_label="actual\\predicted",
    )
    print(
        f"  {split}: accuracy={summary['accuracy']:.4f}, "
        f"macro-F1={summary['macro_f1']:.4f}, MITM recall={summary['mitm_recall']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="fixed", choices=["fixed"])
    parser.add_argument("--output-dir", default="results/stage4")
    parser.add_argument("--model-dir", default="models/stage4")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-rows", type=int, default=5000)
    parser.add_argument("--smoke-validation-rows", type=int, default=1500)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(
        split_name=args.split,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        smoke=args.smoke,
        smoke_train_rows=args.smoke_train_rows,
        smoke_validation_rows=args.smoke_validation_rows,
        n_jobs=args.n_jobs,
        force=args.force,
    )
