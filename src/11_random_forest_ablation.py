"""STAGE 5A: prespecified Random Forest feature ablation.

Full run:

    python src/11_random_forest_ablation.py

Validation-only smoke run (does not predict the fixed test partition):

    python src/11_random_forest_ablation.py --smoke \
        --output-dir /tmp/stage5-rf-smoke --model-dir /tmp/stage5-rf-models

Random Forest is added as a third classical classifier.  The completed Stage
1-4 files are read but never overwritten.  All four feature configurations use
the same deduplicated rows and fixed assignments as the earlier experiments.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from common import FEATURES, MODELS, RESULTS, ROOT, SEED, TARGET, clean_data

augmentation = importlib.import_module("06_feature_augmentation")
stage4_ablation = importlib.import_module("07_feature_ablation")
FEATURE_CONFIGS = augmentation.FEATURE_CONFIGS
MODEL_NAME = "Random_Forest"
DEFAULT_ESTIMATORS = 300


def build_random_forest_pipeline(
    feature_set: str,
    *,
    n_estimators: int = DEFAULT_ESTIMATORS,
    n_jobs: int = 1,
    seed: int = SEED,
) -> Pipeline:
    """Build the frozen untuned Random Forest pipeline for one feature set."""
    augmentation.validate_feature_config(feature_set)
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        class_weight=None,
        random_state=seed,
        n_jobs=n_jobs,
    )
    if feature_set == "F5":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", classifier),
            ]
        )
    return Pipeline(
        [
            (
                "preprocessor",
                augmentation.build_preprocessor(feature_set, "Decision_Tree"),
            ),
            ("classifier", classifier),
        ]
    )


def forest_complexity(model: Pipeline) -> dict[str, Any]:
    """Summarize fitted tree count, node count, depth, and leaf count."""
    forest = model.named_steps["classifier"]
    depths = np.asarray([tree.get_depth() for tree in forest.estimators_], dtype=float)
    leaves = np.asarray([tree.get_n_leaves() for tree in forest.estimators_], dtype=float)
    nodes = np.asarray([tree.tree_.node_count for tree in forest.estimators_], dtype=float)
    return {
        "n_estimators": int(len(forest.estimators_)),
        "forest_total_nodes": int(nodes.sum()),
        "forest_mean_depth": float(depths.mean()),
        "forest_max_depth": int(depths.max()),
        "forest_mean_leaves": float(leaves.mean()),
    }


def importance_tables(
    model: Pipeline,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return encoded and raw-field grouped impurity importance tables."""
    names, _ = augmentation.encoded_feature_metadata(model, feature_set)
    values = model.named_steps["classifier"].feature_importances_
    if len(names) != len(values):
        raise AssertionError("Encoded feature names do not align with forest importances")
    encoded = pd.DataFrame(
        {
            "feature_set": feature_set,
            "encoded_feature": names,
            "importance": values,
        }
    ).sort_values("importance", ascending=False)

    def raw_name(encoded_name: str) -> str:
        if encoded_name in FEATURES:
            return encoded_name
        for categorical in augmentation.CATEGORICAL_FEATURES:
            if encoded_name.startswith(f"{categorical}_"):
                return categorical
        raise ValueError(f"Cannot map encoded feature name to raw input: {encoded_name}")

    grouped = encoded.assign(raw_feature=encoded.encoded_feature.map(raw_name))
    grouped = (
        grouped.groupby(["feature_set", "raw_feature"], as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    return encoded, grouped


def protected_artifact_hashes() -> dict[str, str]:
    """Fingerprint Stage 1-4 evidence that Stage 5 must not modify."""
    paths = [
        augmentation.DATA,
        RESULTS / "fixed_split_assignments.csv",
        RESULTS / "baseline_metrics.csv",
        RESULTS / "stage4" / "ablation_metrics.csv",
        RESULTS / "stage4" / "ablation_predictions.csv",
        MODELS / "Decision_Tree_five_features.joblib",
        MODELS / "kNN_five_features.joblib",
        ROOT / "figures" / "stage4" / "chart_provenance.json",
    ]
    paths.extend(sorted((MODELS / "stage4").glob("*.joblib")))
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing protected Stage 1-4 artifacts: {missing}")
    return {str(path.relative_to(ROOT)): augmentation.sha256_file(path) for path in paths}


def run(
    *,
    output_dir: str | Path = "results/stage5_random_forest",
    model_dir: str | Path = "models/stage5_random_forest",
    n_estimators: int = DEFAULT_ESTIMATORS,
    n_jobs: int = 1,
    smoke: bool = False,
    smoke_train_rows: int = 5000,
    smoke_validation_rows: int = 1500,
    force: bool = False,
) -> dict[str, Path]:
    """Fit four forests, validate all, freeze settings, then test once each."""
    output_path = augmentation.resolve_output_dir(output_dir)
    model_path = augmentation.resolve_output_dir(model_dir)
    manifest_path = output_path / "random_forest_manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite completed run {manifest_path}. Use --force explicitly."
        )
    output_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    protected_before = protected_artifact_hashes()

    raw, clean = clean_data()
    fixed = augmentation.validate_stage4_data(clean)
    y = clean[TARGET].astype(str)
    labels = sorted(y.unique())
    train_positions = fixed["train"]
    validation_positions = fixed["validation"]
    if smoke:
        train_positions = stage4_ablation._sample_positions(
            train_positions, y, smoke_train_rows, SEED
        )
        validation_positions = stage4_ablation._sample_positions(
            validation_positions, y, smoke_validation_rows, SEED + 1
        )

    metric_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    encoded_importances: list[pd.DataFrame] = []
    grouped_importances: list[pd.DataFrame] = []
    fitted: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []

    # Train and validate all four configurations before any test prediction.
    for feature_set in FEATURE_CONFIGS:
        x = augmentation.feature_frame(clean, feature_set)
        model = build_random_forest_pipeline(
            feature_set,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            seed=SEED,
        )
        print(
            f"Fitting {MODEL_NAME}/{feature_set}: {n_estimators} trees on "
            f"{len(train_positions):,} rows...",
            flush=True,
        )
        started = time.perf_counter()
        model.fit(x.iloc[train_positions], y.iloc[train_positions])
        training_seconds = time.perf_counter() - started
        names, categories = augmentation.encoded_feature_metadata(model, feature_set)
        complexity = forest_complexity(model)
        model_destination = model_path / f"{MODEL_NAME}_{feature_set}.joblib"
        joblib.dump(model, model_destination, compress=3)
        model_size_bytes = model_destination.stat().st_size
        encoded, grouped = importance_tables(model, feature_set)
        encoded_importances.append(encoded)
        grouped_importances.append(grouped)
        record = {
            "model": MODEL_NAME,
            "feature_set": feature_set,
            "input_columns": FEATURE_CONFIGS[feature_set],
            "encoded_feature_names": names,
            "encoded_feature_count": len(names),
            "category_mapping": categories,
            "training_rows": int(len(train_positions)),
            "training_seconds": training_seconds,
            "model_path": str(model_destination.relative_to(ROOT))
            if model_destination.is_relative_to(ROOT)
            else str(model_destination),
            "model_size_bytes": model_size_bytes,
            "classifier_settings": model.named_steps["classifier"].get_params(deep=False),
            **complexity,
        }
        metadata.append(record)
        fitted.append(
            {
                "model": model,
                "x": x,
                "feature_set": feature_set,
                "training_seconds": training_seconds,
                "model_size_bytes": model_size_bytes,
                "encoded_feature_count": len(names),
                "details": complexity,
            }
        )
        stage4_ablation._evaluate_one(
            model=model,
            x=x,
            y=y,
            positions=validation_positions,
            split="validation",
            algorithm=MODEL_NAME,
            feature_set=feature_set,
            labels=labels,
            output_path=output_path,
            training_seconds=training_seconds,
            model_size_bytes=model_size_bytes,
            encoded_feature_count=len(names),
            details=complexity,
            metric_rows=metric_rows,
            class_rows=class_rows,
            prediction_frames=prediction_frames,
        )

    frozen_at = datetime.now(timezone.utc).isoformat()
    frozen_path = output_path / "random_forest_frozen_settings.json"
    frozen_path.write_text(
        json.dumps(
            {
                "frozen_before_test_prediction": True,
                "frozen_at_utc": frozen_at,
                "random_seed": SEED,
                "feature_configurations": FEATURE_CONFIGS,
                "random_forest_settings": metadata[0]["classifier_settings"],
                "model_metadata": metadata,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    if not smoke:
        for record in fitted:
            stage4_ablation._evaluate_one(
                model=record["model"],
                x=record["x"],
                y=y,
                positions=fixed["test"],
                split="test",
                algorithm=MODEL_NAME,
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

    metrics = pd.DataFrame(metric_rows).sort_values(["feature_set", "split"])
    classwise = pd.DataFrame(class_rows).sort_values(["feature_set", "split", "class"])
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["feature_set", "split", "row_index"]
    )
    encoded = pd.concat(encoded_importances, ignore_index=True)
    grouped = pd.concat(grouped_importances, ignore_index=True)
    outputs = {
        "metrics": output_path / "random_forest_metrics.csv",
        "classwise": output_path / "random_forest_classwise_metrics.csv",
        "predictions": output_path / "random_forest_predictions.csv",
        "encoded_importances": output_path / "random_forest_encoded_importances.csv",
        "grouped_importances": output_path / "random_forest_grouped_importances.csv",
        "frozen_settings": frozen_path,
    }
    metrics.to_csv(outputs["metrics"], index=False, float_format="%.10f")
    classwise.to_csv(outputs["classwise"], index=False, float_format="%.10f")
    predictions.to_csv(outputs["predictions"], index=False)
    encoded.to_csv(outputs["encoded_importances"], index=False, float_format="%.10f")
    grouped.to_csv(outputs["grouped_importances"], index=False, float_format="%.10f")

    protected_after = protected_artifact_hashes()
    if protected_before != protected_after:
        changed = [
            path for path in protected_before if protected_before[path] != protected_after[path]
        ]
        raise RuntimeError(f"Protected Stage 1-4 artifacts changed: {changed}")
    manifest = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_status": (
            "smoke test; validation subset only"
            if smoke
            else "exploratory/comparative; fixed test was previously examined"
        ),
        "purpose": "Add professor-suggested Random Forest without changing Stage 1-4",
        "selection_policy": "prespecified untuned forest; no test-led tuning",
        "test_predictions_per_frozen_configuration": 0 if smoke else 1,
        "data": {
            "raw_rows": int(len(raw)),
            "deduplicated_rows": int(len(clean)),
            "fixed_split_sizes": {name: int(len(values)) for name, values in fixed.items()},
            "dataset_sha256": augmentation.sha256_file(augmentation.DATA),
            "split_assignment_sha256": augmentation.sha256_file(
                RESULTS / "fixed_split_assignments.csv"
            ),
        },
        "random_seed": SEED,
        "feature_configurations": FEATURE_CONFIGS,
        "random_forest_settings": metadata[0]["classifier_settings"],
        "model_metadata": metadata,
        "frozen_at_utc": frozen_at,
        "package_versions": augmentation.package_versions(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "command": " ".join(sys.argv),
        "protected_stage1_to_stage4_sha256_before_and_after": protected_after,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    outputs["manifest"] = manifest_path
    print(f"Random Forest ablation complete: {outputs['metrics']}")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/stage5_random_forest")
    parser.add_argument("--model-dir", default="models/stage5_random_forest")
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_ESTIMATORS)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-rows", type=int, default=5000)
    parser.add_argument("--smoke-validation-rows", type=int, default=1500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        smoke=args.smoke,
        smoke_train_rows=args.smoke_train_rows,
        smoke_validation_rows=args.smoke_validation_rows,
        force=args.force,
    )
