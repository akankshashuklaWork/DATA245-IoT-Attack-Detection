"""STAGE 5C: traceable Random Forest and three-model comparison figures.

Run after ``12_random_forest_evaluation.py``:

    python src/13_random_forest_visualizations.py

Every PNG is accompanied by its plotted-data CSV and source checksums.
"""
from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT

augmentation = importlib.import_module("06_feature_augmentation")
FEATURE_ORDER = list(augmentation.FEATURE_CONFIGS)
MODEL_ORDER = ["Decision_Tree", "kNN", "Random_Forest"]
FOCUS_CLASSES = ["mitm", "ransomware", "xss"]
COLORS = {
    "F5": "#4C78A8",
    "F5_PROTO": "#F58518",
    "F5_STATE": "#54A24B",
    "F7": "#E45756",
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run Stage 5 evaluation first")
    return pd.read_csv(path)


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _recall_figure(focus: pd.DataFrame, figure_dir: Path) -> dict[str, Path]:
    data = focus[["model", "feature_set", "class", "recall", "support"]].copy()
    data["recall_percent"] = 100 * data.recall
    data_path = figure_dir / "01_three_model_difficult_recall_data.csv"
    data.to_csv(data_path, index=False, float_format="%.6f")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    width = 0.19
    positions = np.arange(len(FOCUS_CLASSES))
    for ax, model_name in zip(axes, MODEL_ORDER):
        subset = data[data.model.eq(model_name)]
        for index, feature_set in enumerate(FEATURE_ORDER):
            ordered = subset[subset.feature_set.eq(feature_set)].set_index("class").loc[FOCUS_CLASSES]
            ax.bar(
                positions + (index - 1.5) * width,
                ordered.recall_percent,
                width,
                color=COLORS[feature_set],
                label=feature_set,
            )
        ax.set_title(model_name.replace("_", " "))
        ax.set_xticks(positions, [name.upper() for name in FOCUS_CLASSES])
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Recall (%)")
    axes[-1].legend(title="Feature set", frameon=False, loc="lower right")
    fig.suptitle("Difficult-class recall: Decision Tree, kNN, and Random Forest", fontsize=14)
    image_path = figure_dir / "01_three_model_difficult_recall.png"
    _finish(fig, image_path)
    return {"image": image_path, "data": data_path}


def _macro_false_alarm_figure(
    comparison: pd.DataFrame, figure_dir: Path
) -> dict[str, Path]:
    data = comparison[
        [
            "model",
            "feature_set",
            "macro_f1",
            "normal_to_attack_fpr",
            "normal_to_attack_false_alarms",
            "normal_rows",
        ]
    ].copy()
    data["macro_f1_percent"] = 100 * data.macro_f1
    data["normal_to_attack_fpr_percent"] = 100 * data.normal_to_attack_fpr
    data_path = figure_dir / "02_three_model_macro_false_alarm_data.csv"
    data.to_csv(data_path, index=False, float_format="%.6f")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1), sharey=True)
    for ax, model_name in zip(axes, MODEL_ORDER):
        subset = data[data.model.eq(model_name)]
        for _, row in subset.iterrows():
            ax.scatter(
                row.normal_to_attack_fpr_percent,
                row.macro_f1_percent,
                color=COLORS[row.feature_set],
                s=100,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.annotate(
                row.feature_set,
                (row.normal_to_attack_fpr_percent, row.macro_f1_percent),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_title(model_name.replace("_", " "))
        ax.set_xlabel("Normal-to-attack FPR (%)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Macro-F1 (%)")
    fig.suptitle("Macro-F1 versus security-alarm false positives", fontsize=14)
    image_path = figure_dir / "02_three_model_macro_false_alarm.png"
    _finish(fig, image_path)
    return {"image": image_path, "data": data_path}


def _latency_figure(comparison: pd.DataFrame, figure_dir: Path) -> dict[str, Path]:
    data = comparison[
        [
            "model",
            "feature_set",
            "macro_f1",
            "median_ms_per_1000_rows",
            "model_size_bytes",
        ]
    ].copy()
    data["macro_f1_percent"] = 100 * data.macro_f1
    data["model_size_megabytes"] = data.model_size_bytes / (1024 * 1024)
    data_path = figure_dir / "03_three_model_efficiency_data.csv"
    data.to_csv(data_path, index=False, float_format="%.6f")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1), sharey=True)
    for ax, model_name in zip(axes, MODEL_ORDER):
        subset = data[data.model.eq(model_name)]
        for _, row in subset.iterrows():
            ax.scatter(
                row.median_ms_per_1000_rows,
                row.macro_f1_percent,
                color=COLORS[row.feature_set],
                s=90,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.annotate(
                row.feature_set,
                (row.median_ms_per_1000_rows, row.macro_f1_percent),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_title(model_name.replace("_", " "))
        ax.set_xlabel("Median ms / 1,000 rows")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Macro-F1 (%)")
    fig.suptitle("Predictive performance versus inference latency", fontsize=14)
    image_path = figure_dir / "03_three_model_efficiency.png"
    _finish(fig, image_path)
    return {"image": image_path, "data": data_path}


def _importance_figure(importances: pd.DataFrame, figure_dir: Path) -> dict[str, Path]:
    data = importances.copy()
    data["importance_percent"] = 100 * data.importance
    data_path = figure_dir / "04_random_forest_grouped_importance_data.csv"
    data.to_csv(data_path, index=False, float_format="%.6f")
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    for ax, feature_set in zip(axes.flat, FEATURE_ORDER):
        subset = data[data.feature_set.eq(feature_set)].sort_values("importance_percent")
        ax.barh(subset.raw_feature, subset.importance_percent, color=COLORS[feature_set])
        ax.set_title(feature_set)
        ax.set_xlabel("Grouped impurity importance (%)")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Random Forest grouped feature importance\n(descriptive, not causal)", fontsize=14
    )
    image_path = figure_dir / "04_random_forest_grouped_importance.png"
    _finish(fig, image_path)
    return {"image": image_path, "data": data_path}


def _optional_stability_figure(result_dir: Path, figure_dir: Path) -> dict[str, Path] | None:
    source = result_dir / "random_forest_stability_paired_deltas.csv"
    if not source.exists():
        return None
    data = pd.read_csv(source)
    focus = data[data["class"].isin(FOCUS_CLASSES)].copy()
    data_path = figure_dir / "05_random_forest_split_stability_data.csv"
    focus.to_csv(data_path, index=False, float_format="%.6f")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, attack_class in zip(axes, FOCUS_CLASSES):
        subset = focus[focus["class"].eq(attack_class)]
        for feature_set, frame in subset.groupby("feature_set"):
            if feature_set == "F5":
                continue
            ax.plot(
                frame.seed.astype(str),
                100 * frame.recall_delta_vs_F5,
                marker="o",
                color=COLORS[feature_set],
                label=feature_set,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(attack_class.upper())
        ax.set_xlabel("Split seed")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Random Forest recall delta versus matched F5 (pp)")
    axes[-1].legend(frameon=False, loc="best")
    image_path = figure_dir / "05_random_forest_split_stability.png"
    _finish(fig, image_path)
    return {"image": image_path, "data": data_path}


def run(
    *,
    result_dir: str | Path = "results/stage5_random_forest",
    figure_dir: str | Path = "figures/stage5_random_forest",
) -> dict[str, Any]:
    """Generate Stage 5 figures and a checksum-based provenance manifest."""
    results = augmentation.resolve_output_dir(result_dir)
    figures = augmentation.resolve_output_dir(figure_dir)
    figures.mkdir(parents=True, exist_ok=True)
    comparison_path = results / "combined_three_model_test_comparison.csv"
    focus_path = results / "combined_difficult_class_metrics.csv"
    importance_path = results / "random_forest_grouped_importances.csv"
    comparison = _read(comparison_path)
    focus = _read(focus_path)
    importances = _read(importance_path)
    generated: dict[str, Any] = {
        "three_model_difficult_recall": _recall_figure(focus, figures),
        "three_model_macro_false_alarm": _macro_false_alarm_figure(comparison, figures),
        "three_model_efficiency": _latency_figure(comparison, figures),
        "random_forest_grouped_importance": _importance_figure(importances, figures),
    }
    optional = _optional_stability_figure(results, figures)
    source_paths = [comparison_path, focus_path, importance_path]
    if optional is not None:
        generated["random_forest_split_stability"] = optional
        source_paths.append(results / "random_forest_stability_paired_deltas.csv")
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_status": "exploratory/comparative fixed-test figures",
        "source_files": {
            str(path.relative_to(ROOT)): augmentation.sha256_file(path)
            for path in source_paths
        },
        "charts": {
            name: {
                "image": str(files["image"].relative_to(ROOT)),
                "companion_data": str(files["data"].relative_to(ROOT)),
            }
            for name, files in generated.items()
        },
        "importance_caution": (
            "Random Forest impurity importance is descriptive, can favor variables with "
            "more split opportunities, and must not be interpreted causally."
        ),
    }
    provenance_path = figures / "chart_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(generated)} Random Forest comparison figures in {figures}")
    return provenance["charts"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="results/stage5_random_forest")
    parser.add_argument("--figure-dir", default="figures/stage5_random_forest")
    args = parser.parse_args()
    run(result_dir=args.result_dir, figure_dir=args.figure_dir)
