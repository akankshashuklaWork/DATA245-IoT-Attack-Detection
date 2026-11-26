"""STAGE 4D: generate traceable final figures from saved result tables.

Run after ``08_comparative_evaluation.py``:

    python src/09_final_visualizations.py

Every PNG has a companion ``*_data.csv`` and is listed with its source result
files in ``figures/stage4/chart_provenance.json``.
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from common import ROOT

augmentation = importlib.import_module("06_feature_augmentation")
FEATURE_ORDER = list(augmentation.FEATURE_CONFIGS)
MODEL_ORDER = list(augmentation.ALGORITHMS)
FOCUS_CLASSES = ["mitm", "ransomware", "xss"]
COLORS = {
    "F5": "#4C78A8",
    "F5_PROTO": "#F58518",
    "F5_STATE": "#54A24B",
    "F7": "#E45756",
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the preceding Stage 4 scripts")
    return pd.read_csv(path)


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _recall_figure(classwise: pd.DataFrame, figure_dir: Path) -> dict[str, Any]:
    data = classwise[
        classwise.split.eq("test") & classwise["class"].isin(FOCUS_CLASSES)
    ][["model", "feature_set", "class", "recall", "support"]].copy()
    data["recall_percent"] = 100 * data["recall"]
    data_path = figure_dir / "01_difficult_class_recall_data.csv"
    data.to_csv(data_path, index=False, float_format="%.6f")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)
    width = 0.19
    positions = np.arange(len(FOCUS_CLASSES))
    for ax, model in zip(axes, MODEL_ORDER):
        subset = data[data.model.eq(model)]
        for index, feature_set in enumerate(FEATURE_ORDER):
            ordered = subset[subset.feature_set.eq(feature_set)].set_index("class").loc[FOCUS_CLASSES]
            ax.bar(
                positions + (index - 1.5) * width,
                ordered.recall_percent,
                width,
                label=feature_set,
                color=COLORS[feature_set],
            )
        ax.set_title(model.replace("_", " "))
        ax.set_xticks(positions, [name.upper() for name in FOCUS_CLASSES])
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlabel("Actual attack class")
    axes[0].set_ylabel("Recall (%)")
    axes[1].legend(title="Feature set", frameon=False, loc="lower right")
    fig.suptitle("Difficult-class recall on the same exploratory fixed test rows", fontsize=14)
    image_path = figure_dir / "01_difficult_class_recall.png"
    _finish(fig, image_path)
    return {"image": image_path, "data": data_path}


def _macro_false_alarm_figure(comparison: pd.DataFrame, figure_dir: Path) -> dict[str, Any]:
    data = comparison[
        ["model", "feature_set", "macro_f1", "normal_to_attack_fpr", "normal_to_attack_false_alarms", "normal_rows"]
    ].copy()
    data["macro_f1_percent"] = 100 * data.macro_f1
    data["normal_to_attack_fpr_percent"] = 100 * data.normal_to_attack_fpr
    data_path = figure_dir / "02_macro_f1_false_alarm_data.csv"
    data.to_csv(data_path, index=False, float_format="%.6f")

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    markers = {"Decision_Tree": "o", "kNN": "s"}
    for _, row in data.iterrows():
        ax.scatter(
            row.normal_to_attack_fpr_percent,
            row.macro_f1_percent,
            color=COLORS[row.feature_set],
            marker=markers[row.model],
            s=95,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.annotate(
            f"{row.model.replace('Decision_Tree', 'DT')} / {row.feature_set}",
            (row.normal_to_attack_fpr_percent, row.macro_f1_percent),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Normal-to-attack false-positive rate (%) — lower is better")
    ax.set_ylabel("Macro-F1 (%) — higher is better")
    ax.set_title("Macro-F1 versus security-alarm false positives")
    ax.grid(alpha=0.2)
    image_path = figure_dir / "02_macro_f1_false_alarm.png"
    _finish(fig, image_path)
    return {"image": image_path, "data": data_path}


def _latency_recall_figure(
    comparison: pd.DataFrame, classwise: pd.DataFrame, figure_dir: Path
) -> dict[str, Any]:
    focus = classwise[
        classwise.split.eq("test") & classwise["class"].isin(FOCUS_CLASSES)
    ][["model", "feature_set", "class", "recall", "support"]]
    data = focus.merge(
        comparison[["model", "feature_set", "median_ms_per_1000_rows", "model_size_bytes"]],
        on=["model", "feature_set"],
        validate="many_to_one",
    )
    data["recall_percent"] = 100 * data.recall
    data_path = figure_dir / "03_latency_recall_tradeoff_data.csv"
    data.to_csv(data_path, index=False, float_format="%.6f")

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharex=False, sharey=True)
    markers = {"Decision_Tree": "o", "kNN": "s"}
    for ax, attack_class in zip(axes, FOCUS_CLASSES):
        subset = data[data["class"].eq(attack_class)]
        for _, row in subset.iterrows():
            ax.scatter(
                row.median_ms_per_1000_rows,
                row.recall_percent,
                color=COLORS[row.feature_set],
                marker=markers[row.model],
                s=90,
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_title(attack_class.upper())
        ax.set_xlabel("Median ms / 1,000 rows")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Recall (%)")
    feature_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[name],
               markeredgecolor="black", label=name, markersize=8)
        for name in FEATURE_ORDER
    ]
    model_handles = [
        Line2D([0], [0], marker=markers[name], color="black", linestyle="none",
               label=name.replace("_", " "), markersize=8)
        for name in MODEL_ORDER
    ]
    fig.legend(
        handles=[*feature_handles, *model_handles],
        ncol=6,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Inference latency versus difficult-class recall", fontsize=14)
    image_path = figure_dir / "03_latency_recall_tradeoff.png"
    _finish(fig, image_path)
    return {"image": image_path, "data": data_path}


def _error_breakdown_figure(predictions: pd.DataFrame, figure_dir: Path) -> dict[str, Any]:
    data = predictions[
        predictions.split.eq("test") & predictions.actual.isin(FOCUS_CLASSES)
    ].copy()
    data["outcome"] = np.where(
        data.predicted.eq(data.actual),
        "correct",
        np.where(data.predicted.eq("normal"), "predicted_normal", "other_attack_class"),
    )
    counts = (
        data.groupby(["model", "feature_set", "actual", "outcome"])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby(["model", "feature_set", "actual"])["count"].transform("sum")
    counts["percent_of_actual_class"] = 100 * counts["count"] / totals
    data_path = figure_dir / "04_difficult_class_error_breakdown_data.csv"
    counts.to_csv(data_path, index=False, float_format="%.6f")

    outcomes = ["correct", "other_attack_class", "predicted_normal"]
    colors = {"correct": "#59A14F", "other_attack_class": "#F28E2B", "predicted_normal": "#E15759"}
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.2), sharex=True)
    labels = [f"{feature_set}\n{attack.upper()}" for feature_set in FEATURE_ORDER for attack in FOCUS_CLASSES]
    x_positions = np.arange(len(labels))
    for ax, model in zip(axes, MODEL_ORDER):
        subset = counts[counts.model.eq(model)]
        bottom = np.zeros(len(labels))
        for outcome in outcomes:
            values = []
            for feature_set in FEATURE_ORDER:
                for attack in FOCUS_CLASSES:
                    match = subset[
                        subset.feature_set.eq(feature_set)
                        & subset.actual.eq(attack)
                        & subset.outcome.eq(outcome)
                    ]
                    values.append(float(match.percent_of_actual_class.iloc[0]) if len(match) else 0.0)
            ax.bar(x_positions, values, bottom=bottom, color=colors[outcome], label=outcome.replace("_", " "))
            bottom += np.asarray(values)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Share of actual class (%)")
        ax.set_title(model.replace("_", " "))
        ax.grid(axis="y", alpha=0.2)
    axes[1].set_xticks(x_positions, labels, fontsize=8)
    axes[1].legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.suptitle("Correct detections and difficult-class error destinations", fontsize=14)
    image_path = figure_dir / "04_difficult_class_error_breakdown.png"
    _finish(fig, image_path)
    return {"image": image_path, "data": data_path}


def _optional_stability_figure(result_dir: Path, figure_dir: Path) -> dict[str, Any] | None:
    source = result_dir / "stability_paired_deltas.csv"
    if not source.exists():
        return None
    data = pd.read_csv(source)
    focus = data[data["class"].isin(FOCUS_CLASSES)].copy()
    data_path = figure_dir / "05_split_stability_data.csv"
    focus.to_csv(data_path, index=False, float_format="%.6f")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), sharey=True)
    for ax, attack_class in zip(axes, FOCUS_CLASSES):
        subset = focus[focus["class"].eq(attack_class)]
        for (model, feature_set), frame in subset.groupby(["model", "feature_set"]):
            ax.plot(
                frame.seed.astype(str),
                100 * frame.recall_delta_vs_F5,
                marker="o",
                label=f"{model.replace('Decision_Tree', 'DT')} / {feature_set}",
                color=COLORS[feature_set],
                linestyle="-" if model == "Decision_Tree" else "--",
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(attack_class.upper())
        ax.set_xlabel("Split seed")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Recall delta versus matched F5 (percentage points)")
    axes[-1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    image_path = figure_dir / "05_split_stability.png"
    _finish(fig, image_path)
    return {"image": image_path, "data": data_path}


def run(
    result_dir: str | Path = "results/stage4",
    figure_dir: str | Path = "figures/stage4",
) -> dict[str, dict[str, str]]:
    """Generate all required Stage 4 figures and provenance records."""
    results = augmentation.resolve_output_dir(result_dir)
    figures = augmentation.resolve_output_dir(figure_dir)
    figures.mkdir(parents=True, exist_ok=True)
    comparison_path = results / "fixed_test_comparison.csv"
    classwise_path = results / "ablation_classwise_metrics.csv"
    predictions_path = results / "ablation_predictions.csv"
    comparison = _read(comparison_path)
    classwise = _read(classwise_path)
    predictions = _read(predictions_path)

    generated: dict[str, Any] = {
        "difficult_class_recall": _recall_figure(classwise, figures),
        "macro_f1_false_alarm": _macro_false_alarm_figure(comparison, figures),
        "latency_recall_tradeoff": _latency_recall_figure(comparison, classwise, figures),
        "difficult_class_error_breakdown": _error_breakdown_figure(predictions, figures),
    }
    optional = _optional_stability_figure(results, figures)
    if optional is not None:
        generated["split_stability"] = optional

    source_paths = [comparison_path, classwise_path, predictions_path]
    stability_source = results / "stability_paired_deltas.csv"
    if optional is not None:
        source_paths.append(stability_source)
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
        "note": "All values are plotted from saved Stage 4 tables; no slide values are hardcoded.",
    }
    provenance_path = figures / "chart_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(generated)} Stage 4 figures in {figures}")
    return provenance["charts"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="results/stage4")
    parser.add_argument("--figure-dir", default="figures/stage4")
    args = parser.parse_args()
    run(args.result_dir, args.figure_dir)
