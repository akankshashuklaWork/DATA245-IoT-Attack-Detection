"""STAGE 4E (optional): matched repeated group-disjoint split sensitivity.

One-seed smoke run before the optional ten-seed extension:

    python src/10_stage4_split_stability.py --seeds 42 \
        --output-dir tests/stage4_one_seed

Full optional run:

    python src/10_stage4_split_stability.py \
        --seeds 42,7,17,23,29,41,73,101,202,313

Observations overlap between seeds.  These are matched split-sensitivity runs,
not independent datasets, new environments, or repeated blind test sets.
"""
from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import ROOT, SEEDS, TARGET, clean_data, feature_group_hashes

augmentation = importlib.import_module("06_feature_augmentation")
ablation = importlib.import_module("07_feature_ablation")
ALGORITHMS = augmentation.ALGORITHMS
FEATURE_CONFIGS = augmentation.FEATURE_CONFIGS


def matched_group_split(
    data: pd.DataFrame,
    seed: int,
) -> dict[str, np.ndarray]:
    """Reproduce the Stage 3 70/15/15 majority-label group split."""
    groups = feature_group_hashes(data)
    labels = data[TARGET].astype(str)
    majority = (
        pd.DataFrame({"group": groups, "label": labels.to_numpy()})
        .groupby(["group", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["group", "count", "label"], ascending=[True, False, True])
        .drop_duplicates("group")
    )
    group_ids = majority.group.to_numpy(dtype=np.uint64)
    group_labels = majority.label.to_numpy()
    label_of_group = majority.set_index("group").label
    train_groups, holdout_groups = train_test_split(
        group_ids,
        test_size=0.30,
        random_state=seed,
        stratify=group_labels,
    )
    holdout_labels = label_of_group.loc[holdout_groups].to_numpy()
    validation_groups, test_groups = train_test_split(
        holdout_groups,
        test_size=0.5,
        random_state=seed + 5000,
        stratify=holdout_labels,
    )
    group_sets = {
        "train": train_groups,
        "validation": validation_groups,
        "test": test_groups,
    }
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if not set(group_sets[left]).isdisjoint(set(group_sets[right])):
            raise RuntimeError(f"F5 group leakage across {left}/{right}")
    positions = {
        name: np.flatnonzero(np.isin(groups, selected))
        for name, selected in group_sets.items()
    }
    if sum(len(values) for values in positions.values()) != len(data):
        raise RuntimeError("Repeated split omitted rows")
    return positions


def _paired_deltas(classwise: pd.DataFrame) -> pd.DataFrame:
    baseline = classwise[classwise.feature_set.eq("F5")][
        ["seed", "model", "class", "precision", "recall", "f1", "support"]
    ].rename(
        columns={
            "precision": "precision_F5",
            "recall": "recall_F5",
            "f1": "f1_F5",
            "support": "support_F5",
        }
    )
    paired = classwise.merge(
        baseline,
        on=["seed", "model", "class"],
        validate="many_to_one",
    )
    for metric in ("precision", "recall", "f1"):
        paired[f"{metric}_delta_vs_F5"] = paired[metric] - paired[f"{metric}_F5"]
    if not (paired.support == paired.support_F5).all():
        raise AssertionError("Matched feature configurations do not share class support")
    return paired


def run(
    seeds: list[int],
    output_dir: str | Path = "results/stage4",
    n_jobs: int = 1,
) -> dict[str, Path]:
    """Run all model/feature configurations on identical rows within each seed."""
    if not seeds:
        raise ValueError("Provide at least one seed")
    destination = augmentation.resolve_output_dir(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _, clean = clean_data()
    y = clean[TARGET].astype(str)
    labels = sorted(y.unique())
    feature_frames = {
        name: augmentation.feature_frame(clean, name) for name in FEATURE_CONFIGS
    }
    metric_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for run_number, seed in enumerate(seeds, start=1):
        positions = matched_group_split(clean, seed)
        split_rows.extend(
            {
                "seed": seed,
                "split": name,
                "rows": len(values),
                "mitm_rows": int(y.iloc[values].eq("mitm").sum()),
            }
            for name, values in positions.items()
        )
        for feature_set, x in feature_frames.items():
            for algorithm in ALGORITHMS:
                model = augmentation.build_stage4_pipeline(
                    feature_set, algorithm, seed=42, n_jobs=n_jobs
                )
                model.fit(x.iloc[positions["train"]], y.iloc[positions["train"]])
                predicted = np.asarray(model.predict(x.iloc[positions["test"]]), dtype=str)
                actual = y.iloc[positions["test"]].to_numpy(dtype=str)
                summary, per_class, _ = ablation.calculate_metrics(actual, predicted, labels)
                metric_rows.append(
                    {
                        "seed": seed,
                        "model": algorithm,
                        "feature_set": feature_set,
                        "train_rows": len(positions["train"]),
                        "validation_rows": len(positions["validation"]),
                        "test_rows": len(positions["test"]),
                        **summary,
                    }
                )
                class_rows.extend(
                    {
                        "seed": seed,
                        "model": algorithm,
                        "feature_set": feature_set,
                        **row,
                    }
                    for row in per_class
                )
        print(f"Completed matched split {run_number}/{len(seeds)} (seed {seed})", flush=True)

    metrics = pd.DataFrame(metric_rows)
    classwise = pd.DataFrame(class_rows)
    paired = _paired_deltas(classwise)
    summary = (
        paired.groupby(["model", "feature_set", "class"])
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
        "metrics": destination / "stability_metrics.csv",
        "classwise": destination / "stability_classwise_metrics.csv",
        "paired_deltas": destination / "stability_paired_deltas.csv",
        "summary": destination / "stability_summary.csv",
        "split_sizes": destination / "stability_split_sizes.csv",
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
        "feature_configurations": augmentation.FEATURE_CONFIGS,
        "algorithms": list(ALGORITHMS),
        "matched_design": "identical train/validation/test row indexes for all eight combinations within a seed",
        "split_design": "70/15/15 group-majority-label stratification; groups are exact F5 hashes",
        "interpretation": (
            "split sensitivity on overlapping observations; not independent cross-environment "
            "validation and not repeated blind tests"
        ),
        "dataset_sha256": augmentation.sha256_file(augmentation.DATA),
        "package_versions": augmentation.package_versions(),
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    manifest_path = destination / "stability_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    outputs["manifest"] = manifest_path
    print(f"Stage 4 split-sensitivity outputs saved in {destination}")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--output-dir", default="results/stage4")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    parsed_seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    run(parsed_seeds, args.output_dir, args.n_jobs)
