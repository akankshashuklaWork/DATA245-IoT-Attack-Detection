"""Automated acceptance checks for the Stage 4 feature-ablation study."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import FEATURES, MODELS, RESULTS, TARGET, clean_data  # noqa: E402

augmentation = importlib.import_module("06_feature_augmentation")
ablation = importlib.import_module("07_feature_ablation")
comparison = importlib.import_module("08_comparative_evaluation")
stability = importlib.import_module("10_stage4_split_stability")


@pytest.fixture(scope="session")
def dataset() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    _, clean = clean_data()
    return clean, augmentation.validate_stage4_data(clean)


def test_exact_feature_configurations_exclude_forbidden_columns() -> None:
    assert augmentation.FEATURE_CONFIGS == {
        "F5": FEATURES,
        "F5_PROTO": [*FEATURES, "proto"],
        "F5_STATE": [*FEATURES, "conn_state"],
        "F7": [*FEATURES, "proto", "conn_state"],
    }
    for columns in augmentation.FEATURE_CONFIGS.values():
        assert TARGET not in columns
        assert "label" not in columns
        assert not {"src_ip", "dst_ip", "src_port", "dst_port", "ts"}.intersection(columns)


def test_fixed_split_is_unchanged_aligned_and_group_disjoint(dataset) -> None:
    clean, split = dataset
    assert len(clean) == 190_474
    assert [len(split[name]) for name in ("train", "validation", "test")] == [
        140_812,
        24_831,
        24_831,
    ]
    assignments = pd.read_csv(RESULTS / "fixed_split_assignments.csv")
    assert np.array_equal(assignments.deduplicated_row_index, np.arange(len(clean)))
    assert np.array_equal(assignments["class"].astype(str), clean[TARGET].astype(str))


@pytest.mark.parametrize("feature_set", ["F5_PROTO", "F5_STATE", "F7"])
@pytest.mark.parametrize("algorithm", augmentation.ALGORITHMS)
def test_unseen_and_missing_categories_are_safe(feature_set: str, algorithm: str) -> None:
    train = pd.DataFrame(
        {
            "duration": np.arange(12, dtype=float),
            "src_bytes": np.arange(12),
            "dst_bytes": np.arange(12) * 2,
            "src_ip_bytes": np.arange(12) * 3,
            "dst_ip_bytes": np.arange(12) * 4,
            "proto": ["tcp", "udp"] * 6,
            "conn_state": ["SF", "S0", "REJ"] * 4,
        }
    )
    target = pd.Series(["normal", "mitm"] * 6)
    model = augmentation.build_stage4_pipeline(feature_set, algorithm)
    model.fit(train[augmentation.FEATURE_CONFIGS[feature_set]], target)
    test = train.iloc[:3].copy()
    test.loc[0, "proto"] = "unseen_protocol"
    test.loc[1, "conn_state"] = "UNSEEN_STATE"
    test.loc[2, "proto"] = np.nan
    test.loc[2, "conn_state"] = np.nan
    predicted = model.predict(test[augmentation.FEATURE_CONFIGS[feature_set]])
    assert len(predicted) == 3
    _, metadata = augmentation.encoded_feature_metadata(model, feature_set)
    learned = metadata["categories"]
    assert all("unseen_protocol" not in values for values in learned.values())
    assert all("UNSEEN_STATE" not in values for values in learned.values())


def test_preprocessing_statistics_are_learned_from_fit_rows_only() -> None:
    train = pd.DataFrame(
        {
            **{feature: [0.0, 1.0, np.nan] for feature in FEATURES},
            "proto": ["tcp", "tcp", "udp"],
        }
    )
    model = augmentation.build_stage4_pipeline("F5_PROTO", "Decision_Tree")
    model.fit(train, pd.Series(["normal", "mitm", "normal"]))
    preprocessor = model.named_steps["preprocessor"]
    assert np.allclose(preprocessor.named_transformers_["numeric"].statistics_, 0.5)
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    assert encoder.categories_[0].tolist() == ["tcp", "udp"]
    transformed = preprocessor.transform(
        pd.DataFrame({**{feature: [9999.0] for feature in FEATURES}, "proto": ["icmp"]})
    )
    assert transformed.shape[0] == 1  # unseen validation value did not refit the encoder
    assert encoder.categories_[0].tolist() == ["tcp", "udp"]


def test_f5_pipeline_preserves_original_knn_search_and_is_reproducible() -> None:
    frame = pd.DataFrame({feature: np.arange(30, dtype=float) for feature in FEATURES})
    labels = pd.Series(["normal", "mitm", "xss"] * 10)
    first = augmentation.build_stage4_pipeline("F5", "kNN")
    second = augmentation.build_stage4_pipeline("F5", "kNN")
    assert first.named_steps["classifier"].algorithm == "kd_tree"
    first.fit(frame, labels)
    second.fit(frame, labels)
    assert np.array_equal(first.predict(frame), second.predict(frame))


def test_security_alarm_metric_definition() -> None:
    actual = np.array(["normal", "normal", "mitm", "mitm"])
    predicted = np.array(["normal", "mitm", "mitm", "normal"])
    summary, classwise, matrix = ablation.calculate_metrics(
        actual, predicted, ["mitm", "normal"]
    )
    assert summary["normal_rows"] == 2
    assert summary["normal_to_attack_false_alarms"] == 1
    assert summary["normal_to_attack_fpr"] == 0.5
    assert summary["missed_actual_attacks"] == 1
    assert sum(row["support"] for row in classwise) == 4
    assert matrix.sum() == 4


def test_matched_repeated_split_uses_all_rows_without_group_overlap(dataset) -> None:
    clean, _ = dataset
    positions = stability.matched_group_split(clean, 42)
    combined = np.concatenate(list(positions.values()))
    assert len(combined) == len(clean)
    assert len(np.unique(combined)) == len(clean)


def test_full_ablation_outputs_trace_to_predictions(dataset) -> None:
    clean, split = dataset
    output = RESULTS / "stage4"
    metrics = pd.read_csv(output / "ablation_metrics.csv")
    classwise = pd.read_csv(output / "ablation_classwise_metrics.csv")
    predictions = pd.read_csv(output / "ablation_predictions.csv")
    assert len(metrics) == 16
    assert len(classwise) == 160
    assert len(predictions) == 16 * 24_831
    assert metrics.groupby(["model", "feature_set", "split"]).size().eq(1).all()
    assert classwise.groupby(["model", "feature_set", "split"]).size().eq(10).all()
    test_mitm = classwise[(classwise.split == "test") & (classwise["class"] == "mitm")]
    assert len(test_mitm) == 8
    assert test_mitm.support.eq(149).all()
    trace = comparison.verify_saved_results(predictions, metrics, classwise, clean, split)
    assert trace["status"] == "passed"


def test_all_eight_models_and_baseline_parity_exist() -> None:
    expected = {
        f"{model}_{feature_set}.joblib"
        for model in augmentation.ALGORITHMS
        for feature_set in augmentation.FEATURE_CONFIGS
    }
    assert expected.issubset({path.name for path in (MODELS / "stage4").glob("*.joblib")})
    metrics = pd.read_csv(RESULTS / "stage4" / "ablation_metrics.csv")
    predictions = pd.read_csv(RESULTS / "stage4" / "ablation_predictions.csv")
    parity = ablation._assert_baseline_parity(metrics, predictions)
    assert parity["status"].startswith("passed")


def test_manifest_and_latency_protocol_are_complete() -> None:
    output = RESULTS / "stage4"
    experiment = json.loads((output / "experiment_manifest.json").read_text())
    assert experiment["test_predictions_per_frozen_configuration"] == 1
    assert experiment["baseline_parity"]["status"].startswith("passed")
    assert experiment["data"]["dataset_sha256"] == augmentation.sha256_file(augmentation.DATA)
    for relative, expected_hash in experiment[
        "protected_artifact_sha256_before_and_after"
    ].items():
        assert augmentation.sha256_file(ROOT / relative) == expected_hash

    latency = pd.read_csv(output / "latency_summary.csv")
    raw = pd.read_csv(output / "latency_measurements.csv")
    assert len(latency) == 8
    assert latency.measured_repeats.ge(2).all()
    assert latency.warmup_runs.ge(1).all()
    assert len(raw) == int(latency.measured_repeats.sum())
    assert latency.median_ms_per_1000_rows.gt(0).all()


def test_every_final_figure_has_traceable_companion_data() -> None:
    figure_dir = ROOT / "figures" / "stage4"
    provenance = json.loads((figure_dir / "chart_provenance.json").read_text())
    assert len(provenance["charts"]) == 5
    for relative, expected_hash in provenance["source_files"].items():
        assert augmentation.sha256_file(ROOT / relative) == expected_hash
    for record in provenance["charts"].values():
        image = ROOT / record["image"]
        data = ROOT / record["companion_data"]
        assert image.exists() and image.stat().st_size > 0
        assert data.exists() and len(pd.read_csv(data)) > 0


def test_ten_seed_stability_outputs_are_matched_and_complete() -> None:
    output = RESULTS / "stage4"
    metrics = pd.read_csv(output / "stability_metrics.csv")
    classwise = pd.read_csv(output / "stability_classwise_metrics.csv")
    paired = pd.read_csv(output / "stability_paired_deltas.csv")
    assert metrics.seed.nunique() == 10
    assert len(metrics) == 10 * 8
    assert len(classwise) == 10 * 8 * 10
    assert len(paired) == len(classwise)
    assert metrics.groupby("seed").test_rows.nunique().eq(1).all()
    assert classwise.groupby(["seed", "class"]).support.nunique().eq(1).all()
