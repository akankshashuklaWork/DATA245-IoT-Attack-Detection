"""STAGE 4A: feature definitions, validation, and train-only preprocessing.

This module does not train or score a classifier when imported.  Its command
line interface writes a data/split audit only:

    python src/06_feature_augmentation.py

The fixed split and full-row deduplication are delegated to ``common.py`` so
Stage 4 cannot silently develop a different data-cleaning path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import (
    DATA,
    FEATURES,
    RESULTS,
    ROOT,
    SEED,
    TARGET,
    build_models,
    clean_data,
    feature_group_hashes,
    load_indices,
)

CATEGORICAL_FEATURES = ["proto", "conn_state"]
FEATURE_CONFIGS = OrderedDict(
    [
        ("F5", list(FEATURES)),
        ("F5_PROTO", [*FEATURES, "proto"]),
        ("F5_STATE", [*FEATURES, "conn_state"]),
        ("F7", [*FEATURES, *CATEGORICAL_FEATURES]),
    ]
)
ALGORITHMS = ("Decision_Tree", "kNN")
REQUIRED_COLUMNS = [*FEATURES, *CATEGORICAL_FEATURES, TARGET]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 checksum for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_output_dir(value: str | Path) -> Path:
    """Resolve a CLI output path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_feature_config(feature_set: str) -> list[str]:
    """Return the exact allowed columns for a named configuration."""
    if feature_set not in FEATURE_CONFIGS:
        raise ValueError(
            f"Unknown feature set {feature_set!r}; choose from {list(FEATURE_CONFIGS)}"
        )
    columns = FEATURE_CONFIGS[feature_set]
    forbidden = {TARGET, "label"}.intersection(columns)
    if forbidden:
        raise ValueError(f"Target-derived columns are forbidden predictors: {forbidden}")
    return list(columns)


def validate_stage4_data(clean: pd.DataFrame) -> dict[str, np.ndarray]:
    """Validate schema, fixed-row alignment, and five/seven-feature separation."""
    missing = [column for column in REQUIRED_COLUMNS if column not in clean.columns]
    if missing:
        raise ValueError(f"Missing Stage 4 columns: {missing}")
    if clean[TARGET].isna().any():
        raise ValueError("Target labels contain missing values")

    split = load_indices(clean)  # also checks row order, labels, and F5 group leakage
    if len(clean) != sum(len(values) for values in split.values()):
        raise ValueError("Fixed split does not cover every deduplicated row exactly once")

    # This is logically implied by F5 separation, but an explicit assertion makes
    # the Stage 4 invariant visible and guards future changes to the split code.
    seven_hashes = pd.util.hash_pandas_object(
        clean[[*FEATURES, *CATEGORICAL_FEATURES]], index=False
    ).to_numpy(dtype=np.uint64)
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if not set(seven_hashes[split[left]]).isdisjoint(set(seven_hashes[split[right]])):
            raise ValueError(f"FEATURE LEAKAGE: duplicate F7 patterns across {left}/{right}")
    return split


def feature_frame(clean: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Create an input frame containing only the prespecified predictor columns."""
    columns = validate_feature_config(feature_set)
    missing = [column for column in columns if column not in clean.columns]
    if missing:
        raise ValueError(f"Missing predictors for {feature_set}: {missing}")
    frame = clean.loc[:, columns].copy()
    frame.loc[:, FEATURES] = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce")
    numeric = frame.loc[:, FEATURES]
    if (numeric < 0).any().any():
        raise ValueError("Negative numeric values are incompatible with kNN log1p")
    return frame


def _dense_one_hot_encoder() -> OneHotEncoder:
    """Build a dense encoder across supported scikit-learn API versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    except TypeError:  # pragma: no cover - compatibility with scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float64)


def build_preprocessor(feature_set: str, algorithm: str) -> ColumnTransformer:
    """Build the augmented train-only transformer for one model/configuration.

    F5 uses the original pipelines directly in :func:`build_stage4_pipeline` to
    guarantee baseline parity.  This function is therefore intended for the
    three configurations that include at least one categorical field.
    """
    columns = validate_feature_config(feature_set)
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}; choose from {ALGORITHMS}")
    categorical = [column for column in CATEGORICAL_FEATURES if column in columns]
    if not categorical:
        raise ValueError("F5 uses the unchanged baseline pipeline, not a ColumnTransformer")

    if algorithm == "Decision_Tree":
        numeric_transformer: Any = SimpleImputer(strategy="median")
    else:
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("log1p", FunctionTransformer(np.log1p, validate=False, feature_names_out="one-to-one")),
                ("standardize", StandardScaler()),
            ]
        )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _dense_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_transformer, list(FEATURES)),
            ("categorical", categorical_transformer, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def build_stage4_pipeline(
    feature_set: str,
    algorithm: str,
    *,
    seed: int = SEED,
    n_jobs: int = 1,
) -> Pipeline:
    """Return one frozen Stage 4 model pipeline.

    The F5 branch is exactly ``common.build_models``.  Augmented kNN uses dense
    one-hot features and ``algorithm='auto'``; this can select a different
    neighbor-search implementation than the F5 baseline's ``kd_tree``.
    """
    validate_feature_config(feature_set)
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}; choose from {ALGORITHMS}")
    if feature_set == "F5":
        return build_models(seed=seed, n_jobs=n_jobs)[algorithm]

    classifier: Any
    if algorithm == "Decision_Tree":
        classifier = DecisionTreeClassifier(criterion="gini", random_state=seed)
    else:
        classifier = KNeighborsClassifier(
            n_neighbors=5,
            algorithm="auto",
            weights="uniform",
            metric="minkowski",
            p=2,
            n_jobs=n_jobs,
        )
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(feature_set, algorithm)),
            ("classifier", classifier),
        ]
    )


def encoded_feature_metadata(model: Pipeline, feature_set: str) -> tuple[list[str], dict[str, Any]]:
    """Return fitted output names and training-derived categorical metadata."""
    if feature_set == "F5":
        return list(FEATURES), {}
    preprocessor = model.named_steps["preprocessor"]
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    cat_pipeline = preprocessor.named_transformers_["categorical"]
    encoder = cat_pipeline.named_steps["encoder"]
    categorical = [column for column in CATEGORICAL_FEATURES if column in FEATURE_CONFIGS[feature_set]]
    mapping = {
        column: [str(value) for value in categories.tolist()]
        for column, categories in zip(categorical, encoder.categories_)
    }
    imputed = {
        column: str(value)
        for column, value in zip(categorical, cat_pipeline.named_steps["imputer"].statistics_)
    }
    return names, {"categories": mapping, "imputation_values": imputed}


def category_audit(clean: pd.DataFrame, split: dict[str, np.ndarray]) -> dict[str, Any]:
    """Describe missing, observed, and train-unseen categorical values."""
    report: dict[str, Any] = {}
    for column in CATEGORICAL_FEATURES:
        train_values = set(clean.iloc[split["train"]][column].dropna().astype(str))
        split_details = {}
        for name, positions in split.items():
            values = clean.iloc[positions][column]
            observed = sorted(set(values.dropna().astype(str)))
            split_details[name] = {
                "rows": int(len(values)),
                "missing": int(values.isna().sum()),
                "observed_category_count": len(observed),
                "observed_categories": observed,
                "unknown_vs_train": sorted(set(observed).difference(train_values)),
            }
        report[column] = split_details
    return report


def build_data_audit(clean: pd.DataFrame, split: dict[str, np.ndarray]) -> dict[str, Any]:
    """Build the serializable Stage 4 preprocessing audit."""
    five_hashes = feature_group_hashes(clean)
    return {
        "study_status": "exploratory/comparative; the fixed test split was examined in Stage 3",
        "dataset_path": str(DATA.relative_to(ROOT)),
        "dataset_sha256": sha256_file(DATA),
        "deduplicated_rows": int(len(clean)),
        "split_sizes": {name: int(len(values)) for name, values in split.items()},
        "unique_five_feature_patterns": int(pd.Series(five_hashes).nunique()),
        "five_feature_group_disjoint": True,
        "seven_feature_group_disjoint": True,
        "target": TARGET,
        "feature_configurations": FEATURE_CONFIGS,
        "categorical_audit": category_audit(clean, split),
        "preprocessing_fit_scope": "training rows only (inside scikit-learn pipelines)",
    }


def package_versions() -> dict[str, str]:
    """Return versions needed to interpret serialized models and results."""
    return {
        "python": __import__("platform").python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def run(output_dir: str | Path = RESULTS / "stage4") -> Path:
    """Validate the fixed experiment inputs and save an audit JSON."""
    _, clean = clean_data()
    split = validate_stage4_data(clean)
    output_path = resolve_output_dir(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    audit = build_data_audit(clean, split)
    audit["package_versions"] = package_versions()
    destination = output_path / "preprocessing_audit.json"
    destination.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Stage 4 data audit saved: {destination}")
    print(json.dumps(audit["split_sizes"], indent=2))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/stage4")
    arguments = parser.parse_args()
    run(arguments.output_dir)
