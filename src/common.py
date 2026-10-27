"""Shared configuration and utilities. No experiment is executed on import.

Run each numbered script from any working directory; paths resolve from this file.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'train_test_network.csv'
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'
DIAGNOSTICS = ROOT / 'diagnostics'
MODELS = ROOT / 'models'
PRESENTATION = ROOT / 'presentation_charts'
FEATURES = ['duration', 'src_bytes', 'dst_bytes', 'src_ip_bytes', 'dst_ip_bytes']
TARGET = 'type'
SEED = 42
SEEDS = [42, 7, 17, 23, 29, 41, 73, 101, 202, 313]


def folders():
    for directory in (RESULTS, FIGURES, DIAGNOSTICS, MODELS, PRESENTATION):
        directory.mkdir(parents=True, exist_ok=True)


def clean_data():
    """Read original CSV, drop exact full-row duplicates BEFORE any data split."""
    if not DATA.exists():
        raise FileNotFoundError(f'Missing dataset: {DATA}')
    df = pd.read_csv(DATA, low_memory=False)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if df[TARGET].isna().any():
        raise ValueError('Target labels contain NaN')
    clean = df.drop_duplicates().reset_index(drop=True)
    X = clean[FEATURES].apply(pd.to_numeric, errors='coerce')
    if (X < 0).any().any():
        raise ValueError('Negative predictor values are incompatible with log1p')
    return df, clean


def numeric_inputs(df):
    return df[FEATURES].apply(pd.to_numeric, errors='coerce')


def feature_group_hashes(df):
    return pd.util.hash_pandas_object(df[FEATURES], index=False).to_numpy(dtype=np.uint64)


def load_indices(clean):
    """Load the exact fixed baseline split; verify alignment with deduplicated rows."""
    path = RESULTS / 'fixed_split_assignments.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing split file {path}; run 01_eda_preprocessing.py first')
    a = pd.read_csv(path)
    required = {'deduplicated_row_index', 'split', 'class'}
    if not required.issubset(a.columns) or len(a) != len(clean):
        raise ValueError('Split assignments do not match deduplicated dataset')
    if not np.array_equal(a['deduplicated_row_index'].to_numpy(), np.arange(len(clean))):
        raise ValueError('Saved split indexes do not match original row order')
    if not np.array_equal(a['class'].astype(str).to_numpy(), clean[TARGET].astype(str).to_numpy()):
        raise ValueError('Saved class labels do not align to dataset')
    hashes = feature_group_hashes(clean)
    indices = {name: np.flatnonzero(a['split'].eq(name).to_numpy())
               for name in ('train', 'validation', 'test')}
    if sum(map(len, indices.values())) != len(clean):
        raise ValueError('Invalid or missing split assignments')
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if not set(hashes[indices[left]]).isdisjoint(set(hashes[indices[right]])):
            raise ValueError(f'FEATURE LEAKAGE: duplicate feature patterns across {left}/{right}')
    return indices


def build_models(seed=SEED, n_jobs=1):
    """Baseline models use only the five original numeric traffic features."""
    return {
        'Decision_Tree': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('classifier', DecisionTreeClassifier(criterion='gini', random_state=seed)),
        ]),
        'kNN': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('log1p', FunctionTransformer(np.log1p, validate=False)),
            ('standardize', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree',
                                               weights='uniform', n_jobs=n_jobs)),
        ]),
    }
