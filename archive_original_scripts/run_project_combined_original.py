"""DATA245 ToN-IoT baseline: dataset audit, EDA, leakage-aware split, DT + kNN.

Run from project root:
    python src/run_project.py

Only the five numerical features selected by Dharini et al. (2026) are inputs.
All other fields are used solely for auditing and exclusion; 'type' is the target.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support,
    accuracy_score, balanced_accuracy_score, f1_score
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'train_test_network.csv'
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'
MODELS = ROOT / 'models'
FEATURES = ['duration', 'src_bytes', 'dst_bytes', 'src_ip_bytes', 'dst_ip_bytes']
TARGET = 'type'
SEED = 42


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=155, bbox_inches='tight')
    plt.close()


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 1 audit and Stage 2 EDA before splitting."""
    for col in FEATURES + [TARGET]:
        if col not in df.columns:
            raise ValueError(f'Required CSV column missing: {col}')
    if df[TARGET].isna().any():
        raise ValueError('Missing target labels must be resolved.')

    counts = df[TARGET].value_counts().sort_values(ascending=False)
    class_table = pd.DataFrame({'count': counts, 'percentage': 100 * counts / len(df)})
    class_table.to_csv(RESULTS / 'class_distribution_original.csv', float_format='%.5f')

    col_missing = df.isna().sum().rename('null_count').to_frame()
    col_missing['pct_missing'] = 100 * col_missing.null_count / len(df)
    col_missing['dtype'] = df.dtypes.astype(str)
    col_missing.to_csv(RESULTS / 'missing_values_by_column.csv')

    numeric = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    numeric.describe(percentiles=[.01,.25,.5,.75,.95,.99]).T.to_csv(RESULTS / 'five_feature_summary.csv')
    if (numeric < 0).any().any():
        raise ValueError('Negative values detected: log1p transform requires nonnegative values.')
    if numeric.isna().any().any():
        print('WARNING: Some feature values are nonnumeric; imputation will be fitted on training only.')

    plt.figure(figsize=(9.5, 5.2))
    counts.sort_values().plot(kind='barh')
    plt.xlabel('Number of network flows')
    plt.ylabel('Traffic class')
    plt.title('ToN-IoT class distribution (before deduplication)')
    savefig('01_class_distribution.png')

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.4))
    for ax, col in zip(axes, FEATURES):
        ax.hist(np.log1p(numeric[col].dropna()), bins=45)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel('log(1 + value)')
    fig.suptitle('Skew in the five original numerical features')
    savefig('02_feature_distributions.png')

    plt.figure(figsize=(7, 5))
    c = np.log1p(numeric).corr(method='spearman')
    plt.imshow(c, vmin=-1, vmax=1, cmap='coolwarm')
    plt.xticks(range(len(FEATURES)), FEATURES, rotation=50, ha='right', fontsize=8)
    plt.yticks(range(len(FEATURES)), FEATURES, fontsize=8)
    plt.colorbar(label='Spearman correlation')
    plt.title('Feature relationships (log-transformed)')
    savefig('03_spearman_correlations.png')

    deduped = df.drop_duplicates().reset_index(drop=True)
    # Distinct 5-feature vectors can have *different target labels*. These
    # contradictory patterns are an irreducible limit for five-feature models.
    group_stats = deduped.groupby(FEATURES, dropna=False, sort=False)[TARGET].agg(['size', 'nunique'])
    collision_rows = int(group_stats.loc[group_stats['nunique'] > 1, 'size'].sum())
    collision_groups = int((group_stats['nunique'] > 1).sum())
    duplicates = int(df.duplicated().sum())
    meta = {
        'original_rows': len(df), 'columns': len(df.columns),
        'exact_full_row_duplicates_removed': duplicates,
        'deduplicated_rows': len(deduped),
        'class_count_including_normal': int(counts.size),
        'mitm_rows_before_deduplication': int((df[TARGET] == 'mitm').sum()),
        'mitm_rows_after_deduplication': int((deduped[TARGET] == 'mitm').sum()),
        'unique_five_feature_patterns_after_deduplication': len(group_stats),
        'identical_feature_patterns_with_conflicting_labels': collision_groups,
        'rows_in_conflicting_label_patterns': collision_rows,
        'input_features': FEATURES,
        'target': TARGET,
        'all_features_nonnegative': True,
    }
    (RESULTS / 'data_audit.json').write_text(json.dumps(meta, indent=2))
    deduped[TARGET].value_counts().to_csv(RESULTS / 'class_distribution_after_dedup.csv', header=['count'])
    print('DATA AUDIT:', json.dumps(meta, indent=2))
    return deduped


def split_by_five_feature_pattern(df: pd.DataFrame):
    """Approx. 71/14/14 split using disjoint exact 5-feature groups.

    We use a fixed 7-fold StratifiedGroupKFold and assign fold 5 to validation,
    fold 6 to test, the other five to train. This avoids identical feature
    vectors appearing in both training and test, an issue in this dataset.
    Unlike ordinary stratification, group folds can have unequal sample sizes.
    """
    hashes = pd.util.hash_pandas_object(df[FEATURES], index=False).to_numpy(dtype=np.uint64)
    cv = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    folds = np.full(len(df), -1, dtype=np.int8)
    for fold_index, (_unused, holdout) in enumerate(cv.split(df[FEATURES], df[TARGET], groups=hashes)):
        folds[holdout] = fold_index
    assert np.all(folds >= 0)
    train_idx = np.flatnonzero((folds != 5) & (folds != 6))
    val_idx = np.flatnonzero(folds == 5)
    test_idx = np.flatnonzero(folds == 6)
    assert len(set(hashes[train_idx]) & set(hashes[val_idx])) == 0
    assert len(set(hashes[train_idx]) & set(hashes[test_idx])) == 0
    assert len(set(hashes[val_idx]) & set(hashes[test_idx])) == 0
    names = np.where(folds == 5, 'validation', np.where(folds == 6, 'test', 'train'))
    pd.DataFrame({'deduplicated_row_index': np.arange(len(df)), 'split': names, 'class': df[TARGET]}).to_csv(
        RESULTS / 'fixed_split_assignments.csv', index=False)
    split_table = pd.crosstab(pd.Series(names, name='split'), df[TARGET])
    split_table.loc[:, 'total'] = split_table.sum(axis=1)
    split_table.to_csv(RESULTS / 'split_class_counts.csv')
    print('\nSPLIT CLASS COUNTS:\n', split_table.to_string())
    return train_idx, val_idx, test_idx


def fit_and_evaluate(df, train_idx, val_idx, test_idx):
    x = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    y = df[TARGET].astype(str)
    labels = sorted(y.unique().tolist())
    mitm_index = labels.index('mitm')

    # Decision Tree operates on the five original raw numeric values.
    dt = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', DecisionTreeClassifier(criterion='gini', random_state=SEED)),
    ])
    # kNN is distance-based: log1p reduces heavy right-skew, and StandardScaler
    # fits only on training rows. These transform only the SAME five features.
    knn = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log1p', FunctionTransformer(np.log1p, validate=False)),
        ('standardize', StandardScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree',
                                             weights='uniform', n_jobs=-1)),
    ])
    models = {'Decision_Tree': dt, 'kNN': knn}
    all_rows, class_rows, predictions = [], [], []

    for name, model in models.items():
        print(f'\nTraining {name} on {len(train_idx):,} rows... ', flush=True)
        start = time.perf_counter()
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        fit_seconds = time.perf_counter() - start
        joblib.dump(model, MODELS / f'{name}_five_features.joblib', compress=3)
        print(f'  fitted in {fit_seconds:.2f}s', flush=True)
        for subset, indices in [('validation', val_idx), ('test', test_idx)]:
            start = time.perf_counter()
            pred = model.predict(x.iloc[indices])
            prediction_seconds = time.perf_counter() - start
            actual = y.iloc[indices].to_numpy()
            report = classification_report(actual, pred, labels=labels,
                                           zero_division=0, output_dict=True)
            result = {
                'model': name, 'split': subset, 'n_rows': len(indices),
                'accuracy': accuracy_score(actual, pred),
                'balanced_accuracy': balanced_accuracy_score(actual, pred),
                'macro_f1': f1_score(actual, pred, labels=labels, average='macro', zero_division=0),
                'weighted_f1': f1_score(actual, pred, labels=labels, average='weighted', zero_division=0),
                'mitm_precision': report['mitm']['precision'],
                'mitm_recall': report['mitm']['recall'],
                'mitm_f1': report['mitm']['f1-score'],
                'mitm_support': int(report['mitm']['support']),
                'training_seconds': fit_seconds,
                'prediction_seconds': prediction_seconds,
                'ms_per_1000_predictions': 1000 * 1000 * prediction_seconds / len(indices),
                'n_model_features': len(FEATURES),
            }
            if name == 'Decision_Tree':
                tree = model.named_steps['classifier']
                result['tree_depth'] = int(tree.get_depth())
                result['tree_leaves'] = int(tree.get_n_leaves())
            else:
                result['knn_reference_rows'] = int(model.named_steps['classifier'].n_samples_fit_)
            all_rows.append(result)
            per_class = pd.DataFrame(report).T
            per_class['class'] = per_class.index
            per_class['model'] = name
            per_class['split'] = subset
            class_rows.append(per_class)

            cm = confusion_matrix(actual, pred, labels=labels)
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(
                RESULTS / f'confusion_{name}_{subset}.csv')
            fig, ax = plt.subplots(figsize=(9.3, 8.5))
            im = ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(len(labels)), labels, rotation=55, ha='right', fontsize=9)
            ax.set_yticks(range(len(labels)), labels, fontsize=9)
            ax.set_xlabel('Predicted class')
            ax.set_ylabel('True class')
            ax.set_title(f'{name.replace("_", " ")} — {subset} confusion matrix (counts)')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            savefig(f'04_confusion_{name}_{subset}.png')
            print(f'  {subset}: accuracy={result["accuracy"]:.3f}, macro-F1={result["macro_f1"]:.3f}, '
                  f'MITM recall={result["mitm_recall"]:.3f}, MITM precision={result["mitm_precision"]:.3f}, '
                  f'predict={prediction_seconds:.1f}s', flush=True)
            predictions.append(pd.DataFrame({'model': name, 'split': subset,
                                             'row_index': indices, 'actual': actual, 'predicted': pred}))

    summary = pd.DataFrame(all_rows)
    summary.to_csv(RESULTS / 'baseline_metrics.csv', index=False, float_format='%.6f')
    pd.concat(class_rows, ignore_index=True).to_csv(RESULTS / 'classwise_metrics.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(RESULTS / 'predictions_val_test.csv', index=False)

    test = summary[summary.split == 'test']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, caption in [(axes[0], 'mitm_recall', 'MITM recall (higher is better)'),
                                (axes[1], 'macro_f1', 'Macro-F1 (higher is better)')]:
        vals = test.set_index('model')[metric]
        vals.plot(kind='bar', ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_xlabel('')
        ax.set_title(caption)
        ax.tick_params(axis='x', labelrotation=15)
    savefig('05_baseline_model_comparison.png')

    tree = dt.named_steps['classifier']
    feat_imp = pd.DataFrame({'feature': FEATURES, 'importance': tree.feature_importances_})
    feat_imp.to_csv(RESULTS / 'decision_tree_feature_importances.csv', index=False)
    plt.figure(figsize=(7, 4))
    feat_imp.sort_values('importance').set_index('feature')['importance'].plot(kind='barh')
    plt.xlabel('Decision Tree impurity-based feature importance')
    plt.title('Baseline Decision Tree: five-feature importance')
    savefig('06_dt_feature_importance.png')
    return summary


def make_report(summary):
    audit_info = json.loads((RESULTS / 'data_audit.json').read_text())
    counts = pd.read_csv(RESULTS / 'split_class_counts.csv').set_index('split')
    tests = summary.query("split == 'test'").set_index('model')
    lines = [
        '# DATA245 — Stage 1–3 Results (Original Five Features Only)', '',
        '## Scope',
        'Stage 1: verify ToN-IoT; Stage 2: clean, perform EDA and create held-out splits; '
        'Stage 3: fit baseline Decision Tree and kNN classifiers for ten-class classification.',
        '',
        'No protocol, connection-state, IP address, port, deep learning, embeddings, '
        'or new engineered predictors were used.',
        '',
        '## Dataset observations',
        f'- Original rows: **{audit_info["original_rows"]:,}**, columns: **{audit_info["columns"]}**.',
        f'- Exact duplicate full rows removed: **{audit_info["exact_full_row_duplicates_removed"]:,}**.',
        f'- Rows remaining: **{audit_info["deduplicated_rows"]:,}**.',
        f'- MITM rows after deduplication: **{audit_info["mitm_rows_after_deduplication"]:,}**.',
        f'- Unique 5-feature patterns: **{audit_info["unique_five_feature_patterns_after_deduplication"]:,}**.',
        f'- Patterns with multiple different class labels: **{audit_info["identical_feature_patterns_with_conflicting_labels"]:,}** '
        f'(covering **{audit_info["rows_in_conflicting_label_patterns"]:,}** rows). '
        'This limits what any deterministic classifier can infer from only these five inputs.',
        '',
        '## Data partitioning',
        'Fixed random seed 42; 7-fold StratifiedGroupKFold. Fold 5 = validation, fold 6 = test, '
        'five other folds = train. All rows sharing exactly the same five-feature vector '
        'remain in the same partition. Approximate target is 71/14/14; actual class and '
        'row counts are in `results/split_class_counts.csv`.',
        '',
        '| Split | Rows | MITM rows |', '|---|---:|---:|',
    ]
    for split in ['train', 'validation', 'test']:
        lines.append(f'| {split} | {int(counts.loc[split, "total"]):,} | {int(counts.loc[split, "mitm"]):,} |')
    lines += ['', '## Baseline models',
              '- **Decision Tree**: Gini splitting, random state 42, default complexity; median imputation fitted on train.',
              '- **kNN**: k=5, uniform neighbors, Euclidean distance; median imputation, log(1+x), '
              'and standardization fitted exclusively on train. Transformations do not add predictors.',
              '', '## Untouched test results',
              '| Model | Accuracy | Macro-F1 | MITM precision | MITM recall | MITM F1 |',
              '|---|---:|---:|---:|---:|---:|']
    for name in ['Decision_Tree','kNN']:
        r=tests.loc[name]
        lines.append(f'| {name.replace("_", " ")} | {r.accuracy:.3f} | {r.macro_f1:.3f} | '
                     f'{r.mitm_precision:.3f} | {r.mitm_recall:.3f} | {r.mitm_f1:.3f} |')
    lines += ['',
        '## What these results mean',
        'The paper reports high classification accuracy on a different data selection and '
        'evaluation split. **These are our actual observed baseline values**, not a '
        'numerical reproduction of the paper. High overall accuracy does not guarantee '
        'MITM detection; use MITM recall, precision, macro-F1 and the confusion matrices.',
        'The dataset has label ambiguity: multiple classes sometimes share identical five-feature '
        'vectors. A model cannot distinguish such rows using only these inputs.',
        '', '## Files to review',
        '- `results/class_distribution_original.csv`: original class sizes.',
        '- `results/five_feature_summary.csv`: statistics including extreme values.',
        '- `results/baseline_metrics.csv`: validation and test metrics and runtime.',
        '- `results/classwise_metrics.csv`: per-class precision, recall, F1 and support.',
        '- `results/confusion_*.csv`, `figures/04_confusion_*.png`: classification errors.',
        '- `models/*joblib`: trained baseline pipelines (load only from trusted files).',
        '', '## Stage 4 (not included)',
        'Protocol or connection-state features, improvements, tuning, feature ablation, '
        'and claims of improved performance are intentionally excluded.',
        '', '## Primary reference',
        'Dharini, N., Janani, V. S., & Katiravan, J. (2026). Efficient detection of '
        'intrusions in TON-IoT dataset using hybrid feature selection approach. '
        'Scientific Reports, 16, 7763. https://doi.org/10.1038/s41598-026-37834-y',
    ]
    (ROOT / 'STAGE_1_TO_3_REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    for folder in [RESULTS, FIGURES, MODELS]: folder.mkdir(exist_ok=True, parents=True)
    print('Loading', DATA)
    dataset = pd.read_csv(DATA, low_memory=False)
    clean = audit(dataset)
    train_idx, val_idx, test_idx = split_by_five_feature_pattern(clean)
    results = fit_and_evaluate(clean, train_idx, val_idx, test_idx)
    make_report(results)
    print('\nSTAGES 1–3 COMPLETE. Results saved under:', ROOT)

if __name__ == '__main__':
    main()
