"""STAGES 1-2: dataset audit, exact-row deduplication, EDA and leak-safe splits.

Run: python src/01_eda_preprocessing.py
     python src/01_eda_preprocessing.py --rebuild-split

By default, reuses the PROVIDED baseline split when valid, preserving the
published slide 10 results. --rebuild-split reproduces the original seven-fold
StratifiedGroupKFold grouping procedure and can take a while.
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from common import (DATA, RESULTS, FIGURES, DIAGNOSTICS, FEATURES, TARGET,
                    SEED, folders, clean_data, numeric_inputs, feature_group_hashes,
                    load_indices)


def plot_save(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=160, bbox_inches='tight')
    plt.close()


def run(rebuild_split=False):
    folders()
    raw, df = clean_data()
    X = numeric_inputs(df)
    counts = raw[TARGET].value_counts()
    pd.DataFrame({'count': counts, 'percentage': 100 * counts / len(raw)}).to_csv(
        RESULTS / 'class_distribution_original.csv', float_format='%.5f')
    df[TARGET].value_counts().to_csv(RESULTS / 'class_distribution_after_dedup.csv', header=['count'])
    missing = raw.isna().sum().rename('null_count').to_frame()
    missing['pct_missing'] = 100 * missing.null_count / len(raw)
    missing['dtype'] = raw.dtypes.astype(str)
    missing.to_csv(RESULTS / 'missing_values_by_column.csv')
    numeric_inputs(raw).describe(percentiles=[.01,.25,.5,.75,.95,.99]).T.to_csv(
        RESULTS / 'five_feature_summary.csv')

    plt.figure(figsize=(9.5, 5.2))
    counts.sort_values().plot(kind='barh')
    plt.xlabel('Network flow records')
    plt.title('ToN-IoT class distribution (before deduplication)')
    plot_save('01_class_distribution.png')

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.4))
    for ax, col in zip(axes, FEATURES):
        values = numeric_inputs(raw)[col].dropna()
        ax.hist(np.log1p(values), bins=45)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel('log(1+value)')
    fig.suptitle('Five original feature distributions')
    plot_save('02_feature_distributions.png')

    plt.figure(figsize=(7, 5))
    corr = np.log1p(numeric_inputs(raw)).corr(method='spearman')
    plt.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    plt.xticks(range(len(FEATURES)), FEATURES, rotation=50, ha='right', fontsize=8)
    plt.yticks(range(len(FEATURES)), FEATURES, fontsize=8)
    plt.colorbar(label='Spearman correlation')
    plt.title('Five-feature correlations')
    plot_save('03_spearman_correlations.png')

    pattern_stats = df.groupby(FEATURES, dropna=False, sort=False)[TARGET].agg(['size', 'nunique'])
    conflicts = pattern_stats[pattern_stats['nunique'] > 1]
    metadata = {
        'original_rows': int(len(raw)), 'columns': int(raw.shape[1]),
        'exact_full_row_duplicates_removed': int(raw.duplicated().sum()),
        'deduplicated_rows': int(len(df)),
        'class_count_including_normal': int(df[TARGET].nunique()),
        'mitm_rows_before_deduplication': int(raw[TARGET].eq('mitm').sum()),
        'mitm_rows_after_deduplication': int(df[TARGET].eq('mitm').sum()),
        'unique_five_feature_patterns_after_deduplication': int(len(pattern_stats)),
        'identical_feature_patterns_with_conflicting_labels': int(len(conflicts)),
        'rows_in_conflicting_label_patterns': int(conflicts['size'].sum()),
        'input_features': FEATURES, 'target': TARGET,
        'all_features_nonnegative': True,
    }
    (RESULTS / 'data_audit.json').write_text(json.dumps(metadata, indent=2))

    conflicts_patterns = conflicts.reset_index()[FEATURES]
    pattern_rows = []
    for cls, sub in df.groupby(TARGET):
        unique = sub[FEATURES].drop_duplicates()
        shared = unique.merge(conflicts_patterns, on=FEATURES, how='inner')
        pattern_rows.append({
            'class': cls, 'records': len(sub),
            'unique_five_feature_patterns': len(unique),
            'pattern_diversity_ratio': len(unique)/len(sub),
            'patterns_shared_with_other_labels': len(shared),
        })
    pd.DataFrame(pattern_rows).sort_values('records').to_csv(
        DIAGNOSTICS / 'class_pattern_diversity.csv', index=False)

    # Reuse the original saved split, unless the user requests full regeneration.
    use_saved = (RESULTS / 'fixed_split_assignments.csv').exists() and not rebuild_split
    if use_saved:
        indices = load_indices(df)
        names = np.full(len(df), '', dtype=object)
        for key, positions in indices.items():
            names[positions] = key
        print('Loaded existing validated group-disjoint split.')
    else:
        print('Generating 7-fold StratifiedGroupKFold split...', flush=True)
        hashes = feature_group_hashes(df)
        splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
        fold_labels = np.full(len(df), -1, dtype=np.int8)
        for fold, (_, holdout) in enumerate(splitter.split(X, df[TARGET], groups=hashes)):
            fold_labels[holdout] = fold
        if np.any(fold_labels < 0):
            raise RuntimeError('Some rows were not assigned to a fold')
        names = np.where(fold_labels == 5, 'validation', np.where(fold_labels == 6, 'test', 'train'))
        pd.DataFrame({'deduplicated_row_index': np.arange(len(df)),
                      'split': names, 'class': df[TARGET]}).to_csv(
                          RESULTS / 'fixed_split_assignments.csv', index=False)
        load_indices(df)  # leakage check
        print('Generated and validated group-disjoint split.')

    cross_tab = pd.crosstab(pd.Series(names, name='split'), df[TARGET])
    cross_tab['total'] = cross_tab.sum(axis=1)
    cross_tab.to_csv(RESULTS / 'split_class_counts.csv')
    print('Dataset audit:', json.dumps(metadata, indent=2))
    print('Class split counts:\n', cross_tab.to_string())
    print('Stage 1-2 complete; data audited, duplicates removed BEFORE splitting.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-split', action='store_true')
    run(**vars(parser.parse_args()))
