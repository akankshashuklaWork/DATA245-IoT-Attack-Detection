"""STAGE 3D: Re-train Decision Tree + kNN over 10 group-disjoint splits.

Run all ten experiments (takes longer):
    python src/05_repeated_split_experiments.py
Quick verification using only one seed, in a SEPARATE output folder:
    python src/05_repeated_split_experiments.py --seeds 42 --output-dir tests/one_seed

This is NOT the fixed held-out test experiment from 03_model_evaluation.py.
Within each run identical exact 5-feature patterns are kept in one partition.
Observations overlap between runs; thus variability reflects sensitivity to
splitting, not new-device/new-environment generalization.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from common import (ROOT, DIAGNOSTICS, FEATURES, TARGET, SEEDS, folders, clean_data,
                    numeric_inputs, feature_group_hashes, build_models)


def run(seeds=SEEDS, output_dir=DIAGNOSTICS):
    folders()
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _, df = clean_data()
    X = numeric_inputs(df)
    y = df[TARGET].astype(str)
    groups = feature_group_hashes(df)
    labels = sorted(y.unique())
    mitm_pos = labels.index('mitm')

    # A group is an exact 5-feature vector. When a group has conflicting
    # labels, use its majority label ONLY to distribute groups across splits.
    # It is never used as an ML feature or as a replacement target label.
    majority = (pd.DataFrame({'group': groups, 'label': y.to_numpy()})
                .groupby(['group', 'label']).size().reset_index(name='count')
                .sort_values(['group', 'count', 'label'], ascending=[True, False, True])
                .drop_duplicates('group'))
    group_ids = majority['group'].to_numpy()
    group_labels = majority['label'].to_numpy()
    label_of_group = majority.set_index('group')['label']

    runs, class_rows = [], []
    for run_number, seed in enumerate(seeds, start=1):
        train_groups, holdout_groups = train_test_split(
            group_ids, test_size=.30, random_state=seed, stratify=group_labels)
        remaining_labels = label_of_group.loc[holdout_groups].to_numpy()
        val_groups, test_groups = train_test_split(
            holdout_groups, test_size=.5, random_state=seed + 5000,
            stratify=remaining_labels)
        train_idx = np.flatnonzero(np.isin(groups, train_groups))
        val_idx = np.flatnonzero(np.isin(groups, val_groups))
        test_idx = np.flatnonzero(np.isin(groups, test_groups))
        if len(train_idx) + len(val_idx) + len(test_idx) != len(df):
            raise RuntimeError('Group split omitted observations')
        for a, b in ((train_groups, val_groups), (train_groups, test_groups),
                     (val_groups, test_groups)):
            if not set(a).isdisjoint(set(b)):
                raise RuntimeError('Group leakage across partitions')
        for model_name, model in [('Decision Tree', build_models(seed=42)['Decision_Tree']),
                                  ('kNN', build_models(seed=42, n_jobs=1)['kNN'])]:
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            predicted = model.predict(X.iloc[test_idx])
            actual = y.iloc[test_idx].to_numpy()
            p, r, f, support = precision_recall_fscore_support(
                actual, predicted, labels=labels, zero_division=0)
            runs.append({
                'seed': seed, 'model': model_name, 'train_rows': len(train_idx),
                'validation_rows': len(val_idx), 'test_rows': len(test_idx),
                'mitm_train': int(y.iloc[train_idx].eq('mitm').sum()),
                'mitm_test': int((actual == 'mitm').sum()),
                'mitm_correct': int(((actual == 'mitm') & (predicted == 'mitm')).sum()),
                'mitm_precision': float(p[mitm_pos]), 'mitm_recall': float(r[mitm_pos]),
                'mitm_f1': float(f[mitm_pos]),
                'accuracy': float(accuracy_score(actual, predicted)),
                'macro_f1': float(f1_score(actual, predicted, average='macro', zero_division=0)),
            })
            class_rows.extend({
                'seed': seed, 'model': model_name, 'class': cls,
                'support': int(s), 'precision': float(pp),
                'recall': float(rr), 'f1': float(ff),
            } for cls, pp, rr, ff, s in zip(labels, p, r, f, support))
        print(f'Completed split {run_number}/{len(seeds)} (seed {seed}, '
              f'test MITM {int((actual == "mitm").sum())})', flush=True)

    metrics = pd.DataFrame(runs)
    metrics.to_csv(output_dir / 'repeated_group_safe_splits.csv', index=False)
    pd.DataFrame(class_rows).to_csv(output_dir / 'repeated_per_class_scores.csv', index=False)
    summary = metrics.groupby('model').agg(
        runs=('seed', 'count'), mitm_test_min=('mitm_test', 'min'),
        mitm_test_max=('mitm_test', 'max'),
        mitm_recall_mean=('mitm_recall', 'mean'),
        mitm_recall_std=('mitm_recall', 'std'),
        mitm_recall_min=('mitm_recall', 'min'),
        mitm_recall_max=('mitm_recall', 'max'),
        mitm_precision_mean=('mitm_precision', 'mean'),
        macro_f1_mean=('macro_f1', 'mean'),
        macro_f1_std=('macro_f1', 'std'),
    ).reset_index()
    summary.to_csv(output_dir / 'repeat_summary.csv', index=False)
    (output_dir / 'repeat_settings.json').write_text(json.dumps({
        'seeds': list(seeds), 'n_runs': len(seeds),
        'holdout_design': '70/15/15 group-majority-label stratification, groups = 5-feature hashes',
        'interpretation': 'Repeated splits overlap; no temporal or new-network generalization test',
    }, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, frame in metrics.groupby('model'):
        axes[0].plot(frame['seed'].astype(str), frame['mitm_recall'], marker='o', label=name)
        axes[1].plot(frame['seed'].astype(str), frame['macro_f1'], marker='o', label=name)
    for ax, name in zip(axes, ['MITM recall', 'Macro-F1']):
        ax.set_title(f'{name} across {len(seeds)} group-disjoint runs')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Random seed')
        ax.set_ylabel('Score')
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'repeated_split_stability.png', dpi=160)
    plt.close(fig)
    print('Stage 3D complete. Summary:\n', summary.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default=','.join(map(str, SEEDS)),
                        help='Comma-separated random seeds (default 10 seeds)')
    parser.add_argument('--output-dir', default=str(DIAGNOSTICS),
                        help='Folder for repeated results (default diagnostics/)')
    args = parser.parse_args()
    run([int(x.strip()) for x in args.seeds.split(',') if x.strip()],
        args.output_dir)
