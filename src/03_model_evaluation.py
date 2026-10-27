"""STAGE 3B: Evaluate saved models on validation and TEST partitions.

Run: python src/03_model_evaluation.py
Reads: models/*.joblib, results/fixed_split_assignments.csv
Writes: results/baseline_metrics.csv, classwise_metrics.csv,
        predictions_val_test.csv, confusion_*.csv; figures/04_*, 05_*, 06_*
Does NOT train any classifier. Does NOT run repeated-split experiments.
"""
import time
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from common import FEATURES, TARGET, RESULTS, FIGURES, MODELS, folders, clean_data, numeric_inputs, load_indices


def save_plot(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, bbox_inches='tight', dpi=155)
    plt.close()


def run():
    folders()
    _, clean = clean_data()
    split = load_indices(clean)
    x = numeric_inputs(clean)
    y = clean[TARGET].astype(str)
    labels = sorted(y.unique())
    training_times = RESULTS / 'training_times.csv'
    times = pd.read_csv(training_times).set_index('model') if training_times.exists() else None
    saved_metrics, per_class_rows, predictions = [], [], []

    for name in ('Decision_Tree', 'kNN'):
        filepath = MODELS / f'{name}_five_features.joblib'
        if not filepath.exists():
            raise FileNotFoundError(f'Missing {filepath}; run 02_train_models.py first')
        # Only load model files you created or trust: joblib uses pickle.
        model = joblib.load(filepath)
        if hasattr(model, 'feature_names_in_') and list(model.feature_names_in_) != FEATURES:
            raise ValueError(f'{name}: model does not match five-feature schema')
        train_seconds = float(times.loc[name, 'training_seconds']) if times is not None else np.nan
        for subset in ('validation', 'test'):
            idx = split[subset]
            start = time.perf_counter()
            pred = model.predict(x.iloc[idx])
            pred_seconds = time.perf_counter() - start
            actual = y.iloc[idx].to_numpy()
            report = classification_report(actual, pred, labels=labels, output_dict=True,
                                           zero_division=0)
            row = {'model': name, 'split': subset, 'n_rows': len(idx),
                   'accuracy': accuracy_score(actual, pred),
                   'balanced_accuracy': balanced_accuracy_score(actual, pred),
                   'macro_f1': f1_score(actual, pred, labels=labels, average='macro', zero_division=0),
                   'weighted_f1': f1_score(actual, pred, labels=labels, average='weighted', zero_division=0),
                   'mitm_precision': report['mitm']['precision'],
                   'mitm_recall': report['mitm']['recall'],
                   'mitm_f1': report['mitm']['f1-score'],
                   'mitm_support': int(report['mitm']['support']),
                   'training_seconds': train_seconds,
                   'prediction_seconds': pred_seconds,
                   'ms_per_1000_predictions': 1_000_000 * pred_seconds / len(idx),
                   'n_model_features': len(FEATURES)}
            if name == 'Decision_Tree':
                tree = model.named_steps['classifier']
                row['tree_depth'] = int(tree.get_depth())
                row['tree_leaves'] = int(tree.get_n_leaves())
            else:
                row['knn_reference_rows'] = int(model.named_steps['classifier'].n_samples_fit_)
            saved_metrics.append(row)
            for cls in labels:
                r = report[cls]
                per_class_rows.append({'model': name, 'split': subset, 'class': cls,
                                       'precision': r['precision'], 'recall': r['recall'],
                                       'f1-score': r['f1-score'], 'support': int(r['support'])})
            predictions.append(pd.DataFrame({'model': name, 'split': subset,
                                             'row_index': idx, 'actual': actual, 'predicted': pred}))
            cm = confusion_matrix(actual, pred, labels=labels)
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(
                RESULTS / f'confusion_{name}_{subset}.csv')
            fig, ax = plt.subplots(figsize=(9, 8))
            im = ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(len(labels)), labels, rotation=55, ha='right', fontsize=9)
            ax.set_yticks(range(len(labels)), labels, fontsize=9)
            ax.set_xlabel('Predicted class')
            ax.set_ylabel('Actual class')
            ax.set_title(f'{name} - {subset} confusion matrix')
            fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
            save_plot(f'04_confusion_{name}_{subset}.png')
            print(f'{name}, {subset}: accuracy={row["accuracy"]:.4f}, '
                  f'macro-F1={row["macro_f1"]:.4f}, MITM recall={row["mitm_recall"]:.4f}', flush=True)

    summary = pd.DataFrame(saved_metrics)
    summary.to_csv(RESULTS / 'baseline_metrics.csv', index=False, float_format='%.6f')
    pd.DataFrame(per_class_rows).to_csv(RESULTS / 'classwise_metrics.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(RESULTS / 'predictions_val_test.csv', index=False)

    test = summary.query('split == "test"').set_index('model')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in [(axes[0], 'mitm_recall', 'MITM recall'),
                              (axes[1], 'macro_f1', 'Macro-F1')]:
        ax.bar(['Decision Tree', 'kNN'], test.loc[['Decision_Tree', 'kNN'], metric])
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_ylabel('Score')
    save_plot('05_baseline_model_comparison.png')

    model = joblib.load(MODELS / 'Decision_Tree_five_features.joblib')
    importance = pd.DataFrame({'feature': FEATURES,
                               'importance': model.named_steps['classifier'].feature_importances_})
    importance.to_csv(RESULTS / 'decision_tree_feature_importances.csv', index=False)
    plt.figure(figsize=(7, 4))
    importance.sort_values('importance').set_index('feature')['importance'].plot(kind='barh')
    plt.title('Decision Tree: five-feature importance')
    save_plot('06_dt_feature_importance.png')
    print('Stage 3B complete. Now run 04_classification_errors.py.')


if __name__ == '__main__':
    run()
