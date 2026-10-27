"""STAGE 3C: Explain WHAT the models misclassify; no training.

Run: python src/04_classification_errors.py
Reads: results/predictions_val_test.csv
Writes: diagnostics/original_stage3_mitm_prediction_breakdown.csv,
        diagnostics/all_class_error_breakdown.csv,
        diagnostics/misclassification_examples.csv,
        figures/07_mitm_error_breakdown.png
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import RESULTS, DIAGNOSTICS, FIGURES, folders


def run():
    folders()
    path = RESULTS / 'predictions_val_test.csv'
    if not path.exists():
        raise FileNotFoundError(f'Run 03_model_evaluation.py first: {path}')
    pred = pd.read_csv(path)
    assert {'model', 'split', 'actual', 'predicted', 'row_index'}.issubset(pred.columns)

    # ALL attack categories; aggregate the actual -> predicted pairs.
    all_errors = (pred[pred.actual.ne(pred.predicted)]
                  .groupby(['model', 'split', 'actual', 'predicted']).size()
                  .reset_index(name='count')
                  .sort_values(['model', 'split', 'count'], ascending=[True, True, False]))
    all_errors.to_csv(DIAGNOSTICS / 'all_class_error_breakdown.csv', index=False)
    # Small row-level example file makes debugging possible.
    pred[pred.actual.ne(pred.predicted)].head(100).to_csv(
        DIAGNOSTICS / 'misclassification_examples.csv', index=False)

    # Keep the EXACT file schema used in the midterm charts.
    mitm = pred[pred.actual.eq('mitm')]
    mitm_counts = (mitm.groupby(['model', 'split', 'predicted']).size()
                   .reset_index(name='n')
                   .rename(columns={'predicted': 'predicted_class'}))
    mitm_counts['true_class'] = 'mitm'
    mitm_counts[['model', 'split', 'true_class', 'predicted_class', 'n']].to_csv(
        DIAGNOSTICS / 'original_stage3_mitm_prediction_breakdown.csv', index=False)

    dt = mitm_counts.query('model == "Decision_Tree" and split == "test"')
    wrong = dt[dt.predicted_class.ne('mitm')].sort_values('n', ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.barh(wrong['predicted_class'], wrong['n'])
    ax.set_xlabel('True MITM cases assigned to another class')
    ax.set_title('Decision Tree: MITM classification errors (held-out test)')
    plt.tight_layout()
    fig.savefig(FIGURES / '07_mitm_error_breakdown.png', dpi=160)
    plt.close(fig)
    print('Stage 3C complete. Original test MITM mistakes:')
    print(wrong[['predicted_class', 'n']].sort_values('n', ascending=False).to_string(index=False))
    print('Total missed MITM:', int(wrong.n.sum()))


if __name__ == '__main__':
    run()
