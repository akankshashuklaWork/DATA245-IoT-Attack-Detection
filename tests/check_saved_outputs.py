"""Check supplied data and saved outputs against midterm figures.

Run from project root: python tests/check_saved_outputs.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from common import ROOT, FEATURES, clean_data, feature_group_hashes, load_indices


def main():
    raw, clean = clean_data()
    assert raw.shape == (211_043, 44), raw.shape
    assert len(clean) == 190_474, len(clean)
    assert raw['type'].eq('mitm').sum() == 1043
    assert clean['type'].eq('mitm').sum() == 1041
    assert len(raw) - len(clean) == 20_569
    assert set(FEATURES).issubset(raw.columns)
    ix = load_indices(clean)
    assert [len(ix[key]) for key in ('train','validation','test')] == [140812,24831,24831]

    metrics = pd.read_csv(ROOT / 'results' / 'baseline_metrics.csv')
    dt = metrics.query('model=="Decision_Tree" and split=="test"').iloc[0]
    knn = metrics.query('model=="kNN" and split=="test"').iloc[0]
    assert round(dt.accuracy * 100, 1) == 93.2
    assert round(knn.accuracy * 100, 1) == 89.1
    assert round(dt.mitm_recall * 100, 1) == 69.8
    assert round(knn.mitm_recall * 100, 1) == 65.8
    predictions = pd.read_csv(ROOT / 'results' / 'predictions_val_test.csv')
    dt_test = predictions.query('model=="Decision_Tree" and split=="test"')
    knn_test = predictions.query('model=="kNN" and split=="test"')
    assert dt_test.actual.eq('mitm').sum() == 149
    assert (dt_test.actual.eq('mitm') & dt_test.predicted.eq('mitm')).sum() == 104
    assert (knn_test.actual.eq('mitm') & knn_test.predicted.eq('mitm')).sum() == 98
    diagnostic = pd.read_csv(ROOT / 'diagnostics' / 'repeated_per_class_scores.csv')
    assert diagnostic['seed'].nunique() == 10
    assert set(diagnostic['model']) == {'Decision Tree','kNN'}
    expected = ['slide07_class_distribution.png','slide08_pattern_diversity.png',
                'slide10_baseline_comparison.png','slide11_mitm_errors.png',
                'slide11_repeated_split_mean_recall.png']
    for filename in expected:
        assert (ROOT / 'presentation_charts' / filename).exists(), filename
    print('PASS: dataset, exact deduplication, group-disjoint split, five-feature baselines,')
    print('      original MITM counts, repeated splits, and five presentation charts.')


if __name__ == '__main__':
    main()
