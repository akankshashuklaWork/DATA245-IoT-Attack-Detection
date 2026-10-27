"""STAGE 3A: Train ONLY the baseline Decision Tree and kNN on the training split.

Run: python src/02_train_models.py
Output: models/*.joblib and results/training_times.csv
Does NOT calculate validation/test metrics; see 03_model_evaluation.py.
"""
import time
import joblib
import pandas as pd
from common import RESULTS, MODELS, FEATURES, TARGET, folders, clean_data, numeric_inputs, load_indices, build_models


def run():
    folders()
    _, clean = clean_data()
    split = load_indices(clean)
    x = numeric_inputs(clean)
    y = clean[TARGET].astype(str)
    rows = []
    for name, model in build_models(n_jobs=1).items():
        print(f'Training {name}: {len(split["train"]):,} rows, {len(FEATURES)} features...', flush=True)
        start = time.perf_counter()
        model.fit(x.iloc[split['train']], y.iloc[split['train']])
        seconds = time.perf_counter() - start
        joblib.dump(model, MODELS / f'{name}_five_features.joblib', compress=3)
        rows.append({'model': name, 'train_rows': len(split['train']), 'training_seconds': seconds,
                     'n_model_features': len(FEATURES)})
        print(f'  Model saved; training time {seconds:.2f} seconds', flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / 'training_times.csv', index=False)
    print('Stage 3A complete. Now run 03_model_evaluation.py.')


if __name__ == '__main__':
    run()
