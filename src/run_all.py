"""Run each step IN ORDER, optionally including 10 repeated-split experiments.

python src/run_all.py
python src/run_all.py --include-repeated
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--include-repeated', action='store_true')
    args = p.parse_args()
    scripts = ['01_eda_preprocessing.py', '02_train_models.py',
               '03_model_evaluation.py', '04_classification_errors.py']
    if args.include_repeated:
        scripts.append('05_repeated_split_experiments.py')
    scripts.append('make_presentation_charts.py')
    for script in scripts:
        print(f'\n--- {script} ---', flush=True)
        subprocess.run([sys.executable, str(SRC / script)], cwd=ROOT, check=True)
    print('\nAll selected steps finished.')


if __name__ == '__main__':
    main()
