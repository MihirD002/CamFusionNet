"""Evaluate saved prediction maps on CHAMELEON / CAMO / COD10K with one metric implementation.

Usage:
    python evaluate_cod.py --pred_root ./results --label original
    python evaluate_cod.py --pred_root ./results_depthwise --label depthwise --output metrics_depthwise.json

Layout expected (produced by test.py):
    <pred_root>/<DATASET>/<name>.png
    <gt_root>/<DATASET>/GT/<name>.png
Metrics: S-measure (S), mean/max/adaptive E-measure, weighted F-measure (wF), MAE.
Results are reported separately for each dataset.
"""
import argparse
import csv
import json
import os

from utils.eva_funcs import evaluate_dataset

COLUMNS = ['S', 'E_mean', 'E_max', 'E_adaptive', 'wF', 'MAE']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_root', type=str, required=True)
    parser.add_argument('--gt_root', type=str, default=os.path.join('data', 'TestDataset', 'TestDataset'))
    parser.add_argument('--datasets', type=str, nargs='+', default=['CAMO', 'COD10K', 'CHAMELEON'])
    parser.add_argument('--label', type=str, default=None, help='name stored with the results (e.g. the MFAM variant)')
    parser.add_argument('--output', type=str, default=None, help='write results to this .json (and .csv next to it)')
    opt = parser.parse_args()

    results = {}
    for ds in opt.datasets:
        results[ds] = evaluate_dataset(os.path.join(opt.pred_root, ds), os.path.join(opt.gt_root, ds, 'GT'))

    header = '{:<10} {:>6} '.format('dataset', 'N') + ' '.join('{:>10}'.format(c) for c in COLUMNS)
    print('Label: {}'.format(opt.label or opt.pred_root))
    print(header)
    for ds, r in results.items():
        print('{:<10} {:>6d} '.format(ds, r['num_images']) + ' '.join('{:>10.4f}'.format(r[c]) for c in COLUMNS))

    if opt.output:
        with open(opt.output, 'w') as f:
            json.dump({'label': opt.label, 'pred_root': opt.pred_root, 'results': results}, f, indent=2)
        with open(os.path.splitext(opt.output)[0] + '.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['label', 'dataset', 'num_images'] + COLUMNS)
            for ds, r in results.items():
                w.writerow([opt.label, ds, r['num_images']] + [round(r[c], 6) for c in COLUMNS])


if __name__ == '__main__':
    main()
