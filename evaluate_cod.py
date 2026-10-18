#!/usr/bin/env python
"""
Unified COD evaluator: S-measure, weighted F-measure, mean E-measure, MAE,
plus boundary F-score, reported overall and per GT-area stratum.

Usage:
  python evaluate_cod.py --pred_root results/baseline \
      --data_root data/TestDataset/TestDataset \
      --datasets CHAMELEON CAMO COD10K --out results/baseline/metrics.json

Predictions are expected at <pred_root>/<dataset>/<image_stem>.png (what test.py writes).
Sanity-check these implementations against a reference (e.g. PySODMetrics) on a
small fixed sample before trusting absolute numbers.
"""
import os
import json
import argparse

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

EPS = np.finfo(np.float64).eps
AREA_BINS = [('small', 0.0, 0.05), ('medium', 0.05, 0.15), ('large', 0.15, 1.01)]


# ---------------------------------------------------------------- S-measure
def _s_object_part(pred, gt):
    x = pred[gt]
    if x.size == 0:
        return 0.0
    mu = x.mean()
    sd = x.std(ddof=1) if x.size > 1 else 0.0
    return 2.0 * mu / (mu ** 2 + 1.0 + sd + EPS)


def s_object(pred, gt):
    fg = pred * gt
    bg = (1.0 - pred) * (~gt)
    u = gt.mean()
    return u * _s_object_part(fg, gt) + (1 - u) * _s_object_part(bg, ~gt)


def _ssim(pred, gt):
    h, w = pred.shape
    n = h * w
    x, y = pred.mean(), gt.mean()
    sx2 = ((pred - x) ** 2).sum() / (n - 1 + EPS)
    sy2 = ((gt - y) ** 2).sum() / (n - 1 + EPS)
    sxy = ((pred - x) * (gt - y)).sum() / (n - 1 + EPS)
    alpha = 4 * x * y * sxy
    beta = (x ** 2 + y ** 2) * (sx2 + sy2)
    if alpha != 0:
        return alpha / (beta + EPS)
    return 1.0 if beta == 0 else 0.0


def s_region(pred, gt):
    h, w = gt.shape
    if gt.sum() == 0:
        X, Y = w // 2, h // 2
    else:
        ys, xs = np.nonzero(gt)
        X, Y = int(round(xs.mean())), int(round(ys.mean()))
    X = max(1, min(w - 1, X))
    Y = max(1, min(h - 1, Y))
    g = gt.astype(np.float64)
    area = float(h * w)
    parts = [
        (pred[:Y, :X], g[:Y, :X], X * Y / area),
        (pred[:Y, X:], g[:Y, X:], (w - X) * Y / area),
        (pred[Y:, :X], g[Y:, :X], X * (h - Y) / area),
        (pred[Y:, X:], g[Y:, X:], (w - X) * (h - Y) / area),
    ]
    return sum(wt * _ssim(p, q) for p, q, wt in parts)


def s_measure(pred, gt):
    y = gt.mean()
    if y == 0:
        return 1.0 - pred.mean()
    if y == 1:
        return pred.mean()
    return max(0.0, 0.5 * s_object(pred, gt) + 0.5 * s_region(pred, gt))


# ------------------------------------------------------- weighted F-measure
_K = cv2.getGaussianKernel(7, 5)
_K = _K @ _K.T


def weighted_fmeasure(pred, gt):
    if gt.sum() == 0:
        return 0.0
    dst, idx = distance_transform_edt(~gt, return_indices=True)
    err = np.abs(pred - gt.astype(np.float64))
    et = err.copy()
    et[~gt] = err[idx[0][~gt], idx[1][~gt]]
    ea = cv2.filter2D(et, -1, _K, borderType=cv2.BORDER_REPLICATE)
    min_e_ea = np.where(gt & (ea < err), ea, err)
    b = np.where(~gt, 2.0 - np.exp(np.log(0.5) / 5 * dst), 1.0)
    ew = min_e_ea * b
    tpw = gt.sum() - ew[gt].sum()
    fpw = ew[~gt].sum()
    r = 1 - ew[gt].mean()
    p = tpw / (EPS + tpw + fpw)
    return 2 * r * p / (EPS + r + p)


# ---------------------------------------------- E-measure (mean over thresholds)
def e_measure(pred_u8, gt):
    gt_size = gt.size
    gt_fg = int(gt.sum())

    fg_fg_hist, _ = np.histogram(pred_u8[gt], bins=256, range=(0, 255))
    fg_bg_hist, _ = np.histogram(pred_u8[~gt], bins=256, range=(0, 255))
    fg_fg = np.cumsum(np.flip(fg_fg_hist)).astype(np.float64)
    fg_bg = np.cumsum(np.flip(fg_bg_hist)).astype(np.float64)
    pred_fg = fg_fg + fg_bg
    pred_bg = gt_size - pred_fg

    if gt_fg == 0:
        total = pred_bg
    elif gt_fg == gt_size:
        total = pred_fg
    else:
        bg_fg = gt_fg - fg_fg
        bg_bg = pred_bg - bg_fg
        parts = [fg_fg, fg_bg, bg_fg, bg_bg]
        mean_pred = pred_fg / gt_size
        mean_gt = gt_fg / gt_size
        dp_fg, dp_bg = 1 - mean_pred, 0 - mean_pred
        dg_fg, dg_bg = 1 - mean_gt, 0 - mean_gt
        combos = [(dp_fg, dg_fg), (dp_fg, dg_bg), (dp_bg, dg_fg), (dp_bg, dg_bg)]
        total = np.zeros(256)
        for numel, (a, g) in zip(parts, combos):
            align = 2 * (a * g) / (a ** 2 + g ** 2 + EPS)
            total += ((align + 1) ** 2 / 4) * numel
    em = total / (gt_size - 1 + EPS)
    return float(em.mean())


# ------------------------------------------------------------------- others
def mae(pred, gt):
    return float(np.abs(pred - gt.astype(np.float64)).mean())


def _boundary(mask_bool, k=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask_bool.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0


def boundary_f(pred_u8, gt, tol=2):
    """Boundary F1 with a +/-tol pixel tolerance band; pred binarised at 128."""
    pb = _boundary(pred_u8 >= 128)
    gb = _boundary(gt)
    if gb.sum() == 0 and pb.sum() == 0:
        return 1.0
    if gb.sum() == 0 or pb.sum() == 0:
        return 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    gd = cv2.dilate(gb.astype(np.uint8), kernel) > 0
    pd = cv2.dilate(pb.astype(np.uint8), kernel) > 0
    prec = (pb & gd).sum() / pb.sum()
    rec = (gb & pd).sum() / gb.sum()
    return float(2 * prec * rec / (prec + rec + EPS))


# --------------------------------------------------------------------- main
METRICS = ['Sm', 'wFm', 'Em', 'MAE', 'BF']


def evaluate_dataset(pred_dir, gt_dir):
    rows = []
    gt_files = sorted(f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.tif')))
    for f in gt_files:
        stem = os.path.splitext(f)[0]
        pp = os.path.join(pred_dir, stem + '.png')
        if not os.path.exists(pp):
            print(f'  [warn] missing prediction for {stem}')
            continue
        gt_img = cv2.imread(os.path.join(gt_dir, f), cv2.IMREAD_GRAYSCALE)
        pr_img = cv2.imread(pp, cv2.IMREAD_GRAYSCALE)
        if pr_img.shape != gt_img.shape:
            pr_img = cv2.resize(pr_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        gt = gt_img > 128
        pred = pr_img.astype(np.float64) / 255.0
        rows.append({
            'name': stem,
            'area': float(gt.mean()),
            'Sm': s_measure(pred, gt),
            'wFm': weighted_fmeasure(pred, gt),
            'Em': e_measure(pr_img, gt),
            'MAE': mae(pred, gt),
            'BF': boundary_f(pr_img, gt),
        })
    return rows


def summarise(rows):
    out = {'n': len(rows)}
    for m in METRICS:
        out[m] = float(np.mean([r[m] for r in rows])) if rows else float('nan')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_root', required=True)
    ap.add_argument('--data_root', default=os.path.join('data', 'TestDataset', 'TestDataset'))
    ap.add_argument('--datasets', nargs='+', default=['CHAMELEON', 'CAMO', 'COD10K'])
    ap.add_argument('--out', default=None, help='write full results (incl. per-image) as JSON')
    opt = ap.parse_args()

    results = {}
    for ds in opt.datasets:
        rows = evaluate_dataset(os.path.join(opt.pred_root, ds), os.path.join(opt.data_root, ds, 'GT'))
        res = {'overall': summarise(rows), 'strata': {}, 'per_image': rows}
        for name, lo, hi in AREA_BINS:
            res['strata'][f'area_{name}'] = summarise([r for r in rows if lo <= r['area'] < hi])
        results[ds] = res

        print(f'\n=== {ds} ===')
        print(f"{'subset':<14}{'n':>5}" + ''.join(f'{m:>9}' for m in METRICS))
        for label, s in [('overall', res['overall'])] + list(res['strata'].items()):
            print(f"{label:<14}{s['n']:>5}" + ''.join(f'{s[m]:>9.4f}' for m in METRICS))

    if opt.out:
        os.makedirs(os.path.dirname(os.path.abspath(opt.out)), exist_ok=True)
        with open(opt.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nSaved {opt.out}')


if __name__ == '__main__':
    main()
