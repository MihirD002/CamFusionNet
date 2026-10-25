#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation metrics for camouflaged object detection.

One numpy implementation of S-measure, E-measure, weighted F-measure and MAE
(following the standard SOD/COD protocol of Fan et al. / Margolin et al.), used
by evaluate_cod.py. Inputs are a prediction map in [0, 1] and a binary ground
truth; torch tensors / PIL images are accepted and converted.

Original author: taozhou. Rewritten so that every metric is implemented once,
runs on CPU, and handles empty/full ground-truth masks.
"""

import os

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

EPS = np.finfo(np.float64).eps
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


###############################################################################
## basic funcs
###############################################################################

def numpy2tensor(numpy):
    """Kept for backward compatibility with older scripts (needs torch + CUDA)."""
    import torch
    return torch.from_numpy(numpy).cuda()


def _to_numpy(x):
    if hasattr(x, 'detach'):            # torch tensor
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return np.squeeze(x)


def prepare_pred(pred):
    """Prediction -> float64 map in [0, 1] (uint8 maps are divided by 255)."""
    pred = _to_numpy(pred)
    if pred.dtype == np.uint8:
        pred = pred.astype(np.float64) / 255.0
    else:
        pred = pred.astype(np.float64)
        if pred.max() > 1.0:
            pred = pred / 255.0
    return np.clip(pred, 0.0, 1.0)


def prepare_gt(gt):
    """Ground truth -> boolean mask (threshold at half of the maximum range)."""
    gt = _to_numpy(gt)
    thr = 127 if gt.dtype == np.uint8 or gt.max() > 1 else 0.5
    return gt > thr


###############################################################################
## MAE
###############################################################################

def eval_mae(pred, gt, cuda=True):
    pred, gt = prepare_pred(pred), prepare_gt(gt)
    return float(np.abs(pred - gt.astype(np.float64)).mean())


###############################################################################
## S-measure (structure measure)
###############################################################################

def _s_object_term(x_vals):
    if x_vals.size == 0:
        return 0.0
    x = x_vals.mean()
    sigma = x_vals.std(ddof=1) if x_vals.size > 1 else 0.0
    return 2.0 * x / (x * x + 1.0 + sigma + EPS)


def fun_S_object(pred, gt):
    o_fg = _s_object_term(pred[gt])
    o_bg = _s_object_term((1.0 - pred)[~gt])
    u = gt.mean()
    return u * o_fg + (1.0 - u) * o_bg


def fun_centroid(gt):
    h, w = gt.shape
    total = gt.sum()
    if total == 0:
        return int(round(w / 2)), int(round(h / 2))
    x = (gt.sum(axis=0) * np.arange(1, w + 1)).sum() / total
    y = (gt.sum(axis=1) * np.arange(1, h + 1)).sum() / total
    return int(np.round(x)), int(np.round(y))


def fun_ssim(pred, gt):
    h, w = gt.shape
    n = h * w
    if n == 0:
        return 0.0
    gt = gt.astype(np.float64)
    x, y = pred.mean(), gt.mean()
    sigma_x2 = ((pred - x) ** 2).sum() / (n - 1 + EPS)
    sigma_y2 = ((gt - y) ** 2).sum() / (n - 1 + EPS)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (n - 1 + EPS)
    alpha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)
    if alpha != 0:
        return alpha / (beta + EPS)
    if alpha == 0 and beta == 0:
        return 1.0
    return 0.0


def fun_S_region(pred, gt):
    h, w = gt.shape
    x, y = fun_centroid(gt)
    area = h * w
    w1, w2, w3 = x * y / area, (w - x) * y / area, x * (h - y) / area
    w4 = 1 - w1 - w2 - w3
    parts = [(slice(0, y), slice(0, x)), (slice(0, y), slice(x, w)),
             (slice(y, h), slice(0, x)), (slice(y, h), slice(x, w))]
    qs = [fun_ssim(pred[r, c], gt[r, c]) for r, c in parts]
    return w1 * qs[0] + w2 * qs[1] + w3 * qs[2] + w4 * qs[3]


def eval_Smeasure(pred, gt, cuda=True):
    pred, gt = prepare_pred(pred), prepare_gt(gt)
    y = gt.mean()
    if y == 0:
        return float(1.0 - pred.mean())
    if y == 1:
        return float(pred.mean())
    q = 0.5 * fun_S_object(pred, gt) + 0.5 * fun_S_region(pred, gt)
    return float(max(q, 0.0))


###############################################################################
## E-measure (enhanced-alignment measure)
###############################################################################

def _em_from_counts(fg_fg, fg_bg, pred_fg, pred_bg, gt_fg, gt_bg, size):
    """E-measure for arrays of per-threshold pixel counts (vectorised over thresholds)."""
    fg_fg, fg_bg = np.asarray(fg_fg, np.float64), np.asarray(fg_bg, np.float64)
    pred_fg, pred_bg = np.asarray(pred_fg, np.float64), np.asarray(pred_bg, np.float64)
    if gt_fg == 0:                       # empty GT: only predicted-background pixels align
        total = pred_bg
    elif gt_fg == size:                  # full GT: only predicted-foreground pixels align
        total = pred_fg
    else:
        bg_fg = gt_fg - fg_fg            # pred bg, gt fg
        bg_bg = gt_bg - fg_bg            # pred bg, gt bg
        mp, mg = pred_fg / size, gt_fg / size
        total = 0.0
        for count, (p, g) in ((fg_fg, (1 - mp, 1 - mg)), (fg_bg, (1 - mp, -mg)),
                              (bg_fg, (-mp, 1 - mg)), (bg_bg, (-mp, -mg))):
            align = 2 * (g * p) / (g * g + p * p + EPS)
            total = total + count * (align + 1) ** 2 / 4
    return total / (size - 1 + EPS)


def eval_emeasure_curve(pred, gt):
    """Per-threshold E-measure (256 thresholds, 255 -> 0) and the adaptive-threshold E-measure."""
    pred, gt = prepare_pred(pred), prepare_gt(gt)
    size = gt.size
    gt_fg = int(gt.sum())
    gt_bg = size - gt_fg
    pred_u8 = np.round(pred * 255).astype(np.uint8)

    hist_fg = np.bincount(pred_u8[gt], minlength=256)
    hist_bg = np.bincount(pred_u8[~gt], minlength=256)
    # counts of pixels with value >= t, for t = 255 ... 0
    fg_fg = np.cumsum(hist_fg[::-1])
    fg_bg = np.cumsum(hist_bg[::-1])
    pred_fg = fg_fg + fg_bg
    pred_bg = size - pred_fg
    curve = _em_from_counts(fg_fg, fg_bg, pred_fg, pred_bg, gt_fg, gt_bg, size)

    thr = min(2 * pred.mean(), 1.0)
    # an all-zero prediction gives thr == 0; `>= 0` would mark every pixel foreground
    binary = pred >= thr if thr > 0 else pred > 0
    a_fg_fg, a_fg_bg = int((binary & gt).sum()), int((binary & ~gt).sum())
    adaptive = _em_from_counts(a_fg_fg, a_fg_bg, a_fg_fg + a_fg_bg, size - a_fg_fg - a_fg_bg,
                               gt_fg, gt_bg, size)
    return curve, float(adaptive)


def eval_emeasure(pred, gt, cuda=True):
    """Single-image E-measure summary: dict with mean / max over thresholds and adaptive."""
    curve, adaptive = eval_emeasure_curve(pred, gt)
    return {'mean': float(curve.mean()), 'max': float(curve.max()), 'adaptive': adaptive}


###############################################################################
## Weighted F-measure (Margolin et al., CVPR 2014)
###############################################################################

def _gaussian_kernel(size=7, sigma=5.0):
    ax = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


_KERNEL = _gaussian_kernel()


def eval_wfmeasure(pred, gt, cuda=True, beta=1.0):
    pred, gt = prepare_pred(pred), prepare_gt(gt)
    if not gt.any():
        return 0.0   # undefined for an empty GT; reported as 0 like the reference implementation
    d_gt = gt.astype(np.float64)
    err = np.abs(pred - d_gt)
    dst, idx = distance_transform_edt(~gt, return_indices=True)
    et = err.copy()
    et[~gt] = et[idx[0][~gt], idx[1][~gt]]
    ea = cv2.filter2D(et, -1, _KERNEL, borderType=cv2.BORDER_REPLICATE)
    min_e_ea = np.where(gt & (ea < err), ea, err)
    b = np.where(~gt, 2.0 - np.exp(np.log(0.5) / 5.0 * dst), 1.0)
    ew = min_e_ea * b
    tpw = d_gt.sum() - ew[gt].sum()
    fpw = ew[~gt].sum()
    r = 1.0 - ew[gt].mean()
    p = tpw / (EPS + tpw + fpw)
    return float((1 + beta ** 2) * r * p / (EPS + r + beta ** 2 * p))


###############################################################################
## dataset-level evaluation
###############################################################################

def _list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


def _read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError("Could not read image: {}".format(path))
    return img


def evaluate_dataset(pred_dir, gt_dir):
    """Evaluate every prediction in pred_dir against the GT with the same file stem.

    Returns a dict with image count and dataset-level S, E (mean/max/adaptive),
    weighted F and MAE. E-measure curves are averaged over images before taking
    the mean/max over thresholds (standard protocol). Missing predictions raise.
    """
    gt_files = {os.path.splitext(f)[0]: f for f in _list_images(gt_dir)}
    pred_files = {os.path.splitext(f)[0]: f for f in _list_images(pred_dir)}
    if not gt_files:
        raise FileNotFoundError("No ground-truth images found in '{}'".format(gt_dir))
    missing = sorted(set(gt_files) - set(pred_files))
    if missing:
        raise FileNotFoundError("{} predictions missing in '{}' (e.g. {})".format(
            len(missing), pred_dir, ', '.join(missing[:3])))

    s_list, wf_list, mae_list, ad_list, curves = [], [], [], [], []
    for stem in sorted(gt_files):
        gt = _read_gray(os.path.join(gt_dir, gt_files[stem]))
        pred = _read_gray(os.path.join(pred_dir, pred_files[stem]))
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        s_list.append(eval_Smeasure(pred, gt))
        wf_list.append(eval_wfmeasure(pred, gt))
        mae_list.append(eval_mae(pred, gt))
        curve, adaptive = eval_emeasure_curve(pred, gt)
        curves.append(curve)
        ad_list.append(adaptive)

    curve = np.mean(curves, axis=0)
    return {
        'num_images': len(gt_files),
        'S': float(np.mean(s_list)),
        'E_mean': float(curve.mean()),
        'E_max': float(curve.max()),
        'E_adaptive': float(np.mean(ad_list)),
        'wF': float(np.mean(wf_list)),
        'MAE': float(np.mean(mae_list)),
    }


class Eval_thread():
    """Thin compatibility wrapper around evaluate_dataset (old interface: run() -> (mae, s))."""
    def __init__(self, pred_dir, gt_dir):
        self.pred_dir, self.gt_dir = pred_dir, gt_dir

    def run(self):
        r = evaluate_dataset(self.pred_dir, self.gt_dir)
        return r['MAE'], r['S']
