"""Efficiency benchmark for the MFAM variants of FAP-Net.

Reports, per variant and under identical conditions (same device, 352x352 input,
batch size 1, warm-up, synchronisation, no_grad, eval mode):
  * total and MFAM-only (rf1..rf5) parameter counts
  * FLOPs at 352x352 (total and MFAM-only), counted with torch.utils.flop_counter
    (FLOPs = 2 x MACs; counts conv/linear/matmul ops, not BN/activations/upsampling)
  * latency (mean / median / std, ms) and FPS
  * peak inference GPU memory (torch.cuda.max_memory_allocated)

GPU-only metrics (peak memory, CUDA-event timing) print 'n/a' when no GPU is used.
Weights are randomly initialised (backbone weights do not affect cost).

Usage:
    python benchmark_model.py                       # all variants
    python benchmark_model.py --variants original depthwise --output bench.json
    python benchmark_model.py --device cpu          # latency measured on CPU (flagged)
"""
import argparse
import csv
import json
import platform
import statistics
import time

import torch
import torch.nn as nn

from lib.model import FAPNet, MFAM_VARIANTS

# input channels / spatial stride of each MFAM instance inside FAPNet
MFAM_INPUTS = {'rf1': (96, 4), 'rf2': (96, 4), 'rf3': (192, 8), 'rf4': (384, 16), 'rf5': (768, 32)}


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def mfam_params(model):
    return sum(count_params(getattr(model, n)) for n in MFAM_INPUTS)


def _hook_flops(module, *inputs):
    """Fallback for torch < 2.1: FLOPs = 2 x MACs of Conv2d and Linear layers."""
    total = [0]
    def conv_hook(m, inp, out):
        kh, kw = m.kernel_size
        total[0] += out.numel() * (m.in_channels // m.groups) * kh * kw
    def linear_hook(m, inp, out):
        total[0] += out.numel() * m.in_features
    handles = []
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(linear_hook))
    try:
        with torch.no_grad():
            module(*inputs)
    finally:
        for h in handles:
            h.remove()
    return 2 * total[0]


def _count_flops(module, *inputs):
    try:
        import importlib
        flop_counter = importlib.import_module("torch.utils.flop_counter")
        FlopCounterMode = flop_counter.FlopCounterMode
    except (ImportError, AttributeError):
        return _hook_flops(module, *inputs)
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            module(*inputs)
    return fc.get_total_flops()


def flops_total(model, size, device):
    return _count_flops(model, torch.randn(1, 3, size, size, device=device))


def flops_mfam(model, size, device):
    total = 0
    for name, (ch, stride) in MFAM_INPUTS.items():
        x = torch.randn(1, ch, size // stride, size // stride, device=device)
        total += _count_flops(getattr(model, name), x)
    return total


def measure_latency(model, size, device, warmup, iters):
    x = torch.randn(1, 3, size, size, device=device)
    use_cuda = device.type == 'cuda'
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if use_cuda:
            torch.cuda.synchronize(device)
        for _ in range(iters):
            if use_cuda:
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                model(x)
                e.record()
                torch.cuda.synchronize(device)
                times.append(s.elapsed_time(e))
            else:
                t0 = time.perf_counter()
                model(x)
                times.append((time.perf_counter() - t0) * 1000)
    mean = statistics.mean(times)
    return {'latency_ms_mean': mean, 'latency_ms_median': statistics.median(times),
            'latency_ms_std': statistics.pstdev(times), 'fps': 1000.0 / mean}


def benchmark_variant(variant, size, device, warmup, iters, seed):
    torch.manual_seed(seed)
    model = FAPNet(channel=64, mfam_variant=variant).to(device).eval()
    res = {'variant': variant, 'params_total': count_params(model), 'params_mfam': mfam_params(model)}
    try:
        res['flops_total'] = flops_total(model, size, device)
        res['flops_mfam'] = flops_mfam(model, size, device)
    except Exception as e:                       # e.g. torch < 2.1 without FlopCounterMode
        res['flops_total'] = res['flops_mfam'] = None
        res['flops_error'] = str(e)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    res.update(measure_latency(model, size, device, warmup, iters))
    res['peak_gpu_mem_mb'] = (torch.cuda.max_memory_allocated(device) / 1024 ** 2) if device.type == 'cuda' else None
    del model
    return res


def fmt(v, scale=1.0, nd=2):
    return 'n/a' if v is None else '{:.{nd}f}'.format(v / scale, nd=nd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', default=list(MFAM_VARIANTS), choices=MFAM_VARIANTS)
    parser.add_argument('--size', type=int, default=352)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=str, default=None, help='write results to .json and .csv')
    opt = parser.parse_args()

    device = torch.device(opt.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but not available; rerun with --device cpu (GPU-only metrics will be n/a).')
    hw = {'device': str(device), 'torch': torch.__version__, 'platform': platform.platform()}
    if device.type == 'cuda':
        hw['gpu'] = torch.cuda.get_device_name(device)
    else:
        print('NOTE: running on CPU. Peak GPU memory is n/a and latency is CPU latency, not GPU latency.')
    print('Hardware/runtime:', json.dumps(hw))

    rows = [benchmark_variant(v, opt.size, device, opt.warmup, opt.iters, opt.seed) for v in opt.variants]

    print('\n{:<11} {:>10} {:>11} {:>11} {:>11} {:>10} {:>8} {:>10}'.format(
        'variant', 'Params(M)', 'MFAM P(M)', 'GFLOPs', 'MFAM GFLOPs', 'Lat(ms)', 'FPS', 'PeakMB'))
    for r in rows:
        print('{:<11} {:>10} {:>11} {:>11} {:>11} {:>10} {:>8} {:>10}'.format(
            r['variant'], fmt(r['params_total'], 1e6, 3), fmt(r['params_mfam'], 1e6, 3),
            fmt(r['flops_total'], 1e9), fmt(r['flops_mfam'], 1e9, 3),
            fmt(r['latency_ms_mean']), fmt(r['fps'], 1, 1), fmt(r['peak_gpu_mem_mb'], 1, 0)))

    if opt.output:
        with open(opt.output, 'w') as f:
            json.dump({'hardware': hw, 'input_size': opt.size, 'warmup': opt.warmup, 'iters': opt.iters,
                       'results': rows}, f, indent=2)
        with open(opt.output.rsplit('.', 1)[0] + '.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()
