"""Checkpoint helpers: associate checkpoints with an MFAM variant and fail loudly on mismatch."""
import json
import os

import torch

from lib.model import MFAM_VARIANTS, validate_mfam_variant


def infer_mfam_variant(state_dict):
    """Infer the MFAM variant from a FAPNet state dict (None if rf1 keys are absent).

    Uses key names and weight shapes of rf1.conv_3_1 / rf1.conv_5_1, which differ
    between variants:
        original   : conv_3_1.weight (3x3), conv_5_1.weight (5x5)
        dilated    : conv_3_1.weight (3x3), conv_5_1.weight (3x3)
        factorized : conv_3_1.weight (3x3), conv_5_1.0.weight (1x5)
        depthwise  : conv_3_1.0.weight (3x3 dw), conv_5_1.0.weight (5x5 dw)
        combined   : conv_3_1.0.weight (3x3 dw), conv_5_1.0.weight (3x3 dw)
    """
    sd = state_dict
    if 'rf1.conv_3_1.weight' in sd:
        if 'rf1.conv_5_1.0.weight' in sd:
            return 'factorized'
        k5 = tuple(sd['rf1.conv_5_1.weight'].shape[-2:])
        return 'original' if k5 == (5, 5) else 'dilated'
    if 'rf1.conv_3_1.0.weight' in sd and 'rf1.conv_5_1.0.weight' in sd:
        k5 = tuple(sd['rf1.conv_5_1.0.weight'].shape[-2:])
        return 'depthwise' if k5 == (5, 5) else 'combined'
    return None


def check_variant_compat(state_dict, requested_variant, source, meta_variant=None):
    """Raise a clear ValueError if the checkpoint was made for a different MFAM variant."""
    validate_mfam_variant(requested_variant)
    inferred = infer_mfam_variant(state_dict)
    found = meta_variant or inferred
    if meta_variant and inferred and meta_variant != inferred:
        raise ValueError(
            "Checkpoint '{}' is inconsistent: metadata says MFAM variant '{}' but its weights look like '{}'."
            .format(source, meta_variant, inferred))
    if found is not None and found != requested_variant:
        raise ValueError(
            "MFAM variant mismatch for '{}': checkpoint was trained with '{}' but '{}' was requested. "
            "Re-run with the matching variant (--mfam_variant {} / MFAM_VARIANT={})."
            .format(source, found, requested_variant, found, found))


def extract_state_dict(checkpoint_data):
    """Return (state_dict, meta) for either a full checkpoint dict or a raw state dict."""
    if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
        return checkpoint_data['model_state_dict'], checkpoint_data
    return checkpoint_data, {}


def load_model_weights(model, path, variant, device='cpu'):
    """Load weights from `path` into `model`, verifying the MFAM variant first.

    Returns the checkpoint metadata dict ({} for raw state dicts).
    """
    data = torch.load(path, map_location=device)
    state_dict, meta = extract_state_dict(data)
    meta_variant = meta.get('mfam_variant') if isinstance(meta, dict) else None
    if meta_variant is None:
        meta_variant = read_sidecar_variant(path)
    check_variant_compat(state_dict, variant, path, meta_variant)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to load '{}' into FAPNet(mfam_variant='{}'): {}".format(path, variant, e)) from e
    return meta


###############################################################################
# Sidecar metadata for raw state-dict snapshots (FAPNet_<epoch>.pth, FAPNet_best.pth)
###############################################################################
META_NAME = 'run_config.json'


def write_run_config(save_dir, variant, config):
    """Write run_config.json next to the checkpoints (variant + full experiment configuration)."""
    os.makedirs(save_dir, exist_ok=True)
    payload = {'mfam_variant': variant, 'config': config}
    with open(os.path.join(save_dir, META_NAME), 'w') as f:
        json.dump(payload, f, indent=2, default=str)


def read_sidecar_variant(path):
    meta_path = os.path.join(os.path.dirname(os.path.abspath(path)), META_NAME)
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path) as f:
            v = json.load(f).get('mfam_variant')
        return v if v in MFAM_VARIANTS else None
    except Exception:
        return None
