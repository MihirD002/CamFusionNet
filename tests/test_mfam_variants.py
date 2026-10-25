"""Smoke / compatibility tests for the MFAM variants. Run: python -m pytest tests -q  (CPU is enough)."""
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.model import FAPNet, MFAM, MFAM_VARIANTS
from utils.checkpoint import infer_mfam_variant, load_model_weights, check_variant_compat, write_run_config

CHANNELS = {'rf1': (96, 4), 'rf2': (96, 4), 'rf3': (192, 8), 'rf4': (384, 16), 'rf5': (768, 32)}


@pytest.mark.parametrize('variant', MFAM_VARIANTS)
def test_mfam_preserves_shape(variant):
    m = MFAM(96, 64, variant).eval()
    out = m(torch.randn(2, 96, 16, 16))
    assert out.shape == (2, 64, 16, 16)


@pytest.mark.parametrize('variant', MFAM_VARIANTS)
def test_fapnet_contract_and_gradients(variant):
    torch.manual_seed(0)
    net = FAPNet(channel=64, mfam_variant=variant)
    net.train()
    x = torch.randn(2, 3, 64, 64)
    outs = net(x)
    assert len(outs) == 5
    for o in outs:
        assert o.shape == (2, 1, 64, 64)
        assert torch.isfinite(o).all()
    loss = sum(o.mean() for o in outs)
    loss.backward()
    for name, p in net.named_parameters():
        if name.startswith('backbone.features') or name.startswith('rf'):
            assert p.grad is None or torch.isfinite(p.grad).all(), name
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.rf1.parameters())
    assert net.mfam_variant == variant
    for rf in ('rf1', 'rf2', 'rf3', 'rf4', 'rf5'):
        assert getattr(net, rf).variant == variant


def test_invalid_variant_rejected():
    with pytest.raises(ValueError):
        FAPNet(mfam_variant='nope')
    with pytest.raises(ValueError):
        MFAM(96, 64, 'nope')


def test_efficient_variants_have_fewer_mfam_params():
    def n(v):
        net = FAPNet(mfam_variant=v)
        return sum(p.numel() for r in CHANNELS for p in getattr(net, r).parameters())
    base = n('original')
    for v in ('depthwise', 'dilated', 'factorized', 'combined'):
        assert n(v) < base, v


def test_original_state_dict_layout_unchanged():
    keys = set(FAPNet().state_dict().keys())
    for k in ('rf1.conv_3_1.weight', 'rf1.conv_3_1.bias', 'rf1.conv_5_1.weight', 'rf5.conv_5_2.bias'):
        assert k in keys
    assert tuple(FAPNet().state_dict()['rf1.conv_5_1.weight'].shape[-2:]) == (5, 5)


@pytest.mark.parametrize('variant', MFAM_VARIANTS)
def test_variant_inference_from_weights(variant):
    assert infer_mfam_variant(FAPNet(mfam_variant=variant).state_dict()) == variant


@pytest.mark.parametrize('variant', MFAM_VARIANTS)
def test_roundtrip_and_mismatch_error(variant):
    with tempfile.TemporaryDirectory() as d:
        net = FAPNet(mfam_variant=variant)
        raw = os.path.join(d, 'FAPNet_best.pth')
        torch.save(net.state_dict(), raw)
        # matching variant loads (raw state dict, variant inferred from weights)
        load_model_weights(FAPNet(mfam_variant=variant), raw, variant)
        # wrong variant -> actionable error naming both variants
        other = next(v for v in MFAM_VARIANTS if v != variant)
        with pytest.raises(ValueError, match='mismatch'):
            load_model_weights(FAPNet(mfam_variant=other), raw, other)
        # full checkpoint with metadata
        full = os.path.join(d, 'checkpoint_best.pth')
        torch.save({'model_state_dict': net.state_dict(), 'mfam_variant': variant, 'epoch': 3}, full)
        meta = load_model_weights(FAPNet(mfam_variant=variant), full, variant)
        assert meta['epoch'] == 3


def test_sidecar_metadata_used_for_raw_checkpoints():
    with tempfile.TemporaryDirectory() as d:
        write_run_config(d, 'dilated', {'seed': 1})
        net = FAPNet(mfam_variant='original')
        p = os.path.join(d, 'FAPNet_5.pth')
        torch.save(net.state_dict(), p)
        with pytest.raises(ValueError, match='dilated'):
            load_model_weights(FAPNet(mfam_variant='original'), p, 'original')


def test_default_constructor_loads_original_checkpoint():
    sd = FAPNet(channel=64).state_dict()          # what an existing baseline checkpoint looks like
    net = FAPNet(channel=64)
    net.load_state_dict(sd)
    check_variant_compat(sd, 'original', 'in-memory')
