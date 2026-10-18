# Boundary and Uncertainty COD Study (append to README.md)

**Backbone note:** the encoder in `lib/model.py` is **ConvNeXt-Tiny** (`torchvision.models.convnext_tiny(weights=None)`),
not Res2Net. Results are *FAP-Net-style* and are not an exact paper reproduction.

## Flags (`train.py`)
| Flag | Values | Hypothesis |
|---|---|---|
| `--boundary_mode` | `file` (legacy edge maps, control), `morph` (derived from transformed GT) | H1 |
| `--boundary_kernel_size` | e.g. 3, 5 | H1 |
| `--lambda_b` | boundary loss weight (sweep 0.5 / 1 / 2) | H1 |
| `--boundary_head` | `single` (control), `multiscale` | H2 |
| `--uncertainty` | `none`, `seg`, `seg_boundary` | H3 |
| `--lambda_u` | log-variance regulariser weight | H3 |
| `--seed`, `--exp_name`, `--config` | reproducibility / per-experiment folders / JSON defaults | all |

Each run writes to `Snapshot/FAPNet/<exp_name>/` with `config.json`, `log.log` and checkpoints (the config is also
stored inside the checkpoints, so `test.py`/`server.py` rebuild the right architecture automatically).

## Run
```bash
python train.py --config experiments/h1_morph_k3.json
python test.py  --model_path Snapshot/FAPNet/h1_morph_k3/FAPNet_best.pth --save_path results/h1_morph_k3 --save_aux
python evaluate_cod.py --pred_root results/h1_morph_k3 --out results/h1_morph_k3/metrics.json
```

## Output contract
`model(x)` returns `(out5, out4, out3, out2, edge_out)`; with uncertainty enabled a 6th element `logvar` is appended.
Index 3 is always the final segmentation logit map. Old checkpoints load with `strict=True` in baseline mode; new heads
only exist when their flag is set.

## Notes
- Training-time mask resizing (including the 0.75x/1.25x multi-scale steps) uses nearest-neighbour; rotation of masks too.
- `--legacy_wbce 1` reproduces the original `reduce='none'` BCE quirk. Re-run the baseline with the updated code so
  every ablation shares the same loss implementation.
- In `morph` mode the boundary is derived before the optional GT salt-and-pepper noise (`--gt_noise`).
- H3 configs in `experiments/` are built on the H2 setting; edit them to whichever H1/H2 configuration wins.
- Requires `opencv-python` and `scipy` (used by the dataloader, `test.py`, `evaluate_cod.py`).
