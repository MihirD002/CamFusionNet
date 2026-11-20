# CamFusionNet

CamFusionNet is an FAP-Net-based implementation for Camouflaged Object Detection (COD). It provides multiple Multi-scale Feature Aggregation Module (MFAM) variants, training and testing scripts, evaluation utilities, efficiency benchmarks, checkpoint compatibility checks, and a React/Vite visualization interface.

**Developers**

- Ajaymohan Kanojia
- Mihir Deshmukh

> **Research lineage:** The implementation builds on the IEEE TIP 2022 FAP-Net paper by Tao Zhou, Yi Zhou, Chen Gong, Jian Yang, and Yu Zhang.

## Overview

Camouflaged object detection identifies objects that blend visually with their surroundings. This project reproduces the FAP-Net framework and modifies the MFAM spatial branch operations to compare three variants:

| Variant | Branch design |
|---|---|
| `original` | Standard 3 × 3 and 5 × 5 spatial convolutions |
| `depthwise` | Depthwise-separable 3 × 3 and 5 × 5 convolutions |
| `dilated` | Standard 3 × 3 branches and 3 × 3 dilated branches |

All variants preserve the remaining FAP-Net topology and return the five-model-output contract used by the training and inference code.

## Implementation Note

The checked-in implementation uses a **ConvNeXt-Tiny backbone** from `torchvision.models` rather than the original Res2Net-50 backbone. The [model implementation](lib/model.py) documents this change explicitly. Backbones and checkpoints are therefore not interchangeable with the original FAP-Net model without matching the architecture.

## Architecture

- **Boundary Guidance Module (BGM):** extracts boundary-enhanced features and supervises edge prediction.
- **Multi-scale Feature Aggregation Module (MFAM):** combines complementary spatial representations while preserving residual feature information.
- **Cross-level Fusion and Propagation Module (CFPM):** fuses adjacent feature levels and uses gated propagation to transfer encoder context to the decoder.

<p align="center">
  <img src="imgs/fapnet.png" alt="FAP-Net architecture" width="850"><br>
  <em>Overall FAP-Net architecture.</em>
</p>

<p align="center">
  <img src="imgs/results.png" alt="Qualitative segmentation results" width="850"><br>
  <em>Qualitative prediction results.</em>
</p>

## Repository Structure

```text
FAPNet/
├── lib/
│   ├── model.py                 # FAPNet architecture and MFAM variants
│   └── Res2Net_v1b.py           # Original/legacy backbone implementation
├── utils/
│   ├── checkpoint.py            # Checkpoint loading and variant validation
│   ├── dataloader.py            # Training and testing data loaders
│   ├── eva_funcs.py             # S/E/wF/MAE metric implementation
│   └── trainer.py               # Learning-rate utilities
├── tests/
│   └── test_mfam_variants.py    # Variant shape, compatibility, and gradient tests
├── ui/
│   ├── src/                     # React visualization components
│   ├── public/samples/          # Bundled demonstration images
│   ├── package.json
│   └── vite.config.js           # Vite server and /api proxy
├── imgs/                        # README architecture and result images
├── Snapshot/                    # Generated training logs and snapshots
├── train.py                     # Model training and resuming
├── test.py                      # Dataset prediction generation
├── evaluate_cod.py              # Python metric evaluation
├── benchmark_model.py           # Parameter, FLOP, latency, and memory benchmark
├── server.py                    # Local HTTP inference server
├── build_full_paper.py          # Research-paper generation utility
├── FAPNet_Research_Paper_IEEE_TAI.docx
├── benchmark and metrics files
├── requirements.txt             # Data, vision, and numerical dependencies
├── requirement.txt              # Original historical pins (not recommended)
└── SETUP_NOTES.md               # Environment and known setup notes
```

Directories such as `data/`, `checkpoints/`, `results/`, `Snapshot/`, `venv/`, and `ui/node_modules/` are generated or locally supplied and are excluded by `.gitignore`.

## Environment Setup

The current setup is tested on **Windows, Python 3.9, PyTorch 1.13.1, CUDA 11.7, and an NVIDIA GPU**. CPU execution is possible for model tests and benchmarking, but training and live inference are designed for CUDA.

Create a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install a matching PyTorch/torchvision build first:

```powershell
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

For systems without a compatible NVIDIA GPU, use the CPU build supported by the installed PyTorch release instead.

## Dataset Layout

Download the training and test datasets separately and arrange them as follows:

```text
data/
└── TestDataset/
    └── TestDataset/
        ├── CAMO/
        │   ├── Imgs/
        │   └── GT/
        ├── CHAMELEON/
        │   ├── Imgs/
        │   └── GT/
        └── COD10K/
            ├── Imgs/
            └── GT/
```

Training is configured for this layout:

```text
data/TrainDataset/Imgs/
data/TrainDataset/GT/
data/TrainDataset/Edge/
```

The script also recognizes the nested `data/TrainDataset/TrainDataset/...` layout. Edge maps are required for the training loss. If the downloaded training dataset does not include `Edge/`, generate edge maps from the ground-truth masks before training.

The repository does not include datasets or model checkpoints.

## Training

Train the original MFAM variant:

```powershell
python train.py --epoch 200 --batchsize 32 --trainsize 352 --lr 0.0001 --mfam_variant original --seed 42
```

Train another variant:

```powershell
python train.py --epoch 200 --batchsize 32 --trainsize 352 --mfam_variant depthwise --seed 42
```

Supported variants:

```text
original
depthwise
dilated
```

Snapshots are written to:

- `Snapshot/FAPNet/` for `original`
- `Snapshot/FAPNet_<variant>/` for other variants

Useful training options include `--resume`, `--start_epoch`, `--amp`, `--num_workers`, `--save_model`, `--train_img_dir`, `--train_gt_dir`, and `--train_eg_dir`. Run `python train.py --help` for the complete list.

## Testing

Generate prediction maps:

```powershell
python test.py --model_path .\Snapshot\FAPNet\FAPNet_best.pth --mfam_variant original --testsize 352 --save_path .\results
```

The default data root is `data/TestDataset/TestDataset`, and the default datasets are `CHAMELEON`, `CAMO`, and `COD10K`. Override them when needed:

```powershell
python test.py --data_root .\data\TestDataset\TestDataset --datasets CAMO CHAMELEON COD10K
```

Prediction masks are saved under `<save_path>/<DATASET>/`.

## Evaluation

Evaluate saved predictions:

```powershell
python evaluate_cod.py --pred_root .\results --gt_root .\data\TestDataset\TestDataset --datasets CAMO CHAMELEON COD10K --label original --output metrics_original.json
```

This reports S-measure, mean/max/adaptive E-measure, weighted F-measure, and MAE. The script writes both JSON and CSV output.

## Efficiency Benchmark

Benchmark every MFAM variant at 352 × 352:

```powershell
python benchmark_model.py --size 352 --output bench.json
```

Benchmark one or more variants:

```powershell
python benchmark_model.py --variants original depthwise --device cuda --output bench_depthwise.json
python benchmark_model.py --device cpu --output bench_cpu.json
```

The benchmark reports total and MFAM-only parameters, FLOPs, latency, FPS, and peak GPU memory. The repository includes benchmark JSON/CSV runs from the development environment.

## Local Web Interface

Start the Python inference server:

```powershell
$env:MODEL_PATH = "Snapshot/FAPNet/checkpoint_best.pth"
$env:MFAM_VARIANT = "original"
python server.py
```

The server listens on `http://127.0.0.1:8000`.

In a second terminal, start the React UI:

```powershell
cd ui
npm install
npm run dev
```

Open `http://localhost:5173`. The Vite development server proxies `/api/*` to the Python server on port 8000.

If no matching checkpoint is available, the server remains online and the UI falls back to client-side edge saliency.

## Checkpoint Compatibility

Checkpoints are validated against the requested MFAM variant. A raw state dict is checked from its tensor shapes, while full checkpoints also include `mfam_variant` metadata and are accompanied by `run_config.json` for snapshots created by `train.py`.

Use the same variant for loading and inference:

```powershell
python test.py --mfam_variant depthwise --model_path .\Snapshot\FAPNet_depthwise\FAPNet_best.pth
```

Do not use an original/checkpoint and an alternative MFAM variant interchangeably.

## Tests

Run the CPU-compatible variant tests:

```powershell
python -m pytest tests -q
```

The tests cover output shapes, gradients, invalid variants, variant parameter efficiency, state-dict compatibility, mismatch detection, and checkpoint metadata.

## Recorded Experimental Metrics

The repository includes the following 10-epoch `original`/baseline evaluation from `metrics_epoch10.json`:

| Dataset | Images | S-measure ↑ | E-mean ↑ | E-max ↑ | E-adaptive ↑ | Weighted F ↑ | MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAMO | 250 | 0.8080 | 0.8562 | 0.8818 | 0.8711 | 0.7090 | 0.0792 |
| CHAMELEON | 76 | 0.8661 | 0.9129 | 0.9480 | 0.8974 | 0.7604 | 0.0379 |
| COD10K | 2,026 | 0.7983 | 0.8623 | 0.8860 | 0.8441 | 0.6373 | 0.0429 |

The repository also contains `metrics_original.json`, which reports a separate original-variant run. Model-quality comparisons for all three variants should use consistently trained checkpoints and the same evaluation command; this README does not invent missing variant metrics.

## Citation

This project is based on the original FAP-Net research. Please cite it when using or building upon the implementation:

```bibtex
@article{zhou2022feature,
  title={Feature Aggregation and Propagation Network for Camouflaged Object Detection},
  author={Zhou, Tao and Zhou, Yi and Gong, Chen and Yang, Jian and Zhang, Yu},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={7036--7047},
  year={2022},
  doi={10.1109/TIP.2022.3217695}
}
```
