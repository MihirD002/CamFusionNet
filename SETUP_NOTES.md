# FAPNet Environment Setup Notes

## Why these versions, not the repo's `requirement.txt`

The original file pins:
```
torch==1.2.0
python==3.6.6
opencv-python==4.5.2
```

`torch==1.2.0` is from 2019 and predates the paper (IEEE TIP 2022) — the
authors almost certainly trained with something newer and never updated this
file. Installing 1.2.0 today is impractical: it needs old CUDA toolkits
(9.2/10.0) that are hard to pair with a current GPU driver, and prebuilt
wheels are hard to source.

This environment instead targets **PyTorch 1.13.1 + CUDA 11.7**, which:
- runs on any GPU with a reasonably current driver (check with `nvidia-smi`)
- keeps `python==3.9` for good library compatibility
- is new enough to be maintained, old enough to match the code style/APIs
  used in `train.py` (e.g. `F.upsample`, which is deprecated but still
  present in this version — it'll emit a warning, not an error)

Document this substitution explicitly in your dissertation's methodology or
reproducibility section — it's a legitimate, disclosed deviation, not a
silent one.

## Two install paths — pick one

**conda (recommended, matches original authors' style):**
```bash
conda env create -f environment.yml
conda activate fapnet
```

**pip / venv:**
```bash
python3.9 -m venv venv
source venv/bin/activate
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

## Checking your CUDA driver before installing

```bash
nvidia-smi
```
Look at the "CUDA Version" field in the top-right of the output — that's the
*maximum* CUDA version your driver supports. Pick a `pytorch-cuda`/`+cuXXX`
build at or below that number. If you're not sure, CUDA 11.7 or 11.8 builds
work with most driver versions released since ~2022.

## Known code-level gotchas (not environment, but will block you next)

1. **`test.py` hardcodes an absolute path**:
   ```python
   '/test/CamouflagedObjectDection/Dataset/TestDataset/{}/Imgs/'
   ```
   Edit this to point at your local `./data/TestDataset/{}/Imgs/` and `GT/`
   before running.

2. **`train.py` expects an `Edge/` directory** (`--train_eg_dir`) alongside
   `Imgs/` and `GT/` in the training set. Confirm the downloaded training
   zip actually includes edge maps — if not, you'll need to generate them
   (e.g. Canny edge detection on the GT masks) before training will run.

3. **Evaluation is MATLAB-only** — there's no Python metrics script in this
   repo. You need MATLAB (or MATLAB-compatible, e.g. Octave with caveats) to
   run `eval/main.m` from the linked `DengPingFan/CODToolbox`.

4. **Batch size 32 is memory-hungry** — the repo's own comment says
   "~500MB per img in GPU" at their settings. On smaller GPUs, lower
   `--batchsize` and note that this is a deviation from the paper's exact
   training regime.
