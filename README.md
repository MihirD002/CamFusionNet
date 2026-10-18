# Feature Aggregation and Propagation Network for Camouflaged Object Detection

> **Authors:** 
> [*Tao Zhou*](https://taozh2017.github.io),
> [*Yi Zhou*](https://cse.seu.edu.cn/2021/0303/c23024a362239/page.htm),
> [*Chen Gong*](https://gcatnjust.github.io/ChenGong/index.html),
> [*Jian Yang*](https://scholar.google.com.hk/citations?user=6CIDtZQAAAAJ&hl=zh-CN),
> and [*Yu Zhang*](https://scholar.google.com.hk/citations?user=oDrTEi0AAAAJ&hl=zh-CN).



## 1. Preface

- This repository provides code for "_**Feature Aggregation and Propagation Network for Camouflaged Object Detection**_" IEEE TIP 2022. [Paper](https://ieeexplore.ieee.org/abstract/document/9940173/) [![Arxiv Page](https://img.shields.io/badge/Arxiv-2105.12555-red?style=flat-square)](https://arxiv.org/pdf/2212.00990.pdf)

> **Important — this checkout is not an exact paper reproduction.** The encoder in `lib/model.py` is
> **ConvNeXt-Tiny** (`torchvision.models.convnext_tiny`), not the Res2Net-50 used in the paper. Results here are
> FAP-Net-*style* baselines. The original paper weights (`checkpoints/FAPNet.pth`) are Res2Net-based and will **not**
> load into this model. Do not claim an exact reproduction unless the Res2Net backbone is restored and validated separately.


## 2. Overview

### 2.1. Introduction

Camouflaged object detection (COD) aims to detect/segment camouflaged objects embedded in the environment, which has attracted increasing attention over the past decades.
Although several COD methods have been developed, they still suffer from unsatisfactory performance due to the intrinsic similarities between the foreground objects and background surroundings. In this paper, we propose a novel Feature Aggregation
and Propagation Network (FAP-Net) for camouflaged object detection. Specifically, we propose a Boundary Guidance Module (BGM) to explicitly model the boundary characteristic, which can provide boundary-enhanced features to boost the COD performance. To capture the scale variations of the camouflaged objects, we propose a Multi-scale Feature Aggregation Module (MFAM) to characterize the multi-scale information from
each layer and obtain the aggregated feature representations. Furthermore, we propose a Cross-level Fusion and Propagation Module (CFPM). In the CFPM, the feature fusion part can effectively integrate the features from adjacent layers to exploit the cross-level correlations, and the feature propagation part can transmit valuable context information from the encoder to the decoder network via a gate unit. Finally, we formulate a unified
and end-to-end trainable framework where cross-level features can be effectively fused and propagated for capturing rich context information. Extensive experiments on three benchmark camouflaged datasets demonstrate that our FAP-Net outperforms other state-of-the-art COD models. Moreover, our model can be extended to the polyp segmentation task, and the comparison
results further validate the effectiveness of the proposed model in segmenting polyps.


### 2.2. Framework Overview

<p align="center">
    <img src="imgs/fapnet.png"/> <br />
    <em> 
    Figure 1: The overall architecture of the proposed FAP-Net, consisting of three key components, i.e., boundary guidance module, multi-scale feature aggregation module, and cross-level fusion and propagation module.
    </em>
</p>

### 2.3. Qualitative Results

<p align="center">
    <img src="imgs/results.png"/> <br />
    <em> 
    Figure 2: Qualitative Results.
    </em>
</p>

## 3. Proposed Method

### 3.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with one NVIDIA Tesla P40 GPU of 24 GB Memory (original paper setup).

1. Configuring your environment (Prerequisites):
       
    + Install PyTorch/torchvision first (**torchvision >= 0.13** is required for ConvNeXt), then the remaining packages: `pip install -r requirements.txt` (see the header of that file for the exact torch command).

1. Downloading necessary data:


    + downloading training dataset and move it into `./data/`, 
    which can be found from [Google Drive](https://drive.google.com/file/d/1Kifp7I0n9dlWKXXNIbN7kgyokoRY4Yz7/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1uyQz0b_r_5yCee0orSw7EA) (extraction code: fapn). 

    + downloading testing dataset and move it into `./data/`, 
    which can be found from [Google Drive](https://drive.google.com/file/d/1SLRB5Wg1Hdy7CQ74s3mTQ3ChhjFRSFdZ/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1F3tVEWYzKYp5NBv3cjiaAg) (extraction code: fapn). 
        
    + (Paper checkpoint, Res2Net-based — **not compatible with the ConvNeXt-Tiny model in this checkout**) `./checkpoints/FAPNet.pth`,
    which can be found from [Google Drive](https://drive.google.com/file/d/1qjb70ZGwExei21x6uMQbUAwZ0Ts4Z-xk/view?usp=share_link) or [(Baidu Drive)](https://pan.baidu.com/s/1BeRx81XNKq_jA7LHut1VZg) (extraction code: fapn).
    
    + (Only needed if you restore the Res2Net backbone) Res2Net weights go to `./lib/res2net50_v1b_26w_4s-3cf99910.pth`,
    which can be found from [Google Drive](https://drive.google.com/file/d/1_1N-cx1UpRQo7Ybsjno1PAg4KE1T9e5J/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1gDNNh7Cad3jv2l7i4_V33Q) (extraction code: fapn). 

1. Training Configuration:

    + After you download the training dataset, run `python train.py` to train the baseline, or pass an experiment config, e.g. `python train.py --config experiments/h1_morph_k3.json` (see Section 5).
    + Each run writes to `Snapshot/FAPNet/<exp_name>/` (config, log, checkpoints).


1. Testing Configuration:

    + After training (or with your own ConvNeXt-Tiny checkpoint) and the testing dataset, run `python test.py --model_path <ckpt> --save_path results/<exp_name>` to generate the prediction maps for CHAMELEON, CAMO and COD10K. Add `--save_aux` to also save predicted boundary / uncertainty maps.

    + You can also download the paper's prediction maps ('CHAMELEON', 'CAMO', 'COD10K') from [Google Drive](https://drive.google.com/file/d/1O5gDTBasHWuwPv4hxd04Nt08y16_Y578/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1nltTLWnU3YZpCQO5LewAIw) (extraction code: fapn)).

    + You can also download the paper's prediction maps (NC4K) from [Google Drive](https://drive.google.com/file/d/139CoLtoQp_9n3T8WXc2G8xI61MJj_bpj/view?usp=share_link) or [Baidu Drive](https://pan.baidu.com/s/1iGcAAnFjbav-HLoG7Dc_FA) (extraction code: fapn)).

### 3.2 Evaluating your trained model:

Use the Python evaluator, which reports S-measure, weighted F-measure, mean E-measure, MAE and boundary F-score, overall and per GT-area stratum:

```bash
python evaluate_cod.py --pred_root results/<exp_name> --out results/<exp_name>/metrics.json
```

Sanity-check it against a reference implementation (e.g. PySODMetrics) on a small fixed sample before trusting absolute numbers.
The original MATLAB toolbox ([link](https://github.com/DengPingFan/CODToolbox), instructions in `./eval/main.m`) can still be used for paper-comparable numbers.


## 4. Citation

Please cite our paper if you find the work useful, thanks!
	
	@article{zhou2022feature,
	   title={Feature Aggregation and Propagation Network for Camouflaged Object Detection},
	   author={Zhou, Tao and Zhou, Yi and Gong, Chen and Yang, Jian and Zhang, Yu},
	   journal={IEEE Transactions on Image Processing},
	   volume={31},
	   pages={7036--7047},
	   year={2022},
	   publisher={IEEE}
	}


## 5. Boundary and Uncertainty COD Study (experimental)

Three hypotheses are tested independently on the fixed ConvNeXt-Tiny baseline (same split, schedule and evaluation):
**H1** morphological boundary target quality, **H2** multi-scale boundary features, **H3** uncertainty-based loss weighting.

| `train.py` flag | Values | Hypothesis |
|---|---|---|
| `--boundary_mode` | `file` (legacy precomputed edge maps, control), `morph` (derived from the transformed GT) | H1 |
| `--boundary_kernel_size` | e.g. 3, 5 | H1 |
| `--lambda_b` | boundary loss weight (sweep 0.5 / 1 / 2) | H1 |
| `--boundary_head` | `single` (control), `multiscale` | H2 |
| `--uncertainty` | `none`, `seg`, `seg_boundary` | H3 |
| `--lambda_u` | log-variance regulariser weight | H3 |
| `--seed`, `--exp_name`, `--config` | reproducibility, per-experiment folders, JSON defaults | all |

Ready-made configs live in `experiments/` (`baseline_edgefile`, `h1_morph_k3`, `h1_morph_k5`, `h1_morph_k3_lb0.5`,
`h1_morph_k3_lb2`, `h2_multiscale_k3`, `h3_unc_seg`, `h3_unc_seg_boundary`). Run H1 first and proceed to H2/H3 only
after the preceding comparison is valid; the H3 configs are built on the H2 setting, so edit them to whichever H1/H2
configuration wins.

```bash
python train.py --config experiments/h1_morph_k3.json
python test.py  --model_path Snapshot/FAPNet/h1_morph_k3/FAPNet_best.pth --save_path results/h1_morph_k3 --save_aux
python evaluate_cod.py --pred_root results/h1_morph_k3 --out results/h1_morph_k3/metrics.json
```

Protocol notes:

- **Output contract:** `model(x)` returns `(out5, out4, out3, out2, edge_out)`; with uncertainty enabled a 6th element
  (`logvar`) is appended. Index 3 is always the final segmentation logit map. Baseline mode keeps the original state_dict
  keys, so old ConvNeXt checkpoints load with `strict=True`; optional heads exist only when their flag is set.
  Checkpoints written by the updated `train.py` store their config, so `test.py` / `server.py` rebuild the right architecture
  (`server.py` also honours `BOUNDARY_HEAD` and `UNCERTAINTY` environment variables).
- **Masks** use nearest-neighbour interpolation for rotation and resizing (including the 0.75x / 1.25x multi-scale training steps).
- **Boundary targets** in `morph` mode are derived from the final transformed binary mask, before the optional GT
  salt-and-pepper noise (`--gt_noise`). The legacy `Edge/` directory is reported as the legacy target source; it is not asserted to be Canny.
- **Loss:** `loss_seg + lambda_b * loss_boundary`, logged separately. The original BCE averaged to a scalar because of
  `reduce='none'`; the updated code uses per-pixel BCE. Re-run the baseline with the updated code so all ablations share one
  loss implementation (`--legacy_wbce 1` reproduces the old behaviour).
- **Model-training initialisation:** the backbone is constructed with `weights=None`; unless your checkpoint provides weights, training starts from a random ConvNeXt.
- **Evaluation:** one consistent evaluator (`evaluate_cod.py`) for all runs; report parameters, FLOPs, latency and peak memory for architecture changes.
- Further detail: `EXPERIMENTS.md`.


**[⬆ back to top](#1-preface)**
