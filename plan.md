## Plan: Boundary and Uncertainty COD Study

Treat this checkout as a ConvNeXt-Tiny, FAP-Net-style baseline and test three hypotheses independently: morphological target quality (H1), multi-scale boundary features (H2), and uncertainty-based loss weighting (H3). Keep the backbone, split, schedule, and evaluation fixed across the primary ablations; do not claim an exact paper reproduction unless the Res2Net backbone is restored and separately validated.

**Steps**
1. **Lock baseline and protocol.** Train/evaluate the current code unchanged, save config/checkpoint/predictions, fix seeds, and document dataset splits and runtime. Current training uses multi-scale input resizing and logs MAE on one selected test set; evaluate all benchmark datasets using the same external or unified metric implementation.
2. **H1: boundary target.** In `utils/dataloader.py`, stop requiring precomputed edge masks for this experiment. Apply geometric augmentations jointly to image and GT, use nearest-neighbor interpolation for masks (including rotation/resize), then derive an optionally configurable morphological boundary from the final transformed binary mask. Add `boundary_mode` and `boundary_kernel_size`; compare at minimum morph-k3 and morph-k5 against the existing edge-target baseline where available. Preserve consistent binary range/dtype and validate target/prediction shape and range.
3. **Make boundary weighting reproducible.** In `train.py`, separate segmentation and boundary losses, add configurable `lambda_b`, log both components, and optimize `loss_seg + lambda_b * loss_boundary`. Run a small lambda sweep (for example 0.5, 1, 2) on the same protocol. H1 comparison changes target only; do not change model features at the same time.
4. **H2: boundary feature scale.** In `lib/model.py`, retain the existing single-stage boundary head as the control. Add an opt-in multi-scale head that projects/resizes a neighboring deeper feature to the shallow feature resolution, fuses them, and predicts a full-resolution boundary map. Keep target, loss weight, decoder fusion, and all other settings fixed. Compare single- versus multi-scale boundary heads with the same morphological target.
5. **H3: uncertainty.** Add an opt-in spatial uncertainty head in `lib/model.py`. Recommended first formulation: predict bounded uncertainty and use it to reweight pixelwise segmentation/boundary losses with an explicit regularizer or a heteroscedastic log-variance likelihood; do not add an undefined standalone uncertainty loss. In `train.py`, compare no uncertainty, segmentation-only uncertainty weighting, and segmentation-plus-boundary weighting while holding the H1/H2 configuration fixed. Track calibration/uncertainty behavior as well as COD metrics.
6. **Keep model users/checkpoints compatible.** Update output unpacking and inference behavior in `train.py`, `test.py`, and `server.py` if model outputs change. Use explicit model flags for optional branches; preserve the original output contract by default where practical. Save each configuration/checkpoint to a separate experiment directory and document strict/non-strict loading behavior for old checkpoints.
7. **Evaluate and analyze.** Use one consistent COD evaluator for S-measure, weighted F-measure, enhanced alignment measure, and MAE on CAMO, COD10K, and CHAMELEON. Report overall and GT-area / boundary-quality strata, plus parameters, FLOPs, inference latency, and peak memory for architecture changes. Produce a compact qualitative grid of GT, target, predicted boundary, segmentation, and uncertainty where applicable.

**Relevant files**
- `D:/GitHub/FAPNet/utils/dataloader.py` — synchronized augmentation, nearest-neighbor mask geometry, on-the-fly boundary target generation, target options.
- `D:/GitHub/FAPNet/train.py` — lambda/config arguments, independent loss terms/logging, validation of tensor shapes/ranges, and controlled training/evaluation runs.
- `D:/GitHub/FAPNet/lib/model.py` — opt-in multi-scale boundary head and later uncertainty head; retain existing boundary branch for ablation control.
- `D:/GitHub/FAPNet/test.py` — adjust inference output handling if the model return contract changes; generate predictions for all evaluation datasets.
- `D:/GitHub/FAPNet/server.py` — update inference unpacking if return values change; preserve app inference behavior.
- `D:/GitHub/FAPNet/README.md` — document that the current encoder is ConvNeXt-Tiny, experimental flags/data requirements, and the exact benchmark protocol.
- `D:/GitHub/FAPNet/requirements.txt` — only if a required evaluation or morphology dependency is missing.

**New files**
- `D:/GitHub/FAPNet/evaluate_cod.py` — recommended if no suitable standard evaluator is already available in the environment; one reproducible entry point for the four COD metrics across datasets.
- `D:/GitHub/FAPNet/experiments/` — recommended experiment configuration files and run metadata (one config per baseline/ablation); directory/files need not be added if existing CLI arguments and run logs are sufficient.
- No new model module is required; boundary target generation can remain in the existing dataloader module unless it grows enough to justify a utility module.

**Verification**
1. Smoke-test a dataloader batch: image/GT/boundary are aligned, GT and boundary are binary, and dimensions remain correct after augmentation and resize.
2. Run a forward/backward smoke test for baseline, multi-scale boundary, and uncertainty flags; verify finite losses/gradients and matching boundary tensor dimensions.
3. Confirm baseline mode loads existing checkpoints and `test.py`/`server.py` still return the intended segmentation output.
4. Run each controlled ablation with identical seeds, split, optimizer, schedule, and evaluation; verify logs include separate losses and full configuration.
5. Check evaluator results on a small fixed sample and then run the full CAMO/COD10K/CHAMELEON evaluation; record compute-cost measurements and qualitative examples.

**Decisions**
- Primary recommendation: use this repository's current ConvNeXt-Tiny model as the explicitly named baseline, and hold it fixed through H1-H3. The README references Res2Net while `lib/model.py` actually constructs ConvNeXt; changing the backbone is a separate study, not part of these ablations.
- Morphology is fixed target generation, not a learnable detector. The learned contribution is the prediction head and its integration.
- Implement H1 first; proceed to H2 and H3 only after the preceding controlled experiment is valid. Do not bundle all changes into one run.
- External-edge/Canny comparison is only valid if a matching, documented edge-target baseline exists; otherwise report the existing edge directory as the legacy target source without asserting it is Canny.
