# ============================================================
# FAPNet Batch Training Script (PowerShell)
# Trains in configurable chunks of epochs using --resume.
#
# Usage:
#   .\train_batch.ps1                        # Run next pending batch
#   .\train_batch.ps1 -TotalEpochs 50        # Extend target to 50 (keeps old checkpoints)
#   .\train_batch.ps1 -ShowStatus            # Show current progress only
#   .\train_batch.ps1 -Reset                 # Wipe all checkpoints and start from epoch 1
#
# Evaluate accuracy at a specific checkpoint:
#   .\train_batch.ps1 -Evaluate -AtEpoch 20  # Generate predictions + print metrics for ep20
#   .\train_batch.ps1 -Evaluate              # Evaluate the best checkpoint so far
# ============================================================

param(
    [switch]$Reset,
    [switch]$ShowStatus,
    [switch]$Evaluate,
    [int]$AtEpoch     = 0,         # epoch checkpoint to evaluate (0 = use best)
    [int]$TotalEpochs = 0          # override total epochs (e.g. extend from 30 to 50)
)

# ---- Config (edit these) -----------------------------------
$DEFAULT_TOTAL  = 30               # default total epochs
$BATCH_SIZE_EP  = 10               # epochs per training session
$BATCHSIZE      = 2                # image batch size (fits ~1.2GB VRAM on GTX 1650)
$TRAINSIZE      = 352
$LR             = 1e-4
$SAVE_EVERY     = 5                # save checkpoint every N epochs
$AMP            = $false           # FP32 standard precision (prevents NaN overflow)
$NUM_WORKERS    = 2                # parallel background data loading
$SINGLE_SCALE   = $true            # 1x scale pass per batch (3x faster, same resolution & output)
$EXP_NAME       = "run_v2"
$SAVE_MODEL     = "./Snapshot/FAPNet"
$DATA_ROOT      = "./data/TestDataset/TestDataset"
$DATASETS       = @("CHAMELEON", "CAMO", "COD10K")
# ------------------------------------------------------------

$SNAPSHOT_DIR  = Join-Path $SAVE_MODEL $EXP_NAME
$PROGRESS_FILE = Join-Path $SNAPSHOT_DIR "progress.txt"
$TOTAL_FILE    = Join-Path $SNAPSHOT_DIR "total_epochs.txt"
$CHECKPOINT    = Join-Path $SNAPSHOT_DIR "checkpoint_latest.pth"
$BEST_CKPT     = Join-Path $SNAPSHOT_DIR "checkpoint_best.pth"

# Helper: get/set last completed epoch
function Get-LastEpoch {
    if (Test-Path $PROGRESS_FILE) { return [int]((Get-Content $PROGRESS_FILE -Raw).Trim()) }
    return 0
}
function Set-LastEpoch($ep) {
    New-Item -ItemType Directory -Force -Path $SNAPSHOT_DIR | Out-Null
    Set-Content -Path $PROGRESS_FILE -Value "$ep"
}

# Helper: get/set total epochs goal
function Get-TotalEpochs {
    if (Test-Path $TOTAL_FILE) { return [int]((Get-Content $TOTAL_FILE -Raw).Trim()) }
    return $DEFAULT_TOTAL
}
function Set-TotalEpochs($ep) {
    New-Item -ItemType Directory -Force -Path $SNAPSHOT_DIR | Out-Null
    Set-Content -Path $TOTAL_FILE -Value "$ep"
}

# ---- Reset -------------------------------------------------
if ($Reset) {
    Write-Host ""
    Write-Host "  [RESET] Removing all checkpoints in:" -ForegroundColor Yellow
    Write-Host "          $SNAPSHOT_DIR" -ForegroundColor Yellow
    if (Test-Path $SNAPSHOT_DIR) {
        Remove-Item -Path (Join-Path $SNAPSHOT_DIR "*") -Recurse -Force
        Write-Host "  Done. Training will restart from epoch 1." -ForegroundColor Green
    } else {
        Write-Host "  Nothing to reset." -ForegroundColor Cyan
    }
    Write-Host ""
    exit 0
}

# ---- Resolve total epochs ----------------------------------
# If -TotalEpochs was passed on the CLI, update/create the file
if ($TotalEpochs -gt 0) {
    $currentTotal = Get-TotalEpochs
    $lastEpoch    = Get-LastEpoch
    if ($TotalEpochs -le $lastEpoch) {
        Write-Host ""
        Write-Host "  [ERROR] -TotalEpochs $TotalEpochs is <= epochs already trained ($lastEpoch)." -ForegroundColor Red
        Write-Host "          Use a higher value to extend, or -Reset to start fresh." -ForegroundColor Yellow
        Write-Host ""
        exit 1
    }
    Set-TotalEpochs $TotalEpochs
    if ($TotalEpochs -ne $currentTotal) {
        Write-Host ""
        Write-Host "  Target updated: $currentTotal --> $TotalEpochs total epochs." -ForegroundColor Green
        Write-Host ""
    }
}
$TOTAL_EPOCHS = Get-TotalEpochs

# ---- Evaluate ----------------------------------------------
if ($Evaluate) {
    $lastEpoch = Get-LastEpoch

    # Determine which checkpoint to use
    if ($AtEpoch -gt 0) {
        $ckptPath = Join-Path $SNAPSHOT_DIR "checkpoint_epoch_$AtEpoch.pth"
        if (-not (Test-Path $ckptPath)) {
            # Fall back to raw state-dict .pth
            $ckptPath = Join-Path $SNAPSHOT_DIR "FAPNet_$AtEpoch.pth"
        }
        if (-not (Test-Path $ckptPath)) {
            Write-Host ""
            Write-Host "  [ERROR] No checkpoint found for epoch $AtEpoch." -ForegroundColor Red
            Write-Host "          Available checkpoints:" -ForegroundColor Yellow
            Get-ChildItem $SNAPSHOT_DIR -Filter "*.pth" | ForEach-Object { Write-Host "    $($_.Name)" }
            Write-Host ""
            exit 1
        }
        $label = "epoch${AtEpoch}"
    } else {
        $ckptPath = $BEST_CKPT
        if (-not (Test-Path $ckptPath)) { $ckptPath = $CHECKPOINT }
        if (-not (Test-Path $ckptPath)) {
            Write-Host ""
            Write-Host "  [ERROR] No checkpoint found. Train at least one batch first." -ForegroundColor Red
            Write-Host ""
            exit 1
        }
        $label = "best"
    }

    $RESULTS_DIR = "./results/$EXP_NAME/$label"

    Write-Host ""
    Write-Host "  Evaluating checkpoint: $ckptPath" -ForegroundColor Cyan
    Write-Host "  Predictions -> $RESULTS_DIR" -ForegroundColor Cyan
    Write-Host ""

    # Step 1: generate predictions
    $test_args = @(
        "test.py",
        "--model_path", $ckptPath,
        "--save_path",  $RESULTS_DIR,
        "--data_root",  $DATA_ROOT,
        "--datasets"
    ) + $DATASETS + @("--testsize", $TRAINSIZE)

    $p1 = Start-Process -FilePath "python" -ArgumentList $test_args -NoNewWindow -Wait -PassThru
    if ($p1.ExitCode -ne 0) {
        Write-Host "  [ERROR] test.py failed (exit $($p1.ExitCode))." -ForegroundColor Red
        exit 1
    }

    # Step 2: compute metrics
    Write-Host ""
    Write-Host "  Computing metrics..." -ForegroundColor Cyan
    $metricsOut = "$RESULTS_DIR/metrics.json"
    $eval_args = @(
        "evaluate_cod.py",
        "--pred_root", $RESULTS_DIR,
        "--data_root", $DATA_ROOT,
        "--datasets"
    ) + $DATASETS + @("--out", $metricsOut)

    $p2 = Start-Process -FilePath "python" -ArgumentList $eval_args -NoNewWindow -Wait -PassThru
    if ($p2.ExitCode -eq 0) {
        Write-Host ""
        Write-Host "  Metrics saved to: $metricsOut" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] evaluate_cod.py failed (exit $($p2.ExitCode))." -ForegroundColor Yellow
    }

    Write-Host ""
    exit 0
}

# ---- Status header -----------------------------------------
$lastEpoch    = Get-LastEpoch

Write-Host ""
Write-Host "  FAPNet Batch Training" -ForegroundColor Cyan
Write-Host "  ================================================"
Write-Host "  Total target  : $TOTAL_EPOCHS epochs"
Write-Host "  Batch size    : $BATCH_SIZE_EP epochs per run"
Write-Host "  Completed     : $lastEpoch / $TOTAL_EPOCHS epochs"

if ($lastEpoch -ge $TOTAL_EPOCHS) {
    Write-Host ""
    Write-Host "  Training already complete ($lastEpoch / $TOTAL_EPOCHS)." -ForegroundColor Green
    Write-Host "  To extend: .\train_batch.ps1 -TotalEpochs 50" -ForegroundColor Yellow
    Write-Host "  To reset : .\train_batch.ps1 -Reset" -ForegroundColor Yellow
    Write-Host "  To eval  : .\train_batch.ps1 -Evaluate" -ForegroundColor Yellow
    Write-Host ""
    exit 0
}

$startEpoch = $lastEpoch + 1
$endEpoch   = [Math]::Min($lastEpoch + $BATCH_SIZE_EP, $TOTAL_EPOCHS)

Write-Host "  Next batch    : epochs $startEpoch --> $endEpoch"
Write-Host "  Exp dir       : $SNAPSHOT_DIR"
Write-Host "  ================================================"

if ($ShowStatus) { Write-Host ""; exit 0 }

Write-Host ""
Write-Host "  Starting training batch (epochs $startEpoch to $endEpoch)..." -ForegroundColor Green
Write-Host ""

# ---- Build training args -----------------------------------
$train_args = @(
    "train.py",
    "--epoch",       $endEpoch,
    "--batchsize",   $BATCHSIZE,
    "--trainsize",   $TRAINSIZE,
    "--lr",          $LR,
    "--save_epoch",  $SAVE_EVERY,
    "--save_model",  $SAVE_MODEL,
    "--exp_name",    $EXP_NAME,
    "--num_workers", $NUM_WORKERS
)

if ($SINGLE_SCALE) {
    $train_args += "--single_scale"
    Write-Host "  Scale mode    : Single scale (1.0x) - 3x faster" -ForegroundColor Cyan
} else {
    Write-Host "  Scale mode    : Multi scale [0.75, 1, 1.25]" -ForegroundColor Cyan
}

Write-Host "  Batch size    : $BATCHSIZE" -ForegroundColor Cyan
Write-Host "  Workers       : $NUM_WORKERS" -ForegroundColor Cyan

if ($lastEpoch -gt 0 -and (Test-Path $CHECKPOINT)) {
    $train_args += @("--resume", $CHECKPOINT)
    Write-Host "  Resuming from : $CHECKPOINT" -ForegroundColor Cyan
} else {
    Write-Host "  Fresh start   : epoch 1" -ForegroundColor Cyan
}

if ($AMP) {
    $train_args += "--amp"
    Write-Host "  AMP           : ON (mixed precision)" -ForegroundColor Cyan
}

Write-Host ""

# ---- Run training ------------------------------------------
$proc = Start-Process -FilePath "python" -ArgumentList $train_args -NoNewWindow -Wait -PassThru

if ($proc.ExitCode -eq 0) {
    Set-LastEpoch $endEpoch
    # Also persist the total target so future runs remember it
    Set-TotalEpochs $TOTAL_EPOCHS

    Write-Host ""
    Write-Host "  Batch complete! Epochs $startEpoch - $endEpoch done." -ForegroundColor Green

    $remaining = $TOTAL_EPOCHS - $endEpoch
    if ($remaining -gt 0) {
        $nextEnd = [Math]::Min($endEpoch + $BATCH_SIZE_EP, $TOTAL_EPOCHS)
        Write-Host "  $remaining epochs remaining." -ForegroundColor Yellow
        Write-Host "  Run .\train_batch.ps1 for epochs $($endEpoch+1) to $nextEnd." -ForegroundColor Yellow
    } else {
        Write-Host "  All $TOTAL_EPOCHS epochs complete!" -ForegroundColor Green
        Write-Host "  Best model : $SNAPSHOT_DIR\FAPNet_best.pth" -ForegroundColor Green
        Write-Host "  Evaluate   : .\train_batch.ps1 -Evaluate" -ForegroundColor Cyan
    }
} else {
    Write-Host ""
    Write-Host "  Training failed (exit code $($proc.ExitCode))." -ForegroundColor Red
    Write-Host "  Progress NOT updated - will retry this batch on next run." -ForegroundColor Yellow
}

Write-Host ""
