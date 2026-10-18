import argparse
import json
import logging
import os
import random
import re
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# Enable cuDNN benchmark for faster convolutions on fixed input resolutions
torch.backends.cudnn.benchmark = True

from lib.model import FAPNet
from utils.dataloader import get_loader, test_dataset
from utils.trainer import adjust_lr


best_mae   = 1
best_epoch = 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def structure_loss(pred, mask, logvar=None, lambda_u=1.0, legacy_wbce=False):
    """
    Weighted BCE + weighted IoU.

    logvar (H3): bounded per-pixel log-variance s (same shape as pred). The BCE
    term becomes exp(-s) * bce, with the regulariser lambda_u * mean(s) so that
    s cannot grow without bound (heteroscedastic form). s = 0 -> unchanged loss.

    legacy_wbce: reproduces the original `reduce='none'` quirk, which (since
    reduce was truthy) averaged the BCE to a scalar before weighting.
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    if legacy_wbce:
        assert logvar is None, 'legacy_wbce cannot be combined with uncertainty weighting'
        bce  = F.binary_cross_entropy_with_logits(pred, mask)            # scalar mean
        wbce = (weit * bce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    else:
        bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        if logvar is not None:
            assert logvar.shape == pred.shape, (logvar.shape, pred.shape)
            bce = bce * torch.exp(-logvar)
        wbce = (weit * bce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        if logvar is not None:
            wbce = wbce + lambda_u * logvar.mean(dim=(2, 3))

    pred  = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou  = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def boundary_loss(edge_out, edge_gt, logvar=None, lambda_u=1.0):
    """Pixelwise BCE on the boundary map; optionally uncertainty-weighted (H3)."""
    bce = F.binary_cross_entropy_with_logits(edge_out, edge_gt, reduction='none')
    if logvar is None:
        return bce.mean()
    return (bce * torch.exp(-logvar) + lambda_u * logvar).mean()


def check_batch(images, gts, egs, opt):
    """Smoke checks: alignment, shapes, value range, binary targets."""
    assert images.shape[0] == gts.shape[0] == egs.shape[0], 'batch size mismatch'
    assert images.shape[-2:] == gts.shape[-2:] == egs.shape[-2:], \
        f'spatial mismatch: {images.shape} {gts.shape} {egs.shape}'
    for name, t in (('gt', gts), ('boundary', egs)):
        assert t.min() >= 0 and t.max() <= 1, f'{name} out of [0,1]'
        if opt.boundary_mode == 'morph':
            assert torch.all((t == 0) | (t == 1)), f'{name} is not binary'
    print(f'[check] batch OK  img{tuple(images.shape)} gt{tuple(gts.shape)} boundary{tuple(egs.shape)} '
          f'boundary_pixels={egs.mean().item():.4f}')


# ---------------------------------------------------------------------------
# Train / test
# ---------------------------------------------------------------------------
def train(train_loader, model, optimizer, epoch, opt, total_step, save_path, scaler=None):
    model.train()

    size_rates = [1.0] if getattr(opt, 'single_scale', False) else [0.75, 1, 1.25]
    use_seg_u  = opt.uncertainty in ('seg', 'seg_boundary')
    use_b_u    = opt.uncertainty == 'seg_boundary'

    for step, data_pack in enumerate(train_loader):
        images, gts, egs = data_pack
        if epoch == 1 and step == 0:
            check_batch(images, gts, egs, opt)
        images = images.cuda(non_blocking=True)
        gts    = gts.cuda(non_blocking=True)
        egs    = egs.cuda(non_blocking=True)

        for rate in size_rates:

            optimizer.zero_grad()

            # ---- rescale (masks: nearest-neighbour so they stay binary) ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images_r = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts_r    = F.interpolate(gts,    size=(trainsize, trainsize), mode='nearest')
                egs_r    = F.interpolate(egs,    size=(trainsize, trainsize), mode='nearest')
            else:
                images_r, gts_r, egs_r = images, gts, egs

            with autocast(enabled=(scaler is not None)):
                outs = model(images_r)
                sal1, sal2, sal3, sal4, edge_out = [o.float() for o in outs[:5]]
                logvar = outs[5].float() if len(outs) > 5 else None

                seg_u = logvar if use_seg_u else None
                b_u   = logvar if use_b_u else None

                loss_boundary = boundary_loss(edge_out, egs_r, b_u, opt.lambda_u)

                loss_seg = sum(structure_loss(s, gts_r, seg_u, opt.lambda_u, bool(opt.legacy_wbce))
                               for s in (sal1, sal2, sal3, sal4))

                loss_total = loss_seg + opt.lambda_b * loss_boundary

            if not torch.isfinite(loss_total):
                msg = (f'Non-finite loss at epoch {epoch} step {step} rate {rate}: '
                       f'seg={loss_seg.item():.4f} boundary={loss_boundary.item():.4f} — skipping step')
                print(f'[WARN] {msg}')
                logging.warning(msg)
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.update()   # keep scaler state consistent
                continue

            if scaler is not None:
                scaler.scale(loss_total).backward()
                if opt.clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                if opt.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()

        if step % 20 == 0 or step == total_step:
            msg = ('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss_seg: {:.4f} Loss_boundary: {:.4f} '
                   '(lambda_b={}) Loss_all: {:.4f}'.format(
                       epoch, opt.epoch, step, total_step, loss_seg.item(), loss_boundary.item(),
                       opt.lambda_b, loss_total.item()))
            if logvar is not None:
                msg += ' logvar_mean: {:.4f}'.format(logvar.mean().item())
            print('[{}] => {}'.format(datetime.now(), msg))
            logging.info('#TRAIN#:' + msg)

    if epoch % opt.save_epoch == 0:
        torch.save(model.state_dict(), os.path.join(save_path, 'FAPNet_%d.pth' % epoch))
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mae': best_mae,
            'best_epoch': best_epoch,
            'config': vars(opt),
        }
        torch.save(checkpoint, os.path.join(save_path, 'checkpoint_epoch_%d.pth' % epoch))
        torch.save(checkpoint, os.path.join(save_path, 'checkpoint_latest.pth'))


def test(test_loader, model, epoch, save_path, opt, optimizer=None):
    global best_mae, best_epoch
    model.eval()

    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)

            image = image.cuda()

            res = model(image)[3]                                  # final segmentation logits
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size

        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if mae < best_mae:
            best_mae   = mae
            best_epoch = epoch

            torch.save(model.state_dict(), os.path.join(save_path, 'FAPNet_best.pth'))
            if optimizer is not None:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mae': best_mae,
                    'best_epoch': best_epoch,
                    'config': vars(opt),
                }
                torch.save(checkpoint, os.path.join(save_path, 'checkpoint_best.pth'))
            print('best epoch:{}'.format(epoch))

        model.train()
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='JSON file of argument defaults (see experiments/*.json); CLI flags override it')
    parser.add_argument('--exp_name', type=str, default='',
                        help='experiment name; checkpoints/logs go to <save_model>/<exp_name>/')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='init learning rate')
    parser.add_argument('--batchsize', type=int, default=32, help='training batch size (Note: ~500MB per img in GPU)')
    parser.add_argument('--trainsize', type=int, default=352, help='the size of training image')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate per decay step')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every N epochs decay lr')
    parser.add_argument('--gpu', type=int, default=0, help='choose which gpu you use')
    parser.add_argument('--save_epoch', type=int, default=5, help='every N epochs save your trained snapshot')
    parser.add_argument('--save_model', type=str, default='./Snapshot/FAPNet/')

    # ---- experiment options ----
    parser.add_argument('--boundary_mode', type=str, default='file', choices=['file', 'morph'],
                        help="H1: 'file' = legacy precomputed edge maps (control); 'morph' = morphological boundary of the transformed GT")
    parser.add_argument('--boundary_kernel_size', type=int, default=3, help='H1: morphological kernel size (e.g. 3 or 5)')
    parser.add_argument('--lambda_b', type=float, default=1.0, help='weight of the boundary loss')
    parser.add_argument('--boundary_head', type=str, default='single', choices=['single', 'multiscale'],
                        help="H2: 'single' = original boundary branch (control); 'multiscale' = fused deeper feature")
    parser.add_argument('--uncertainty', type=str, default='none', choices=['none', 'seg', 'seg_boundary'],
                        help='H3: uncertainty-weighted loss on segmentation only, or segmentation + boundary')
    parser.add_argument('--lambda_u', type=float, default=1.0, help='H3: weight of the log-variance regulariser')
    parser.add_argument('--gt_noise', type=int, default=1, help='salt-and-pepper noise on GT (legacy behaviour)')
    parser.add_argument('--legacy_wbce', type=int, default=0,
                        help='1 = reproduce the original reduce=\'none\' BCE quirk (scalar-mean BCE)')

    default_train_img = './data/TrainDataset/TrainDataset/Imgs/' if os.path.exists('./data/TrainDataset/TrainDataset/Imgs/') else './data/TrainDataset/Imgs/'
    default_train_gt  = './data/TrainDataset/TrainDataset/GT/'   if os.path.exists('./data/TrainDataset/TrainDataset/GT/')   else './data/TrainDataset/GT/'
    default_train_eg  = './data/TrainDataset/TrainDataset/Edge/' if os.path.exists('./data/TrainDataset/TrainDataset/Edge/') else './data/TrainDataset/Edge/'

    default_test_img  = './data/TestDataset/TestDataset/CAMO/Imgs/' if os.path.exists('./data/TestDataset/TestDataset/CAMO/Imgs/') else './data/TestDataset/CAMO/Imgs/'
    default_test_gt   = './data/TestDataset/TestDataset/CAMO/GT/'   if os.path.exists('./data/TestDataset/TestDataset/CAMO/GT/')   else './data/TestDataset/CAMO/GT/'
    default_test_eg   = './data/TestDataset/TestDataset/CAMO/Edge/' if os.path.exists('./data/TestDataset/TestDataset/CAMO/Edge/') else './data/TestDataset/CAMO/Edge/'

    parser.add_argument('--train_img_dir', type=str, default=default_train_img)
    parser.add_argument('--train_gt_dir',  type=str, default=default_train_gt)
    parser.add_argument('--train_eg_dir',  type=str, default=default_train_eg)

    parser.add_argument('--test_img_dir',  type=str, default=default_test_img)
    parser.add_argument('--test_gt_dir',   type=str, default=default_test_gt)
    parser.add_argument('--test_eg_dir',   type=str, default=default_test_eg)
    parser.add_argument('--start_epoch',   type=int, default=None,
                        help='epoch to start/resume training from (default: 1 or auto-detected from checkpoint)')
    parser.add_argument('--resume',        type=str, default=None,
                        help='path to checkpoint (.pth) to resume training from')
    parser.add_argument('--amp',           action='store_true',
                        help='Use automatic mixed precision (FP16) to reduce GPU memory usage')
    parser.add_argument('--num_workers',   type=int, default=0,
                        help='number of dataloader workers (0 is safest on Windows to avoid RAM exhaustion)')
    parser.add_argument('--single_scale',  action='store_true',
                        help='Train at 1x resolution instead of 3 multi-scale passes [0.75, 1, 1.25] (3x faster)')

    # JSON config provides defaults; explicit CLI flags still win
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=None)
    known, _ = pre.parse_known_args()
    if known.config:
        with open(known.config) as f:
            cfg = json.load(f)
        valid = {a.dest for a in parser._actions}
        unknown = set(cfg) - valid
        assert not unknown, f'unknown keys in {known.config}: {sorted(unknown)}'
        parser.set_defaults(**cfg)

    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu)
    set_seed(opt.seed)

    ## experiment directory + log
    save_path = os.path.join(opt.save_model, opt.exp_name) if opt.exp_name else opt.save_model
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(vars(opt), f, indent=2)

    logging.basicConfig(filename=os.path.join(save_path, 'log.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("COD-Train")
    logging.info("Config: " + json.dumps(vars(opt), sort_keys=True))

    model     = FAPNet(channel=64, boundary_head=opt.boundary_head,
                       uncertainty=(opt.uncertainty != 'none')).cuda()
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    # AMP scaler (None = FP32 mode, no change in behaviour)
    scaler = GradScaler() if opt.amp else None
    if scaler is not None:
        print("[AMP] Mixed precision (FP16) enabled — GPU memory usage will be ~40-50% lower.")

    torch.cuda.empty_cache()

    start_epoch = 1
    if opt.resume:
        if os.path.isfile(opt.resume):
            print(f"=> Loading checkpoint '{opt.resume}'")
            logging.info(f"=> Loading checkpoint '{opt.resume}'")
            checkpoint = torch.load(opt.resume, map_location='cuda')

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("=> Restored optimizer state from checkpoint")
                if 'best_mae' in checkpoint:
                    best_mae = checkpoint['best_mae']
                if 'best_epoch' in checkpoint:
                    best_epoch = checkpoint['best_epoch']
                if opt.start_epoch is not None:
                    start_epoch = opt.start_epoch
                elif 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                print(f"=> Loaded checkpoint successfully (trained through epoch {checkpoint.get('epoch', 'N/A')}, best MAE: {best_mae:.4f} at epoch {best_epoch})")
            else:
                # Direct model weights (e.g. FAPNet_15.pth or FAPNet_best.pth)
                model.load_state_dict(checkpoint)
                print("=> Loaded model weights from state dictionary")

                base_dir = os.path.dirname(opt.resume)
                file_name = os.path.basename(opt.resume)
                match = re.search(r'FAPNet_(\d+)\.pth', file_name)
                matched_epoch = int(match.group(1)) if match else None

                if matched_epoch is not None:
                    ckpt_alt = os.path.join(base_dir, f'checkpoint_epoch_{matched_epoch}.pth')
                    if os.path.isfile(ckpt_alt):
                        print(f"=> Found companion checkpoint '{ckpt_alt}', restoring optimizer and stats...")
                        ckpt_data = torch.load(ckpt_alt, map_location='cuda')
                        if 'optimizer_state_dict' in ckpt_data:
                            optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
                        if 'best_mae' in ckpt_data:
                            best_mae = ckpt_data['best_mae']
                        if 'best_epoch' in ckpt_data:
                            best_epoch = ckpt_data['best_epoch']

                if opt.start_epoch is not None:
                    start_epoch = opt.start_epoch
                elif matched_epoch is not None:
                    start_epoch = matched_epoch + 1
                else:
                    start_epoch = 1

            print(f"=> Resuming training at epoch {start_epoch}")
            logging.info(f"=> Resuming training at epoch {start_epoch}")
        else:
            raise FileNotFoundError(f"No checkpoint found at '{opt.resume}'")
    elif opt.start_epoch is not None:
        start_epoch = opt.start_epoch

    train_loader = get_loader(opt.train_img_dir, opt.train_gt_dir,
                              opt.train_eg_dir if opt.boundary_mode == 'file' else None,
                              batchsize=opt.batchsize, trainsize=opt.trainsize,
                              num_workers=opt.num_workers, pin_memory=(opt.num_workers > 0),
                              boundary_mode=opt.boundary_mode,
                              boundary_kernel_size=opt.boundary_kernel_size,
                              gt_noise=bool(opt.gt_noise), seed=opt.seed)
    test_loader  = test_dataset(opt.test_img_dir, opt.test_gt_dir, testsize=opt.trainsize)
    total_step   = len(train_loader)

    print('--------------------starting-------------------')
    print('-' * 30, "\n[Training Dataset INFO]\nexp: {}\nimg_dir: {}\ngt_dir: {}\nboundary: {} (k={}) lambda_b={}\n"
                    "boundary_head: {}\nuncertainty: {} (lambda_u={})\nLearning Rate: {}\nBatch Size: {}\n"
                    "Training Save: {}\ntotal_num: {}\nStarting Epoch: {}\nTotal Epochs: {}\n".format(
                        opt.exp_name or '(default)', opt.train_img_dir, opt.train_gt_dir,
                        opt.boundary_mode, opt.boundary_kernel_size, opt.lambda_b,
                        opt.boundary_head, opt.uncertainty, opt.lambda_u,
                        opt.lr, opt.batchsize, save_path, total_step, start_epoch, opt.epoch), '-' * 30)

    for epoch_iter in range(start_epoch, opt.epoch + 1):
        adjust_lr(optimizer, epoch_iter, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch_iter, opt, total_step, save_path, scaler)
        test(test_loader, model, epoch_iter, save_path, opt, optimizer)
