import os
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.model import load_fapnet
from utils.dataloader import tt_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize',   type=int, default=352)
parser.add_argument('--model_path', type=str, default='./Snapshot/FAPNet/FAPNet_best.pth')
parser.add_argument('--save_path',  type=str, default='./results/',
                    help='predictions go to <save_path>/<dataset>/ (use one folder per experiment)')
parser.add_argument('--data_root',  type=str, default=os.path.join('data', 'TestDataset', 'TestDataset'))
parser.add_argument('--datasets',   type=str, nargs='+', default=['CHAMELEON', 'CAMO', 'COD10K'])
parser.add_argument('--gpu',        type=int, default=0)
parser.add_argument('--strict',     type=int, default=1,
                    help='1 = strict state_dict loading (baseline); 0 = non-strict')
parser.add_argument('--boundary_head', type=str, default=None, choices=['single', 'multiscale'],
                    help='default: read from the checkpoint config, else "single"')
parser.add_argument('--uncertainty', type=int, default=None, choices=[0, 1],
                    help='default: read from the checkpoint config, else 0')
parser.add_argument('--save_aux',   action='store_true',
                    help='also save predicted boundary / uncertainty maps (for the qualitative grid)')
opt = parser.parse_args()

device = torch.device(f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu')

model, ckpt, (missing, unexpected) = load_fapnet(
    opt.model_path, device,
    boundary_head=opt.boundary_head,
    uncertainty=None if opt.uncertainty is None else bool(opt.uncertainty),
    strict=bool(opt.strict))
if missing or unexpected:
    print(f'[load] missing: {missing}\n[load] unexpected: {unexpected}')
print(f'Loaded {opt.model_path}  (boundary_head={model.boundary_head}, uncertainty={model.uncertainty}'
      + (f", epoch={ckpt.get('epoch', '?')}, best MAE={ckpt.get('best_mae', 'N/A')}" if ckpt else '') + ')')


def to_u8(x):
    """Min-max normalise a float map to uint8."""
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return (x * 255).astype(np.uint8)


def up_sigmoid(t, size):
    t = F.interpolate(t, size=size, mode='bilinear', align_corners=False)
    return t.sigmoid().squeeze().cpu().numpy()


with torch.no_grad():
    for dataset in opt.datasets:
        save_dir = os.path.join(opt.save_path, dataset)
        os.makedirs(save_dir, exist_ok=True)
        if opt.save_aux:
            os.makedirs(os.path.join(save_dir, 'edge'), exist_ok=True)
            if model.uncertainty:
                os.makedirs(os.path.join(save_dir, 'uncertainty'), exist_ok=True)

        loader = tt_dataset(
            os.path.join(opt.data_root, dataset, 'Imgs') + os.sep,
            os.path.join(opt.data_root, dataset, 'GT') + os.sep,
            opt.testsize,
        )
        print(f'{dataset}: {loader.size} images')

        for _ in range(loader.size):
            image, gt, name = loader.load_data()
            gt_shape = np.asarray(gt, np.float32).shape      # (H, W)
            image = image.to(device)

            outs = model(image)
            # Contract: outs[3] = final segmentation, outs[4] = boundary logits,
            # outs[5] (only with uncertainty) = log-variance map.
            cam = outs[3]

            cv2.imwrite(os.path.join(save_dir, name), to_u8(up_sigmoid(cam, gt_shape)))

            if opt.save_aux:
                cv2.imwrite(os.path.join(save_dir, 'edge', name),
                            to_u8(up_sigmoid(outs[4], gt_shape)))
                if len(outs) > 5:
                    unc = F.interpolate(outs[5], size=gt_shape, mode='bilinear', align_corners=False)
                    cv2.imwrite(os.path.join(save_dir, 'uncertainty', name),
                                to_u8(unc.squeeze().cpu().numpy()))
