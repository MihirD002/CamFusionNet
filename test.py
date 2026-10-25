import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse


from lib.model import FAPNet, MFAM_VARIANTS
from utils.dataloader import tt_dataset
from utils.checkpoint import load_model_weights
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--testsize',   type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str, default='./Snapshot/FAPNet/FAPNet_best.pth',
                    help='path to trained model weights (.pth)')
parser.add_argument('--mfam_variant', type=str, default='original', choices=MFAM_VARIANTS,
                    help='MFAM variant the checkpoint was trained with (default: original)')
parser.add_argument('--save_path',  type=str, default=None,
                    help="prediction output dir (default: ./results/ for 'original', "
                         "./results_<variant>/ otherwise)")
parser.add_argument('--data_root',  type=str, default=os.path.join('data', 'TestDataset', 'TestDataset'),
                    help='folder containing <DATASET>/Imgs and <DATASET>/GT')
parser.add_argument('--datasets',   type=str, nargs='+', default=['CHAMELEON', 'CAMO', 'COD10K'])

opt   = parser.parse_args()
if opt.save_path is None:
    opt.save_path = './results/' if opt.mfam_variant == 'original' else './results_{}/'.format(opt.mfam_variant)

model = FAPNet(channel=64, mfam_variant=opt.mfam_variant).cuda()

# Supports full checkpoints (model_state_dict key) and raw state dicts; raises a clear
# error if the checkpoint was trained with a different MFAM variant.
meta = load_model_weights(model, opt.model_path, opt.mfam_variant, device='cuda')
if meta:
    print(f"Loaded checkpoint from epoch {meta.get('epoch', '?')} "
          f"(best MAE: {meta.get('best_mae', 'N/A')}, MFAM variant: {opt.mfam_variant})")
else:
    print(f"Loaded model weights from: {opt.model_path} (MFAM variant: {opt.mfam_variant})")
model.eval()
        
    
################################################################

for dataset in opt.datasets:
    save_path = os.path.join(opt.save_path, dataset)
    os.makedirs(save_path, exist_ok=True)

    test_loader = tt_dataset(
        os.path.join(opt.data_root, dataset, 'Imgs') + os.sep,
        os.path.join(opt.data_root, dataset, 'GT') + os.sep,
        opt.testsize
    )
    print(f"[DEBUG] {dataset}: found {test_loader.size} images")

    for iteration in range(test_loader.size):
            
            
        image, gt, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
            

        with torch.no_grad():
            _,_,_, cam,_ = model(image)
            
        res = F.interpolate(cam, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        
################################################################
        # Round to uint8 (cv2.imwrite on float truncates) so the saved map is what gets evaluated.
        cv2.imwrite(os.path.join(save_path, name), np.uint8(np.round(res * 255)))