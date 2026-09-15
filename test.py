import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse


from lib.model import FAPNet
from utils.dataloader import tt_dataset
from utils.eva_funcs import eval_Smeasure,eval_mae,numpy2tensor
import scipy.io as scio 
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--testsize',   type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str, default='./Snapshot/FAPNet/FAPNet_best.pth',
                    help='path to trained model weights (.pth)')
parser.add_argument('--save_path',  type=str, default='./results/')

opt   = parser.parse_args()
model = FAPNet(channel=64).cuda()

# Support both full checkpoints (with model_state_dict key) and raw state dicts
cur_model_path = opt.model_path
checkpoint_data = torch.load(cur_model_path, map_location='cuda')
if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
    model.load_state_dict(checkpoint_data['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint_data.get('epoch', '?')} "
          f"(best MAE: {checkpoint_data.get('best_mae', 'N/A')})")
else:
    model.load_state_dict(checkpoint_data)
    print(f"Loaded model weights from: {cur_model_path}")
model.eval()
        
    
################################################################

for dataset in ['CHAMELEON', 'CAMO', 'COD10K']:
    save_path = os.path.join(opt.save_path, dataset)
    os.makedirs(save_path, exist_ok=True)

    test_loader = tt_dataset(
    os.path.join('data', 'TestDataset', 'TestDataset', dataset, 'Imgs') + os.sep,
    os.path.join('data', 'TestDataset', 'TestDataset', dataset, 'GT') + os.sep,
    opt.testsize
)
    print(f"[DEBUG] {dataset}: found {test_loader.size} images")

    for iteration in range(test_loader.size):
            
            
        image, gt, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
            

        _,_,_, cam,_ = model(image)
            
        res = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        
################################################################
        cv2.imwrite(os.path.join(save_path, name), res*255)
        


 