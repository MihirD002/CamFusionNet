
import torch
import os
import numpy as np
import torch.nn.functional as F
import argparse
import logging
from torch.cuda.amp import autocast, GradScaler

# Ensure cuDNN benchmark is False so it doesn't allocate huge workspaces to benchmark algorithms
torch.backends.cudnn.benchmark = False

from lib.model import FAPNet
from utils.dataloader import get_loader,test_dataset
from utils.trainer import adjust_lr
from datetime import datetime


best_mae   = 1
best_epoch = 0


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()


def train(train_loader, model, optimizer, epoch, opt, loss_func, total_step, scaler=None):
    """
    Training iteration
    :param train_loader:
    :param model:
    :param optimizer:
    :param epoch:
    :param opt:
    :param loss_func:
    :param total_step:
    :param scaler: GradScaler for AMP mixed precision (or None to use FP32)
    :return:
    """
    model.train()
    
    size_rates = [0.75, 1, 1.25]
    
    for step, data_pack in enumerate(train_loader):
        images, gts, egs = data_pack
        images = images.cuda()
        gts    = gts.cuda()
        egs    = egs.cuda()

        for rate in size_rates:

            optimizer.zero_grad()

            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images_r = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts_r    = F.interpolate(gts,    size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                egs_r    = F.interpolate(egs,    size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            else:
                images_r, gts_r, egs_r = images, gts, egs

            with autocast(enabled=(scaler is not None)):
                sal1,sal2,sal3,sal4,edge_out = model(images_r)
                
                loss_edge = loss_func(edge_out, egs_r)
            
                loss1 = structure_loss(sal1, gts_r)
                loss2 = structure_loss(sal2, gts_r)
                loss3 = structure_loss(sal3, gts_r)
                loss4 = structure_loss(sal4, gts_r)
                
                loss_obj = loss1 + loss2 + loss3 + loss4

                loss_total  = loss_obj + loss_edge

            if scaler is not None:
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                optimizer.step()

        if step % 20 == 0 or step == total_step:
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_obj: {:.4f} Loss_edge: {:0.4f} Loss_all: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss_obj.data,loss_edge.data,  loss_total.data))
            
            logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss_obj: {:.4f} Loss_edge: {:0.4f} Loss_all: {:0.4f}'.
                    format( epoch, opt.epoch, step, total_step, loss_obj.data,loss_edge.data,  loss_total.data))
                                                        

    if (epoch) % opt.save_epoch == 0:
        torch.save(model.state_dict(), os.path.join(save_path, 'FAPNet_%d.pth' % (epoch)))
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mae': best_mae,
            'best_epoch': best_epoch,
        }
        torch.save(checkpoint, os.path.join(save_path, 'checkpoint_epoch_%d.pth' % (epoch)))
        torch.save(checkpoint, os.path.join(save_path, 'checkpoint_latest.pth'))


def test(test_loader, model, epoch, save_path, optimizer=None):
    
    global best_mae, best_epoch
    model.eval()
    
    with torch.no_grad():
        mae_sum=0
        for i in range(test_loader.size):
            image, gt, name,_ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            
            gt /= (gt.max() + 1e-8)
            
            image = image.cuda()

            _,_,_,res,_ = model(image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum +=np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])
            
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
                }
                torch.save(checkpoint, os.path.join(save_path, 'checkpoint_best.pth'))
            print('best epoch:{}'.format(epoch))
            
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch,mae,best_epoch,best_mae))
                
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200,
                        help='epoch number, default=30')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='init learning rate, try `lr=1e-4`')
    parser.add_argument('--batchsize', type=int, default=32,
                        help='training batch size (Note: ~500MB per img in GPU)')
    parser.add_argument('--trainsize', type=int, default=352,
                        help='the size of training image, try small resolutions for speed (like 256)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='decay rate of learning rate per decay step')
    parser.add_argument('--decay_epoch', type=int, default=30,
                        help='every N epochs decay lr')
    parser.add_argument('--gpu', type=int, default=0,
                        help='choose which gpu you use')
    parser.add_argument('--save_epoch', type=int, default=5,
                        help='every N epochs save your trained snapshot')
    parser.add_argument('--save_model', type=str, default='./Snapshot/FAPNet/')
    
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
    
    
    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu)
    
    ## log

    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)
    
    
    logging.basicConfig(filename=opt.save_model+'/log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("COD-Train")
    logging.info("Config")
    logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};save_path:{};decay_epoch:{};resume:{};start_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.save_model, opt.decay_epoch, opt.resume, opt.start_epoch))

    

    #
    model     = FAPNet(channel=64).cuda()
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    LogitsBCE = torch.nn.BCEWithLogitsLoss()

    # AMP scaler (None = FP32 mode, no change in behaviour)
    scaler = GradScaler() if opt.amp else None
    if scaler is not None:
        print("[AMP] Mixed precision (FP16) enabled — GPU memory usage will be ~40-50% lower.")

    # Free any leftover GPU memory before allocating the model
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

                # Look for companion checkpoint in the same directory to restore optimizer and best stats
                base_dir = os.path.dirname(opt.resume)
                file_name = os.path.basename(opt.resume)
                import re
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

    #net, optimizer = amp.initialize(model_SINet, optimizer, opt_level='O1')     # NOTES: Ox not 0x

    train_loader = get_loader(opt.train_img_dir,  opt.train_gt_dir, opt.train_eg_dir,
                              batchsize=opt.batchsize, trainsize=opt.trainsize,
                              num_workers=opt.num_workers, pin_memory=(opt.num_workers > 0))
    test_loader  = test_dataset(opt.test_img_dir, opt.test_gt_dir,  testsize=opt.trainsize)
    total_step   = len(train_loader)
    
    print('--------------------starting-------------------')

    print('-' * 30, "\n[Training Dataset INFO]\nimg_dir: {}\ngt_dir: {}\nLearning Rate: {}\nBatch Size: {}\n"
                    "Training Save: {}\ntotal_num: {}\nStarting Epoch: {}\nTotal Epochs: {}\n".format(
                        opt.train_img_dir, opt.train_gt_dir, opt.lr, opt.batchsize, opt.save_model, total_step, start_epoch, opt.epoch), '-' * 30)

    for epoch_iter in range(start_epoch, opt.epoch + 1):
        
        adjust_lr(optimizer, epoch_iter, opt.decay_rate, opt.decay_epoch)
        
        train(train_loader, model, optimizer,  epoch_iter, opt, LogitsBCE, total_step, scaler)
        test(test_loader,   model, epoch_iter, opt.save_model, optimizer)
