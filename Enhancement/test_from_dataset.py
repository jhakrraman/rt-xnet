# RT-X Net: RGB-Thermal cross attention network for Low-Light Image Enhancement
# Raman Jha, Adithya Lenka, Mani Ramanagopal, Aswin Sankaranarayanan, Kaushik Mitra
# International Conference on Image Processing, IEEE ICIP 2025
# https://arxiv.org/abs/2505.24705
# https://github.com/jhakrraman/rt-xnet

from ast import arg
import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils

from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte
from pdb import set_trace as stx
from skimage import metrics

from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse

def self_ensemble(x_tuple, model):
    def forward_transformed(x_tuple, hflip, vflip, rotate, model):
        x_rgb, x_thermal = x_tuple
        
        if hflip:
            x_rgb = torch.flip(x_rgb, (-2,))
            x_thermal = torch.flip(x_thermal, (-2,))
        if vflip:
            x_rgb = torch.flip(x_rgb, (-1,))
            x_thermal = torch.flip(x_thermal, (-1,))
        if rotate:
            x_rgb = torch.rot90(x_rgb, dims=(-2, -1))
            x_thermal = torch.rot90(x_thermal, dims=(-2, -1))
            
        result = model((x_rgb, x_thermal))
        
        if rotate:
            result = torch.rot90(result, dims=(-2, -1), k=3)
        if vflip:
            result = torch.flip(result, (-1,))
        if hflip:
            result = torch.flip(result, (-2,))
        return result
    
    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x_tuple, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)


parser = argparse.ArgumentParser(
    description='Image Enhancement using RTxNet')

parser.add_argument('--input_dir', default='./Enhancement/Datasets',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/',
                    type=str, help='Directory for results')
parser.add_argument('--output_dir', default='',
                    type=str, help='Directory for output')
parser.add_argument(
    '--opt', type=str, default='Options/RTxNet_LLVIP.yml', help='Path to option YAML file.')
parser.add_argument('--weights', default='pretrained_weights/LLVIP_best.pth',
                    type=str, help='Path to weights')
parser.add_argument('--dataset', default='LLVIP', type=str,
                    help='Test Dataset') 
parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')
parser.add_argument('--GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')
parser.add_argument('--self_ensemble', action='store_true', help='Use self-ensemble to obtain better results')

args = parser.parse_args()

# gpu
gpu_list = ','.join(str(x) for x in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

####### Load yaml #######
yaml_file = args.opt
weights = args.weights
print(f"dataset {args.dataset}")

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt = parse(args.opt, is_train=False)
opt['dist'] = False


x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')
##########################


model_restoration = create_model(opt).net_g

checkpoint = torch.load(weights)

try:
    model_restoration.load_state_dict(checkpoint['params'])
except:
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)

print("===>Testing using weights: ", weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

factor = 4
dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
output_dir = args.output_dir
# stx()
os.makedirs(result_dir, exist_ok=True)
if args.output_dir != '':
    os.makedirs(output_dir, exist_ok=True)

psnr = []
ssim = []

input_dir_lq = opt['datasets']['val']['dataroot_lq']
input_dir_th = opt['datasets']['val']['dataroot_th']
target_dir = opt['datasets']['val']['dataroot_gt']
print(input_dir_lq)
print(input_dir_th)
print(target_dir)

input_paths_lq = natsorted(
    glob(os.path.join(input_dir_lq, '*.png')) + glob(os.path.join(input_dir_lq, '*.jpg')))

input_paths_th = natsorted(
    glob(os.path.join(input_dir_th, '*.png')) + glob(os.path.join(input_dir_th, '*.jpg')))


target_paths = natsorted(glob(os.path.join(
    target_dir, '*.png')) + glob(os.path.join(target_dir, '*.jpg')))

with torch.inference_mode():
    for inp_path_lq, inp_path_th, tar_path in tqdm(zip(input_paths_lq, input_paths_th, target_paths), total=len(target_paths)):

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img_lq = np.float32(utils.load_img(inp_path_lq)) / 255.
        img_th = np.float32(utils.load_img(inp_path_th)) / 255.

        target = np.float32(utils.load_img(tar_path)) / 255.

        img_rgb = torch.from_numpy(img_lq).permute(2, 0, 1)
        input_rgb = img_rgb.unsqueeze(0).cuda()

        img_th = torch.from_numpy(img_th).permute(2, 0, 1)
        input_thermal = img_th.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 4
        # Padding for both inputs
        #input_rgb, input_thermal = input_tuple
        h, w = input_rgb.shape[2], input_rgb.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0

        input_rgb = F.pad(input_rgb, (0, padw, 0, padh), 'reflect')
        input_thermal = F.pad(input_thermal, (0, padw, 0, padh), 'reflect')
        input_tuple = (input_rgb, input_thermal)


        if h < 3000 and w < 3000:
            # Replace single input with tuple
            if args.self_ensemble:
                restored = self_ensemble(input_tuple, model_restoration)
            else:
                restored = model_restoration(input_tuple)

        else:
            # split and test
            input_1 = input_rgb[:, :, :, 1::2]
            input_2 = input_rgb[:, :, :, 0::2]
            if args.self_ensemble:
                restored_1 = self_ensemble(input_1, model_restoration)
                restored_2 = self_ensemble(input_2, model_restoration)
            else:
                restored_1 = model_restoration(input_1)
                restored_2 = model_restoration(input_2)
            restored = torch.zeros_like(input_rgb)
            restored[:, :, :, 1::2] = restored_1
            restored[:, :, :, 0::2] = restored_2

        # Unpad images to original dimensions
        restored = restored[:, :, :h, :w]

        restored = torch.clamp(restored, 0, 1).cpu(
        ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        if args.GT_mean:
            # This test setting is the same as KinD, LLFlow, and recent diffusion models
            # Please refer to Line 73 (https://github.com/zhangyhuaee/KinD/blob/master/evaluate_LOLdataset.py)
            mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
            mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
            restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

        psnr.append(utils.PSNR(target, restored))
        ssim.append(utils.calculate_ssim(
            img_as_ubyte(target), img_as_ubyte(restored)))
        if output_dir != '':
            utils.save_img((os.path.join(output_dir, os.path.splitext(
                os.path.split(inp_path_lq)[-1])[0] + '.png')), img_as_ubyte(restored))
        else:
            utils.save_img((os.path.join(result_dir, os.path.splitext(
                os.path.split(inp_path_lq)[-1])[0] + '.png')), img_as_ubyte(restored))

psnr = np.mean(np.array(psnr))
ssim = np.mean(np.array(ssim))
print("PSNR: %f " % (psnr))
print("SSIM: %f " % (ssim))
