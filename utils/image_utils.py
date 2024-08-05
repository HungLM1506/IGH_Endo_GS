#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import torch
from typing import List
import re
import subprocess
import os
import tempfile
from PIL import Image
import torch.nn.functional as F


def tensor2array(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor
    
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

@torch.no_grad()
def psnr(img1, img2, mask=None):
    if mask is None:
        mse_mask = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    else:
        if mask.shape[1] == 3:
            mse_mask = (((img1-img2)**2)*mask).sum() / ((mask.sum()+1e-10))
        else:
            mse_mask = (((img1-img2)**2)*mask).sum() / ((mask.sum()+1e-10)*3.0)

    return 20 * torch.log10(1.0 / torch.sqrt(mse_mask))

def rmse(a, b, mask=None):
    """Compute rmse.
    """
    if torch.is_tensor(a):
        a = tensor2array(a)
    if torch.is_tensor(b):
        b = tensor2array(b)
    if torch.is_tensor(mask):
        mask = tensor2array(mask)

    if mask is None:
        rmse = (((a - b)**2).sum() / (a.shape[-1]*a.shape[-2]))**0.5
    else:
        if len(mask.shape) == len(a.shape) - 1:
            mask = mask[..., None]
        mask_sum = np.sum(mask) + 1e-10
        rmse = (((a - b)**2 * mask).sum() / (mask_sum))**0.5
    
    return rmse



def flip(pred_frames: List[np.ndarray], gt_frames: List[np.ndarray], interval: int = 10) -> float:
    
    def extract_from_result(text: str, prompt: str):
        m = re.search(prompt, text)
        return float(m.group(1))

    all_results = []

    pred_frames = [e.squeeze(0).permute(1, 2, 0).cpu().numpy() for e in pred_frames]
    gt_frames = [e.squeeze(0).permute(1, 2, 0).cpu().numpy() for e in gt_frames]

    with tempfile.TemporaryDirectory() as tmpdir:
        pred_fname = os.path.join(tmpdir, "pred.png")
        gt_fname = os.path.join(tmpdir, "gt.png")
        for i in range(len(pred_frames)):
            write_png(pred_fname, pred_frames[i])
            write_png(gt_fname, gt_frames[i])
            result = subprocess.check_output(
                ['python', 'flip/flip.py', '--reference', gt_fname, '--test', pred_fname]
            ).decode()
            all_results.append(extract_from_result(result, r'Mean: (\d+\.\d+)'))
    return sum(all_results) / len(all_results)


def write_png(path, data):
    """Writes an PNG image to some path.

    Args:
        path (str): Path to save the PNG.
        data (np.array): HWC image.

    Returns:
        (void): Writes to path.
    """
    Image.fromarray(data).save(path)


def ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)