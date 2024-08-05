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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn
from utils.general_utils import build_rotation

def TV_loss(x):
    B, C, H, W = x.shape
    tv_h = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).sum()
    tv_w = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).sum()
    return (tv_h + tv_w) / (B * C * H * W)


def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()

def l1_loss(network_output, gt, mask=None):
    loss = torch.abs((network_output - gt))
    if mask is not None:
        if mask.ndim == 4:
            mask = mask.repeat(1, network_output.shape[1], 1, 1)
        elif mask.ndim == 3:
            mask = mask.repeat(network_output.shape[1], 1, 1)
        else:
            raise ValueError('the dimension of mask should be either 3 or 4')
    
        try:
            loss = loss[mask!=0]
        except:
            print(loss.shape)
            print(mask.shape)
            print(loss.dtype)
            print(mask.dtype)
    return loss.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)








#EXTENT
def get_smallest_axis(rot, scaling, return_idx=False):
    """Returns the smallest axis of the Gaussians.

    Args:
        return_idx (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    rotation_matrices = build_rotation(rot)
    # print(rotation_matrices)
    # print(rotation_matrices.shape)
    # print(scaling.shape)
    # print((scaling.min(dim=-1)[1]).shape)
    # print(scaling)
    smallest_axis_idx = scaling.min(
        dim=-1)[1][..., None, None].expand(-1, 3, -1)
    smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
    # print(smallest_axis)
    # print(smallest_axis.shape)
    # exit()

    if return_idx:
        return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
    return smallest_axis.squeeze(dim=2)


from torchmetrics.regression import PearsonCorrCoef
def confidence_loss(gt, pred, confidence, mask):
    # mask = torch.logical_or(torch.isnan(pred),
    #                         ~mask.expand(-1, pred.shape[1], -1, -1))
    # mask = ~torch.logical_or(mask,
    #                          torch.isnan(confidence).expand(-1, pred.shape[1], -1, -1))
    # gt = gt[mask]
    # pred = pred[mask]
    # confidence = confidence[mask[:, 0:1]]
    if min(pred.shape) == 0 or min(confidence.shape) == 0:
        return 0
    loss = F.mse_loss(gt, pred, reduction='mean')
    # print(torch.isnan(loss))
    loss = loss / ((2*confidence**2).mean()) + torch.log(confidence).mean()
    return loss


def mae_loss(img1, img2, mask):
    mae_map = -torch.sum(img1 * img2, dim=1, keepdims=True) + 1
    # loss_map = torch.abs(mae_map * mask)
    loss_map = torch.abs(mae_map)
    loss = torch.mean(loss_map)
    return loss

def grad(x):
    # x.shape : n, c, h, w
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    # z = torch.ones_like(diff_x)
    # normal = torch.cat([-diff_x, -diff_y, -z], dim=1)
    # normal = F.normalize(normal, dim=1)
    # cv2.imwrite('norm.png', (normal*255).squeeze(0).permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
    # print(diff_x.max(), diff_y.max())
    mag = diff_x**2 + diff_y**2
    # angle_ratio
    angle = torch.atan(diff_y / (diff_x + 1e-10))
    return mag, angle

def extract_key(prediction, key):
    if isinstance(prediction, dict):
        return prediction[key]
    return prediction
KEY_OUTPUT = 'metric_depth'

class GradL1Loss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(GradL1Loss, self).__init__()
        self.name = 'GradL1'
        self.pc_loss = PearsonCorrCoef().cuda()

    def forward(self, input, target, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        grad_gt = grad(target)
        # print(grad_gt.shape)
        grad_pred = grad(input)

        grad_pred_0 = grad_pred[0].reshape(-1)
        grad_gt_0 = grad_gt[0].reshape(-1)

        loss = 1 - self.pc_loss(grad_pred_0[:, None], grad_gt_0[:, None])
        

        if not return_interpolated:
            return loss
        return loss, intr_input
    

