# -*- coding: utf-8 -*-
"""
Khoảng cách bề mặt & SDF/DTM bổ sung. Dọn lại để tránh trùng.
- compute_dtm: DTM foreground hoặc fg+bg.
- hd_loss: Hausdorff-like loss (dựa trọng số DTM), cần torch.
- compute_sdf: import từ util để tránh trùng logic.
- (Bỏ boundary_loss/sdf_loss cũ để đơn giản hoá; nếu cần có thể thêm lại sau).
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from .util import compute_sdf_numpy


def compute_dtm(img_gt, out_shape, normalize=False, fg=False):
    """
    Distance Transform Map cho mask nhị phân.
    img_gt: ndarray [B,D,H,W] hoặc [D,H,W]
    fg=False: tổng fg+bg; fg=True: chỉ fg
    """
    arr = img_gt.astype(np.uint8)
    if arr.ndim == 3:
        arr = arr[None, ...]
    B = arr.shape[0]
    fg_dtm = np.zeros((B, *arr.shape[1:]), dtype=np.float32)

    for b in range(B):
        posmask = arr[b].astype(bool)
        if not fg:
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                if normalize:
                    posdis = (posdis - posdis.min()) / max(1e-6, (posdis.max() - posdis.min()))
                    negdis = (negdis - negdis.min()) / max(1e-6, (negdis.max() - negdis.min()))
                    fg_dtm[b] = posdis + negdis
                else:
                    fg_dtm[b] = posdis + negdis
                fg_dtm[b][boundary == 1] = 0
        else:
            if posmask.any():
                posdis = distance(posmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                if normalize:
                    posdis = (posdis - posdis.min()) / max(1e-6, (posdis.max() - posdis.min()))
                fg_dtm[b] = posdis
                fg_dtm[b][boundary == 1] = 0
    return fg_dtm.reshape(out_shape).astype(np.float32)


def hd_loss(seg_soft, gt, gt_dtm=None, one_side=True, seg_dtm=None):
    """
    Hausdorff-like boundary loss (Karimi et al. style).
    seg_soft: torch [B,D,H,W] (prob của lớp 1)
    gt: torch [B,D,H,W] (0/1)
    gt_dtm: torch/ndarray [B,D,H,W], distance transform của gt
    one_side=True: chỉ dùng gt_dtm; False: cộng thêm seg_dtm
    """
    if gt_dtm is None:
        # tính trên CPU (numpy) rồi đưa lên device
        gt_np = gt.detach().cpu().numpy().astype(np.uint8)
        dtm_np = compute_dtm(gt_np, gt_np.shape, normalize=False, fg=True)
        gt_dtm = torch.from_numpy(dtm_np).to(seg_soft.device, seg_soft.dtype)

    delta_s = (seg_soft - gt.float()) ** 2
    if one_side:
        dtm = gt_dtm ** 2
    else:
        if seg_dtm is None:
            with torch.no_grad():
                seg_bin = (seg_soft > 0.5).float()
                seg_np = seg_bin.detach().cpu().numpy().astype(np.uint8)
                seg_dtm_np = compute_dtm(seg_np, seg_np.shape, normalize=False, fg=True)
                seg_dtm = torch.from_numpy(seg_dtm_np).to(seg_soft.device, seg_soft.dtype)
        dtm = (gt_dtm ** 2) + (seg_dtm ** 2)

    multipled = delta_s * dtm
    return multipled.mean()


# Giữ tên hàm compute_sdf để backward-compat (gọi bản thống nhất ở util)
def compute_sdf(img_gt, out_shape=None):
    sdf = compute_sdf_numpy(img_gt)
    if out_shape is not None:
        sdf = sdf.reshape(out_shape)
    return sdf.astype(np.float32)
