# -*- coding: utf-8 -*-
"""
Metrics cho segmentation: Dice/Jaccard/HD95/ASD.
- Mặc định dùng medpy.metric; nếu không có hoặc lỗi HD, bạn có thể
  chuyển sang gói 'surface-distance' trong phiên bản mở rộng.
"""
import numpy as np
from medpy import metric


def cal_dice(prediction, label, num=2):
    """
    Tính dice cho từng lớp (1..num-1).
    prediction, label: ndarray int (0..C-1)
    """
    total_dice = np.zeros(num - 1, dtype=np.float32)
    for i in range(1, num):
        p = (prediction == i).astype(float)
        g = (label == i).astype(float)
        denom = (p.sum() + g.sum())
        total_dice[i - 1] = (2.0 * (p * g).sum() / denom) if denom > 0 else 1.0
    return total_dice


def calculate_metric_percase(pred, gt):
    """
    Trả về: (dc, jc, hd95, asd) dạng float
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    dc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    # hd95/asd có thể lỗi nếu mask rỗng → xử lý an toàn:
    try:
        hd = metric.binary.hd95(pred, gt)
    except Exception:
        hd = np.inf
    try:
        asd = metric.binary.asd(pred, gt)
    except Exception:
        asd = np.inf

    return float(dc), float(jc), float(hd), float(asd)


def dice(input, target, ignore_index=None):
    """
    Dice cho 2 mask nhị phân (tensor/ndarray flatten) — phiên bản đơn giản.
    """
    if hasattr(input, "detach"):  # torch tensor
        iflat = input.detach().reshape(-1).float()
    else:
        iflat = np.asarray(input).reshape(-1).astype(np.float32)

    if hasattr(target, "detach"):
        tflat = target.detach().reshape(-1).float()
    else:
        tflat = np.asarray(target).reshape(-1).astype(np.float32)

    if ignore_index is not None:
        if hasattr(tflat, "clone"):
            mask = (tflat == ignore_index)
            tflat[mask] = 0.0
            iflat[mask] = 0.0
        else:
            mask = (tflat == ignore_index)
            tflat = tflat.copy(); iflat = iflat.copy()
            tflat[mask] = 0.0; iflat[mask] = 0.0

    inter = (iflat * tflat).sum()
    denom = iflat.sum() + tflat.sum()
    return float((2.0 * inter + 1.0) / (denom + 1.0))

# === Add these helper wrappers ===

def jaccard(pred, gt):
    """
    Jaccard (IoU) giữa 2 mask nhị phân (numpy array)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 1.0


# def asd(pred, gt):
#     """
#     Average Surface Distance — dùng medpy.metric.binary.asd
#     """
#     pred = pred.astype(bool)
#     gt = gt.astype(bool)
#     try:
#         return float(metric.binary.asd(pred, gt))
#     except Exception:
#         return np.inf


# def hd95(pred, gt):
#     """
#     95th percentile Hausdorff Distance — dùng medpy.metric.binary.hd95
#     """
#     pred = pred.astype(bool)
#     gt = gt.astype(bool)
#     try:
#         return float(metric.binary.hd95(pred, gt))
#     except Exception:
#         return np.inf

import numpy as np
from medpy import metric

# def dice(input, target, ignore_index=None):
#     # giữ nguyên phần cũ
#     if hasattr(input, "detach"):
#         iflat = input.detach().reshape(-1).float()
#     else:
#         iflat = np.asarray(input).reshape(-1).astype(np.float32)
#     if hasattr(target, "detach"):
#         tflat = target.detach().reshape(-1).float()
#     else:
#         tflat = np.asarray(target).reshape(-1).astype(np.float32)
#     if ignore_index is not None:
#         mask = (tflat == ignore_index)
#         tflat = tflat.clone() if hasattr(tflat, "clone") else tflat.copy()
#         iflat = iflat.clone() if hasattr(iflat, "clone") else iflat.copy()
#         tflat[mask] = 0.0; iflat[mask] = 0.0
#     inter = (iflat * tflat).sum()
#     denom = iflat.sum() + tflat.sum()
#     return float((2.0 * inter + 1.0) / (denom + 1.0))

# def jaccard(pred, gt):
#     pred = pred.astype(bool)
#     gt = gt.astype(bool)
#     inter = np.logical_and(pred, gt).sum()
#     union = np.logical_or(pred, gt).sum()
#     return float(inter / union) if union > 0 else 1.0

def asd(pred, gt, spacing=None):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() or not gt.any():
        return np.inf
    try:
        return float(metric.binary.asd(pred, gt, voxelspacing=spacing))
    except Exception:
        return np.inf

def hd95(pred, gt, spacing=None):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() or not gt.any():
        return np.inf
    try:
        return float(metric.binary.hd95(pred, gt, voxelspacing=spacing))
    except Exception:
        return np.inf
