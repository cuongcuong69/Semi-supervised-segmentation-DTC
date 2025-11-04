# -*- coding: utf-8 -*-
"""
Hàm tiện ích dùng chung (phiên bản torch nếu cần).
"""
import torch
from .util import compute_sdf_numpy


@torch.no_grad()
def compute_sdf_torch(mask: torch.Tensor) -> torch.Tensor:
    """
    Tính SDF trên CPU bằng numpy rồi trả torch tensor (dùng cho inference/val).
    mask: [N,1,D,H,W] hoặc [N,D,H,W] (0/1)
    """
    if mask.dim() == 5:
        mask_np = mask[:, 0].detach().cpu().numpy()
    else:
        mask_np = mask.detach().cpu().numpy()
    sdf_np = compute_sdf_numpy(mask_np)  # [N,D,H,W]
    sdf = torch.from_numpy(sdf_np).to(mask.device, mask.dtype)
    return sdf.unsqueeze(1)  # [N,1,D,H,W]
