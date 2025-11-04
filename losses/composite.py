# -*- coding: utf-8 -*-
"""
Bộ loss hợp thành cho DTC (seg + sdf + consistency).
- DiceCE: CE + Dice (ổn định)
- SDFReg: SmoothL1 giữa sdf_pred và sdf_gt
- DualTaskConsistency: ép seg ↔ mask-from-SDF (logits) với ramp-up
- LossDTC: gói tất cả cho tiện gọi trong trainer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import dice_loss, softmax_mse_loss
from .ramps import sigmoid_rampup


class DiceCELoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.wce = float(weight_ce)
        self.wdice = float(weight_dice)

    def forward(self, logits, target):
        """
        logits: [N,2,D,H,W]
        target: [N,1,D,H,W] hoặc [N,D,H,W] (0/1)
        """
        if target.dim() == 5:
            target = target[:, 0]  # [N,D,H,W]
        ce = self.ce(logits, target.long())

        prob1 = torch.softmax(logits, dim=1)[:, 1:2]   # [N,1,...]
        mask1 = target.unsqueeze(1).float()            # [N,1,...]
        d = dice_loss(prob1, mask1)
        return self.wce * ce + self.wdice * d


class SDFRegLoss(nn.Module):
    def __init__(self, mode="smoothl1", weight=0.5):
        super().__init__()
        self.weight = float(weight)
        self.fn = nn.SmoothL1Loss() if mode.lower() == "smoothl1" else nn.L1Loss()

    def forward(self, pred_sdf, gt_sdf):
        # pred_sdf có thể là [N,2,...] (do model), gt_sdf là [N,1,...]
        if pred_sdf.dim() == 5 and pred_sdf.size(1) != 1:
            pred_sdf = pred_sdf[:, :1]   # lấy kênh đầu
        return self.weight * self.fn(pred_sdf, gt_sdf)


class DualTaskConsistency(nn.Module):
    """
    Ép nhất quán: seg logits (2 kênh) ~ mask-from-SDF logits (2 kênh).
    Thường áp lên cả labeled & unlabeled (nhưng weight ramp-up nhẹ).
    """
    def __init__(self, max_w=1.0, ramp_len=0.3, use_kl=False):
        super().__init__()
        self.max_w = float(max_w)
        self.ramp_len = float(ramp_len)
        self.use_kl = bool(use_kl)

    @staticmethod
    def sdf_to_mask_logits(sdf: torch.Tensor) -> torch.Tensor:
        """
        sdf: [N,1,D,H,W] in ~[-1,1]
        Trả về logits 2 kênh: [N,2,D,H,W]
        - Lớp 1 (lung) có logit cao khi sdf < 0 (inside)
        """
        # đảm bảo 1 kênh
        if sdf.dim() == 5 and sdf.size(1) != 1:
            sdf = sdf[:, :1]
        mult = 6.0
        logit1 = (-mult * sdf).clamp(-12, 12)  # lung
        logit0 = (+mult * sdf).clamp(-12, 12)  # background
        return torch.cat([logit0, logit1], dim=1)  # [N,2,D,H,W]

    def forward(self, seg_logits, sdf_pred, iter_frac: float):
        target_logits = self.sdf_to_mask_logits(sdf_pred)
        if self.use_kl:
            loss = F.kl_div(
                F.log_softmax(seg_logits, dim=1),
                F.softmax(target_logits, dim=1),
                reduction='batchmean'
            )
        else:
            loss = softmax_mse_loss(seg_logits, target_logits).mean()
        w = self.max_w * sigmoid_rampup(iter_frac, self.ramp_len)
        return w * loss


class LossDTC(nn.Module):
    def __init__(self, w_seg=1.0, w_sdf=0.1, w_cons=0.1, cons_type="mse"):
        super().__init__()
        self.w_seg  = float(w_seg)
        self.w_sdf  = float(w_sdf)
        self.w_cons = float(w_cons)
        if cons_type == "mse":
            self.cons_crit = softmax_mse_loss  # (N,C,...) vs (N,C,...)
        elif cons_type == "kl":
            self.cons_crit = softmax_kl_loss
        else:
            raise ValueError(f"cons_type={cons_type}")

        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.l1  = nn.SmoothL1Loss(beta=1.0, reduction="mean")  # cho SDF

    def _seg_loss(self, seg_logits, mask):
        # hỗ trợ 1 kênh (sigmoid) hoặc 2 kênh (softmax)
        if seg_logits.shape[1] == 1:
            prob = torch.sigmoid(seg_logits[:, 0:1])
            y    = mask.float()
            ce   = self.bce(seg_logits[:,0], y[:,0])
            # dice trên prob so với mask nhị phân
            inter = (prob * y).sum()
            denom = prob.sum() + y.sum() + 1e-7
            dice  = 1.0 - (2.0 * inter / denom)
        else:
            prob1 = torch.softmax(seg_logits, dim=1)[:,1:2]
            y     = mask.float()
            # CE 2 lớp: lấy nhãn 0/1 từ y
            ce    = F.cross_entropy(seg_logits, y[:,0].long())
            inter = (prob1 * y).sum()
            denom = prob1.sum() + y.sum() + 1e-7
            dice  = 1.0 - (2.0 * inter / denom)
        return ce + dice  # tổng hai thành phần

    def _sdf_loss(self, sdf_pred, sdf_gt):
        return self.l1(sdf_pred, sdf_gt)

    def _consistency_from_sdf(self, seg_logits, sdf_pred):
        """
        DTC-style: ép khoảng cách → xác suất foreground.
        dis_to_mask = sigmoid(-alpha * sdf_pred)
        so sánh với prob(seg_logits)
        """
        alpha = 1500.0
        dis2mask = torch.sigmoid(-alpha * sdf_pred)     # (B,1,D,H,W)
        if seg_logits.shape[1] == 1:
            prob = torch.sigmoid(seg_logits)            # (B,1,...)
        else:
            prob = torch.softmax(seg_logits, dim=1)[:,1:2]
        return torch.mean((dis2mask - prob) ** 2)

    def forward(self, batch_lab=None, batch_unlab=None, iter_frac=1.0, compute_consistency=True):
        device = None
        L_seg = torch.tensor(0., device="cpu")
        L_sdf = torch.tensor(0., device="cpu")
        Lc_lab = torch.tensor(0., device="cpu")
        Lc_unl = torch.tensor(0., device="cpu")

        if batch_lab is not None:
            # bắt buộc phải có đủ khóa này
            seg_logits = batch_lab["seg_logits"]
            sdf_pred   = batch_lab["sdf_pred"]
            mask       = batch_lab["mask"]
            sdf_gt     = batch_lab.get("sdf_gt", None)
            device = seg_logits.device

            if sdf_gt is None:
                sdf_gt = torch.zeros_like(sdf_pred, device=seg_logits.device)

            L_seg = self._seg_loss(seg_logits, mask)
            L_sdf = self._sdf_loss(sdf_pred, sdf_gt)

            if compute_consistency:
                Lc_lab = self._consistency_from_sdf(seg_logits, sdf_pred)

        if compute_consistency and (batch_unlab is not None):
            # chỉ cần logits & sdf_pred
            seg_logits_u = batch_unlab["seg_logits"]
            sdf_pred_u   = batch_unlab["sdf_pred"]
            device = seg_logits_u.device
            Lc_unl = self._consistency_from_sdf(seg_logits_u, sdf_pred_u)

        # tổng
        loss = self.w_seg*L_seg + self.w_sdf*L_sdf + self.w_cons*(Lc_lab + Lc_unl)

        return {
            "loss":  loss,
            "L_seg": L_seg.detach(),
            "L_sdf": L_sdf.detach(),
            "L_cons": (Lc_lab + Lc_unl).detach(),
        }