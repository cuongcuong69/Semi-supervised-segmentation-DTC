# -*- coding: utf-8 -*-
"""
Smoke test cho pipeline DTC (VNet-SDF):
- Lấy 1 batch labeled + 1 batch unlabeled
- Forward cả 2 head (SDF + Seg), tính loss DTC
- Backward + optimizer step
- In ra các losses và lưu checkpoint tạm
- Mini-train: chạy vài step theo tqdm (tắt wandb)

Chạy:
  python -m trainers.smoke_train
"""

from __future__ import annotations
import os
from pathlib import Path
from collections import defaultdict

import torch
import torch.optim as optim
# === phần đầu file ===
try:
    # PyTorch >= 2.0
    from torch.amp import autocast, GradScaler
except ImportError:
    # PyTorch < 2.0 fallback
    from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

# project modules
from data.dataloader import build_train_loaders, build_val_loader
from models.vnet_sdf import VNet as VNet_SDF
from losses.composite import LossDTC
from losses.metrics import dice as dice_binary

# =========================
# cấu hình test nhanh (không CLI)
# =========================
CFG = {
    "seed": 2025,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Data (để chắc chắn chạy trên Windows, đặt workers=0)
    "crop": (112, 112, 80),
    "batch_lab": 2,
    "batch_unlab": 2,
    "num_workers": 0,
    "sampling_mode": "mixed",
    "mixed_weights": {"center_fg": 0.6, "random": 0.4},

    # Model
    "in_channels": 1,
    "n_classes": 2,
    "n_filters": 16,
    "normalization": "instancenorm",
    "has_dropout": True,

    # Optim & AMP
    "base_lr": 1e-3,
    "weight_decay": 1e-4,
    "betas": (0.9, 0.999),
    "use_amp": True,

    # Loss weights & ramp-up
    "w_seg": 1.0,
    "w_sdf": 0.5,
    "w_cons": 1.0,
    "ramp_ratio": 0.3,

    # mini-train
    "do_minitrain": True,   # True → chạy 1 epoch ngắn với tqdm
    "max_steps": 4,         # số step trong mini-train
    "save_dir": "experiments/smoke_check",  # chỗ lưu ckpt tạm
}

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def set_seed(seed: int = 2025):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

@torch.no_grad()
def quick_val(model, val_loader, device: str) -> float:
    """Val cực nhanh trên đúng 1 batch để xem Dice > 0 ?"""
    model.eval()
    for xb, yb, sb, nb in val_loader:
        xb = xb.to(device); yb = yb.to(device)
        sdf_pred, seg_logits = model(xb, turnoff_drop=True)
        prob = torch.softmax(seg_logits, dim=1)[:, 1:2]
        pred_bin = (prob > 0.5).float()
        d = dice_binary(pred_bin[0], yb[0])
        return float(d)
    return 0.0

def main():
    root = _project_root()
    set_seed(CFG["seed"])
    device = CFG["device"]

    # --- Data
    lab_loader, unlab_loader = build_train_loaders(
        crop=CFG["crop"],
        batch_lab=CFG["batch_lab"],
        batch_unlab=CFG["batch_unlab"],
        num_workers=CFG["num_workers"],
        sampling_mode=CFG["sampling_mode"],
        mixed_weights=CFG["mixed_weights"],
    )
    val_loader = build_val_loader(
        crop=CFG["crop"],
        batch_val=1,
        num_workers=CFG["num_workers"]
    )

    # --- Model, loss, optim
    model = VNet_SDF(
        n_channels=CFG["in_channels"],
        n_classes=CFG["n_classes"],
        n_filters=CFG["n_filters"],
        normalization=CFG["normalization"],
        has_dropout=CFG["has_dropout"],
    ).to(device)

    loss_fn = LossDTC(
        w_seg=CFG["w_seg"], w_sdf=CFG["w_sdf"], w_cons=CFG["w_cons"],
        ramp=CFG["ramp_ratio"], use_kl=False
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=CFG["base_lr"],
        weight_decay=CFG["weight_decay"], betas=CFG["betas"]
    )
    scaler = GradScaler(enabled=CFG["use_amp"])

    # --- 1) Smoke step: đúng 1 step duy nhất
    print("==== [SMOKE STEP] 1 forward/backward/step ====")
    x_lab, y_lab, sdf_gt, _ = next(iter(lab_loader))
    x_unlab, _ = next(iter(unlab_loader))
    x_lab = x_lab.to(device); y_lab = y_lab.to(device); sdf_gt = sdf_gt.to(device)
    x_unlab = x_unlab.to(device)

    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", enabled=CFG["use_amp"]):
        sdf_pred_lab, seg_logits_lab = model(x_lab)
        sdf_pred_un,  seg_logits_un  = model(x_unlab)
        out = loss_fn(
            batch_lab   = {'seg_logits': seg_logits_lab, 'sdf_pred': sdf_pred_lab, 'mask': y_lab, 'sdf_gt': sdf_gt},
            batch_unlab = {'seg_logits': seg_logits_un,  'sdf_pred': sdf_pred_un},
            iter_frac   = 0.0   # đầu training
        )
        loss = out['loss']
    scaler.scale(loss).backward()
    scaler.step(optimizer); scaler.update()

    print(f"[OK] smoke step done | "
          f"loss={float(loss):.4f} | L_seg={float(out['L_seg']):.4f} | "
          f"L_sdf={float(out['L_sdf']):.4f} | L_cons={float(out['L_cons']):.4f}")

    # --- optional: val nhanh trên 1 batch để check Dice hợp lý
    dice1 = quick_val(model, val_loader, device)
    print(f"[Quick-VAL] dice (1 batch) = {dice1:.4f}")

    # --- 2) Mini-train vài step với tqdm + lưu ckpt tạm
    if CFG["do_minitrain"]:
        save_dir = (root / CFG["save_dir"]).resolve()
        (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        steps = min(CFG["max_steps"], len(lab_loader))
        pbar = tqdm(range(steps), desc="Mini-train", ncols=100)
        losses_meter = defaultdict(float)

        lab_iter = iter(lab_loader)
        unlab_iter = iter(unlab_loader)
        for step in pbar:
            try:
                x_lab, y_lab, sdf_gt, _ = next(lab_iter)
            except StopIteration:
                lab_iter = iter(lab_loader)
                x_lab, y_lab, sdf_gt, _ = next(lab_iter)
            try:
                x_unlab, _ = next(unlab_iter)
            except StopIteration:
                unlab_iter = iter(unlab_loader)
                x_unlab, _ = next(unlab_iter)

            x_lab = x_lab.to(device); y_lab = y_lab.to(device); sdf_gt = sdf_gt.to(device)
            x_unlab = x_unlab.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=CFG["use_amp"]):
                sdf_pred_lab, seg_logits_lab = model(x_lab)
                sdf_pred_un,  seg_logits_un  = model(x_unlab)
                out = loss_fn(
                    batch_lab   = {'seg_logits': seg_logits_lab, 'sdf_pred': sdf_pred_lab, 'mask': y_lab, 'sdf_gt': sdf_gt},
                    batch_unlab = {'seg_logits': seg_logits_un,  'sdf_pred': sdf_pred_un},
                    iter_frac   = (step + 1) / max(1, steps)  # 0..1 trong mini-train
                )
                loss = out['loss']

            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

            losses_meter["loss"]   += float(loss.detach())
            losses_meter["L_seg"]  += float(out["L_seg"])
            losses_meter["L_sdf"]  += float(out["L_sdf"])
            losses_meter["L_cons"] += float(out["L_cons"])
            avg = losses_meter["loss"] / (step + 1)
            pbar.set_postfix({"loss": f"{avg:.4f}"})

        # Lưu checkpoint tạm (last.ckpt)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if CFG["use_amp"] else None,
            "epoch": 0,
            "global_step": steps,
            "best_dice": dice1,
            "cfg": CFG,
        }
        last_path = save_dir / "checkpoints" / "last.ckpt"
        torch.save(state, last_path)
        print(f"[SAVE] wrote {last_path}")

        # Val nhanh lại sau mini-train
        dice2 = quick_val(model, val_loader, device)
        print(f"[Quick-VAL] dice after mini-train = {dice2:.4f}")

    print("[DONE] smoke_train finished.")


if __name__ == "__main__":
    main()
