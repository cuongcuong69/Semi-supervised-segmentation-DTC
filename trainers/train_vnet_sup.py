# -*- coding: utf-8 -*-
"""
Train supervised VNet (segmentation only) on labeled data.
- Data: dùng data/dataloader_sup.py (chỉ ảnh có nhãn)
- Model: models/vnet.VNet (KHÔNG có nhánh SDF)
- Loss: Dice + CE; tùy chọn thêm Boundary(SDF) loss nếu dataset cung cấp SDF
- AMP, checkpoint, optional validation, cosine/none scheduler, LR decay mốc 30-60-90...
"""

from __future__ import annotations
import os, sys, math
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
try:
    # PyTorch >= 2.0
    from torch.amp import autocast, GradScaler
except Exception:
    from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# -------------------------------------------------------------------------
# Project imports
# -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.vnet import VNet
from data.dataloader_sup import (
    build_sup_train_loader,
    build_sup_val_loader,
    build_sup_test_loader,
)
# (tùy chọn) dùng Dice/Jaccard ngoài evaluate tổng hợp
from losses.metrics import dice as dice_metric

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
CFG: Dict = dict(
    exp_name="vnet_sup_nsclc",
    seed=2025,

    # Device & AMP
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_amp=True if torch.cuda.is_available() else False,

    # Data
    crop=(144, 144, 112),
    batch_size=2,
    num_workers=0,
    with_sdf=True,              # dataset sinh SDF GT (để dùng boundary loss)
    sampling_mode="mixed",      # "random" | "rejection" | "center_fg" | "mixed"
    rejection_thresh=0.01,
    rejection_max=8,
    mixed_weights={"center_fg": 0.6, "random": 0.4},

    # Model
    n_channels=1,
    n_classes=2,
    n_filters=16,
    normalization="batchnorm",
    has_dropout=True,

    # Optim
    optimizer="adamw",          # "adamw"|"adam"|"sgd"
    lr=1e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
    momentum=0.9,

    # Scheduler
    sched="none",               # "cosine"|"none"
    warmup_epochs=5,
    max_epochs=300,

    # Manual multi-step decay (recommend ×0.1 at 30, 60, 90…)
    decay_milestones=[50, 100, 150, 200, 250, 300],
    decay_gamma=0.5,

    # Loss weights
    w_dice=1.0,
    w_ce=1.0,
    use_sdf_loss=True,          # ✅ tùy chọn bật/tắt boundary(SDF) loss
    w_sdf=1.0,

    # Logging & ckpt
    log_wandb=True,
    proj_name="VNet-Supervised",
    run_name=None,
    ckpt_dir=str(ROOT / "experiments" / "vnet_sup" / "checkpoints"),
    resume=True,

    # Eval / save
    do_eval=True,               # có đánh giá trên val trong lúc train không
    eval_per_epochs=10,
)

# -------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_run_name() -> str:
    import datetime as dt
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return (CFG["run_name"] or f'{CFG["exp_name"]}-{stamp}')

def maybe_init_wandb() -> Optional[object]:
    if not CFG["log_wandb"]:
        return None
    try:
        import wandb
        wandb.init(project=CFG["proj_name"], name=get_run_name(), config=CFG, mode="online")
        return wandb
    except Exception as e:
        print(f"[wandb] disabled: {e}")
        return None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def build_optimizer(params) -> optim.Optimizer:
    opt = CFG["optimizer"].lower()
    if opt == "adamw":
        return optim.AdamW(params, lr=CFG["lr"], weight_decay=CFG["weight_decay"], betas=CFG["betas"])
    if opt == "adam":
        return optim.Adam(params, lr=CFG["lr"], weight_decay=CFG["weight_decay"], betas=CFG["betas"])
    if opt == "sgd":
        return optim.SGD(params, lr=CFG["lr"], momentum=CFG["momentum"],
                         weight_decay=CFG["weight_decay"], nesterov=True)
    raise ValueError(f"Unknown optimizer: {CFG['optimizer']}")

def build_scheduler(optimizer):
    if CFG["sched"] == "none":
        return None
    if CFG["sched"] == "cosine":
        def lr_lambda(epoch):
            if epoch < CFG["warmup_epochs"]:
                return float(epoch + 1) / float(max(1, CFG["warmup_epochs"]))
            progress = (epoch - CFG["warmup_epochs"]) / float(max(1, CFG["max_epochs"] - CFG["warmup_epochs"]))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Unknown scheduler: {CFG['sched']}")

def save_ckpt(path: Path, model: nn.Module, optimizer, epoch: int, best_dice: float):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_dice": best_dice,
        "cfg": CFG
    }, path)

def load_ckpt(path: Path, model: nn.Module, optimizer=None) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = ckpt.get("epoch", 0)
    best = ckpt.get("best_dice", 0.0)
    print(f"[RESUME] loaded '{path}' (epoch={epoch}, best_dice={best:.4f})")
    return epoch, best

# -------------------------------------------------------------------------
# Losses
# -------------------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,C,D,H,W), target: (B,1,D,H,W) in {0,1}
        Compute foreground Dice (class-1) if C=2, else binary on channel-0.
        """
        if logits.shape[1] > 1:
            probs = torch.softmax(logits, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(logits[:, 0])
        target = target.float().squeeze(1)
        probs = probs.contiguous().view(probs.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        inter = (probs * target).sum(dim=1)
        denom = probs.sum(dim=1) + target.sum(dim=1)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()

class BCEOrCEWithLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce  = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            return self.bce(logits[:, 0], target.float().squeeze(1))
        # target: (B,1,D,H,W) -> long labels
        return self.ce(logits, target.squeeze(1).long())

class BoundaryLoss(nn.Module):
    """
    Boundary / Surface loss (Kervadec et al.):
      L = mean( prob_fg * sdf_gt )
    Với SDF: âm trong, dương ngoài (chúng ta dùng chuẩn từ dataloader_sup).
    """
    def forward(self, logits: torch.Tensor, sdf_gt: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] > 1:
            probs = torch.softmax(logits, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(logits[:, 0])
        return (probs * sdf_gt.squeeze(1)).mean()

# -------------------------------------------------------------------------
# Evaluation (Dice trên foreground)
# -------------------------------------------------------------------------
@torch.inference_mode()
def evaluate(model, val_loader, device="cuda") -> float:
    model.eval()
    dices = []
    for batch in val_loader:
        x = batch["image"].to(device).float()
        y = batch["label"].to(device).float()
        logits = model(x, turnoff_drop=True)
        if logits.shape[1] > 1:
            prob_fg = torch.softmax(logits, dim=1)[:, 1]
        else:
            prob_fg = torch.sigmoid(logits[:, 0])
        pred = (prob_fg > 0.5).float()
        d = dice_metric(pred, y)
        dices.append(float(d))
    model.train()
    return float(np.mean(dices)) if dices else 0.0

# -------------------------------------------------------------------------
# Main training
# -------------------------------------------------------------------------
def main():
    set_seed(CFG["seed"])
    device = CFG["device"]
    print(f"[Device] {device} | AMP={CFG['use_amp']}")

    # Data
    train_loader = build_sup_train_loader(
        crop=CFG["crop"], batch_size=CFG["batch_size"], num_workers=CFG["num_workers"],
        with_sdf=CFG["with_sdf"], sampling_mode=CFG["sampling_mode"],
        rejection_thresh=CFG["rejection_thresh"], rejection_max=CFG["rejection_max"],
        mixed_weights=CFG["mixed_weights"]
    )
    val_loader = build_sup_val_loader(
        crop=CFG["crop"], batch_size=1, num_workers=CFG["num_workers"], with_sdf=CFG["with_sdf"]
    ) if CFG["do_eval"] else None

    # Model
    model = VNet(
        n_channels=CFG["n_channels"],
        n_classes=CFG["n_classes"],
        n_filters=CFG["n_filters"],
        normalization=CFG["normalization"],
        has_dropout=CFG["has_dropout"],
    ).to(device)

    # Optim & sched
    optimizer = build_optimizer(model.parameters())
    scheduler = build_scheduler(optimizer)
    scaler = GradScaler(enabled=CFG["use_amp"])

    # Losses
    dice_loss = DiceLoss()
    ce_loss   = BCEOrCEWithLogits()
    bnd_loss  = BoundaryLoss() if CFG["use_sdf_loss"] else None

    # Logging & ckpt
    wandb = maybe_init_wandb()
    if wandb:
        wandb.watch(model, log="all", log_freq=50)

    ckpt_dir = Path(CFG["ckpt_dir"]); ensure_dir(ckpt_dir)
    last_ckpt = ckpt_dir / "last.ckpt"
    best_ckpt = ckpt_dir / "best.ckpt"

    start_epoch, best_dice = 0, 0.0
    if CFG["resume"] and last_ckpt.exists():
        try:
            start_epoch, best_dice = load_ckpt(last_ckpt, model, optimizer)
        except Exception as e:
            print(f"[WARN] resume failed: {e}")

    # Train loop
    for epoch in range(start_epoch + 1, CFG["max_epochs"] + 1):
        model.train()
        ep_loss, n_steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG['max_epochs']}")

        for batch in pbar:
            x = batch["image"].to(device).float()
            y = batch["label"].to(device)          # (B,1,D,H,W)
            sdf_gt = batch.get("sdf", None)
            if sdf_gt is not None:
                sdf_gt = sdf_gt.to(device).float()

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=CFG["use_amp"]):
                logits = model(x)                  # (B,C,D,H,W)
                L_dice = dice_loss(logits, y) * CFG["w_dice"]
                L_ce   = ce_loss(logits, y)   * CFG["w_ce"]
                loss = L_dice + L_ce
                L_sdf = torch.tensor(0.0, device=device)
                if CFG["use_sdf_loss"] and (sdf_gt is not None):
                    L_sdf = bnd_loss(logits, sdf_gt) * CFG["w_sdf"]
                    # Boundary loss thường có dấu cộng — nếu muốn “kéo” ranh giới, có thể dùng + hoặc abs.
                    loss = loss + L_sdf

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            n_steps += 1
            step_loss = float(loss.detach().cpu())
            ep_loss += step_loss

            postfix = {"loss": f"{step_loss:.4f}", "L_dice": f"{float(L_dice):.4f}", "L_ce": f"{float(L_ce):.4f}"}
            if CFG["use_sdf_loss"]:
                postfix["L_sdf"] = f"{float(L_sdf):.4f}"
            pbar.set_postfix(postfix)

            if wandb:
                log = {"train/step_loss": step_loss, "train/L_dice": float(L_dice), "train/L_ce": float(L_ce),
                       "lr": optimizer.param_groups[0]["lr"]}
                if CFG["use_sdf_loss"]:
                    log["train/L_sdf"] = float(L_sdf)
                wandb.log(log)

        # Cosine or other scheduler
        if scheduler is not None:
            scheduler.step()

        # Manual multi-step decay
        if epoch in set(CFG["decay_milestones"]):
            old_lr = optimizer.param_groups[0]["lr"]
            new_lr = old_lr * CFG["decay_gamma"]
            for g in optimizer.param_groups:
                g["lr"] = new_lr
            print(f"[LR] decay @epoch {epoch}: {old_lr:.6f} → {new_lr:.6f}")
            if wandb: wandb.log({"lr_decay": new_lr, "epoch": epoch})

        mean_train_loss = ep_loss / max(1, n_steps)
        print(f"[Train] epoch={epoch} | loss={mean_train_loss:.4f}")
        if wandb:
            wandb.log({"train/epoch_loss": mean_train_loss, "epoch": epoch})

        # Save last
        save_ckpt(last_ckpt, model, optimizer, epoch, best_dice)
        print(f"[SAVE] last -> {last_ckpt}")

        # Validation
        if CFG["do_eval"] and (epoch % CFG["eval_per_epochs"] == 0):
            val_d = evaluate(model, val_loader, device=device)
            print(f"[Val] epoch={epoch} Dice={val_d:.4f} | best={best_dice:.4f}")
            if wandb:
                wandb.log({"val/dice": val_d, "epoch": epoch})
            if val_d > best_dice:
                best_dice = val_d
                save_ckpt(best_ckpt, model, optimizer, epoch, best_dice)
                print(f"[SAVE] best -> {best_ckpt}")

    print("[DONE] Training finished.")

    # Test (tuỳ chọn)
    try:
        test_loader = build_sup_test_loader(
            crop=CFG["crop"], batch_size=1, num_workers=CFG["num_workers"], with_sdf=False
        )
        best_model = VNet(
            n_channels=CFG["n_channels"], n_classes=CFG["n_classes"],
            n_filters=CFG["n_filters"], normalization=CFG["normalization"], has_dropout=False
        ).to(device)
        if best_ckpt.exists():
            load_ckpt(best_ckpt, best_model, optimizer=None)
        else:
            best_model.load_state_dict(model.state_dict(), strict=True)
        test_d = evaluate(best_model, test_loader, device=device)
        print(f"[Test] Dice={test_d:.4f}")
        if wandb: wandb.log({"test/dice": test_d})
    except Exception as e:
        print(f"[WARN] test phase skipped: {e}")

    if wandb:
        wandb.finish()

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
