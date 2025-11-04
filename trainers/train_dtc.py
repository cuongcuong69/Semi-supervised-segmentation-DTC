# trainers/train_dtc.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

# -----------------------------------------------------------------------------
# Import project modules
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataloader import (
    build_train_loader_twostream,
    build_val_loader,
    build_test_loader,
)
from models.vnet_sdf import VNet
from losses.composite import LossDTC

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
CFG: Dict = dict(
    exp_name="dtc_nsclc_vnet_sdf",
    seed=2025,

    # Device & AMP
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_amp=True if torch.cuda.is_available() else False,

    # Data
    crop=(144, 144, 112),
    batch_size=4,
    sec_batch=2,
    num_workers=0,
    with_sdf=True,
    sampling_mode="mixed",  # "random" | "rejection" | "center_fg" | "mixed"
    rejection_thresh=0.01,
    rejection_max=8,
    mixed_weights={"center_fg": 0.3, "random": 0.7},

    # Model
    n_channels=1,
    n_classes=2,
    n_filters=16,
    normalization="batchnorm",
    has_dropout=True,

    # Optim 
    optimizer="adamw",
    lr=1e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
    momentum=0.9,

    # LR sched & decay
    sched="none",           # "cosine" | "none"
    warmup_epochs=5,
    max_epochs=300,
    lr_decay_epoch=100,      # giảm ×0.1 mỗi 30 epoch
    lr_decay_factor=0.5,

    # Loss weights
    w_seg=1.0,
    w_sdf=1.0,
    w_cons=0.1,

    # Logging & ckpt
    log_wandb=True,
    proj_name="DTC-NSCLC",
    run_name=None,
    ckpt_dir=str(ROOT / "experiments" / "dtc_nsclc_vnet_sdf1" / "checkpoints"),
    resume=True,

    # Eval / save
    do_eval=True,           # 🔘 Bật/tắt evaluate trên val set
    eval_per_epochs=10,
    eval_portion=0.3 # = dùng toàn bộ; 0.5 = dùng 1/2; 1/3 ~ 0.333...
)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
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

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

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

def stack_batch_dict(batch_list: List[Dict], keys: List[str]) -> Dict[str, Optional[torch.Tensor]]:
    out = {}
    for k in keys:
        if any(k not in b for b in batch_list):
            out[k] = None
        else:
            out[k] = torch.stack([b[k] for b in batch_list], dim=0)
    return out

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
# @torch.inference_mode()
# def evaluate(model, val_loader, loss_fn=None, device="cuda"):
#     model.eval()
#     dices, losses = [], []

#     for batch in tqdm(val_loader, desc="[Val]"):
#         x = batch["image"].to(device).float()
#         y = batch["label"].to(device).float()
#         sdf_gt = batch["sdf"].to(device).float() if "sdf" in batch else None

#         sdf_pred, seg_logits = model(x, turnoff_drop=True)
#         prob_fg = torch.sigmoid(seg_logits[:, 0]) if seg_logits.shape[1] == 1 \
#                    else torch.softmax(seg_logits, dim=1)[:, 1]
#         pred_bin = (prob_fg > 0.5).float()

#         inter = (pred_bin * y).sum()
#         denom = pred_bin.sum() + y.sum() + 1e-7
#         d = (2.0 * inter / denom).item()
#         dices.append(d)

#         if loss_fn is not None and sdf_gt is not None:
#             batch_lab = {
#                 "seg_logits": seg_logits,
#                 "sdf_pred": sdf_pred,
#                 "mask": y,
#                 "sdf_gt": sdf_gt,
#             }
#             out = loss_fn(batch_lab=batch_lab, batch_unlab=None,
#                           iter_frac=1.0, compute_consistency=False)
#             losses.append(float(out["loss"]))

#     model.train()
#     val_dice = float(np.mean(dices)) if dices else 0.0
#     val_loss = float(np.mean(losses)) if losses else 0.0
#     return val_dice, val_loss

@torch.inference_mode()
def evaluate(model, val_loader, loss_fn=None, device="cuda", portion: float = 1.0):
    """
    Đánh giá trên một phần val_loader để tiết kiệm thời gian.
    - portion trong (0, 1]: 1.0 = toàn bộ; 0.5 = một nửa; 0.333.. = một phần ba, ...
    """
    model.eval()
    dices, losses = [], []

    # Xác định số batch sẽ dùng
    try:
        total_batches = len(val_loader)
    except TypeError:
        total_batches = None  # fallback nếu loader không có __len__

    if total_batches is not None and total_batches > 0:
        use_batches = max(1, int(math.ceil(total_batches * max(0.0, min(1.0, portion)))))
    else:
        # Không biết trước độ dài -> dùng counter để dừng theo tỉ lệ xấp xỉ  (fallback: 32 batch tối đa)
        use_batches = None  # nghĩa là sẽ dựa vào counter và ngắt theo portion nếu có thể

    used = 0
    for batch in tqdm(val_loader, desc="[Val]"):
        # Nếu đã đạt số batch mục tiêu thì dừng
        if use_batches is not None and used >= use_batches:
            break
        used += 1

        x = batch["image"].to(device).float()
        y = batch["label"].to(device).float()
        sdf_gt = batch["sdf"].to(device).float() if "sdf" in batch else None

        sdf_pred, seg_logits = model(x, turnoff_drop=True)
        prob_fg = torch.sigmoid(seg_logits[:, 0]) if seg_logits.shape[1] == 1 \
                   else torch.softmax(seg_logits, dim=1)[:, 1]
        pred_bin = (prob_fg > 0.5).float()

        inter = (pred_bin * y).sum()
        denom = pred_bin.sum() + y.sum() + 1e-7
        d = (2.0 * inter / denom).item()
        dices.append(d)

        if loss_fn is not None and sdf_gt is not None:
            batch_lab = {
                "seg_logits": seg_logits,
                "sdf_pred": sdf_pred,
                "mask": y,
                "sdf_gt": sdf_gt,
            }
            out = loss_fn(batch_lab=batch_lab, batch_unlab=None,
                          iter_frac=1.0, compute_consistency=False)
            losses.append(float(out["loss"]))

    model.train()
    val_dice = float(np.mean(dices)) if dices else 0.0
    val_loss = float(np.mean(losses)) if losses else 0.0
    return val_dice, val_loss

# -----------------------------------------------------------------------------
# Main train
# -----------------------------------------------------------------------------
def main():
    set_seed(CFG["seed"])
    device = CFG["device"]
    print(f"[Device] {device} | AMP={CFG['use_amp']}")

    # ---------------- Data
    train_loader = build_train_loader_twostream(
        crop=CFG["crop"],
        batch_size=CFG["batch_size"],
        secondary_batch_size=CFG["sec_batch"],
        num_workers=CFG["num_workers"],
        with_sdf=True,
        sampling_mode=CFG["sampling_mode"],
        rejection_thresh=CFG["rejection_thresh"],
        rejection_max=CFG["rejection_max"],
        mixed_weights=CFG["mixed_weights"],
    )

    val_loader = None
    if CFG["do_eval"]:
        val_loader = build_val_loader(crop=CFG["crop"], batch_val=1,
                                      num_workers=CFG["num_workers"])

    # ---------------- Model
    model = VNet(
        n_channels=CFG["n_channels"],
        n_classes=CFG["n_classes"],
        n_filters=CFG["n_filters"],
        normalization=CFG["normalization"],
        has_dropout=CFG["has_dropout"]
    ).to(device)

    optimizer = build_optimizer(model.parameters())
    scheduler = build_scheduler(optimizer)
    scaler = GradScaler(enabled=CFG["use_amp"])

    loss_fn = LossDTC(
        w_seg=CFG["w_seg"],
        w_sdf=CFG["w_sdf"],
        w_cons=CFG["w_cons"],
    )

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

    # ---------------- Train loop
    for epoch in range(start_epoch + 1, CFG["max_epochs"] + 1):
        model.train()
        ep_loss, n_steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG['max_epochs']}")

        for batch_list in pbar:
            lab_list = [b for b in batch_list if b.get("has_label", False)]
            un_list  = [b for b in batch_list if not b.get("has_label", False)]
            lab = stack_batch_dict(lab_list, ["image", "label", "sdf"])
            un  = stack_batch_dict(un_list,  ["image"])

            x_lab = lab["image"].to(device).float() if lab["image"] is not None else None
            y_lab = lab["label"].to(device).float() if lab["label"] is not None else None
            sdf_gt = lab["sdf"].to(device).float()  if lab["sdf"] is not None else None
            x_un  = un["image"].to(device).float()  if un["image"] is not None else None

            optimizer.zero_grad(set_to_none=True)
            iter_frac = (epoch - 1) / max(1, (CFG["max_epochs"] - 1))

            with autocast(device_type="cuda", enabled=CFG["use_amp"]):
                batch_lab = None
                if x_lab is not None:
                    sdf_pred_lab, seg_logits_lab = model(x_lab)
                    batch_lab = {
                        "seg_logits": seg_logits_lab,
                        "sdf_pred":   sdf_pred_lab,
                        "mask":       y_lab,
                        "sdf_gt":     sdf_gt if sdf_gt is not None else torch.zeros_like(sdf_pred_lab),
                    }

                batch_unlab = None
                if x_un is not None:
                    sdf_pred_un, seg_logits_un = model(x_un)
                    batch_unlab = {
                        "seg_logits": seg_logits_un,
                        "sdf_pred":   sdf_pred_un,
                    }

                out = loss_fn(batch_lab=batch_lab, batch_unlab=batch_unlab, iter_frac=iter_frac)
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ep_loss += float(loss.detach().cpu())
            n_steps += 1

            disp = {"loss": f"{float(loss):.4f}"}
            for k in ("L_seg", "L_sdf", "L_cons"):
                if k in out: disp[k] = f"{float(out[k]):.4f}"
            pbar.set_postfix(disp)

            if wandb:
                log = {"train/step_loss": float(loss), "lr": optimizer.param_groups[0]["lr"]}
                for k in ("L_seg", "L_sdf", "L_cons"):
                    if k in out: log[f"train/{k}"] = float(out[k])
                wandb.log(log)

        if scheduler is not None:
            scheduler.step()

        # Manual LR decay
        if epoch % CFG["lr_decay_epoch"] == 0:
            old_lr = optimizer.param_groups[0]["lr"]
            new_lr = old_lr * CFG["lr_decay_factor"]
            for g in optimizer.param_groups:
                g["lr"] = new_lr
            print(f"[LR] Decayed learning rate from {old_lr:.6f} → {new_lr:.6f}")
            if wandb: wandb.log({"lr_decay": new_lr, "epoch": epoch})

        mean_loss = ep_loss / max(1, n_steps)
        print(f"[Train] epoch={epoch} | loss={mean_loss:.4f}")
        if wandb: wandb.log({"train/epoch_loss": mean_loss, "epoch": epoch})

        save_ckpt(last_ckpt, model, optimizer, epoch, best_dice)
        print(f"[SAVE] last -> {last_ckpt}")

        # ---------------- Validation
        if CFG["do_eval"] and (epoch % CFG["eval_per_epochs"] == 0):
            val_dice, val_loss = evaluate(model, val_loader, loss_fn=loss_fn, device=device)
            print(f"[Val] epoch={epoch} dice={val_dice:.4f} | loss={val_loss:.4f} | best={best_dice:.4f}")
            if wandb:
                wandb.log({"val/dice": val_dice, "val/loss": val_loss, "epoch": epoch})
            if val_dice > best_dice:
                best_dice = val_dice
                save_ckpt(best_ckpt, model, optimizer, epoch, best_dice)
                print(f"[SAVE] best -> {best_ckpt}")

    print("[DONE] Training finished.")

    # ---------------- Test
    try:
        test_loader = build_test_loader(crop=CFG["crop"], batch_test=2, num_workers=CFG["num_workers"])
        best_model = VNet(
            n_channels=CFG["n_channels"],
            n_classes=CFG["n_classes"],
            n_filters=CFG["n_filters"],
            normalization=CFG["normalization"],
            has_dropout=False
        ).to(device)
        if best_ckpt.exists():
            load_ckpt(best_ckpt, best_model, optimizer=None)
        else:
            best_model.load_state_dict(model.state_dict(), strict=True)
        test_dice, test_loss = evaluate(best_model, test_loader, loss_fn=loss_fn, device=device)
        print(f"[Test] dice={test_dice:.4f} | loss={test_loss:.4f}")
        if wandb: wandb.log({"test/dice": test_dice, "test/loss": test_loss})
    except Exception as e:
        print(f"[WARN] test phase skipped: {e}")

    if wandb:
        wandb.finish()

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
