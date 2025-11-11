# -*- coding: utf-8 -*-
"""
Inference + evaluation cho mô hình VNet (supervised)
- Tái sử dụng hàm từ inference/infer_dtc.py
- Không dùng nhánh SDF
- Sliding-window + Hann blending
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

# --------------------------------------------------------------------------
# Import project modules và tái sử dụng hàm từ infer_dtc
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.vnet import VNet                          # ✅ model supervised
from losses.metrics import dice, jaccard, asd, hd95 
from inference.infer_dtc import (
    make_hann_window, make_starts, iter_test_list
)
import torch.nn.functional as F

@torch.no_grad()
def sliding_window_logits(
    model: torch.nn.Module,
    vol: torch.Tensor,                 # (1,1,D,H,W)
    patch_size: tuple[int,int,int],
    stride: tuple[int,int,int],
    n_classes: int,
) -> torch.Tensor:
    """
    Sliding-window inference dành cho VNet supervised (chỉ có output logits).
    """
    model.eval()
    _, _, D, H, W = vol.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    out = torch.zeros((1, n_classes, D, H, W), dtype=torch.float32, device=vol.device)
    acc = torch.zeros((1, 1, D, H, W), dtype=torch.float32, device=vol.device)
    win = make_hann_window(patch_size).to(vol.device)

    for z0 in make_starts(D, pd, sd):
        for y0 in make_starts(H, ph, sh):
            for x0 in make_starts(W, pw, sw):
                patch = vol[..., z0:z0+pd, y0:y0+ph, x0:x0+pw]
                dz, dy, dx = patch.shape[-3:]
                need_pad = (dz,dy,dx) != (pd,ph,pw)
                if need_pad:
                    pad = [0, pw-dx, 0, ph-dy, 0, pd-dz]
                    patch = F.pad(patch, pad, mode="constant", value=0.0)

                seg_logits = model(patch)  # chỉ 1 output

                if need_pad:
                    seg_logits = seg_logits[..., :dz, :dy, :dx]
                    w = win[..., :dz, :dy, :dx]
                else:
                    w = win

                out[..., z0:z0+dz, y0:y0+dy, x0:x0+dx] += seg_logits * w
                acc[..., z0:z0+dz, y0:y0+dy, x0:x0+dx] += w

    out = out / torch.clamp(acc, min=1e-8)
    return out


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
CFG = dict(
    ckpt_path = str(ROOT / "experiments" / "vnet_sup" / "checkpoints" / "best.ckpt"),
    test_list = str(ROOT / "configs" / "splits_lung_test.txt"),
    save_dir  = str(ROOT / "experiments" / "inference_results_vnet_supervised"),
    device    = "cuda" if torch.cuda.is_available() else "cpu",

    patch_size = (144, 144, 112),
    stride     = (64, 64, 64),

    n_channels = 1,
    n_classes  = 2,
    n_filters  = 16,
    normalization = "batchnorm",
)

# --------------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------------
def load_model() -> torch.nn.Module:
    ckpt_path = Path(CFG["ckpt_path"])
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    model = VNet(
        n_channels=CFG["n_channels"],
        n_classes=CFG["n_classes"],
        n_filters=CFG["n_filters"],
        normalization=CFG["normalization"],
        has_dropout=False,
    ).to(CFG["device"])

    ckpt = torch.load(ckpt_path, map_location=CFG["device"])
    model.load_state_dict(ckpt["model"], strict=True)
    print(f"[LOAD] Supervised weights from {ckpt_path}")
    return model

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
@torch.no_grad()
def main():
    device = CFG["device"]
    save_dir = Path(CFG["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    model = load_model()

    metrics = {"dice": [], "jaccard": [], "asd": [], "hd95": []}

    for img_path, msk_path, case in tqdm(list(iter_test_list(CFG["test_list"])), desc="[Inference-Supervised]"):
        nii = nib.load(img_path)
        vol = nii.get_fdata().astype(np.float32)
        vol_t = torch.from_numpy(vol)[None, None].to(device)

        # ---- Sliding window inference (tái sử dụng từ infer_dtc)
        logits = sliding_window_logits(
            model, vol_t,
            patch_size=CFG["patch_size"],
            stride=CFG["stride"],
            n_classes=CFG["n_classes"],
        )

        # ---- Foreground prob & threshold
        if logits.shape[1] == 1:
            prob_fg = torch.sigmoid(logits[:, 0])
        else:
            prob_fg = torch.softmax(logits, dim=1)[:, 1]
        pred_bin = (prob_fg > 0.5).float().cpu().numpy()[0]

        # ---- Evaluation
        if msk_path and Path(msk_path).exists():
            zooms = (2.0, 2.0, 2.0)
            gt = nib.load(msk_path).get_fdata().astype(np.uint8)
            d = dice(pred_bin, gt)
            j = jaccard(pred_bin, gt)
            a = asd(pred_bin, gt, spacing=zooms)
            h = hd95(pred_bin, gt, spacing=zooms)
            metrics["dice"].append(d)
            metrics["jaccard"].append(j)
            metrics["asd"].append(a)
            metrics["hd95"].append(h)
            print(f"[{case}] Dice={d:.4f}  Jaccard={j:.4f}  ASD={a:.3f}  HD95={h:.3f}")

        # ---- Save prediction
        out_path = save_dir / f"{case}_pred.nii.gz"
        nib.save(nib.Nifti1Image(pred_bin.astype(np.uint8), nii.affine, nii.header), out_path)

    # ---- Summary
    if metrics["dice"]:
        print("\n=== Evaluation Summary ===")
        for k, v in metrics.items():
            print(f"{k:8s}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    print(f"Results saved to {save_dir}")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
