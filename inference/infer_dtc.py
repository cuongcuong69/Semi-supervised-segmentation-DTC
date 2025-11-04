# inference/infer_dtc.py
# -*- coding: utf-8 -*-
"""
Inference + evaluation for trained DTC model (full-volume)
- Đọc trực tiếp NIfTI gốc (không crop)
- Sliding-window 3D với stride = (16,16,16) (config được)
- Ghép bằng Hann weighting trên logits
- Tính Dice, Jaccard, ASD, 95HD (dùng losses.metrics)
- Lưu NIfTI dự đoán với affine/header gốc
"""

from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Tuple, Iterable, Dict

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm

# --------------------------------------------------------------------------
# Import project modules
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.vnet_sdf import VNet
from losses.metrics import dice, jaccard, asd, hd95  # đảm bảo các hàm có trong metrics.py

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
CFG: Dict = dict(
    ckpt_path = str(ROOT / "experiments" / "dtc_nsclc_vnet_sdf1" / "checkpoints" / "best.ckpt"),
    test_list = str(ROOT / "configs" / "splits_lung_test.txt"),   # mỗi dòng: img_path | (optional) mask_path  (đường dẫn tương đối từ repo)
    save_dir  = str(ROOT / "experiments" / "inference_results"),
    device    = "cuda" if torch.cuda.is_available() else "cpu",

    patch_size = (144, 144, 112),  # (D,H,W)
    stride     = (64, 64, 64),    # (D,H,W)

    n_channels = 1,
    n_classes  = 2,
    n_filters  = 16,
    normalization = "batchnorm",
)

# --------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------
def case_id_from_path(p: str) -> str:
    return Path(p).parent.name

def make_hann_window(ps: Tuple[int,int,int]) -> torch.Tensor:
    """3D Hann window normalized to max=1, shape (1,1,D,H,W)"""
    dz, dy, dx = ps
    wz = np.hanning(max(dz, 2))
    wy = np.hanning(max(dy, 2))
    wx = np.hanning(max(dx, 2))
    w3 = np.outer(wz, wy).reshape(dz, dy, 1) * wx.reshape(1, 1, dx)
    w3 = w3 / (w3.max() + 1e-8)
    return torch.from_numpy(w3.astype(np.float32))[None, None]  # (1,1,D,H,W)

def make_starts(L: int, P: int, S: int) -> Iterable[int]:
    """Sinh danh sách vị trí bắt đầu để phủ kín chiều dài L bằng patch P và stride S."""
    if L <= P:
        return [0]
    starts = list(range(0, L - P + 1, S))
    if starts[-1] != L - P:
        starts.append(L - P)
    return starts

# --------------------------------------------------------------------------
# Sliding-window inference with weighted blending on logits
# --------------------------------------------------------------------------
@torch.no_grad()
def sliding_window_logits(
    model: torch.nn.Module,
    vol: torch.Tensor,                 # (1,1,D,H,W)
    patch_size: Tuple[int,int,int],
    stride: Tuple[int,int,int],
    n_classes: int,
) -> torch.Tensor:
    model.eval()
    _, _, D, H, W = vol.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    # output logits + weight accumulator
    out = torch.zeros((1, n_classes, D, H, W), dtype=torch.float32, device=vol.device)
    acc = torch.zeros((1, 1,        D, H, W), dtype=torch.float32, device=vol.device)

    win = make_hann_window(patch_size).to(vol.device)

    for z0 in make_starts(D, pd, sd):
        for y0 in make_starts(H, ph, sh):
            for x0 in make_starts(W, pw, sw):
                patch = vol[..., z0:z0+pd, y0:y0+ph, x0:x0+pw]
                dz, dy, dx = patch.shape[-3:]
                need_pad = (dz,dy,dx) != (pd,ph,pw)
                if need_pad:
                    pad = [0, pw-dx, 0, ph-dy, 0, pd-dz]  # (W, H, D)
                    patch = F.pad(patch, pad, mode="constant", value=0.0)

                # forward
                _, seg_logits = model(patch)  # (1,C, pd,ph,pw)

                # unpad window if needed
                if need_pad:
                    seg_logits = seg_logits[..., :dz, :dy, :dx]
                    w = win[..., :dz, :dy, :dx]
                else:
                    w = win

                out[..., z0:z0+dz, y0:y0+dy, x0:x0+dx] += seg_logits * w
                acc[..., z0:z0+dz, y0:y0+dy, x0:x0+dx] += w

    out = out / torch.clamp(acc, min=1e-8)
    return out  # (1,C,D,H,W)

# --------------------------------------------------------------------------
# IO for test list
# --------------------------------------------------------------------------
def iter_test_list(list_file: str):
    """
    Mỗi dòng trong test_list:
      - 'path_img'  hoặc
      - 'path_img|path_gt'
    Đường dẫn được hiểu tương đối so với ROOT.
    """
    list_path = Path(list_file)
    assert list_path.exists(), f"test_list not found: {list_path}"
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            img_p = str((ROOT / parts[0]).resolve())
            msk_p = str((ROOT / parts[1]).resolve()) if len(parts) > 1 else None
            yield img_p, msk_p, case_id_from_path(img_p)

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
    print(f"[LOAD] weights from {ckpt_path}")
    return model

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    device = CFG["device"]
    save_dir = Path(CFG["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    model = load_model()

    metrics = {"dice": [], "jaccard": [], "asd": [], "hd95": []}

    for img_path, msk_path, case in tqdm(list(iter_test_list(CFG["test_list"])), desc="[Inference]"):
        nii = nib.load(img_path)
        vol = nii.get_fdata().astype(np.float32)             # (D,H,W) hoặc (H,W,D) tùy dataset
        # nếu trục khác, đảm bảo định dạng (D,H,W) như khi train (chỉnh tại đây nếu cần)
        vol_t = torch.from_numpy(vol)[None, None].to(device)  # (1,1,D,H,W)

        # ---- sliding window (logits blending)
        logits = sliding_window_logits(
            model, vol_t,
            patch_size=CFG["patch_size"],
            stride=CFG["stride"],
            n_classes=CFG["n_classes"],
        )

        # ---- prob foreground & binarize
        if logits.shape[1] == 1:
            prob_fg = torch.sigmoid(logits[:, 0:1])[:, 0]     # (1,D,H,W) -> (1,D,H,W)
        else:
            prob_fg = torch.softmax(logits, dim=1)[:, 1]       # (1,D,H,W)
        pred_bin = (prob_fg > 0.5).float().cpu().numpy()[0]    # (D,H,W)

        # ---- evaluate if GT available
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

        # ---- save prediction with original affine/header
        out_path = save_dir / f"{case}_pred.nii.gz"
        nib.save(nib.Nifti1Image(pred_bin.astype(np.uint8), nii.affine, nii.header), out_path)

    # ---- summary
    if metrics["dice"]:
        print("\n=== Evaluation Summary ===")
        for k, v in metrics.items():
            print(f"{k:8s}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    print(f"Results saved to {save_dir}")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
