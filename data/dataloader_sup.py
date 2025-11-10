# -*- coding: utf-8 -*-
"""
Dataloader 3D NIfTI cho phân đoạn GIÁM SÁT (supervised).
- Đọc danh sách ảnh|mask từ các file splits_* (đường dẫn tương đối tính từ root repo)
- Hỗ trợ crop 3D và các chế độ sampling cho TRAIN:
    {"random", "rejection", "center_fg", "mixed"}
- VAL/TEST: CenterCrop (tuỳ chọn) hoặc giữ nguyên kích thước (nếu output_size <= volume)
- Tuỳ chọn sinh SDF để dùng các loss dạng LSF/DTF
- Tích hợp sẵn các builder: build_sup_train_loader / build_sup_val_loader / build_sup_test_loader
- Có self-test: python -m data.dataloader_sup
"""

from __future__ import annotations
import itertools
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any, Union

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

from scipy.ndimage import distance_transform_edt as distance
import matplotlib.pyplot as plt

# =============================================================================
# SELF-TEST CONFIG
# =============================================================================
SELFTEST_SEED = 2025
SELFTEST_CROP = (112, 112, 80)
SELFTEST_BATCH = 2
SELFTEST_NUM_WORKERS = 0
SELFTEST_VIS_DIR = "experiments/vis_sup"
SELFTEST_SLICE = "middle"  # hoặc int
# =============================================================================


# ========= Helpers =========

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _abs_from_root(rel: str) -> str:
    return str((_project_root() / rel).resolve())

def set_seed(seed: int = 2025):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_nii(path: str) -> np.ndarray:
    return nib.load(path).get_fdata().astype(np.float32)

def _ensure_min_size(vol: np.ndarray, out_size: Tuple[int, int, int]) -> np.ndarray:
    D, H, W = vol.shape
    d, h, w = out_size
    pads = []
    for S, s in zip((D, H, W), (d, h, w)):
        need = max(0, s - S)
        pads.append((need // 2, need - need // 2))
    if any(p[0] or p[1] for p in pads):
        vol = np.pad(vol, pads, mode="constant", constant_values=0)
    return vol

def compute_sdf(mask3d: np.ndarray, out_shape: Tuple[int, int, int]) -> np.ndarray:
    """SDF ~ [-1, 1] với biên = 0 (chuẩn DTC/LSF)."""
    mask = (mask3d > 0.5).astype(np.uint8)
    sdf = np.zeros(out_shape, dtype=np.float32)
    pos = mask.astype(bool)
    if pos.any():
        neg = ~pos
        posdis = distance(pos)
        negdis = distance(neg)
        posdis = (posdis - posdis.min()) / (posdis.max() - posdis.min() + 1e-8)
        negdis = (negdis - negdis.min()) / (negdis.max() - negdis.min() + 1e-8)
        sdf = (negdis - posdis).astype(np.float32)
    return sdf

def _random_crop_coords(D, H, W, d, h, w):
    z = np.random.randint(0, max(1, D - d + 1))
    y = np.random.randint(0, max(1, H - h + 1))
    x = np.random.randint(0, max(1, W - w + 1))
    return z, y, x

def _centered_crop_coords(zc, yc, xc, D, H, W, d, h, w):
    zs = int(np.clip(zc - d // 2, 0, max(0, D - d)))
    ys = int(np.clip(yc - h // 2, 0, max(0, H - h)))
    xs = int(np.clip(xc - w // 2, 0, max(0, W - w)))
    return zs, ys, xs


# ========= Transforms (đã giản lược cho ổn định) =========

class CenterCrop(object):
    def __init__(self, output_size: Tuple[int, int, int]):
        self.output_size = tuple(map(int, output_size))

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample["image"]
        label = sample.get("label", None)
        sdf   = sample.get("sdf", None)

        D, H, W = image.shape
        d, h, w = self.output_size
        if D < d or H < h or W < w:
            pw = max((d - D) // 2 + 3, 0)
            ph = max((h - H) // 2 + 3, 0)
            pd = max((w - W) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if label is not None: label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], 'constant', constant_values=0)
            if sdf   is not None: sdf   = np.pad(sdf,   [(pw, pw), (ph, ph), (pd, pd)], 'constant', constant_values=0)

        D, H, W = image.shape
        z1 = int(round((D - d) / 2.))
        y1 = int(round((H - h) / 2.))
        x1 = int(round((W - w) / 2.))

        out = {"image": image[z1:z1+d, y1:y1+h, x1:x1+w]}
        if label is not None: out["label"] = label[z1:z1+d, y1:y1+h, x1:x1+w]
        if sdf   is not None: out["sdf"]   = sdf  [z1:z1+d, y1:y1+h, x1:x1+w]
        out["case"] = sample.get("case")
        return out


class ToTensor(object):
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample["image"].reshape(1, *sample["image"].shape).astype(np.float32)  # [1,D,H,W]
        out: Dict[str, Any] = {"image": torch.from_numpy(image)}
        if "label" in sample:
            out["label"] = torch.from_numpy(sample["label"].reshape(1, *sample["label"].shape)).long()
        if "sdf" in sample:
            out["sdf"]   = torch.from_numpy(sample["sdf"].reshape(1, *sample["sdf"].shape)).float()
        out["case"] = sample.get("case")
        return out


# ========= Dataset (Supervised) =========

class LungSupervised(Dataset):
    """
    Dataset GIÁM SÁT: mỗi dòng trong list_txt là 'img_rel|mask_rel'
    sampling_mode chỉ áp dụng cho TRAIN:
        - random: crop ngẫu nhiên
        - rejection: từ chối patch có mean(mask) < threshold (thử tối đa rejection_max lần)
        - center_fg: crop quanh điểm foreground
        - mixed: trộn theo trọng số mixed_weights
    """
    def __init__(
        self,
        list_txt: str,
        output_size: Tuple[int, int, int] = (112, 112, 80),
        mode: str = "train",      # 'train'|'val'|'test'
        with_sdf: bool = False,
        use_center_crop_eval: bool = True,
        sampling_mode: str = "random",               # "random"|"rejection"|"center_fg"|"mixed"
        rejection_thresh: float = 0.01,
        rejection_max: int = 8,
        mixed_weights: Optional[Dict[str, float]] = None,  # e.g., {"center_fg":0.6, "random":0.4}
    ):
        self.mode = mode
        self.output_size = tuple(map(int, output_size))
        self.with_sdf = bool(with_sdf)
        self.use_center_crop_eval = bool(use_center_crop_eval)

        self.sampling_mode = sampling_mode
        self.rejection_thresh = float(rejection_thresh)
        self.rejection_max = int(rejection_max)
        if mixed_weights is None:
            mixed_weights = {"center_fg": 0.6, "random": 0.4}
        valid_keys = {"random", "rejection", "center_fg"}
        mixed_weights = {k: v for k, v in mixed_weights.items() if k in valid_keys and v > 0}
        s = sum(mixed_weights.values()) or 1.0
        self.mixed_weights = {k: v / s for k, v in mixed_weights.items()}

        # đọc danh sách
        items: List[Dict[str, Any]] = []
        with open(list_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_rel, mask_rel = line.split("|")
                items.append({
                    "img": _abs_from_root(img_rel),
                    "mask": _abs_from_root(mask_rel),
                })

        self.items = items
        print(f"[LungSupervised] mode={mode} | total={len(self.items)}")

        # eval transforms
        if mode in ("val", "test"):
            self.transforms_eval = ([CenterCrop(self.output_size)] if self.use_center_crop_eval else []) + [ToTensor()]

    def __len__(self) -> int:
        return len(self.items)

    # ---- sampling helpers (TRAIN) ----
    def _choose_mixed_mode(self) -> str:
        r = random.random()
        acc = 0.0
        for k, p in self.mixed_weights.items():
            acc += p
            if r <= acc:
                return k
        return list(self.mixed_weights.keys())[-1]

    def _crop_random(self, img, msk):
        D, H, W = img.shape
        d, h, w = self.output_size
        z, y, x = _random_crop_coords(D, H, W, d, h, w)
        return img[z:z+d, y:y+h, x:x+w], msk[z:z+d, y:y+h, x:x+w]

    def _crop_rejection(self, img, msk):
        D, H, W = img.shape
        d, h, w = self.output_size
        for _ in range(self.rejection_max):
            z, y, x = _random_crop_coords(D, H, W, d, h, w)
            sub = msk[z:z+d, y:y+h, x:x+w]
            if sub.mean() >= self.rejection_thresh:
                return img[z:z+d, y:y+h, x:x+w], sub
        return self._crop_random(img, msk)

    def _crop_center_fg(self, img, msk):
        pts = np.argwhere(msk > 0.5)
        if len(pts) == 0:
            return self._crop_random(img, msk)
        D, H, W = img.shape
        d, h, w = self.output_size
        zc, yc, xc = pts[np.random.randint(len(pts))]
        zs, ys, xs = _centered_crop_coords(int(zc), int(yc), int(xc), D, H, W, d, h, w)
        return img[zs:zs+d, ys:ys+h, xs:xs+w], msk[zs:zs+d, ys:ys+h, xs:xs+w]

    def _sample_patch(self, img, msk):
        if self.mode != "train":
            return self._crop_random(img, msk)
        mode = self.sampling_mode
        if mode == "mixed":
            mode = self._choose_mixed_mode()
        if mode == "rejection":
            return self._crop_rejection(img, msk)
        if mode == "center_fg":
            return self._crop_center_fg(img, msk)
        return self._crop_random(img, msk)

    # ---- __getitem__ ----
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        img = load_nii(item["img"])
        msk = (load_nii(item["mask"]) > 0.5).astype(np.uint8)

        img = _ensure_min_size(img, self.output_size)
        msk = _ensure_min_size(msk, self.output_size)
        case = Path(item["img"]).parent.name

        if self.mode in ("val", "test"):
            sample: Dict[str, Any] = {"image": img, "label": msk, "case": case}
            if self.with_sdf:
                sample["sdf"] = compute_sdf(msk, msk.shape)
            for t in self.transforms_eval:
                sample = t(sample)
            return sample

        # train: sampling patch
        img_p, msk_p = self._sample_patch(img, msk)
        sample: Dict[str, Any] = {"image": img_p, "label": msk_p, "case": case}
        if self.with_sdf:
            sample["sdf"] = compute_sdf(msk_p, msk_p.shape)
        return ToTensor()(sample)


# ========= Builders =========

def build_sup_train_loader(
    crop: Tuple[int, int, int] = (112, 112, 80),
    batch_size: int = 2,
    num_workers: int = 0,
    seed: int = 2025,
    with_sdf: bool = False,
    sampling_mode: str = "mixed",
    rejection_thresh: float = 0.01,
    rejection_max: int = 8,
    mixed_weights: Optional[Dict[str, float]] = None,
) -> DataLoader:
    set_seed(seed)
    root = _project_root()
    list_txt = str(root / "configs" / "supervised" / "splits_lung_train_labeled.txt")
    ds = LungSupervised(
        list_txt, output_size=crop, mode="train", with_sdf=with_sdf,
        sampling_mode=sampling_mode, rejection_thresh=rejection_thresh,
        rejection_max=rejection_max, mixed_weights=mixed_weights,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True)

def build_sup_val_loader(
    crop: Tuple[int, int, int] = (112, 112, 80),
    batch_size: int = 1,
    num_workers: int = 0,
    with_sdf: bool = False,
) -> DataLoader:
    root = _project_root()
    list_txt = str(root / "configs" / "supervised" / "splits_lung_val.txt")
    ds = LungSupervised(list_txt, output_size=crop, mode="val",
                        with_sdf=with_sdf, use_center_crop_eval=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)

def build_sup_test_loader(
    crop: Tuple[int, int, int] = (112, 112, 80),
    batch_size: int = 1,
    num_workers: int = 0,
    with_sdf: bool = False,
) -> DataLoader:
    root = _project_root()
    list_txt = str(root / "configs" / "supervised" / "splits_lung_test.txt")
    ds = LungSupervised(list_txt, output_size=crop, mode="test",
                        with_sdf=with_sdf, use_center_crop_eval=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


# ========= Visualization (tái dùng cho debug) =========
def visualize_batch(
    x: torch.Tensor,
    names: List[str],
    y: Optional[torch.Tensor] = None,
    sdf: Optional[torch.Tensor] = None,
    out_dir: Union[str, Path] = "experiments/vis_sup",
    slice_idx: Union[int, str] = "middle",
    prefix: str = "sup",
    plane: str = "axial",
    dpi: int = 150,
) -> List[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def _to_np(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        if t is None: return None
        return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

    x_np = _to_np(x)     # (N,1,D,H,W)
    y_np = _to_np(y)     # (N,1,D,H,W) or None
    s_np = _to_np(sdf)   # (N,1,D,H,W) or None
    assert x_np.ndim == 5 and x_np.shape[1] == 1

    N, _, D, H, W = x_np.shape
    plane = plane.lower().strip()
    assert plane in ("axial", "coronal", "sagittal")

    def _resolve_idx():
        if slice_idx == "middle":
            return {"axial": D // 2, "coronal": H // 2, "sagittal": W // 2}[plane]
        i = int(slice_idx)
        if plane == "axial":    i = max(0, min(D - 1, i))
        if plane == "coronal":  i = max(0, min(H - 1, i))
        if plane == "sagittal": i = max(0, min(W - 1, i))
        return i

    idx = _resolve_idx()

    def _slice2d(vol3d: np.ndarray, plane: str, idx: int) -> np.ndarray:
        if plane == "axial":     return vol3d[idx, :, :]
        if plane == "coronal":   return vol3d[:, idx, :]
        return vol3d[:, :, idx]

    saved = []
    for i in range(N):
        img2d = _slice2d(x_np[i, 0], plane, idx)
        ncols = 1 + int(y_np is not None) + int(s_np is not None)
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
        if ncols == 1: axes = [axes]

        ax = axes[0]
        ax.imshow(img2d, cmap="gray"); ax.axis("off")
        ax.set_title(f"{names[i]} - img")

        c = 1
        if y_np is not None:
            m = _slice2d(y_np[i, 0], plane, idx)
            ax = axes[c]; ax.imshow(img2d, cmap="gray"); ax.imshow(m, cmap="Reds", alpha=0.35, vmin=0, vmax=1)
            ax.set_title("mask"); ax.axis("off"); c += 1

        if s_np is not None:
            s = _slice2d(s_np[i, 0], plane, idx)
            ax = axes[c]; im = ax.imshow(s, cmap="jet", vmin=-1, vmax=1); ax.axis("off"); ax.set_title("SDF")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        out_path = Path(out_dir) / f"{prefix}_{i:02d}_{names[i]}_{plane}{idx}.png"
        plt.tight_layout(); plt.savefig(out_path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
        saved.append(out_path)
    return saved


# ========= Self-test =========
if __name__ == "__main__":
    set_seed(SELFTEST_SEED)
    crop = SELFTEST_CROP
    root = _project_root()

    # Train loader
    train_loader = build_sup_train_loader(
        crop=crop, batch_size=SELFTEST_BATCH, num_workers=SELFTEST_NUM_WORKERS,
        with_sdf=True, sampling_mode="mixed", mixed_weights={"center_fg": 0.6, "random": 0.4}
    )
    b = next(iter(train_loader))
    xb, yb = b["image"], b["label"]
    sb = b.get("sdf", None)
    nb = list(b["case"])
    print(f"[SUP-TRAIN] x={tuple(xb.shape)} y={tuple(yb.shape)} sdf={None if sb is None else tuple(sb.shape)} names={nb}")
    visualize_batch(xb, nb, y=yb, sdf=sb, out_dir=root / SELFTEST_VIS_DIR, prefix="sup_train", plane="sagittal")

    # Val loader
    val_loader = build_sup_val_loader(crop=crop, batch_size=1, num_workers=SELFTEST_NUM_WORKERS, with_sdf=True)
    vb = next(iter(val_loader))
    xv, yv = vb["image"], vb["label"]
    sv = vb.get("sdf", None); nv = list(vb["case"])
    print(f"[SUP-VAL] x={tuple(xv.shape)} y={tuple(yv.shape)} sdf={None if sv is None else tuple(sv.shape)} names={nv}")
    visualize_batch(xv, nv, y=yv, sdf=sv, out_dir=root / SELFTEST_VIS_DIR, prefix="sup_val", plane="axial")

    print("[OK] supervised dataloader self-test + visualize done.")
