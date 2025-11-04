# data/dataloader.py
# -*- coding: utf-8 -*-
"""
Dataloader 3D NIfTI cho semi-supervised lung segmentation (DTC-style) với sampling_mode:
  {"random", "rejection", "center_fg", "mixed"} và KHÔNG thêm nhiễu (đã bỏ RandomNoise).

- Labeled (có nhãn) -> sampling theo sampling_mode
- Unlabeled (không nhãn) -> luôn random (fallback nếu user chọn mode khác)
- TwoStreamBatchSampler đúng chuẩn (gắn vào batch_sampler)
- visualize_batch: lưu PNG (image, overlay, SDF)
- Self-test: python -m data.dataloader
"""

from __future__ import annotations
import itertools
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from scipy.ndimage import distance_transform_edt as distance
import matplotlib.pyplot as plt


# =============================================================================
# SELF-TEST CONFIG
# =============================================================================
SELFTEST_SEED = 2025
SELFTEST_CROP = (112, 112, 80)
SELFTEST_BATCH_LAB = 2
SELFTEST_BATCH_UNLAB = 4
SELFTEST_TWOSTREAM_BS = 6
SELFTEST_TWOSTREAM_SEC = 2
SELFTEST_NUM_WORKERS = 0
SELFTEST_VIS_DIR = "experiments/vis"
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
    """SDF ~ [-1, 1] với biên = 0 (chuẩn DTC)."""
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


# ========= Transforms (giống repo, đã bỏ RandomNoise) =========

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
        out["has_label"] = sample.get("has_label", label is not None)
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
        out["has_label"] = bool(sample.get("has_label", "label" in sample))
        return out


# ========= Dataset với sampling_mode =========

class LungUnified(Dataset):
    """
    - labeled_list (txt): mỗi dòng 'img_rel|mask_rel'
    - unlabeled_list (txt): mỗi dòng 'img_rel'
    sampling_mode áp dụng cho **labeled**; unlabeled luôn random (fallback).
    """
    def __init__(
        self,
        labeled_list_txt: Optional[str],
        unlabeled_list_txt: Optional[str],
        output_size: Tuple[int, int, int] = (112, 112, 80),
        mode: str = "train",   # 'train'|'val'|'test'
        with_sdf: bool = True,
        use_center_crop_val: bool = True,
        sampling_mode: str = "random",               # "random"|"rejection"|"center_fg"|"mixed"
        rejection_thresh: float = 0.01,
        rejection_max: int = 8,
        mixed_weights: Optional[Dict[str, float]] = None,  # e.g., {"center_fg":0.6,"random":0.4}
    ):
        self.mode = mode
        self.output_size = tuple(map(int, output_size))
        self.with_sdf = with_sdf
        self.use_center_crop_val = use_center_crop_val

        # sampling config (áp dụng cho train + labeled)
        self.sampling_mode = sampling_mode
        self.rejection_thresh = float(rejection_thresh)
        self.rejection_max = int(rejection_max)
        if mixed_weights is None:
            mixed_weights = {"center_fg": 0.6, "random": 0.4}
        # chuẩn hóa và lọc key hợp lệ
        valid_keys = {"random", "rejection", "center_fg"}
        mixed_weights = {k: v for k, v in mixed_weights.items() if k in valid_keys and v > 0}
        s = sum(mixed_weights.values()) or 1.0
        self.mixed_weights = {k: v / s for k, v in mixed_weights.items()}

        items: List[Dict[str, Any]] = []
        if labeled_list_txt is not None:
            with open(labeled_list_txt, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    img_rel, mask_rel = line.split("|")
                    items.append({
                        "img": _abs_from_root(img_rel),
                        "mask": _abs_from_root(mask_rel),
                        "has_label": True
                    })
        if unlabeled_list_txt is not None:
            with open(unlabeled_list_txt, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append({
                        "img": _abs_from_root(line),
                        "mask": None,
                        "has_label": False
                    })

        self.items = items
        print(f"[LungUnified] mode={mode} | total={len(self.items)} | "
              f"labeled={(labeled_list_txt is not None)} | unlabeled={(unlabeled_list_txt is not None)}")

        # Compose transforms cho eval
        if mode in ("val", "test"):
            if self.use_center_crop_val:
                self.transforms_eval = [CenterCrop(self.output_size), ToTensor()]
            else:
                self.transforms_eval = [ToTensor()]

    def __len__(self) -> int:
        return len(self.items)

    # ---------- sampling helpers (chỉ dùng cho TRAIN + Labeled) ----------
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
        return img[z:z+d, y:y+h, x:x+w], (None if msk is None else msk[z:z+d, y:y+h, x:x+w])

    def _crop_rejection(self, img, msk):
        if msk is None:
            return self._crop_random(img, msk)
        D, H, W = img.shape
        d, h, w = self.output_size
        for _ in range(self.rejection_max):
            z, y, x = _random_crop_coords(D, H, W, d, h, w)
            sub = msk[z:z+d, y:y+h, x:x+w]
            if sub.mean() >= self.rejection_thresh:
                return img[z:z+d, y:y+h, x:x+w], sub
        return self._crop_random(img, msk)

    def _crop_center_fg(self, img, msk):
        if msk is None:
            return self._crop_random(img, msk)
        pts = np.argwhere(msk > 0.5)
        if len(pts) == 0:
            return self._crop_random(img, msk)
        D, H, W = img.shape
        d, h, w = self.output_size
        zc, yc, xc = pts[np.random.randint(len(pts))]
        zs, ys, xs = _centered_crop_coords(int(zc), int(yc), int(xc), D, H, W, d, h, w)
        return img[zs:zs+d, ys:ys+h, xs:xs+w], msk[zs:zs+d, ys:ys+h, xs:xs+w]

    def _sample_patch(self, img, msk, has_label: bool):
        if not has_label or self.mode != "train":
            return self._crop_random(img, msk)
        mode = self.sampling_mode
        if mode == "mixed":
            mode = self._choose_mixed_mode()
        if mode == "rejection":
            return self._crop_rejection(img, msk)
        if mode == "center_fg":
            return self._crop_center_fg(img, msk)
        return self._crop_random(img, msk)

    # ---------- __getitem__ ----------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        img = load_nii(item["img"])
        img = _ensure_min_size(img, self.output_size)
        case = Path(item["img"]).parent.name
        has_label = item["has_label"]

        if self.mode in ("val", "test"):
            sample: Dict[str, Any] = {"image": img, "case": case, "has_label": has_label}
            if has_label and item["mask"] is not None:
                mask = (load_nii(item["mask"]) > 0.5).astype(np.uint8)
                mask = _ensure_min_size(mask, self.output_size)
                sample["label"] = mask
                if self.with_sdf:
                    sample["sdf"] = compute_sdf(mask, mask.shape)
            for t in self.transforms_eval:
                sample = t(sample)
            return sample

        # TRAIN
        if has_label and item["mask"] is not None:
            mask = (load_nii(item["mask"]) > 0.5).astype(np.uint8)
            mask = _ensure_min_size(mask, self.output_size)
        else:
            mask = None

        # sampling patch theo sampling_mode
        img_p, mask_p = self._sample_patch(img, mask, has_label)

        sample: Dict[str, Any] = {"image": img_p, "case": case, "has_label": has_label}
        if has_label and mask_p is not None:
            sample["label"] = mask_p
            if self.with_sdf:
                sample["sdf"] = compute_sdf(mask_p, mask_p.shape)

        # Train: chỉ ToTensor (đã bỏ RandomNoise/RotFlip để đúng yêu cầu "bỏ nhiễu đi")
        sample = ToTensor()(sample)
        return sample


# ========= TwoStreamBatchSampler =========

class TwoStreamBatchSampler(Sampler[List[int]]):
    """
    Lặp hai dãy chỉ số: primary (labeled) + secondary (unlabeled).
    Một 'epoch' duyệt hết primary; secondary lặp vô hạn.
    """
    def __init__(self, primary_indices: List[int], secondary_indices: List[int],
                 batch_size: int, secondary_batch_size: int):
        assert len(primary_indices) > 0 and len(secondary_indices) > 0
        assert 0 < secondary_batch_size < batch_size
        self.primary_indices = np.array(primary_indices)
        self.secondary_indices = np.array(secondary_indices)
        self.primary_batch = batch_size - secondary_batch_size
        self.secondary_batch = secondary_batch_size

    def __iter__(self):
        primary_iter = self._iterate_once(self.primary_indices)
        secondary_iter = self._iterate_eternally(self.secondary_indices)
        for (p, s) in zip(self._grouper(primary_iter, self.primary_batch),
                          self._grouper(secondary_iter, self.secondary_batch)):
            yield list(p) + list(s)

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch

    @staticmethod
    def _iterate_once(indices):
        return (i for i in np.random.permutation(indices))

    @staticmethod
    def _iterate_eternally(indices):
        def infinite_shuffles():
            while True:
                yield np.random.permutation(indices)
        return itertools.chain.from_iterable(infinite_shuffles())

    @staticmethod
    def _grouper(iterable, n):
        args = [iter(iterable)] * n
        return zip(*args)


# ========= Builders =========

def build_train_loaders(
    crop: Tuple[int, int, int] = (112, 112, 80),
    batch_lab: int = 2,
    batch_unlab: int = 2,
    num_workers: int = 0,
    seed: int = 2025,
    sampling_mode: str = "mixed",
    rejection_thresh: float = 0.01,
    rejection_max: int = 8,
    mixed_weights: Optional[Dict[str, float]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Trả 2 DataLoader riêng: labeled và unlabeled (dict đã collate).
    sampling_mode áp dụng cho Labeled.
    """
    set_seed(seed)
    root = _project_root()
    lab_list = str(root / "configs" / "splits_lung_train_labeled.txt")
    unlab_list = str(root / "configs" / "splits_lung_train_unlabeled.txt")

    ds_lab = LungUnified(
        lab_list, None, output_size=crop, mode="train", with_sdf=True,
        sampling_mode=sampling_mode, rejection_thresh=rejection_thresh,
        rejection_max=rejection_max, mixed_weights=mixed_weights,
    )
    ds_un  = LungUnified(
        None, unlab_list, output_size=crop, mode="train", with_sdf=False,
        sampling_mode="random"  # unlabeled luôn random
    )

    lab_loader = DataLoader(ds_lab, batch_size=batch_lab, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    unlab_loader = DataLoader(ds_un, batch_size=batch_unlab, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    return lab_loader, unlab_loader


def build_train_loader_twostream(
    crop: Tuple[int, int, int] = (112, 112, 80),
    batch_size: int = 4,
    secondary_batch_size: int = 2,
    num_workers: int = 0,
    seed: int = 2025,
    with_sdf: bool = True,
    sampling_mode: str = "mixed",
    rejection_thresh: float = 0.01,
    rejection_max: int = 8,
    mixed_weights: Optional[Dict[str, float]] = None,
) -> DataLoader:
    """
    Trả 1 DataLoader dùng TwoStreamBatchSampler. Mỗi batch là list[dict].
    """
    set_seed(seed)
    root = _project_root()
    lab_list = str(root / "configs" / "splits_lung_train_labeled.txt")
    unlab_list = str(root / "configs" / "splits_lung_train_unlabeled.txt")

    ds = LungUnified(
        lab_list, unlab_list, output_size=crop, mode="train", with_sdf=with_sdf,
        sampling_mode=sampling_mode, rejection_thresh=rejection_thresh,
        rejection_max=rejection_max, mixed_weights=mixed_weights
    )

    primary_indices, secondary_indices = [], []
    for i, it in enumerate(ds.items):
        (primary_indices if it["has_label"] else secondary_indices).append(i)

    sampler = TwoStreamBatchSampler(
        primary_indices, secondary_indices,
        batch_size=batch_size, secondary_batch_size=secondary_batch_size
    )

    def _identity_collate(batch):
        return batch

    loader = DataLoader(
        ds,
        batch_sampler=sampler,     # LƯU Ý
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_identity_collate,
    )
    return loader


def build_val_loader(
    crop: Tuple[int, int, int] = (112, 112, 80),
    batch_val: int = 1,
    num_workers: int = 0
) -> DataLoader:
    root = _project_root()
    val_list = str(root / "configs" / "splits_lung_val.txt")
    ds_val = LungUnified(val_list, None, output_size=crop, mode="val",
                         with_sdf=True, use_center_crop_val=True)
    return DataLoader(ds_val, batch_size=batch_val, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


def build_test_loader(
    crop: Tuple[int, int, int] = (112, 112, 80),
    batch_test: int = 1,
    num_workers: int = 0
) -> DataLoader:
    root = _project_root()
    test_list = str(root / "configs" / "splits_lung_test.txt")
    ds_test = LungUnified(test_list, None, output_size=crop, mode="test",
                          with_sdf=True, use_center_crop_val=True)
    return DataLoader(ds_test, batch_size=batch_test, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


# ========= Visualization =========

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import matplotlib.pyplot as plt


def visualize_batch(
    x: torch.Tensor,
    names: List[str],
    y: Optional[torch.Tensor] = None,
    sdf: Optional[torch.Tensor] = None,
    out_dir: Union[str, Path] = "experiments/vis",
    slice_idx: Union[int, str] = "middle",
    prefix: str = "batch",
    plane: str = "axial",           # "axial"(Z), "coronal"(Y), "sagittal"(X)
    dpi: int = 150,
) -> List[Path]:
    """
    Vẽ và lưu ảnh 2D từ batch 3D (N, C=1, D, H, W).
    - x: ảnh input [N,1,D,H,W]
    - names: tên từng case (dùng đặt file)
    - y: (tùy chọn) mask [N,1,D,H,W] (0/1)
    - sdf: (tùy chọn) signed distance map [N,1,D,H,W] ~ [-1,1]
    - out_dir: thư mục lưu PNG
    - slice_idx: "middle" hoặc chỉ số int (áp dụng theo trục của plane)
    - prefix: tiền tố tên file
    - plane: "axial" (cắt theo Z, nhìn từ trên xuống – thường dùng),
             "coronal" (cắt theo Y, nhìn trước-sau),
             "sagittal" (cắt theo X, nhìn trái-phải)
    - dpi: độ phân giải lưu ảnh

    Trả về: danh sách đường dẫn ảnh đã lưu (Path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- đảm bảo định dạng numpy
    def _to_np(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        if t is None:
            return None
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        return t

    x_np = _to_np(x)     # (N,1,D,H,W)
    y_np = _to_np(y)     # (N,1,D,H,W) hoặc None
    s_np = _to_np(sdf)   # (N,1,D,H,W) hoặc None

    assert x_np.ndim == 5 and x_np.shape[1] == 1, "Kỳ vọng x có shape [N,1,D,H,W]."
    N, C, D, H, W = x_np.shape

    # --- xác định chỉ số lát theo plane
    plane = plane.lower().strip()
    if plane not in ("axial", "coronal", "sagittal"):
        raise ValueError("plane must be 'axial'|'coronal'|'sagittal'")

    def _resolve_idx():
        if slice_idx == "middle":
            idx = {"axial": D // 2, "coronal": H // 2, "sagittal": W // 2}[plane]
        else:
            i = int(slice_idx)
            if plane == "axial":    i = max(0, min(D - 1, i))
            if plane == "coronal":  i = max(0, min(H - 1, i))
            if plane == "sagittal": i = max(0, min(W - 1, i))
            idx = i
        return idx

    idx = _resolve_idx()

    # --- helper cắt 2D theo plane
    def _slice2d(vol3d: np.ndarray, plane: str, idx: int) -> np.ndarray:
        # vol3d: (D,H,W)
        if plane == "axial":      # Z-fixed -> (H,W)
            return vol3d[idx, :, :]
        elif plane == "coronal":  # Y-fixed -> (D,W)
            return vol3d[:, idx, :]
        else:                     # sagittal: X-fixed -> (D,H)
            return vol3d[:, :, idx]

    saved_paths: List[Path] = []

    for i in range(N):
        vol = x_np[i, 0]  # (D,H,W)

        img2d = _slice2d(vol, plane, idx)
        # dựng layout: cột 1 = img, cột 2 (nếu có) = overlay mask, cột 3 (nếu có) = SDF
        ncols = 1 + int(y_np is not None) + int(s_np is not None)
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
        if ncols == 1:
            axes = [axes]

        # --- cột 1: ảnh xám
        ax = axes[0]
        ax.imshow(img2d, cmap="gray")
        axis_name = {"axial": "z", "coronal": "y", "sagittal": "x"}[plane]
        ax.set_title(f"{names[i]} - img ({axis_name}={idx})")
        ax.axis("off")

        col = 1

        # --- cột 2: overlay mask
        if y_np is not None:
            msk2d = _slice2d(y_np[i, 0], plane, idx)
            ax = axes[col]
            ax.imshow(img2d, cmap="gray")
            ax.imshow(msk2d, cmap="Reds", alpha=0.35, vmin=0, vmax=1)
            ax.set_title("mask overlay")
            ax.axis("off")
            col += 1

        # --- cột 3: SDF
        if s_np is not None:
            sdf2d = _slice2d(s_np[i, 0], plane, idx)
            ax = axes[col]
            im = ax.imshow(sdf2d, cmap="jet", vmin=-1, vmax=1)
            ax.set_title("SDF")
            ax.axis("off")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel("distance (norm)", rotation=270, labelpad=12)

        # --- lưu file
        out_path = out_dir / f"{prefix}_{i:02d}_{names[i]}_{plane}{idx}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths



# ========= Self-test =========

if __name__ == "__main__":
    set_seed(SELFTEST_SEED)
    crop = SELFTEST_CROP

    # --- 2 loader: Labeled & Unlabeled
    lab_loader, un_loader = build_train_loaders(
        crop=crop,
        batch_lab=SELFTEST_BATCH_LAB,
        batch_unlab=SELFTEST_BATCH_UNLAB,
        num_workers=SELFTEST_NUM_WORKERS,
        sampling_mode="mixed",                # thử mixed default
        rejection_thresh=0.01,
        rejection_max=8,
        mixed_weights={"center_fg": 0.6, "random": 0.4},
    )

    lab_batch = next(iter(lab_loader))
    xb = lab_batch["image"]; yb = lab_batch["label"]; sb = lab_batch["sdf"]; nb = lab_batch["case"]
    print(f"[LAB] x={tuple(xb.shape)} y={tuple(yb.shape)} sdf={tuple(sb.shape)} names={list(nb)}")

    un_batch = next(iter(un_loader))
    xu = un_batch["image"]; nu = un_batch["case"]
    print(f"[UNLAB] x={tuple(xu.shape)} names={list(nu)}")

    vis_dir = _project_root() / SELFTEST_VIS_DIR
    visualize_batch(xb, list(nb), y=yb, sdf=sb, out_dir=vis_dir, plane="sagittal", slice_idx=SELFTEST_SLICE, prefix="train_lab")
    visualize_batch(xu, list(nu), y=None, sdf=None, out_dir=vis_dir, plane="sagittal", slice_idx=SELFTEST_SLICE, prefix="train_unlab")
    print(f"[VIS] Saved PNGs to: {vis_dir}")

    # --- 1 loader: TwoStream (list[dict] mỗi batch)
    tl = build_train_loader_twostream(
        crop=crop,
        batch_size=SELFTEST_TWOSTREAM_BS,
        secondary_batch_size=SELFTEST_TWOSTREAM_SEC,
        num_workers=SELFTEST_NUM_WORKERS,
        sampling_mode="mixed",
        mixed_weights={"center_fg": 0.6, "random": 0.4},
    )
    batch_list = next(iter(tl))

    lab_b = [b for b in batch_list if b["has_label"]]
    un_b  = [b for b in batch_list if not b["has_label"]]
    if lab_b:
        x_lab = torch.stack([b["image"] for b in lab_b], dim=0)
        y_lab = torch.stack([b["label"] for b in lab_b], dim=0)
        s_lab = torch.stack([b["sdf"]   for b in lab_b], dim=0)
        n_lab = [b["case"] for b in lab_b]
        visualize_batch(x_lab, n_lab, y=y_lab, sdf=s_lab, out_dir=vis_dir, plane="sagittal", slice_idx=SELFTEST_SLICE, prefix="twostream_lab")
        
    if un_b:
        x_un  = torch.stack([b["image"] for b in un_b], dim=0)
        n_un  = [b["case"] for b in un_b]
        visualize_batch(x_un, n_un, y=None, sdf=None, out_dir=vis_dir, plane="sagittal", slice_idx=SELFTEST_SLICE, prefix="twostream_unlab")

    print("[OK] dataloader self-test + visualize done.")
