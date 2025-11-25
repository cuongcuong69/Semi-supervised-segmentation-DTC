# -*- coding: utf-8 -*-
"""
export_obj_mesh.py
Tạo mesh 3D từ kết quả segmentation (Prediction & Ground-truth) và xuất ra file .obj
(có thể mở bằng Blender / MeshLab / Paraview)

- Đầu vào: các file NIfTI (.nii hoặc .nii.gz)
- Sử dụng marching_cubes để sinh lưới 3D
- Xuất ra 2 file .obj: *_pred.obj và *_gt.obj (nếu có)
"""

import os
import numpy as np
import nibabel as nib
from skimage.measure import marching_cubes
from pathlib import Path

# ============================= CONFIG ========================================
CASE_LIST = [str(i).zfill(3) for i in range(81, 101)]

pred_root = r"d:\Semi-supervised segmentation\experiments\inference_results"
data_root = r"d:\Semi-supervised segmentation\data\processed"
label_name = "lungmask.nii.gz"

out_dir = os.path.join(pred_root, "obj_mesh")
os.makedirs(out_dir, exist_ok=True)

level = 0.5  # threshold cho marching_cubes
# ============================================================================

def load_nii(path):
    img = nib.load(path)
    arr = img.get_fdata().astype(np.float32)
    zooms = img.header.get_zooms()[:3]
    return arr, zooms

def bbox_from_mask(mask, pad=2):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return (0, *mask.shape)
    z0, y0, x0 = coords.min(0)
    z1, y1, x1 = coords.max(0) + 1
    z0, y0, x0 = max(z0 - pad, 0), max(y0 - pad, 0), max(x0 - pad, 0)
    z1, y1, x1 = min(z1 + pad, mask.shape[0]), min(y1 + pad, mask.shape[1]), min(x1 + pad, mask.shape[2])
    return z0, z1, y0, y1, x0, x1

def mask_to_mesh(mask_bin, spacing, level=0.5):
    """
    Convert mask (z,y,x) → verts, faces
    """
    vol = np.transpose(mask_bin, (2, 1, 0))  # (x,y,z)
    sx, sy, sz = spacing
    verts, faces, _, _ = marching_cubes(vol, level=level, spacing=(sx, sy, sz))
    return verts, faces

def export_obj(path_obj, verts, faces, color=(1.0, 0.0, 0.0)):
    """
    Ghi lưới sang file .obj với màu RGB (comment)
    """
    with open(path_obj, "w") as f:
        f.write("# OBJ file generated from segmentation mask\n")
        f.write(f"# Color: {color}\n")
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

# ============================= MAIN LOOP ====================================
for cid in CASE_LIST:
    case_name = f"LUNG1-{cid}"
    pred_path = os.path.join(pred_root, f"{case_name}_pred.nii.gz")
    gt_path = os.path.join(data_root, case_name, label_name)

    if not os.path.exists(pred_path):
        print(f"[SKIP] Không tìm thấy {pred_path}")
        continue
    if not os.path.exists(gt_path):
        print(f"[SKIP] Không tìm thấy {gt_path}")
        continue

    print(f"\n=== Exporting meshes for {case_name} ===")

    pred_arr, spacing_pred = load_nii(pred_path)
    gt_arr, spacing_gt = load_nii(gt_path)
    spacing = spacing_gt

    pred_bin = (pred_arr > 0.5).astype(np.uint8)
    gt_bin = (gt_arr > 0.5).astype(np.uint8)

    z0p, z1p, y0p, y1p, x0p, x1p = bbox_from_mask(pred_bin)
    z0g, z1g, y0g, y1g, x0g, x1g = bbox_from_mask(gt_bin)
    z0, y0, x0 = min(z0p,z0g), min(y0p,y0g), min(x0p,x0g)
    z1, y1, x1 = max(z1p,z1g), max(y1p,y1g), max(x1p,x1g)

    crop_pred = pred_bin[z0:z1, y0:y1, x0:x1]
    crop_gt   = gt_bin[z0:z1, y0:y1, x0:x1]

    if np.sum(crop_pred) > 0:
        verts_p, faces_p = mask_to_mesh(crop_pred, spacing, level)
        verts_p += np.array([x0*spacing[0], y0*spacing[1], z0*spacing[2]])
        out_pred = os.path.join(out_dir, f"{case_name}_pred.obj")
        export_obj(out_pred, verts_p, faces_p, color=(1.0, 0.0, 0.0))
        print(f"[OUT] Saved {out_pred}")

    if np.sum(crop_gt) > 0:
        verts_g, faces_g = mask_to_mesh(crop_gt, spacing, level)
        verts_g += np.array([x0*spacing[0], y0*spacing[1], z0*spacing[2]])
        out_gt = os.path.join(out_dir, f"{case_name}_gt.obj")
        export_obj(out_gt, verts_g, faces_g, color=(0.0, 1.0, 0.0))
        print(f"[OUT] Saved {out_gt}")

print("\n✅ Hoàn tất export OBJ cho tất cả case.")
