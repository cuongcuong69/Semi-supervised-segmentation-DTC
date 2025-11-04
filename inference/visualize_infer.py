# -*- coding: utf-8 -*-
"""
visualize_infer_batch.py
Tạo visualization 3D (prediction vs ground-truth) cho nhiều bệnh nhân trong danh sách CASE_LIST.
Kết quả mỗi bệnh nhân: 1 file HTML (và PNG nếu có kaleido)
"""

import os
import numpy as np
import nibabel as nib
from skimage.measure import marching_cubes
import plotly.graph_objects as go

# ==== DANH SÁCH CASE CẦN VISUALIZE ==========================================
CASE_LIST = CASE_LIST = [str(i).zfill(3) for i in range(81, 101)]


pred_root = r"d:\Semi-supervised segmentation\experiments\inference_results"
data_root = r"d:\Semi-supervised segmentation\data\processed"
label_name = "lungmask.nii.gz"
out_dir = os.path.join(pred_root, "viz_batch")
os.makedirs(out_dir, exist_ok=True)

pred_color = "#E74C3C"
gt_color = "#2ECC71"
opacity_pred = 0.45
opacity_gt = 0.55


# ==== HÀM PHỤ ================================================================

def load_nii(path):
    img = nib.load(path)
    arr = img.get_fdata().astype(np.float32)
    zooms = img.header.get_zooms()[:3]
    return arr, zooms

def dice_coefficient(pred, gt, eps=1e-8):
    inter = np.sum((pred > 0) & (gt > 0))
    return (2 * inter + eps) / (np.sum(pred > 0) + np.sum(gt > 0) + eps)

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
    vol = np.transpose(mask_bin, (2,1,0))
    sx, sy, sz = spacing
    spacing_zyx = (sz, sy, sx)
    verts, faces, _, _ = marching_cubes(vol, level=level, spacing=spacing_zyx)
    return verts, faces

def make_mesh3d(verts, faces, color, name, opacity):
    i, j, k = faces.T
    return go.Mesh3d(
        x=verts[:,2], y=verts[:,1], z=verts[:,0],
        i=i, j=j, k=k,
        color=color,
        name=name,
        opacity=opacity,
        lighting=dict(ambient=0.6, diffuse=0.8),
        flatshading=True
    )


# ==== CHẠY CHO TỪNG CASE =====================================================
for cid in CASE_LIST:
    case_name = f"LUNG1-{cid.zfill(3)}"
    pred_path = os.path.join(pred_root, f"{case_name}_pred.nii.gz")
    gt_path = os.path.join(data_root, case_name, label_name)
    out_html = os.path.join(out_dir, f"visualize_{case_name}.html")
    out_png  = os.path.join(out_dir, f"visualize_{case_name}.png")

    if not os.path.exists(pred_path):
        print(f"[SKIP] Không tìm thấy {pred_path}")
        continue
    if not os.path.exists(gt_path):
        print(f"[SKIP] Không tìm thấy {gt_path}")
        continue

    print(f"\n=== Processing {case_name} ===")
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
    dice = dice_coefficient(crop_pred, crop_gt)
    print(f"[METRIC] Dice = {dice:.4f}")

    meshes = []
    sx, sy, sz = spacing

    if np.sum(crop_gt) > 0:
        verts_gt, faces_gt = mask_to_mesh(crop_gt, spacing)
        verts_gt += np.array([z0*sz, y0*sy, x0*sx])
        meshes.append(make_mesh3d(verts_gt, faces_gt, gt_color, f"GT ({case_name})", opacity_gt))

    if np.sum(crop_pred) > 0:
        verts_pr, faces_pr = mask_to_mesh(crop_pred, spacing)
        verts_pr += np.array([z0*sz, y0*sy, x0*sx])
        meshes.append(make_mesh3d(verts_pr, faces_pr, pred_color, f"Pred ({case_name})", opacity_pred))

    if not meshes:
        print("[WARN] Không có mask hợp lệ, bỏ qua.")
        continue

    fig = go.Figure(data=meshes)
    fig.update_layout(
        title=f"{case_name} | Dice={dice:.4f}",
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.write_html(out_html)
    print(f"[OUT] Lưu HTML: {out_html}")

    try:
        fig.write_image(out_png, scale=2, width=1280, height=900)
        print(f"[OUT] Lưu PNG: {out_png}")
    except Exception as e:
        print(f"[WARN] Không xuất PNG (cần kaleido): {e}")

print("\n Hoàn tất visualize hàng loạt.")
