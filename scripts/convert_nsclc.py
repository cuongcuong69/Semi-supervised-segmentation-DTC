import os
import glob
import numpy as np
import pydicom
import SimpleITK as sitk

# ===================== CONFIG CỦA BẠN =====================
# Thư mục raw gốc (có LUNG1-001, LUNG1-002, ...)
RAW_ROOT = r"D:\NSCLC_Radiomics\NSCLC-Radiomics"

# Thư mục output trong dự án
OUT_ROOT = r"D:\Semi-supervised segmentation\data\processed"

# Chạy từ bệnh nhân nào tới bệnh nhân nào
START_PATIENT = 1      # -> LUNG1-001
END_PATIENT = 100       # -> LUNG1-010

# Resample CT về spacing này
TARGET_SPACING = (1.0, 1.0, 1.0)

# HU window + normalize
HU_MIN = -700
HU_MAX = 500

# Chỉ giữ segment nào trong SEG mà tên (lower) có chứa 1 trong các keyword này
LABEL_KEYWORDS = ("lung",)

# Có muốn fill/mịn mask không? (để đỡ nhìn như sọc trong coronal/sagittal)
# cần scipy nếu bật lên
FILL_MASK = True
# ===========================================================


# ----------------- CT utils ----------------- #
def find_study_folder(patient_dir: str):
    """Trong mỗi LUNG1-xxx chỉ có 1 folder study ⇒ lấy luôn cái đầu."""
    subdirs = [
        os.path.join(patient_dir, d)
        for d in os.listdir(patient_dir)
        if os.path.isdir(os.path.join(patient_dir, d))
    ]
    return subdirs[0] if subdirs else None


def list_series_dirs(study_folder: str):
    """
    Trong study có vài series:
      - CT: '0.000000-NA-...', '1.000000-NA-...'
      - Seg: '300.000000-Segmentation-...'
    Ta tách riêng.
    """
    ct_dirs, seg_dirs = [], []
    for d in os.listdir(study_folder):
        full = os.path.join(study_folder, d)
        if not os.path.isdir(full):
            continue
        if "Segmentation" in d or "SEG" in d:
            seg_dirs.append(full)
        else:
            ct_dirs.append(full)
    return ct_dirs, seg_dirs


def pick_ct_series(ct_dirs):
    """
    Có thể có 2-3 series CT. Lấy cái có nhiều file DICOM nhất.
    """
    if not ct_dirs:
        return None
    best_dir, best_cnt = None, -1
    for d in ct_dirs:
        n = len(glob.glob(os.path.join(d, "*.dcm")))
        if n > best_cnt:
            best_cnt = n
            best_dir = d
    return best_dir


def load_dicom_series_as_image(series_dir: str):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(series_dir)
    reader.SetFileNames(dicom_files)
    return reader.Execute()


def window_and_normalize(image_sitk, hu_min=-700, hu_max=500):
    arr = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
    arr = np.clip(arr, hu_min, hu_max)
    arr = (arr - hu_min) / (hu_max - hu_min)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image_sitk)
    return out


def resample_to_spacing(image_sitk, out_spacing=(1.0, 1.0, 1.0), is_label=False):
    orig_spacing = image_sitk.GetSpacing()
    orig_size = image_sitk.GetSize()

    out_spacing = tuple(out_spacing)
    out_size = [
        int(round(osz * (ospc / nspc)))
        for osz, ospc, nspc in zip(orig_size, orig_spacing, out_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(out_size)
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(image_sitk)


def save_nifti(image_sitk, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sitk.WriteImage(image_sitk, out_path)
    print(f"[+] Saved: {out_path}")


# ----------------- SEG (QIICR) → mask ----------------- #
def load_qiicr_seg_as_lung_mask(seg_file: str, keywords=("lung",)):
    """
    Đọc 1 file DICOM SEG kiểu QIICR (Modality=SEG),
    chỉ lấy những frame thuộc các SegmentLabel có chứa từ khóa.
    Trả về:
        - sitk.Image mask (spacing/origin/direction lấy từ chính SEG)
        - set các segmentNumber đã dùng
    Nếu không có segment nào phù hợp => (None, None)
    """
    ds = pydicom.dcmread(seg_file)

    # phải là SEG
    if ds.Modality != "SEG":
        return None, None
    if not hasattr(ds, "SegmentSequence"):
        return None, None

    # 1) tìm segmentNumber cần lấy
    wanted_segments = set()
    for seg in ds.SegmentSequence:
        label = seg.SegmentLabel
        if any(kw in label.lower() for kw in keywords):
            wanted_segments.add(seg.SegmentNumber)

    if not wanted_segments:
        return None, None

    # 2) lấy info chung
    shared_fg = ds.SharedFunctionalGroupsSequence[0]
    pxm = shared_fg.PixelMeasuresSequence[0]
    row_spacing = float(pxm.PixelSpacing[0])
    col_spacing = float(pxm.PixelSpacing[1])
    slice_thickness = float(getattr(pxm, "SliceThickness", 1.0))

    ori = shared_fg.PlaneOrientationSequence[0].ImageOrientationPatient
    row_cos = [float(x) for x in ori[0:3]]
    col_cos = [float(x) for x in ori[3:6]]
    normal = np.cross(row_cos, col_cos)
    direction = row_cos + col_cos + list(normal)

    # 3) data
    all_frames = ds.pixel_array  # (num_frames, rows, cols)
    per_frame_seq = ds.PerFrameFunctionalGroupsSequence

    slices = []  # list of (z, mask2d, ipp)
    for fi, frame in enumerate(per_frame_seq):
        seg_id = frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber
        if seg_id not in wanted_segments:
            continue

        ipp = frame.PlanePositionSequence[0].ImagePositionPatient
        z = float(ipp[2])
        mask2d = (all_frames[fi] > 0).astype(np.uint8)
        slices.append((z, mask2d, ipp))

    if not slices:
        return None, None

    # sắp xếp theo z
    slices.sort(key=lambda x: x[0])
    masks_2d = [s[1] for s in slices]
    ipps = [s[2] for s in slices]

    volume = np.stack(masks_2d, axis=0)  # (z, y, x)

    # tính z spacing
    if len(ipps) >= 2:
        z_positions = [float(p[2]) for p in ipps]
        z_diffs = np.diff(z_positions)
        z_spacing = float(np.mean(np.abs(z_diffs)))
    else:
        z_spacing = slice_thickness

    origin = [float(ipps[0][0]), float(ipps[0][1]), float(ipps[0][2])]

    sitk_mask = sitk.GetImageFromArray(volume.astype(np.uint8))
    sitk_mask.SetSpacing((col_spacing, row_spacing, z_spacing))
    sitk_mask.SetOrigin(origin)
    sitk_mask.SetDirection(direction)

    return sitk_mask, wanted_segments


def optional_fill_mask(mask_img: sitk.Image):
    """
    (tùy chọn) làm mịn / fill mask để đỡ bị sọc.
    Cần scipy. Nếu không cần thì đừng bật FILL_MASK.
    """
    try:
        import scipy.ndimage as ndi
    except ImportError:
        print("[!] scipy chưa cài, bỏ qua fill.")
        return mask_img

    arr = sitk.GetArrayFromImage(mask_img).astype(np.uint8)
    # đóng/mở nhẹ
    arr = ndi.binary_closing(arr, iterations=5)
    arr = ndi.binary_fill_holes(arr)
    arr = arr.astype(np.uint8)

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(mask_img)
    return out


# ----------------- MAIN ----------------- #
def process_one_patient(pid_int: int):
    pid = f"LUNG1-{pid_int:03d}"
    patient_dir = os.path.join(RAW_ROOT, pid)
    if not os.path.isdir(patient_dir):
        print(f"[!] Missing: {patient_dir}")
        return

    print(f"\n=== Processing {pid} ===")
    study_folder = find_study_folder(patient_dir)
    if study_folder is None:
        print(f"[!] No study folder in {patient_dir}")
        return

    ct_dirs, seg_dirs = list_series_dirs(study_folder)
    ct_dir = pick_ct_series(ct_dirs)
    if ct_dir is None:
        print(f"[!] No CT series for {pid}")
        return

    # ---- CT ----
    ct_img = load_dicom_series_as_image(ct_dir)
    ct_img_win = window_and_normalize(ct_img, hu_min=HU_MIN, hu_max=HU_MAX)
    ct_img_res = resample_to_spacing(ct_img_win, out_spacing=TARGET_SPACING, is_label=False)

    patient_out_dir = os.path.join(OUT_ROOT, pid)
    ct_out_path = os.path.join(patient_out_dir, "image.nii.gz")
    save_nifti(ct_img_res, ct_out_path)

    # ---- LABEL (DICOM SEG) ----
    if not seg_dirs:
        print(f"[!] No segmentation folder for {pid}")
        return

    seg_dir = seg_dirs[0]
    dcm_files = glob.glob(os.path.join(seg_dir, "*.dcm"))
    if not dcm_files:
        print(f"[!] Segmentation folder empty for {pid}")
        return

    seg_file = dcm_files[0]
    lung_mask_sitk, used_segments = load_qiicr_seg_as_lung_mask(seg_file, keywords=LABEL_KEYWORDS)
    if lung_mask_sitk is None:
        print(f"[!] {pid}: SEG has NO segment with keywords {LABEL_KEYWORDS}")
        return

    # (tùy chọn) làm mịn
    if FILL_MASK:
        lung_mask_sitk = optional_fill_mask(lung_mask_sitk)

    # QUAN TRỌNG: resample mask đúng theo CT (dùng CT làm reference)
    lung_mask_on_ct = sitk.Resample(
        lung_mask_sitk,
        ct_img_res,                      # reference
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        lung_mask_sitk.GetPixelID()
    )

    label_out_path = os.path.join(patient_out_dir, "lungmask.nii.gz")
    save_nifti(lung_mask_on_ct, label_out_path)
    print(f"    used segments: {used_segments}")


def main():
    for pid_int in range(START_PATIENT, END_PATIENT + 1):
        process_one_patient(pid_int)


if __name__ == "__main__":
    main()
