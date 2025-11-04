# -*- coding: utf-8 -*-
"""
Sinh 4 file:
- configs/splits_lung_train_labeled.txt     (LUNG1-001..010)
- configs/splits_lung_train_unlabeled.txt   (LUNG1-011..060)
- configs/splits_lung_val.txt               (LUNG1-061..080)
- configs/splits_lung_test.txt              (LUNG1-081..100)

Định dạng:
- labeled/val/test:  <path_img>|<path_mask>
- unlabeled:         <path_img>
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # d:\Semi-supervised segmentation
DATA_ROOT = ROOT / "data" / "processed"
CONFIGS = ROOT / "configs"
CONFIGS.mkdir(parents=True, exist_ok=True)

LABELED_TXT   = CONFIGS / "splits_lung_train_labeled.txt"
UNLABELED_TXT = CONFIGS / "splits_lung_train_unlabeled.txt"
VAL_TXT       = CONFIGS / "splits_lung_val.txt"
TEST_TXT      = CONFIGS / "splits_lung_test.txt"

IMG_CANDIDATES  = ["img.nii.gz", "image.nii.gz", "ct.nii.gz", "ct_image.nii.gz"]
MASK_CANDIDATES = ["lungmask.nii.gz", "mask.nii.gz", "seg.nii.gz", "label.nii.gz"]

def find_first_existing(folder: Path, names):
    for n in names:
        p = folder / n
        if p.exists():
            return p
    return None

def case_dir(i: int) -> Path:
    return DATA_ROOT / f"LUNG1-{i:03d}"

def to_rel_from_root(p: Path) -> str:
    # -> "data/processed/..."
    parts = p.as_posix().split("/")
    if "data" in parts:
        idx = parts.index("data")
        return "/".join(parts[idx:])
    return p.as_posix()

def write_labeled_range(start, end, fh):
    for i in range(start, end + 1):
        cd = case_dir(i)
        img = find_first_existing(cd, IMG_CANDIDATES)
        msk = find_first_existing(cd, MASK_CANDIDATES)
        if not img:
            print(f"[WARN] Missing image for {cd}; skip.")
            continue
        if not msk:
            print(f"[WARN] Missing mask for {cd}; skip (needs mask).")
            continue
        fh.write(f"{to_rel_from_root(img)}|{to_rel_from_root(msk)}\n")

def write_unlabeled_range(start, end, fh):
    for i in range(start, end + 1):
        cd = case_dir(i)
        img = find_first_existing(cd, IMG_CANDIDATES)
        if not img:
            print(f"[WARN] Missing image for {cd}; skip.")
            continue
        fh.write(f"{to_rel_from_root(img)}\n")

def main():
    with open(LABELED_TXT, "w", encoding="utf-8") as f:
        write_labeled_range(1, 10, f)     # train labeled

    with open(UNLABELED_TXT, "w", encoding="utf-8") as f:
        write_unlabeled_range(11, 60, f)  # train unlabeled

    with open(VAL_TXT, "w", encoding="utf-8") as f:
        write_labeled_range(61, 80, f)    # validation (with labels)

    with open(TEST_TXT, "w", encoding="utf-8") as f:
        write_labeled_range(81, 100, f)   # test (with labels)

    print("[OK] Wrote splits:")
    print(" -", LABELED_TXT)
    print(" -", UNLABELED_TXT)
    print(" -", VAL_TXT)
    print(" -", TEST_TXT)

if __name__ == "__main__":
    main()
