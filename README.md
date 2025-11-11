
## Project: Semi-supervised segmentation (DTC)

Mô tả ngắn
------------
Đây là một repository triển khai phương pháp phân đoạn bán giám sát dựa trên DTC (Dual-Task Consistency) sử dụng mô hình V-Net để cải thiện chất lượng phân đoạn phổi trên ảnh CT. Project chứa mã tiền xử lý dữ liệu, chia tập dữ liệu, mô hình, hàm mất mát, training loop cho cả supervised và semi-supervised, cùng các script inference và trực quan hóa kết quả.

Mục tiêu
---------
- Triển khai pipeline tiền xử lý DICOM -> NIfTI chuẩn hóa (HU clipping, normalization, resampling) để có voxel isotropic 1x1x1 mm.
- Xây dựng mô hình V-Net (và V-Net với SDF) cho bài toán phân đoạn phổi/khối u.
- Hỗ trợ huấn luyện supervised và semi-supervised (chia labeled/unlabeled) với loss composite và các ramp-up schedule.
- Hỗ trợ inference, lưu checkpoint và trực quan hóa kết quả (HTML/visualizations).

Tóm tắt các bước tiền xử lý dữ liệu
----------------------------------
1. Chuyển DICOM -> NIfTI bằng các script trong `scripts/` (ví dụ `convert_nsclc.py`, `convert_nsclc2.py`).
2. Áp dụng clipping giá trị HU: giới hạn trong phạm vi [-700, 500].
3. Chuẩn hóa voxel intensity về khoảng [0, 1] (min-max trên cửa sổ HU đã clipping) để đảm bảo ổn định huấn luyện.
4. Resample lại ảnh thành isotropic spacing 1x1x1 mm.
5. Chuẩn bị các thư mục dữ liệu processed trong `data/processed/` theo cấu trúc từng ca (ví dụ `LUNG1-001`, ...).

Chia tập (splits)
------------------
- Các file split có trong `configs/` (ví dụ `splits_lung_train_labeled.txt`, `splits_lung_train_unlabeled.txt`, `splits_lung_val.txt`, `splits_lung_test.txt`).
- Ngoài ra có thư mục `configs/supervised/` chứa các split cho kịch bản huấn luyện hoàn toàn supervised.

Cấu trúc repository (mô tả thư mục chính)
---------------------------------------
- `data/`:
  - `dataloader.py` và `dataloader_sup.py`: dataloader cho chế độ semi-supervised và supervised tương ứng. Chịu trách nhiệm đọc NIfTI, áp augmentation, crop/resample nếu cần.
  - `processed/`: nơi chứa các NIfTI đã xử lý, tổ chức theo ca bệnh `LUNG1-***`.
  - `tempCodeRunnerFile.py`: file tạm (IDE).

- `scripts/`:
  - `convert_nsclc.py`, `convert_nsclc2.py`: script chuyển đổi dữ liệu thô (DICOM) sang NIfTI và áp tiền xử lý cơ bản.
  - `make_splits.py`: script tạo các file split (train/val/test, labeled/unlabeled).

- `models/`:
  - `vnet.py`: định nghĩa kiến trúc V-Net chính.
  - `vnet_sdf.py`: biến thể V-Net làm việc với SDF hoặc head SDF kết hợp.

- `losses/`:
  - `losses.py`, `losses_2.py`, `composite.py`, `functional.py`: các định nghĩa hàm mất mát (Dice, BCE, SDF-related losses), utility và ramps (điều chỉnh trọng số loss theo epoch).
  - `metrics.py`: các metric đánh giá (Dice, IoU, v.v.).

- `trainers/`:
  - `train_vnet_sup.py`: trainer cho huấn luyện supervised với V-Net.
  - `train_dtc.py`: trainer cho phương pháp DTC (semi-supervised) kết hợp labeled + unlabeled.

- `inference/`:
  - `infer_vnet_sup.py`, `infer_dtc.py`: script thực hiện inference từ checkpoint được huấn luyện.
  - `visualize_infer.py`: script tạo kết quả trực quan (ví dụ HTML, ảnh overlay) để đánh giá.

- `experiments/`:
  - Lưu các kết quả và checkpoint: `experiments/<exp_name>/checkpoints/{best.ckpt,last.ckpt}`.
  - `inference_results/`, `vis/` chứa kết quả inference và file trực quan hóa (ví dụ `visualize_LUNG1-081.html`).

- `notebooks/`: notebook thử nghiệm và debug (ví dụ `test.ipynb`).


Các script chính và cách chạy (ví dụ)
-----------------------------------
Lưu ý: các script trong repo thường sử dụng argparse hoặc cấu hình nội bộ; dưới đây là ví dụ tổng quát cách chạy (tùy script có thể có flag khác nhau):

- Chạy huấn luyện supervised (ví dụ):

```powershell
python trainers\\train_vnet_sup.py
```

- Chạy huấn luyện semi-supervised DTC (ví dụ):

```powershell
python trainers\\train_dtc.py
```

- Thực hiện inference với checkpoint đã lưu:

```powershell
python inference\\infer_vnet_sup.py --checkpoint experiments\\vnet_sup\\checkpoints\\best.ckpt
python inference\\infer_dtc.py --checkpoint experiments\\dtc_nsclc_vnet_sdf\\checkpoints\\best.ckpt
```

- Trực quan hóa kết quả inference (HTML/overlay):

```powershell
python inference\\visualize_infer.py --results_dir experiments\\inference_results
```

Ghi chú: Kiểm tra các script để biết các tham số chính xác (đường dẫn dữ liệu, checkpoint, batch size, device...).

Chi tiết triển khai (ghi chú kỹ thuật)
-------------------------------------
- DataLoader (`data/dataloader.py`, `data/dataloader_sup.py`):
  - Xử lý đọc NIfTI, normalization, augmentation/patch sampling.
  - Hỗ trợ chế độ labeled/unlabeled cho training bán giám sát.

- Mô hình (`models/vnet.py`, `models/vnet_sdf.py`):
  - Kiến trúc V-Net dạng 3D UNet-like dành cho ảnh y tế 3D.
  - `vnet_sdf.py` bổ sung head để dự đoán SDF (nếu sử dụng loss SDF).

- Losses & metrics (`losses/`):
  - Dice loss, BCE, combined/composite losses cho labeled và unlabeled.
  - `ramps.py` để điều chỉnh hệ số loss (ví dụ tăng dần trọng số unsupervised consistency).

- Trainers (`trainers/`):
  - Quản lý vòng epoch, logging, lưu checkpoint (best, last), tính metric val.
  - `train_dtc.py` có logic kết hợp labeled + unlabeled (consistency loss, pseudo-labels hoặc SDF-based constraint).

Kiểm tra và kết quả experiments
--------------------------------
- Các kết quả, checkpoint và visualizations mẫu đã được lưu trong `experiments/`.
- Mỗi experiment chứa folder `checkpoints/` và kết quả inference; ví dụ: `experiments/dtc_nsclc_vnet_sdf/checkpoints/best.ckpt`.


