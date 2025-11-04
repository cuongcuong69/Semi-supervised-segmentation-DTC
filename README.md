
1. Conver data từ dicom sang định dạng NifTi
- Ảnh CT: 
  - Cắt ngưỡng HU [-700, 500]
  - Chuẩn hóa giá trị voxel [0, 1]
  - Resample thành 1x1x1mm
2. Chia data thành train (labeled/unlabeled)/val/test