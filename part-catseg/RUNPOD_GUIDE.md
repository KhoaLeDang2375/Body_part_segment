# 🚀 Hướng Dẫn Triển Khai PartCATSeg Gradio Demo Trên RunPod

Tài liệu chi tiết hướng dẫn thiết lập, cài đặt phụ thuộc, khởi chạy ứng dụng Gradio Demo và xử lý tất cả các lỗi môi trường thường gặp khi triển khai dự án **PartCATSeg** trên **RunPod**.

---

## 📌 1. Chuẩn Bị Trên RunPod

### 1.1. Tạo Pod trên RunPod
1. **Template**: Chọn `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04` (hoặc bất kỳ template PyTorch 2.x nào).
2. **GPU**: Chọn bất kỳ GPU nào (RTX 3090, RTX 4090, L4, A40, A100...).
3. **HTTP Port**: Đảm bảo mở port **`7860`** trong cấu hình Pod (Expose HTTP Ports: `7860`).
4. **Kết nối**: Bấm **Connect** ➔ **Connect to Web Terminal** (hoặc SSH / VS Code Remote).

---

## 📥 2. Clone Repository & Kích Hoạt Conda

### 2.1. Clone mã nguồn
```bash
git clone https://github.com/KhoaLeDang2375/part-catseg.git
cd part-catseg

# Chuyển sang nhánh dev1 (nhánh chứa Gradio Demo & các bản sửa lỗi)
git fetch origin
git checkout -B dev1 origin/dev1
```

### 2.2. Kích hoạt Conda
Nếu gặp lỗi `bash: conda: command not found`, hãy nạp Conda vào môi trường:

```bash
# Nạp Conda (chọn đường dẫn phù hợp với container của bạn)
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null

# Nếu chưa có Conda trên máy, cài đặt Miniconda siêu nhanh:
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
# bash /tmp/miniconda.sh -b -p /root/miniconda3
# source /root/miniconda3/etc/profile.d/conda.sh
```

---

## 📦 3. Cài Đặt Môi Trường & Thư Viện

### 3.1. Tạo môi trường Conda
```bash
# Tạo môi trường Python 3.11 + PyTorch 2.2.2 + CUDA 12.1
conda env create -f environment.yml

# Kích hoạt môi trường
conda activate partcatseg

# Cập nhật các thư viện phụ trợ (bao gồm cloudpickle, matplotlib, scipy, pandas, pyyaml...)
pip install -r requirements.txt
```

### 3.2. Cài đặt Detectron2
Do Detectron2 cần biên dịch C++/CUDA extensions với PyTorch hiện tại, **bắt buộc cài sau khi đã kích hoạt conda env `partcatseg`** và dùng cờ `--no-build-isolation`:

```bash
# 1. Khóa chuẩn bị thư mục & clone detectron2 về /tmp (xóa thư mục cũ nếu có)
rm -rf /tmp/detectron2
git clone https://github.com/facebookresearch/detectron2.git /tmp/detectron2

# 2. Cài đặt chế độ editable không dùng build isolation
pip install --no-build-isolation --no-deps -e /tmp/detectron2

# 3. Kiểm tra cài đặt thành công
python -c "import detectron2; print('Detectron2 OK:', detectron2.__version__)"
```

---

## 🚀 4. Khởi Chạy Ứng Dụng Gradio

Chạy duy nhất lệnh sau để tự động tải weights `partcatseg_voc.pth` (~885 MB) và tạo **Link Public Gradio**:

```bash
bash start_demo.sh --share
```

Khi chạy xong, terminal sẽ xuất hiện đường link public:
👉 **`https://xxxxxxxxx.gradio.live`**

Bạn chỉ cần click vào link đó trên trình duyệt máy tính của bạn để sử dụng giao diện!

---

## 🎨 5. Tùy Chọn Tính Năng Trên Giao Diện

1. **Upload Ảnh**: Chọn ảnh nhân vật / đối tượng (Độ phân giải khuyến nghị: `512px` - `1080px`).
2. **Object Class**: Chọn lớp đối tượng chính (ví dụ: `person`, `car`, `cat`, `dog`...).
3. **Target Parts Filter (Tùy chọn)**:
   - **Để trống**: Phân đoạn **TẤT CẢ** các bộ phận (`head`, `torso`, `hand`, `leg`, `hair`...).
   - **Chọn cụ thể**: Ví dụ chọn `head` và `hand` ➔ Model sẽ **chỉ phân đoạn 2 bộ phận này**, ẩn toàn bộ phần còn lại.
4. **Settings (Confidence & Opacity)**:
   - `Confidence Threshold`: Ngưỡng độ tin cậy tối thiểu để hiển thị mask.
   - `Overlay Opacity`: Độ trong suốt của lớp màu đè lên ảnh.

---

## 🛠️ 6. Bảng Tra Cứu Sự Cố & Cách Khắc Phục (Troubleshooting)

| STT | Lỗi Thường Gặp | Nguyên Nhân | Cách Khắc Phục |
|---|---|---|---|
| 1 | `bash: conda: command not found` | Conda chưa được thêm vào PATH của shell | Run: `source /opt/conda/etc/profile.d/conda.sh` hoặc `source /root/miniconda3/etc/profile.d/conda.sh` |
| 2 | `ModuleNotFoundError: No module named 'torch'` khi cài Detectron2 | `pip` mặc định tạo môi trường cách ly không có `torch` | Thêm cờ `--no-build-isolation` khi cài: `pip install --no-build-isolation --no-deps -e /tmp/detectron2` |
| 3 | `Failed building wheel for detectron2` | `pip` thất bại khi đóng gói `.whl` | Dùng chế độ editable `-e /tmp/detectron2` và cài trước `pip install fvcore iopath pycocotools` |
| 4 | `ModuleNotFoundError: No module named 'detectron2'` khi chạy `app.py` | Python không tìm thấy thư mục `/tmp/detectron2` | Run: `export PYTHONPATH="/tmp/detectron2:$PYTHONPATH"` (Đã tự động hóa trong `start_demo.sh`) |
| 5 | `AttributeError: module 'PIL.Image' has no attribute 'LINEAR'` | Pillow 10+ đã xóa thuộc tính cũ `Image.LINEAR` | Đã fix trong code (`getattr(Image, 'LINEAR', Image.BILINEAR)`). Cập nhật bằng `git pull` |
| 6 | `ModuleNotFoundError: No module named 'cloudpickle'` / `'matplotlib'` | Thiếu thư viện phụ thuộc của Detectron2 / Visualizer | Run: `pip install -r requirements.txt` |
| 7 | Quên mở Port `7860` trên RunPod | Tạo Pod chưa cấu hình port | Thêm cờ `--share` vào lệnh: `bash start_demo.sh --share` để dùng link `.gradio.live` |

---

## 📂 7. Cấu Trúc Các File Chính

```
part-catseg/
├── app.py                  # Giao diện Gradio 4.x
├── inference_engine.py     # Wrapper nạp model & xử lý inference / render mask
├── app_config.py           # Config danh sách 20 object classes & part labels
├── start_demo.sh           # Script 1-lệnh tự nạp weights, check GPU & bật app
├── environment.yml         # File khởi tạo môi trường Conda (Python 3.11 + PyTorch 2.2.2)
├── requirements.txt        # Danh sách thư viện Python tương thích
└── RUNPOD_GUIDE.md         # Tài liệu hướng dẫn này
```
