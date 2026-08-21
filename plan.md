# 🚀 Kế Hoạch Thực Hiện MLOps Lab: Từ Thực Nghiệm Đến Triển Khai Liên Tục

Bài lab thuộc khóa học **AIInAction - VinUni (Day 21 - CI/CD cho AI Systems)**.
Mục tiêu: Đạt **100/100 điểm** (80 điểm chính + 20 điểm Bonus nâng cao).

---

## 📊 Bảng Điểm Mục Tiêu

| Hạng mục | Tiêu chí | Điểm tối đa |
|---|---|:---:|
| **Bước 1** | MLflow tracking cục bộ (>= 3 runs, log đủ metrics/params/model) | 24 |
| **Bước 2** | Quản lý dữ liệu DVC, CI/CD GitHub Actions, Eval gate (>= 0.70), FastAPI Serving | 44 |
| **Bước 3** | Continuous Training tự động khi bổ sung dữ liệu mới | 12 |
| **Bonus 1** | Tracking MLflow từ xa trên DagsHub | 4 |
| **Bonus 2** | Thí nghiệm hỗ trợ nhiều thuật toán ML (`model_type`) | 4 |
| **Bonus 3** | Báo cáo hiệu suất tự động (`report.txt` + Artifact) | 4 |
| **Bonus 4** | Model Fallback (Chỉ deploy khi accuracy mới >= accuracy cũ) | 4 |
| **Bonus 5** | Cảnh báo lệch lạc dữ liệu (Data Drift / Class Imbalance Warning) | 4 |
| **TỔNG ĐIỂM** | | **100 / 100** |

---

## 📌 DANH SÁCH CÔNG VIỆC CHI TIẾT

### 🔵 BƯỚC 1: Thực Nghiệm Cục Bộ & MLflow Tracking (24 điểm)
- [ ] **1.1 Cấu hình môi trường & Dữ liệu**
  - [ ] Chạy `python generate_data.py` để tạo `train_phase1.csv`, `eval.csv`, `train_phase2.csv` trong `data/`.
  - [ ] Khai báo biến môi trường: `MLFLOW_TRACKING_URI=sqlite:///mlflow.db` và `MLFLOW_ARTIFACT_ROOT=./mlartifacts`.
- [ ] **1.2 Hoàn thiện [src/train.py](file:///d:/K3-Track2-Day21-2A202601890-DinhLeHoangDanh/src/train.py)**
  - [ ] Đọc dữ liệu `train_phase1.csv` & `eval.csv`.
  - [ ] Tách đặc trưng ($X$) và nhãn ($y$).
  - [ ] Log siêu tham số từ [params.yaml](file:///d:/K3-Track2-Day21-2A202601890-DinhLeHoangDanh/params.yaml) vào MLflow (`mlflow.log_params`).
  - [ ] Huấn luyện mô hình, tính `accuracy` và `f1_score` (weighted), log metrics vào MLflow (`mlflow.log_metric`).
  - [ ] Lưu mô hình vào MLflow artifact (`mlflow.sklearn.log_model`).
  - [ ] Xuất kết quả cục bộ: `outputs/metrics.json` và `models/model.pkl`.
- [ ] **1.3 Chạy thử nghiệm & chọn Siêu tham số tối ưu**
  - [ ] Chạy `python src/train.py` **ít nhất 3 lần** với các bộ siêu tham số khác nhau trong `params.yaml`.
  - [ ] Mở MLflow UI (`mlflow ui --backend-store-uri sqlite:///mlflow.db`), so sánh kết quả.
  - [ ] Cập nhật bộ siêu tham số tốt nhất vào `params.yaml`.
  - [ ] Chụp màn hình MLflow UI (cần nộp bài).

---

### 🟢 BƯỚC 2: Quản Lý Dữ Liệu DVC & Pipeline CI/CD Tự Động (44 điểm)
- [ ] **2.1 Quản lý phiên bản dữ liệu với DVC**
  - [ ] Tạo Bucket trên Cloud Storage (GCS / S3 / Azure) & Service Account / Access Key.
  - [ ] Khởi tạo DVC: `dvc init`.
  - [ ] Cấu hình DVC remote: `dvc remote add -d myremote gs://<BUCKET>/dvc` và cài `credentialpath`.
  - [ ] Track dữ liệu: `dvc add data/train_phase1.csv data/eval.csv data/train_phase2.csv`.
  - [ ] Đẩy dữ liệu lên Cloud: `dvc push`.
  - [ ] Commit file `.dvc` vào Git.
- [ ] **2.2 Hoàn thiện Unit Tests trong [tests/test_train.py](file:///d:/K3-Track2-Day21-2A202601890-DinhLeHoangDanh/tests/test_train.py)**
  - [ ] Viết hàm tạo dữ liệu giả lập `_make_temp_data()`.
  - [ ] Hoàn thiện 3 test functions: `test_train_returns_float`, `test_metrics_file_created`, `test_model_file_created`.
  - [ ] Chạy `pytest tests/ -v` xác nhận 3/3 tests pass.
- [ ] **2.3 Hoàn thiện REST API Serving trong [src/serve.py](file:///d:/K3-Track2-Day21-2A202601890-DinhLeHoangDanh/src/serve.py)**
  - [ ] Tải `model.pkl` từ Cloud Storage khi server khởi chạy (`download_model`).
  - [ ] Đỉnh nghĩa endpoint `GET /health` trả về `{"status": "ok"}`.
  - [ ] Định nghĩa endpoint `POST /predict` kiểm tra 12 đặc trưng và trả về kết quả dự đoán (0 -> "thap", 1 -> "trung_binh", 2 -> "cao").
- [ ] **2.4 Thiết lập Cloud VM & GitHub Secrets**
  - [ ] Tạo Compute VM trên Cloud, cài đặt Python & dependencies.
  - [ ] Cấu hình Systemd Service `mlops-serve.service`.
  - [ ] Tạo SSH Key và nạp 5 Secrets vào GitHub Repo (`CLOUD_CREDENTIALS`, `CLOUD_BUCKET`, `VM_HOST`, `VM_USER`, `VM_SSH_KEY`).
- [ ] **2.5 Tạo Workflow GitHub Actions (`.github/workflows/mlops.yml`)**
  - [ ] **Job 1 (Test)**: Chạy `pytest tests/ -v`.
  - [ ] **Job 2 (Train)**: Xây dựng môi trường, `dvc pull`, chạy `src/train.py`, upload model lên Cloud Storage, lưu `metrics.json` artifact.
  - [ ] **Job 3 (Eval)**: Kiểm tra `accuracy >= 0.70` (Eval Gate). Nếu không đạt, dừng pipeline.
  - [ ] **Job 4 (Deploy)**: SSH vào VM, restart service `mlops-serve`, kiểm tra `curl /health`.
  - [ ] Push code lên GitHub, xác nhận 4 jobs đều xanh.
  - [ ] Kiểm tra API thực tế bằng `curl http://VM_IP:8000/predict`.

---

### 🟡 BƯỚC 3: Continuous Training - Huấn Luyện Liên Tục (12 điểm)
- [ ] **3.1 Bổ sung dữ liệu mới**
  - [ ] Chạy `python add_new_data.py` (tăng dữ liệu huấn luyện từ 2,998 lên 5,996 mẫu).
- [ ] **3.2 Cập nhật DVC & Trigger Pipeline**
  - [ ] Chạy `dvc add data/train_phase1.csv`.
  - [ ] Đẩy dữ liệu mới lên Cloud trước: `dvc push`.
  - [ ] Commit file `.dvc`: `git add data/train_phase1.csv.dvc && git commit -m "data: bổ sung dữ liệu phase 2"`.
  - [ ] Push Git: `git push origin main`.
- [ ] **3.3 Xác nhận kết quả**
  - [ ] Kiểm tra GitHub Actions tự động chạy do commit dữ liệu `.dvc`.
  - [ ] Xác nhận mô hình mới được train & deploy tự động.
  - [ ] Điền bảng so sánh `accuracy` & `f1_score` giữa Bước 2 và Bước 3.

---

### 🔥 THÁCH THỨC NÂNG CAO (BONUS - 20 điểm)

- [ ] **Bonus 1: Tracking MLflow Từ Xa Với DagsHub (4 điểm)**
  - [ ] Đăng ký DagsHub, liên kết repo GitHub.
  - [ ] Thêm credentials DagsHub vào GitHub Secrets (`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`).
  - [ ] Cập nhật `mlops.yml` đẩy log trực tiếp lên DagsHub Server.
- [ ] **Bonus 2: Thí Nghiệm Nhiều Thuật Toán (4 điểm)**
  - [ ] Thêm `model_type` vào `params.yaml` (`random_forest`, `gradient_boosting`, `logistic_regression`).
  - [ ] Nâng cấp [src/train.py](file:///d:/K3-Track2-Day21-2A202601890-DinhLeHoangDanh/src/train.py) hỗ trợ khởi tạo mô hình linh hoạt theo `model_type`.
  - [ ] So sánh hiệu năng giữa các thuật toán trên MLflow UI.
- [ ] **Bonus 3: Báo Cáo Hiệu Suất Tự Động (4 điểm)**
  - [ ] Tính Confusion Matrix, `precision`, `recall` cho từng lớp trong `train.py`.
  - [ ] Xuất file `outputs/report.txt`.
  - [ ] Đính kèm `report.txt` vào GitHub Actions Artifact trong `mlops.yml`.
- [ ] **Bonus 4: Model Fallback - Chống Giam Hiệu Suất Mô Hình (4 điểm)**
  - [ ] Tải `metrics.json` của phiên bản mô hình cũ từ Cloud Storage trong Job Eval.
  - [ ] So sánh `accuracy_moi >= accuracy_cu`. Hủy Deploy nếu mô hình mới kém hơn mô hình cũ.
- [ ] **Bonus 5: Cảnh Báo Lệch Lạc Dữ Liệu - Data Drift Warning (4 điểm)**
  - [ ] Kiểm tra tỷ lệ phân phối nhãn (lớp 0, 1, 2) trước khi train trong `train.py`.
  - [ ] Cảnh báo `WARNING` nếu có lớp $< 10\%$ số lượng mẫu.
  - [ ] Ghi tỷ lệ phân phối nhãn vào `outputs/metrics.json`.

---

## 📑 BẰNG CHỨNG NỘP BÀI
- [ ] URL GitHub Repository công khai.
- [ ] Ảnh chụp màn hình MLflow UI (>= 3 thí nghiệm).
- [ ] Ảnh chụp màn hình GitHub Actions Tab (4 jobs màu xanh ở Bước 2 & 3).
- [ ] Ảnh chụp màn hình lệnh `curl http://VM_IP:8000/health` và `/predict`.
- [ ] Ảnh chụp màn hình Cloud Storage Console (file `dvc/` và `models/latest/model.pkl`).
- [ ] File báo cáo tóm tắt (dưới 1 trang A4).
