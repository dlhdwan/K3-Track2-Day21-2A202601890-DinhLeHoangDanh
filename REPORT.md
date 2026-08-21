# BÁO CÁO KẾT QUẢ BÀI LAB MLOPS DAY 21
**Môn học / Bài lab**: Continuous Integration, Continuous Training & Continuous Deployment  
**Họ và tên**: Đinh Lê Hoàng Danh  
**Mã số sinh viên**: 2A202601890  
**GCP Project ID**: `mlops-lab-day21`  
**GitHub Repository**: [https://github.com/dlhdwan/K3-Track2-Day21-2A202601890-DinhLeHoangDanh](https://github.com/dlhdwan/K3-Track2-Day21-2A202601890-DinhLeHoangDanh)  

---

## 1. Bộ Siêu Tham Số Đã Chọn Và Lý Do

Dựa trên các lần chạy thí nghiệm và theo dõi chỉ số trên **MLflow UI** (`sqlite:///mlflow.db`), thuật toán **`RandomForestClassifier`** kết hợp với **`StandardScaler`** đã được lựa chọn. Bộ siêu tham số tối ưu trong `params.yaml`:

- `model_type: "random_forest"`
- `n_estimators: 500`: Tăng số lượng cây quyết định lên 500 giúp mô hình giảm thiểu hiện tượng variance và gia tăng độ ổn định trên tập kiểm thử.
- `max_depth: 25`: Cho phép cây phân nhánh đủ sâu để nắm bắt được các mối quan hệ phi tuyến phức tạp giữa 12 đặc trưng hóa lý của rượu.
- `min_samples_split: 2` & `max_features: 0.4`: Hạn chế overfitting trên tập train nhưng vẫn giữ được độ phong phú đặc trưng.
- `class_weight: "balanced"`: Xử lý triệt để tình trạng mất cân bằng phân bố giữa các nhãn chất lượng rượu (thấp: 0, trung bình: 1, cao: 2).

**Lý do chọn**: Trên MLflow UI, cấu hình này cho kết quả vượt trội so với các tham số mặc định (Accuracy mặc định < 0.65), đạt **Accuracy: 0.7040** và **F1-Score: 0.7033**, chính thức vượt qua ngưỡng đánh giá khắt khe Eval Gate ($\ge 0.70$).

---

## 2. So Sánh Hiệu Suất Giữa 2 Lần Huấn Luyện (Phase 1 vs Phase 2)

| Lần chạy | Tập dữ liệu | Số lượng mẫu | Accuracy | F1-Score (Weighted) | Đánh giá |
|---|---|---|---|---|---|
| **Lần 1 (Bước 2)** | `train_phase1.csv` ban đầu | 2,998 mẫu | **0.7040** | **0.7033** | Vượt qua Eval Gate ($\ge 0.70$), đủ điều kiện Deploy lần đầu. |
| **Lần 2 (Bước 3)** | Bổ sung `train_phase2.csv` | 5,996 mẫu | **0.7500** | **0.7496** | Hiệu suất tăng **+4.60%** nhờ bổ sung dữ liệu mới. |

**Nhận xét**: Khi tập dữ liệu huấn luyện được bổ sung lên 5,996 mẫu qua script `add_new_data.py`, khả năng tổng quát hóa của mô hình cải thiện rõ rệt. Chỉ số F1-Score gia tăng đồng đều chứng minh mô hình phân loại chính xác hơn trên cả 3 nhãn phân loại rượu.

---

## 3. Khó Khăn Gặp Phải Và Cách Giải Quyết

1. **Phân quyền và bảo mật Bucket trên Google Cloud Platform (GCP)**:
   - *Khó khăn*: Ban đầu gán quyền `roles/storage.admin` có nguy cơ cho phép xóa toàn bộ bucket, vi phạm nguyên tắc an toàn dữ liệu.
   - *Cách giải quyết*: Tạo Service Account `mlops-lab-sa@mlops-lab-day21.iam.gserviceaccount.com` và chỉ cấp quyền `roles/storage.objectAdmin` trên đúng bucket `gs://mlops-lab-day21-2a202601890-bucket` theo nguyên tắc quyền tối thiểu (Minimum Privilege).

2. **Lỗi xác thực DVC trong GitHub Actions Runner (`401 Invalid Credentials`)**:
   - *Khó khăn*: File `sa-key.json` bị chặn bởi `.gitignore` nên không có sẵn trên môi trường ảo GitHub Actions runner, khiến lệnh `dvc pull` bị từ chối truy cập.
   - *Cách giải quyết*: Cập nhật bước `Authenticate to Cloud Storage` trong [.github/workflows/mlops.yml](file:///d:/K3-Track2-Day21-2A202601890-DinhLeHoangDanh/.github/workflows/mlops.yml) để tự động khởi tạo file `sa-key.json` từ GitHub Secret `GCP_SA_KEY` ngay tại thư mục làm việc root trước khi thực thi `dvc pull`.

3. **Khởi động FastAPI Model Serving trên Compute Engine VM**:
   - *Khó khăn*: Service systemd `mlops-serve` bị thoát đột ngột (`Connection refused` tại port 8000) khi tiến trình khởi tạo chưa kịp tải xong mô hình từ GCS.
   - *Cách giải quyết*: Bổ sung khối xử lý ngoại lệ `try-except` an toàn trong [src/serve.py](file:///d:/K3-Track2-Day21-2A202601890-DinhLeHoangDanh/src/serve.py) và tạo cơ chế fallback model cục bộ, giúp dịch vụ FastAPI luôn liên tục lắng nghe tại cổng `8000`.

---

## 4. Báo Cáo Thực Hiện Các Thách Thức Nâng Cao (Bonus Challenges - +20 Điểm)

- **Bonus 1 (Tracking MLflow Từ Xa)**: Đã sẵn sàng cấu hình URL và credentials cho MLflow remote tracking server.
- **Bonus 2 (Thử nghiệm nhiều thuật toán)**: Đã mở rộng [src/train.py](file:///d:/K3-Track2-Day21-2A202601890-DinhLeHoangDanh/src/train.py) và [params.yaml](file:///d:/K3-Track2-Day21-2A202601890-DinhLeHoangDanh/params.yaml) hỗ trợ tham số `model_type` chọn giữa `random_forest`, `gradient_boosting`, và `logistic_regression`.
- **Bonus 3 (Báo cáo hiệu suất tự động)**: Tự động tính toán Confusion Matrix, Precision, Recall và xuất file [outputs/report.txt](file:///d:/K3-Track2-Day21-2A202601890-DinhLeHoangDanh/outputs/report.txt), đồng thời upload làm Artifact trên GitHub Actions.
- **Bonus 4 (Cơ chế an toàn Rollback)**: Tải file metrics lịch sử từ GCS và so sánh accuracy. Nếu mô hình mới có accuracy thấp hơn phiên bản đang chạy, pipeline tự động chặn Deploy để bảo vệ hệ thống.
- **Bonus 5 (Cảnh báo Data Drift)**: Tự động phân tích tỷ lệ phân phối nhãn của tập train (Lớp 0: 36.86%, Lớp 1: 43.51%, Lớp 2: 19.63%), đưa ra cảnh báo nếu có lớp < 10% và ghi vào `outputs/metrics.json`.
