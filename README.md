# MLOps eKYC

Phân loại ảnh giấy tờ mô phỏng MIDV-2020 thành `ID_Card` hoặc `Passport`.
Chạy từ thư mục gốc trên Linux với Python 3.11.

| Thư mục | Trách nhiệm |
|---|---|
| `data_pipeline/` | Kiểm tra ảnh, lấy mẫu và tạo manifest theo tập chia. |
| `labeling/` | Cấu hình Label Studio, nhập nhãn và xuất bản manifest đã duyệt. |
| `modeling/` | Huấn luyện, đánh giá, đóng gói và phục vụ dự đoán. |
| `demo/` | Giao diện thử dự đoán và hàng đợi duyệt. |
| `azure/` | Cấu hình chạy trên Azure ML và các phép đo vận hành. |
| `checks/` | Kiểm thử xuyên suốt giữa các thành phần. |

Các kiểm thử riêng nằm cạnh mã của từng thành phần. Đặt dữ liệu đầu vào
trong `Data/` và kết quả cục bộ trong `runs/`.

```bash
python3.11 -m venv .venv/release
source .venv/release/bin/activate
python -m pip install -r requirements.txt
```

## 1. Chuẩn bị dữ liệu

Đặt `source_dir` theo vị trí đã giải nén. Chọn `run` mới cho mỗi lần chạy.

```bash
source_dir=Data/MIDV2020_sample_5GB
run=runs/midv-5gb
python -m data_pipeline.prepared_midv --source "$source_dir" --output "$run/inputs"
```

Pipeline đọc tại chỗ năm thư mục `clips`, `photo`, `scan_rotated`,
`scan_upright`, `templates`. Mỗi giấy tờ có ba frame video cách nhau ít nhất
10 frame và một ảnh từ mỗi nguồn tĩnh. Split theo template cố định, loại
`rus_internalpassport` và kiểm tra JPEG trùng byte, ảnh lỗi và các nguồn
thiếu giấy tờ. File `provenance.json` ghi kích thước và SHA-256 của các
manifest. Nhãn chất lượng để trống trước khi duyệt.

Với đầu vào `clips.tar`, tạo một bộ nhỏ để chạy thử.

```bash
python -m data_pipeline.midv2020 --source /path/to/clips.tar \
  --output "$run/clips" --max-clips-per-template 2
python -m modeling.train --dataset-root "$run/clips" \
  --manifest "$run/clips/manifest.csv" --output-dir "$run/clips-model" --device cpu
```

Bundle của bộ chạy thử nằm ở `"$run/clips-model"` và dùng được ở bước 3.

## 2. Huấn luyện và đánh giá

Với bộ đã giải nén từ bước 1, so sánh hai cấu hình trên validation, khóa
lựa chọn rồi đánh giá test. Lần đầu cần tải trọng số ImageNet.

```bash
holdout_hash=$(sha256sum "$run/inputs/held-out.csv" | cut -d ' ' -f 1)
python -m modeling.quality_compare compare --dataset-root "$source_dir" \
  --development "$run/inputs/development.csv" --held-out-sha256 "$holdout_hash" \
  --output "$run/comparison-cpu" --device cpu --workers 0 --epochs 8
python -m modeling.quality_compare seal --comparison "$run/comparison-cpu" \
  --output "$run/selected"
python -m modeling.quality_compare evaluate --dataset-root "$source_dir" \
  --base-manifest "$run/inputs/manifest.csv" --held-out "$run/inputs/held-out.csv" \
  --output "$run/selected" --device cpu --workers 0
```

Bundle nằm ở `"$run/selected/bundle"`. Kết quả đánh giá lưu accuracy,
macro-F1, recall, confusion matrix và số mẫu. Huấn luyện chất lượng ảnh
cần [nhãn Label Studio](labeling/README.md).

## 3. Chạy API và demo

```bash
azmlinfsrv --entry_script modeling/score.py \
  --model_dir "$(realpath "$run/selected/bundle")" --port 5001
```

Trong terminal thứ hai, ở thư mục gốc của repo, chạy các lệnh sau.

```bash
source .venv/release/bin/activate
export EKYC_SCORING_URL=http://127.0.0.1:5001/score
python -m demo.server --port 8080
```

Mở [demo](http://127.0.0.1:8080) và gửi JPEG hoặc PNG mô phỏng tối đa 8 MiB.
Xem [hàng đợi duyệt và huấn luyện lại](labeling/README.md) hoặc
[triển khai Azure ML](azure/README.md).

## Kiểm thử cục bộ

```bash
python -m unittest data_pipeline.test_release data_pipeline.test_sample_midv \
  data_pipeline.test_prepared_midv labeling.test_release labeling.test_hitl \
  modeling.test_release modeling.test_quality_compare azure.test_release \
  demo.test_release demo.test_review_queue
```

Đặt `EKYC_RELEASE_DATASET` trỏ đến fixture do `data_pipeline.midv2020` tạo
và đặt `EKYC_CHECK_AZURE_SCHEMAS=1` để chạy thêm kiểm tra tích hợp.

```bash
python -m checks.release_smoke --source /path/to/clips.tar --output "$run/check-release"
python -m checks.hitl_smoke --dataset-root "$run/clips" --output "$run/check-review" --runtime podman
```

Kiểm thử hàng đợi dùng Label Studio chạy trong Podman hoặc Docker
(`--runtime docker`).
