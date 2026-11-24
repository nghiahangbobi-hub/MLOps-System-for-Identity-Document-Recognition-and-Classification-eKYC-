# Tái lập eKYC trên MIDV-2020 đã giải nén

## Giai đoạn 1 — Dữ liệu

Chạy từ thư mục gốc của repo trong môi trường Python 3.11 của dự án. Cài
dependency bằng `python -m pip install -r ops/release-requirements.txt` trong
virtualenv. Chọn thư mục đầu ra mới cho mỗi lần chạy.

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

Lần chạy này chọn 1.827 JPEG (2.006.018.038 byte) từ 261 giấy tờ và chín mẫu
layout. Dung lượng ảnh được chọn là một phần của gói nguồn khoảng 5 GB.

| Split | ID_Card | Passport |
|---|---:|---:|
| Train | 609 | 406 |
| Validation | 203 | 203 |
| Test | 203 | 203 |

## Giai đoạn 2 — Huấn luyện và đánh giá

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

So sánh MobileNetV3 Small tiền huấn luyện ImageNet trên validation. Mô hình
frozen đạt 74,38%. Mô hình fine-tuned đạt 80,05% ở epoch 8 và được khóa trước
khi đánh giá test. Số liệu test áp dụng cho hai layout giữ riêng thuộc MIDV-2020.

| Chỉ số test | Kết quả |
|---|---:|
| Accuracy | 98,28% (399/406) |
| Macro-F1 | 0,98276 |
| Recall ID_Card | 99,01% (201/203) |
| Recall Passport | 97,54% (198/203) |
| ECE sau hiệu chuẩn | 7,59% |
| ECE trước hiệu chuẩn | 2,09% |
| Tỷ lệ giữ lại theo ngưỡng phân loại | 94,83% (385/406) |
| Accuracy trong nhóm giữ lại | 98,70% (380/385) |

Confusion matrix theo thứ tự ID_Card, Passport, hàng là nhãn thật và cột
là dự đoán. `[[201, 2], [5, 198]]`. Nhánh chất lượng chưa có ảnh được duyệt,
nên tỷ lệ giữ lại của API đầy đủ là 0% và mọi dự đoán cần duyệt.

## Giai đoạn 3 — API, demo và vòng duyệt

Dùng bundle ở `"$run/selected/bundle"` theo [hướng dẫn triển khai](../azure/README.vi.md).
Demo cấu hình allowlist bằng một manifest nhỏ chứa các ảnh mock thuộc train,
giữ nguyên cột `image_id` và `image_path`. Điền đường dẫn dataset
`"$source_dir"` vào các lệnh snapshot và publish trong
[hướng dẫn vòng duyệt](hitl.vi.md). Mở project Label Studio cho lô mới và nhập
`review-batch/tasks.json` với thư mục media của lô. Người duyệt chọn lại loại
giấy tờ và chất lượng hoặc `Outside_Scope`, xuất JSON thô rồi phê duyệt theo
hướng dẫn trước khi xuất bản manifest mới và chạy huấn luyện thủ công.

| Kiểm tra | Trạng thái | Minh chứng cục bộ |
|---|---|---|
| API và demo với bundle mới | Đạt | `runs/midv-5gb/demo-result.json` |
| JPEG lỗi và thao tác Report lặp | Đạt | HTTP 400 và cùng mã sự kiện trong kiểm thử demo |
| Snapshot hàng đợi từ dự đoán thật | Đạt | `runs/midv-5gb/review-batch/` |
| Nhãn thật, phê duyệt và huấn luyện theo lineage | Bị chặn | Lô đang chờ người duyệt |
| Tải dữ liệu train/validation lên Azure Blob | Đạt | 1.423 đối tượng, 1.515.052.747 byte và SHA-256 của hai manifest đã được đối chiếu |
| Chạy thử huấn luyện trên Azure ML | Đạt | Job `cr-20261124070808-a6574e65` hoàn tất. MLflow ghi nhận 14 ảnh train, 14 ảnh validation và SHA-256 của manifest. |
| Suy luận Azure ML trên bundle đã khóa | Đạt | HTTP 200 trên một replica Standard_DS2_v2 ở Japan East. Ảnh mock được dự đoán là Passport và chuyển sang duyệt. |
| Độ trễ từ máy trạm đến Japan East | 99/100 lượt thành công | p50 880,00 ms, p95 914,38 ms trên các lượt thành công. Một lượt thành công mất 15,31 giây, một lượt bị timeout. |
| Xóa tài nguyên staging | Đạt | Azure xác nhận endpoint và deployment đã được xóa lúc 07:33:10 UTC. |
