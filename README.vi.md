# Hệ thống MLOps eKYC

[English](README.md) · **Tiếng Việt**

Hệ thống phân loại khung hình MIDV-2020 thành `ID_Card` (căn cước) hoặc `Passport` (hộ chiếu), đánh giá chất lượng ảnh khi có nhãn đã duyệt và chuyển các ảnh chất lượng kém hoặc có dự đoán thiếu tin cậy sang hàng đợi duyệt.

## Trình bày pipeline trong ba bước

| Bước | Chạy hoặc mở | Kết quả cần cho thấy |
|---|---|---|
| 1. Chuẩn bị dữ liệu | `python -m data_pipeline.prepared_midv` với thư mục MIDV-2020 đã giải nén | Manifest chia train, validation và test theo mẫu giấy tờ. |
| 2. Huấn luyện | `python -m modeling.quality_compare` với lần lượt `compare`, `seal` và `evaluate` | So sánh hai cấu hình MobileNetV3 trên validation, chọn bundle rồi đo trên test. |
| 3. Suy luận và duyệt | Chạy scoring server, mở `python -m demo.server` và tải một ảnh MIDV-2020 | Hiển thị dự đoán và đưa ảnh cần xem lại vào hàng đợi Label Studio. |

[Lệnh đầy đủ cho bộ dữ liệu đã giải nén](docs/midv-5gb.md) · [Duyệt ảnh và huấn luyện lại](docs/hitl.vi.md) · [Triển khai Azure](azure/README.vi.md)

## Cấu trúc dự án

| Thư mục | Trách nhiệm |
|---|---|
| `data_pipeline/` | Đọc MIDV ở chế độ chỉ đọc, chia theo mẫu giấy tờ, tạo manifest và kiểm tra dữ liệu. |
| `labeling/` | Giao diện Label Studio, nhập bản duyệt, phê duyệt và truy vết phiên bản dữ liệu. |
| `modeling/` | Huấn luyện, chọn trên validation, tính chỉ số và suy luận. |
| `azure/` | Hướng dẫn Blob và Azure ML, cấu hình job và endpoint, benchmark và giám sát. |
| `demo/` | Tải ảnh qua trình duyệt, proxy suy luận và hàng đợi duyệt. |
| `ops/` | Công cụ tái lập và xác minh. |
| `docs/` | Tài liệu quy trình. Lưu minh chứng cục bộ ở ngoài bản phát hành source. |

Khi tái lập, dùng môi trường Python mới và thư mục riêng bên ngoài repo để
lưu dataset, mô hình và log chạy. Xem
[hướng dẫn đánh giá mô hình](modeling/QUALITY.vi.md) để chạy thí nghiệm đầy đủ.

## Chạy pipeline trên máy

Chạy từ thư mục gốc của dự án. Tạo môi trường Python riêng và cài các phiên bản PyTorch, torchvision tương thích với CPU. Có thể dùng Python 3.11.

```bash
python -m pip install -r data_pipeline/requirements.txt
run_dir=$(mktemp -d)
python -m data_pipeline.midv2020 --output "$run_dir/dataset" \
  --max-clips-per-template 2 --frames-per-clip 3 --min-frame-gap 10
python -m labeling.label_studio make-tasks \
  --manifest "$run_dir/dataset/manifest.csv" \
  --dataset-root "$run_dir/dataset" --media-root "$run_dir" \
  --output "$run_dir/tasks.json"
```

Dùng `--source /duong/dan/clips.tar` nếu tệp nằm ở vị trí khác. Bỏ giới hạn số clip khi chuẩn bị toàn bộ tập. Giữ mỗi nhóm tài liệu trong một tập train, validation hoặc test và dùng cùng manifest, seed khi so sánh các cấu hình.

Để huấn luyện đầu ra chất lượng, chạy Label Studio Community trên máy, bật phục vụ tệp cục bộ và trỏ thư mục gốc của Label Studio đến cùng thư mục ảnh dùng để tạo tác vụ. Dùng [quality.xml](labeling/quality.xml) làm giao diện, đăng ký thư mục ảnh làm nguồn lưu trữ và nhập `tasks.json`. Gán `Clear` (rõ), `Blur` (mờ), `Glare` (lóa), `Occluded` (che khuất) hoặc `Outside_Scope` (ngoài phạm vi). Xuất JSON thô và nhập nhãn vào manifest như sau. Giữ JSON và manifest đã duyệt làm minh chứng. [Hướng dẫn Label Studio](labeling/README.vi.md)

```bash
python -m labeling.label_studio apply-reviews \
  --manifest "$run_dir/dataset/manifest.csv" \
  --review-export "$run_dir/reviews.json" \
  --output "$run_dir/dataset/reviewed.csv"
```

Lệnh dưới đây chạy thử phần huấn luyện phân loại giấy tờ. Dùng `reviewed.csv` thay cho manifest gốc khi huấn luyện đầu ra chất lượng.

```bash
python -m pip install -r modeling/requirements.txt
export MLFLOW_ALLOW_FILE_STORE=true
python -m modeling.train \
  --dataset-root "$run_dir/dataset" \
  --manifest "$run_dir/dataset/manifest.csv" \
  --output-dir "$run_dir/model-baseline" \
  --device cpu --epochs 1 --batch-size 4 \
  --tracking-uri "file://$run_dir/mlruns" --experiment ekyc-smoke
```

Gói mô hình chứa trọng số PyTorch, cấu hình tiền xử lý và thông tin lớp. So sánh các cấu hình trên tập validation và đánh giá cấu hình được chọn trên tập test giữ riêng. Đánh giá chất lượng ảnh bằng mẫu đã duyệt và báo cáo số mẫu cùng chỉ số.

API nhận JSON có trường `image_base64` và trả về `label`, `status`, `confidence`, `needs_review`. Ảnh kém chất lượng, ảnh chưa được mô hình đánh giá chất lượng hoặc dự đoán thiếu tin cậy có thể cần duyệt thủ công. Gói chỉ phân loại giấy tờ trả về `status: null`.

Nếu Label Studio chạy ở cổng 8080, cấu hình `EKYC_SCORING_URL` theo hướng dẫn Azure rồi chạy trang demo bằng `python -m demo.server --port 8081`. [Hướng dẫn Azure](azure/README.vi.md)

## Duyệt thủ công (Human-in-the-loop)

Trang demo đưa dự đoán cần xem lại và báo cáo dự đoán sai cho ảnh mô phỏng vào hàng đợi. Người gán nhãn xử lý ảnh trong Label Studio. Sau khi nhãn được duyệt, ETL tạo phiên bản dữ liệu có thông tin truy vết. Người vận hành chủ động gửi job huấn luyện. [Hướng dẫn Human-in-the-loop](docs/hitl.vi.md)

## Azure và kiểm thử

Tạo workspace, tải ảnh lên Blob Storage và đăng ký data asset. Sau đó dùng runbook Azure để triển khai mô hình, cấu hình trang demo và chạy benchmark. Lưu dữ liệu, gói mô hình, thông tin xác thực và kết quả đánh giá trong nơi lưu trữ riêng. [Thiết lập Blob Storage](azure/blob-storage.md) · [Triển khai Azure](azure/README.vi.md)

Sau khi cài phụ thuộc trong môi trường Python riêng, chạy kiểm thử hồi quy theo từng nhóm:

```bash
python -m unittest data_pipeline.test_release labeling.test_release labeling.test_hitl
python -m unittest modeling.test_release modeling.test_quality_compare
python -m unittest azure.test_release demo.test_release demo.test_review_queue
```
