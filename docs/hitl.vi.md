# Duyệt thủ công (Human-in-the-loop)

[English](hitl.md) · **Tiếng Việt**

Đặt `work` thành thư mục làm việc riêng và `dataset` thành thư mục chứa ảnh MIDV-2020 mô phỏng và manifest đã chuẩn bị. Hàng đợi chỉ lưu JPEG hoặc PNG có nội dung khớp chính xác với manifest. Lưu dữ liệu, bản duyệt, hồ sơ phê duyệt và gói mô hình trong thư mục riêng. Giữ nguyên các mẫu giấy tờ thuộc tập validation và test.

## Đưa dự đoán vào hàng đợi

Khởi động dịch vụ suy luận cục bộ và cấu hình URL cho demo. [Lệnh khởi động dịch vụ](../azure/README.vi.md#thu-thập-minh-chứng)

```bash
export EKYC_SCORING_URL=http://127.0.0.1:5001/score
python -m demo.server --port 8081 \
  --review-queue "$work/queue" \
  --review-manifest "$dataset/manifest.csv" \
  --review-dataset-root "$dataset"
```

Mở [demo cục bộ](http://127.0.0.1:8081), chọn ảnh mô phỏng trong manifest và gửi dự đoán. Khi API trả `needs_review=true`, demo lưu ảnh và dự đoán vào hàng đợi. Nút **Report incorrect prediction** thêm lý do báo sai. Các lần gửi cùng ảnh dùng chung sự kiện theo SHA-256. Có thể thử lại khi lỗi ghi hàng đợi. Sau khi khởi động lại demo, gửi dự đoán mới để lấy mã báo sai hiện hành. Sự kiện đã lưu vẫn còn.

## Chuyển lô ảnh sang Label Studio

```bash
python -m labeling.hitl batch --queue-root "$work/queue" \
  --base-manifest "$dataset/manifest.csv" --dataset-root "$dataset" \
  --output "$work/batch"
```

Tạo dự án Label Studio Community với [hitl.xml](../labeling/hitl.xml). Đặt thư mục phục vụ ảnh là thư mục lô hoặc gắn chỉ đọc vào container. Thêm thư mục con `images` làm nguồn Local Files ở chế độ Tasks và lưu nguồn. Nhập `batch/tasks.json`. Người duyệt xác nhận hoặc sửa loại giấy tờ và chất lượng. Chọn `Outside_Scope` khi cần loại ảnh. Hoàn thành mọi tác vụ trong lô rồi xuất JSON thô.

Chỉ sự kiện train đã ghi hoàn tất được đưa vào lô. Sự kiện validation hoặc test vẫn có thể xuất hiện trong lịch sử hàng đợi. Các dòng validation và test giữ nguyên khi tạo phiên bản mới.

## Phê duyệt và tạo phiên bản

Đặt `reviewer_id` thành định danh người phê duyệt. Giữ nguyên nội dung tệp JSON đã xuất. Gán nhãn và phê duyệt là hai thao tác riêng.

```bash
python -m labeling.hitl approve --batch "$work/batch" \
  --review-export "$work/reviews.json" --reviewer "$reviewer_id" \
  --output "$work/approval.json"
python -m labeling.hitl publish --batch "$work/batch" \
  --review-export "$work/reviews.json" --approval "$work/approval.json" \
  --base-manifest "$dataset/manifest.csv" --dataset-root "$dataset" \
  --versions-root "$work/versions"
```

Thư mục mang mã phiên bản vừa in chứa `manifest.csv` và `lineage.json`. Hồ sơ phê duyệt ghi người duyệt và SHA-256 của bản xuất. Lineage ghi nguồn, kết quả, nhãn trước và sau cùng các trường hợp bị loại. Chạy lại với cùng đầu vào đã duyệt sẽ dùng lại phiên bản đó.

## Gửi job huấn luyện thủ công

Kiểm tra manifest và lineage, rồi dùng mã phiên bản `version_id` để chạy huấn luyện. Các đường dẫn ảnh vẫn tính từ thư mục dataset ban đầu.

```bash
python -m modeling.train --dataset-root "$dataset" \
  --manifest "$work/versions/$version_id/manifest.csv" \
  --output-dir "$work/manual-model" --device cpu --experiment ekyc-reviewed
```

Lưu ID của lần huấn luyện, mã phiên bản, hash manifest và số mẫu chất lượng ảnh. Đánh giá mô hình chất lượng cần nhãn người duyệt ở tập đánh giá. Nghiệm thu cần người thật duyệt lô ảnh và người vận hành chạy kiểm tra Azure.

## Tái lập kiểm thử kỹ thuật

Dùng môi trường kiểm thử và thư mục làm việc riêng mới. Đặt `source_path` trỏ tới `clips.tar`. Bài kiểm thử dùng dự án Label Studio tạm, một ca sửa nhãn và một ca loại ảnh với dữ liệu kiểm thử.

```bash
set -euo pipefail
py=.venv/release/bin/python
work=$(mktemp -d)
"$py" -m data_pipeline.midv2020 --source "$source_path" --output "$work/data" \
  --max-clips-per-template 2
"$py" -u -m ops.hitl_smoke --dataset-root "$work/data" --output "$work/smoke" \
  2>&1 | tee "$work/hitl.log"
```
