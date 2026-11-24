# Duyệt chất lượng ảnh MIDV-2020

[English](README.md) · **Tiếng Việt**

Chạy từ thư mục gốc. Lưu dataset, tác vụ và bản xuất trong thư mục riêng ngoài repo. Cài Pillow bằng `python -m pip install -r data_pipeline/requirements.txt`.

```bash
run=$(mktemp -d /tmp/ekyc-data.XXXXXX)
python -m data_pipeline.midv2020 --output "$run/dataset" \
  --max-clips-per-template 2 --frames-per-clip 3 --min-frame-gap 10
python -m labeling.label_studio make-tasks \
  --manifest "$run/dataset/manifest.csv" --dataset-root "$run/dataset" \
  --media-root "$run/dataset" --output "$run/tasks.json"
```

Dùng `--source /path/to/clips.tar` để đổi vị trí tệp. Bỏ giới hạn clip để xử lý toàn bộ phần hợp lệ. `template_id` là mẫu giấy tờ, ví dụ `est_id`. `document_id` là giấy tờ mô phỏng có số thứ tự và clip tương ứng. Mỗi mẫu chỉ thuộc một tập, mỗi tập có đủ hai lớp. Khoảng cách giữa ảnh được tính bằng số thứ tự frame gốc. Nhãn chất lượng để trống trước khi duyệt.

Trước khi chạy Label Studio, bật `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` và đặt `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` thành thư mục ảnh. Nếu chạy container, gắn dataset chỉ đọc và dùng đường dẫn bên trong container. Tạo dự án với giao diện `labeling/quality.xml`. Thêm thư mục `images` vào Local Files, chọn chế độ Tasks, lưu và bỏ qua đồng bộ. Nhập `tasks.json` trong Data Manager. [Label Studio Local Files](https://labelstud.io/guide/storage_local)

Chọn lỗi ảnh ảnh hưởng khả năng đọc nhiều nhất. Chọn Clear khi đọc được giấy tờ và không có lỗi chính. Chọn Outside_Scope nếu ảnh không có giấy tờ cần tìm hoặc ngoài hai lớp hỗ trợ. Xuất JSON thô vào `$run/reviews.json`.

```bash
python -m labeling.label_studio apply-reviews \
  --manifest "$run/dataset/manifest.csv" --review-export "$run/reviews.json" \
  --output "$run/dataset/reviewed.csv"
```

Mỗi bản xuất được áp dụng như một snapshot đầy đủ trên manifest gốc. Dòng bị bỏ qua giữ nhãn trống, nhãn đã thống nhất được lưu, dòng Outside_Scope bị loại. Nếu nhãn mâu thuẫn, xử lý trong Label Studio rồi xuất lại. Hai manifest dùng chung dataset root. Ghi ra tệp riêng. Chạy lại cùng bản xuất cho kết quả giống nhau. Nếu nhiều lượt ghi cùng đích, lần xuất bản thành công cuối quyết định toàn bộ nội dung.

## Kiểm thử với Label Studio Community

Chạy từ repo sau khi cài thư viện dữ liệu và chuẩn bị Docker hoặc rootless Podman.

```bash
set -o pipefail
python -u -m labeling.smoke 2>&1 | tee /tmp/ekyc-label-studio-smoke.log
```

Có thể chỉ định `--source /path/to/clips.tar` hoặc `--runtime podman`. Lệnh tạo lô 54 frame trong thư mục tạm, chạy image Community đã ghim trên một cổng loopback tự chọn, nhập tác vụ và đối chiếu từng JPEG với nguồn. Lệnh thêm một nhãn kiểm thử, tải JSON thô và chạy nhập nhãn. Container, dữ liệu dự án và fixture được dọn sau khi chạy. Log ghi trạng thái HTTP và số lượng. Nhãn fixture chỉ xác minh kỹ thuật. Phê duyệt cần thao tác của người thật.

## Duyệt dự đoán và xuất bản manifest train

Dùng dự án khác cho hàng đợi. Đặt `queue_root` là thư mục hàng đợi, `dataset_root` là thư mục dataset MIDV gốc, `work` là thư mục riêng đã tạo. Lượt đầu dùng manifest gốc, lượt sau có thể dùng manifest đã xuất bản từ lượt trước.

```bash
base_manifest="$dataset_root/manifest.csv"
python -m labeling.hitl batch \
  --queue-root "$queue_root" --base-manifest "$base_manifest" \
  --dataset-root "$dataset_root" --output "$work/batch"
```

Chỉ đọc sự kiện đã hoàn tất theo digest. Nội dung ảnh phải khớp digest và ảnh nguồn. Sự kiện lỗi làm lô thất bại. Validation và test được ghi nhận là giữ riêng, chỉ sự kiện train tạo tác vụ. Lô không có sự kiện train trả lỗi kèm số lượng giữ riêng. Giữ nguyên thư mục lô.

Bật phục vụ Local Files với root `$work/batch` hoặc mount chỉ đọc tương ứng. Thêm nguồn `images`, chọn Tasks và lưu mà không đồng bộ. Dùng `labeling/hitl.xml`, nhập `$work/batch/tasks.json`, sửa loại và chất lượng hoặc chọn riêng Outside_Scope để loại dòng train. Xuất JSON thô vào `$work/export.json`. Mỗi tác vụ cần một quyết định hoàn tất được thống nhất. Kết quả bị hủy, bị bỏ qua và nhãn dự đoán không được tính.

Người phê duyệt chạy lệnh riêng với định danh của mình. Hồ sơ lưu định danh, SHA-256 metadata lô, manifest gốc và bản xuất nguyên vẹn.

```bash
python -m labeling.hitl approve --batch "$work/batch" \
  --review-export "$work/export.json" --reviewer "$reviewer_id" \
  --output "$work/approval.json"
python -m labeling.hitl publish --batch "$work/batch" \
  --review-export "$work/export.json" --approval "$work/approval.json" \
  --base-manifest "$base_manifest" --dataset-root "$dataset_root" \
  --versions-root "$work/versions"
```

Lệnh in đường dẫn manifest. `lineage.json` lưu phê duyệt, hash manifest, ID sự kiện, dự đoán, lý do vào hàng đợi, nhãn trước và sau, cùng các dòng bị loại. Dòng test và validation giữ nguyên. Nhãn train có thể sửa loại trong khi layout ID và split giữ nguyên. Kết quả lưu nguồn gốc nhãn chất lượng.

ID phiên bản là SHA-256 của tệp phê duyệt. Lặp lại đầu vào cho cùng kết quả. Tệp khác nội dung ở một phiên bản có sẵn gây lỗi. Thay đổi bản xuất hoặc manifest gốc cần phê duyệt mới. Lưu bản xuất và phê duyệt mới ở đường dẫn riêng. Gửi job thủ công với manifest đã xuất bản và dataset root gốc, giữ các tệp truy vết kèm hồ sơ huấn luyện.

Chạy hồi quy bằng `python -m unittest -v labeling.test_hitl`.
