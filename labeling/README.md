# Label Studio

## Kết nối thư mục ảnh cho từng dự án

Sau khi tạo dự án, mở **Settings → Cloud Storage → Add Source Storage → Local Files**. Thêm từng thư mục con được tác vụ sử dụng, chẳng hạn `/label-studio/media/clips/images` hoặc `/label-studio/media/photo/images`. Với bộ MIDV nhiều nguồn, thêm cả `clips`, `photo`, `scan_rotated`, `scan_upright` và `templates` theo mẫu `/label-studio/media/<nguồn>/images`.

Chọn chế độ **Tasks** và lưu nguồn. Khi đã nhập `tasks.json`, mở lại một tác vụ để kiểm tra ảnh. Cài đặt này phải được thực hiện riêng cho mỗi dự án gán nhãn hoặc duyệt trực tiếp.

## Duyệt trực tiếp từ demo

Tạo dự án Label Studio với [hitl.xml](hitl.xml). Điền ID dự án vào `EKYC_LIVE_REVIEW_PROJECT` và thông tin tài khoản dự án vào `EKYC_LABEL_STUDIO_EMAIL` và `EKYC_LABEL_STUDIO_PASSWORD` trong `.env` cục bộ. Gắn cùng thư mục ảnh qua `EKYC_REVIEW_MEDIA` và `EKYC_REVIEW_DATASET_ROOT` rồi cấu hình allowlist như bên dưới.

Demo tạo hoặc mở lại tác vụ khi ảnh trong allowlist có trạng thái **Cần duyệt**, hoặc khi người dùng bấm **Báo dự đoán sai**. Tác vụ mở trực tiếp trong Label Studio để chọn `Passport`, `ID_Card` hoặc `Outside_Scope`, sau đó chọn chất lượng nếu ảnh thuộc phạm vi. Dùng allowlist CSV riêng trong `EKYC_REVIEW_AUDIT_MANIFEST` cho ảnh chỉ dùng đối chiếu. Hai allowlist phải chứa các ảnh khác nhau.

## Khởi động Label Studio

Đặt `media_root` thành đường dẫn tuyệt đối của dataset khi gán nhãn ban đầu,
hoặc thư mục `batch` khi duyệt dự đoán. Chạy container ở terminal riêng.

```bash
runtime=docker # Hoặc podman
media_root="$(realpath /path/to/media-root)"
"$runtime" volume create ekyc-label-studio
"$runtime" run --rm -p 127.0.0.1:8080:8080 \
  -v ekyc-label-studio:/label-studio/data \
  -v "$media_root:/label-studio/media:ro" \
  -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
  -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/media \
  docker.io/heartexlabs/label-studio@sha256:844f82c5b08de3f77a57c66f480e0036d3c76c6eff53699e8d47686613faa922
```

Mở [Label Studio](http://127.0.0.1:8080). Nếu SELinux chặn mount ảnh, dùng
bản sao lô riêng và thêm `,Z` sau `:ro` trong tùy chọn mount.

## Gán nhãn chất lượng ban đầu

Chạy từ thư mục gốc. Lưu dataset, tác vụ và bản xuất trong thư mục riêng ngoài repo. Cài Pillow bằng `python -m pip install -r data_pipeline/requirements.txt`.

```bash
run=$(mktemp -d /tmp/ekyc-data.XXXXXX)
python -m data_pipeline.midv2020 --source /path/to/clips.tar --output "$run/dataset" \
  --max-clips-per-template 2 --frames-per-clip 3 --min-frame-gap 10
python -m labeling.label_studio make-tasks \
  --manifest "$run/dataset/manifest.csv" --dataset-root "$run/dataset" \
  --media-root "$run/dataset" --output "$run/tasks.json"
```

Dùng `--source /path/to/clips.tar` để đổi vị trí tệp. Bỏ giới hạn clip để xử lý toàn bộ phần hợp lệ. `template_id` là mẫu giấy tờ, ví dụ `est_id`. `document_id` là giấy tờ mô phỏng có số thứ tự và clip tương ứng. Mỗi mẫu chỉ thuộc một tập, mỗi tập có đủ hai lớp. Khoảng cách giữa ảnh được tính bằng số thứ tự frame gốc. Nhãn chất lượng để trống trước khi duyệt.

Trước khi chạy Label Studio, bật `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` và đặt `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` thành thư mục ảnh. Nếu chạy container, gắn dataset chỉ đọc và dùng đường dẫn bên trong container. Tạo dự án với giao diện [quality.xml](quality.xml). Thêm thư mục `/label-studio/media/images` vào Local Files, chọn chế độ Tasks, lưu và bỏ qua đồng bộ. Nhập `tasks.json` trong Data Manager. [Label Studio Local Files](https://labelstud.io/guide/storage_local)

Chọn lỗi ảnh ảnh hưởng khả năng đọc nhiều nhất. Chọn Clear khi đọc được giấy tờ và không có lỗi chính. Chọn Outside_Scope nếu ảnh không có giấy tờ cần tìm hoặc ngoài hai lớp hỗ trợ. Xuất JSON thô vào `$run/reviews.json`.

```bash
python -m labeling.label_studio apply-reviews \
  --manifest "$run/dataset/manifest.csv" --review-export "$run/reviews.json" \
  --output "$run/dataset/reviewed.csv"
```

Mỗi bản xuất được áp dụng như một snapshot đầy đủ trên manifest gốc. Dòng bị bỏ qua giữ nhãn trống, nhãn đã thống nhất được lưu, dòng Outside_Scope bị loại. Nếu nhãn mâu thuẫn, xử lý trong Label Studio rồi xuất lại. Hai manifest dùng chung dataset root. Ghi ra tệp riêng. Chạy lại cùng bản xuất cho kết quả giống nhau. Nếu nhiều lượt ghi cùng đích, lần xuất bản thành công cuối quyết định toàn bộ nội dung.

## Đưa dự đoán vào hàng đợi

Khởi động [API suy luận](../README.md#3-chạy-api-và-demo) trước.
Ví dụ sau dùng đầu ra từ bộ MIDV đã giải nén. Đổi đường dẫn theo lần chạy.

```bash
dataset=/path/to/MIDV2020_sample_5GB
run="$HOME/.local/share/mlops-ekyc/runs/midv-5gb"
base_manifest="$run/inputs/manifest.csv"
work=$(mktemp -d "$run/review.XXXXXX")
python - "$dataset" "$base_manifest" "$work/allowlist.csv" <<'PYCODE'
import csv
import sys
from pathlib import Path

dataset, manifest, allowlist = map(Path, sys.argv[1:])
with manifest.open(newline="") as source, allowlist.open("w", newline="") as target:
    writer = csv.DictWriter(target, fieldnames=["image_id", "image_path"])
    writer.writeheader()
    for row in csv.DictReader(source):
        if row["split"] != "train":
            continue
        size = (dataset / row["image_path"]).stat().st_size
        if 0 < size <= 8 * 1024**2:
            writer.writerow({key: row[key] for key in writer.fieldnames})
PYCODE
```

```bash
export EKYC_SCORING_URL=http://127.0.0.1:5002/score
python -m demo.server --monitoring-threshold "${EKYC_MONITORING_THRESHOLD:-0.7}" \
  --review-queue "$work/queue" \
  --review-manifest "$work/allowlist.csv" \
  --review-dataset-root "$dataset"
```

Mở [demo cục bộ](http://127.0.0.1:8081), chọn ảnh mô phỏng trong manifest và gửi dự đoán. Khi API trả `needs_review=true`, demo lưu ảnh và dự đoán vào hàng đợi. Nút **Báo dự đoán sai** thêm lý do báo sai. Các lần gửi cùng ảnh dùng chung sự kiện theo SHA-256. Có thể thử lại khi lỗi ghi hàng đợi. Sau khi khởi động lại demo, gửi dự đoán mới để lấy mã báo sai hiện hành. Sự kiện đã lưu vẫn còn.

## Chuyển lô ảnh sang Label Studio

```bash
python -m labeling.hitl batch --queue-root "$work/queue" \
  --base-manifest "$base_manifest" --dataset-root "$dataset" \
  --output "$work/batch"
```

Tạo dự án Label Studio Community với [hitl.xml](hitl.xml). Khởi động Label Studio với `media_root="$work/batch"`. Thêm `/label-studio/media/images` làm nguồn Local Files ở chế độ Tasks và lưu nguồn. Nhập `batch/tasks.json`. Người duyệt xác nhận hoặc sửa loại giấy tờ và chất lượng. Chọn `Outside_Scope` khi cần loại ảnh. Hoàn thành mọi tác vụ trong lô rồi xuất JSON thô.

Chỉ sự kiện train đã ghi hoàn tất được đưa vào lô. Sự kiện validation hoặc test vẫn có thể xuất hiện trong lịch sử hàng đợi. Các dòng validation và test giữ nguyên khi tạo phiên bản mới.

## Phê duyệt và tạo phiên bản

Đặt `reviewer_id` thành định danh người phê duyệt. Giữ nguyên nội dung tệp JSON đã xuất. Gán nhãn và phê duyệt là hai thao tác riêng.

```bash
python -m labeling.hitl approve --batch "$work/batch" \
  --review-export "$work/reviews.json" --reviewer "$reviewer_id" \
  --output "$work/approval.json"
python -m labeling.hitl publish --batch "$work/batch" \
  --review-export "$work/reviews.json" --approval "$work/approval.json" \
  --base-manifest "$base_manifest" --dataset-root "$dataset" \
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

## Duyệt trong webapp

Webapp dùng lô đã tạo và dự án Label Studio đã nhập tác vụ theo các bước trên.
Sau khi cấu hình hàng đợi và máy chủ chấm điểm như trên, bổ sung
`--review-batch "$work/batch" --label-studio-project "$project_id"`
vào lệnh chạy demo. `EKYC_LABEL_STUDIO_URL` mặc định là
`http://127.0.0.1:8080`. Khi trình duyệt hỏi đăng nhập, nhập email và mật khẩu
của tài khoản Label Studio có quyền trên dự án. Có thể đặt
`EKYC_LABEL_STUDIO_EMAIL` và `EKYC_LABEL_STUDIO_PASSWORD` ở máy chủ để
Monitoring đọc trạng thái lô.

Nếu chạy ở chế độ token, đặt `EKYC_LABEL_STUDIO_TOKEN` cho tài khoản ghi nhãn
và thêm `EKYC_LABEL_STUDIO_TOKEN_TYPE=personal` nếu dùng PAT. Ở chế độ này,
hộp thoại dùng `EKYC_REVIEW_USER` (mặc định `reviewer`) và
`EKYC_REVIEW_PASSWORD` của demo.

Mở [Vận hành → Duyệt ảnh](http://127.0.0.1:8081/review). Chọn ảnh, sửa nhãn
và bấm **Lưu nhãn**. Trang hiển thị ID annotation và tài khoản Label Studio
ghi nhãn. Nếu ảnh mới vào
hàng đợi sau khi tạo lô, tạo lô mới trước khi duyệt. Sau khi ghi nhãn xong,
xuất JSON từ Label Studio rồi chạy bước phê duyệt và tạo phiên bản ở trên.
