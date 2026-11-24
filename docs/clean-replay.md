# Clean eKYC replay / Tái lập eKYC từ môi trường sạch

## Inputs and isolation / Đầu vào và phạm vi

Obtain access from the publisher and cite Bulatov et al. when using the
benchmark. Xin quyền tải từ nhà phát hành và trích dẫn Bulatov và cộng sự khi
sử dụng bộ dữ liệu. [MIDV-2020 dataset access](https://l3i-share.univ-lr.fr/MIDV2020/midv2020.html) · [Benchmark paper](https://www.computeroptics.ru/KO/PDF/KO46-2/460212.pdf).

**EN.** Wait for the MIDV-2020 downloads to finish. Keep the source archives
read-only. Verify the canonical filenames, formats, and full TAR end markers.
Use the verified JPEG TARs `clips.tar`, `photo.tar`, `scan_rotated.tar`,
`scan_upright.tar`, and `templates.tar` for sampling. Record video and TIFF in
source inventory. JPEGs are accepted directly by the scoring API and retain
their source bytes. Video frame extraction and TIFF transcoding would add
preprocessing variables. Run the comparison on the sampled JPEGs.

**VI.** Chờ tải xong MIDV-2020. Giữ archive nguồn ở chế độ chỉ đọc. Xác minh
tên chính thức, định dạng và phần kết thúc TAR. Lấy mẫu từ năm TAR JPEG đã
xác minh gồm `clips.tar`, `photo.tar`, `scan_rotated.tar`,
`scan_upright.tar` và `templates.tar`. Ghi video và TIFF vào danh mục nguồn.
JPEG được API scoring nhận trực tiếp và giữ nguyên byte nguồn. Trích khung
hình từ video và chuyển TIFF sẽ thêm biến tiền xử lý. Chạy so sánh trên tập
JPEG đã lấy mẫu.

**EN.** Use a fresh source snapshot with a recorded Git base and SHA-256 list.
Build separate CPU release and CUDA training environments. Keep the source
archive inventory, private dataset, parameters, logs, manifests and model
artifacts outside Git. Save the package versions and device details for each
environment.

**VI.** Dùng snapshot mã nguồn mới, ghi commit gốc và danh sách SHA-256.
Tạo môi trường CPU cho kiểm thử release và môi trường CUDA riêng cho huấn
luyện. Giữ danh mục archive, tập dữ liệu, tham số, log, manifest và mô hình
ngoài Git. Lưu phiên bản gói và thông tin thiết bị của từng môi trường.

Run the regression suite from the snapshot after installing
`ops/release-requirements.txt`. Set `EKYC_RELEASE_DATASET` to the small
fixture dataset root to exercise the local Azure inference runtime and
`EKYC_CHECK_AZURE_SCHEMAS=1` to check Azure schemas.

Chạy bộ kiểm thử từ snapshot sau khi cài `ops/release-requirements.txt`.
Đặt `EKYC_RELEASE_DATASET` trỏ tới bộ fixture nhỏ để kiểm thử runtime
Azure inference cục bộ và `EKYC_CHECK_AZURE_SCHEMAS=1` để kiểm tra schema.

```bash
python -m unittest -v \
  data_pipeline.test_release data_pipeline.test_sample_midv \
  labeling.test_hitl labeling.test_release \
  modeling.test_quality_compare modeling.test_release \
  azure.test_release demo.test_release demo.test_review_queue
```

## Dataset and checks / Lấy mẫu và kiểm tra

Recreate the split record locally from the verified clips archive.
Tạo lại bản ghi split cục bộ từ archive clips đã xác minh.

```bash
python -m data_pipeline.midv2020 \
  --source "$MIDV_ARCHIVE_DIR/clips.tar" --output "$REPLAY_ROOT/clips"
python ops/prepare_model_quality.py \
  --manifest "$REPLAY_ROOT/clips/manifest.csv" --output "$REPLAY_ROOT/fixed-split"
sha256sum "$REPLAY_ROOT/fixed-split/split.json"
```

Compare with the SHA-256 in the sampling instructions before using the
record. So sánh với SHA-256 trong hướng dẫn lấy mẫu trước khi dùng bản ghi.

Follow the [sampling CLI](../data_pipeline/SAMPLING.md) with the sealed
historical split record. Run twice to distinct private output directories.
Compare `manifest.csv` and `provenance.json` byte for byte. Recalculate each
JPEG digest and the total payload size from disk. Confirm the
5,000,000,000-byte ceiling, both document classes in every split, and one
split per template and mock document. Keep train and validation rows separate
from the held-out manifest. Save the SHA-256 of each manifest.

Chạy [CLI lấy mẫu](../data_pipeline/SAMPLING.md) với bản ghi split lịch sử
đã khóa. Chạy hai lần vào hai thư mục riêng. So từng byte của `manifest.csv`
và `provenance.json`. Tính lại hash của từng JPEG và tổng số byte ảnh từ
đĩa. Kiểm tra trần 5.000.000.000 byte, đủ hai lớp giấy tờ trong từng split
và mỗi template, giấy tờ giả lập chỉ thuộc một split. Tách train và validation
khỏi manifest held-out. Lưu SHA-256 từng manifest.

## Model selection and deployment / Chọn mô hình và triển khai

Use [model comparison instructions](../modeling/QUALITY.md) to compare
pretrained frozen and fine-tuned candidates with validation metrics and to
seal the selection before opening the held-out manifest. Report macro-F1,
accuracy, per-class counts and recall, confusion, calibration, and retained
coverage. Report capture-domain results where available. Document the
number of distinct test layouts and any domain or template imbalance.

Dùng [hướng dẫn so sánh mô hình](../modeling/QUALITY.vi.md) để so sánh cấu
hình pretrained frozen và fine-tuned bằng validation. Khóa lựa chọn trước
khi mở manifest held-out. Báo cáo macro-F1, accuracy, số mẫu và recall từng
lớp, ma trận nhầm lẫn, calibration và tỷ lệ giữ lại. Báo cáo theo miền nếu
có, kèm số layout test và mọi chênh lệch số mẫu theo miền hoặc template.

Use the [Azure runbook](../azure/clean-room.md) for inventory, reversible job
archive, new Blob prefix and experiment, model registration, staging test,
and cleanup. Record the exact Blob manifest digest recovered from the job.
Use versioned assets and distinguish a bounded CPU Blob-to-train smoke from
local full-sample CUDA training. Preserve the endpoint response, benchmark
distribution, browser verification and telemetry query results with UTC
timestamps. Record the cleanup confirmation for the new endpoint.

Dùng [quy trình Azure](../azure/clean-room.md) để kiểm kê, archive job có thể
khôi phục, dùng prefix Blob và experiment mới, đăng ký mô hình, kiểm thử
staging và dọn dẹp. Lưu digest chính xác của manifest Blob mà job đọc. Dùng
asset có phiên bản và ghi rõ phạm vi smoke Blob-to-train chạy CPU và lượt
huấn luyện tập đầy đủ chạy CUDA cục bộ. Lưu phản hồi endpoint, phân phối
latency, xác minh browser và kết quả truy vấn telemetry cùng mốc UTC. Lưu
xác nhận đã dọn endpoint mới.

## Review and reporting / Duyệt nhãn và báo cáo

Follow the [review procedure](hitl.md) for a mock prediction that enters the
queue, correction in Label Studio and import. Document fixture cases as
technical evidence. Human approval requires the actual reviewer identity
and export digest. After approval, version the corrected training manifest
with lineage and submit a manual retraining run. Record quality-head
evaluation and the human approval step as pending until evidence exists.

Theo [quy trình duyệt nhãn](hitl.vi.md) cho dự đoán trên ảnh giả lập đi qua
queue, sửa nhãn trong Label Studio và nhập lại. Ghi ca fixture là minh chứng
kỹ thuật. Phê duyệt cần danh tính người duyệt thực tế và digest của bản
xuất. Sau khi được duyệt, tạo phiên bản manifest train đã chỉnh sửa kèm
lineage và gửi lượt huấn luyện lại thủ công. Giữ trạng thái đánh giá quality
head và phê duyệt là đang chờ cho đến khi có minh chứng.
