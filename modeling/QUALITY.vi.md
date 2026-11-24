# So sánh MobileNetV3 trên cùng cách chia dữ liệu

[English](QUALITY.md) · **Tiếng Việt**

Chạy từ thư mục gốc bằng Python 3.11 có các thư viện của mô hình. Chỉ chọn CUDA sau khi thử thành công một phép toán tensor CUDA. Mặc định dùng sáu luồng Torch CPU, một luồng interop và hai tiến trình tải dữ liệu. Chạy tuần tự nếu tài nguyên chưa được phân bổ cho nhiều job.

Dùng cách chia theo mẫu giấy tờ của ETL MIDV-2020 với seed 42 và CSV chỉ chứa train và validation. Hai cấu hình cùng khởi tạo từ MobileNetV3 Small pretrained ImageNet, dùng ảnh RGB 224×224, cùng phép tăng cường và thứ tự lớp. Thử nghiệm `--crop-mode center_square` cắt vùng vuông ở giữa trước khi resize để giữ tỉ lệ và giảm nền. Gói mô hình lưu chế độ này để API dùng cùng phép biến đổi. Cấu hình frozen chỉ học classifier. Cấu hình fine-tune cập nhật bốn khối feature cuối với learning rate 3e-5 và classifier với 1e-3. Cả hai dùng AdamW và giữ thống kê BatchNorm pretrained. Xếp hạng bằng macro-F1, accuracy, rồi log loss thấp hơn trên validation. Hiệu chỉnh temperature và ngưỡng duyệt từ logits validation của checkpoint được chọn.

## So sánh trên train và validation

Dùng CSV phát triển với đường dẫn ảnh tương đối theo dataset root. Giữ tập test riêng. Nếu có SHA-256 của CSV test, truyền `--held-out-sha256` để ràng buộc đúng tệp test khi đánh giá.

```bash
set -o pipefail
runtime=.venv/model-release/bin/python
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
"$runtime" -u -m modeling.quality_compare compare \
  --dataset-root "$DATASET_ROOT" --development "$DEVELOPMENT_CSV" \
  --output "$RUN_DIR" --device cpu --threads 6 --epochs 8 \
  2>&1 | tee "$RUN_DIR.compare.log"
```

Thư mục đầu ra phải mới. Thư mục lưu bundle, snapshot dữ liệu đã dùng, JSONL từng epoch, MLflow run và cấu hình chọn được. So sánh các phép tiền xử lý bổ sung bằng validation, rồi chốt cấu hình trước khi mở tập test. Truyền một `--comparison` cho mỗi lần thử. Lệnh `seal` xếp hạng theo validation và sao chép bundle cùng snapshot của cấu hình thắng vào thư mục mới.

```bash
"$runtime" -m modeling.quality_compare seal \
  --comparison "$FULL_FRAME_RUN" --comparison "$CENTER_SQUARE_RUN" \
  --output "$FINAL_RUN_DIR"
```

## Huấn luyện lại từ phiên bản đã được duyệt

Cung cấp thư mục chứa `manifest.csv` và `lineage.json` đã duyệt, bản xuất Label Studio thô, JSON phê duyệt, thư mục lô và manifest ETL gốc có đầy đủ các tập. `DEVELOPMENT_CSV` vẫn là CSV train/validation gốc. Chương trình dựng lại phiên bản đã duyệt từ đầu vào gốc và đối chiếu từng byte của hai tệp trước khi train. Chỉ các dòng train được sửa hoặc loại. Người vận hành chạy lệnh riêng để bắt đầu huấn luyện.

```bash
"$runtime" -u -m modeling.quality_compare compare \
  --dataset-root "$DATASET_ROOT" --development "$DEVELOPMENT_CSV" \
  --approved-version "$APPROVED_VERSION_DIR" --approval "$APPROVAL_JSON" \
  --review-export "$RAW_LABEL_STUDIO_EXPORT" --batch "$BATCH_DIR" \
  --base-manifest "$BASE_MANIFEST" --output "$RUN_DIR" \
  --device cpu --threads 6 --epochs 8 2>&1 | tee "$RUN_DIR.compare.log"
```

## Đánh giá trên tập test giữ riêng

Sau khi chốt cấu hình, dùng CSV test và manifest ETL đầy đủ. Chương trình kiểm tra hash bundle và CSV phát triển, danh sách test, sự hiện diện của hai lớp và cách ly giữa các tập. Dấu ghi độc quyền chặn các lần đánh giá lặp hoặc đồng thời kể cả khi lần trước bị gián đoạn. Nếu bị gián đoạn sau khi tạo dấu ghi, người điều phối phải xử lý theo quy trình đánh giá. Giữ dấu ghi và log gốc.

```bash
"$runtime" -u -m modeling.quality_compare evaluate \
  --dataset-root "$DATASET_ROOT" --held-out "$HELD_OUT_CSV" \
  --base-manifest "$BASE_MANIFEST" --output "$FINAL_RUN_DIR" \
  --device cpu --threads 6 2>&1 | tee "$FINAL_RUN_DIR.evaluate.log"
```

`held-out.json` ghi accuracy toàn tập, macro-F1, recall và số mẫu theo lớp, ma trận nhầm lẫn với hàng là nhãn thật và cột là dự đoán, kết quả theo mẫu giấy tờ, ECE top-label 15 bin trước và sau hiệu chỉnh, độ bao phủ và accuracy ở ngưỡng đã chọn trên validation. Ngưỡng tối đa hóa độ bao phủ tại accuracy mục tiêu mặc định 85%, giữ các điểm confidence bằng nhau cùng nhóm. Khi không tìm được ngưỡng đạt mục tiêu, bundle chuyển mọi ảnh sang duyệt. Báo cáo ghi riêng độ bao phủ theo confidence giấy tờ và độ bao phủ API, trong đó API còn đưa ảnh thiếu kết quả chất lượng hoặc có chất lượng khác `Clear` sang duyệt.

Chỉ số chất lượng ảnh được báo cáo riêng. Kết quả chờ nếu thiếu nhãn test đã duyệt hoặc thiếu đầu ra phân loại chất lượng. Kết quả được đánh dấu một phần khi tập đánh giá thiếu lớp trong bốn lớp chất lượng. Tiêu chí accuracy giấy tờ 85% tính trên toàn bộ tập test, kể cả ảnh chuyển duyệt.

```bash
OMP_NUM_THREADS=6 "$runtime" -m unittest -v modeling.test_quality_compare modeling.test_release
```

## Đánh giá trên điều kiện thu nhận bổ sung

Dùng mô hình đã khóa để chấm một ảnh chụp hoặc scan cho mỗi giấy tờ mô phỏng
thuộc tập test. Cung cấp `split.json` gốc từ `ops.prepare_model_quality` và
các TAR có cùng danh sách giấy tờ. Công cụ đọc archive tại chỗ và giữ nguyên
crop, nhiệt độ hiệu chỉnh và ngưỡng. Báo cáo riêng từng điều kiện thu nhận và
lưu JSON trong thư mục riêng. Nhiều ảnh của cùng giấy tờ là các quan sát ghép cặp.

```bash
"$runtime" -m ops.evaluate_capture_domains --bundle "$FINAL_RUN_DIR/bundle" \
  --split-record "$SPLIT_RECORD" --archive "$MIDV_DIR/photo.tar" \
  --archive "$MIDV_DIR/scan_rotated.tar" --archive "$MIDV_DIR/scan_upright.tar" \
  --archive "$MIDV_DIR/templates.tar" --output "$CAPTURE_RESULTS"
```

## Chuẩn bị bộ dữ liệu fine-tune đa nguồn

Dùng manifest development gốc, bản ghi chia tập và mã SHA-256 đã lưu của bản
ghi đó. Lệnh tạo dataset riêng chỉ có train và validation, với cột `domain` và
nhãn chất lượng để trống. Chấm đối chứng giữ nguyên và từng epoch fine-tune
trên cùng các dòng validation. Báo cáo macro-F1 theo nguồn và đặt tiêu chí
validation trước khi quyết định phát hành gói mô hình.

```bash
"$runtime" -m data_pipeline.multisource \
  --clips-manifest "$CLIPS_DEVELOPMENT_CSV" --clips-root "$CLIPS_DATASET_ROOT" \
  --split-record "$SPLIT_RECORD" --split-sha256 "$SPLIT_SHA256" \
  --archive-dir "$MIDV_DIR" --output "$PRIVATE_DATASET_ROOT"
```
