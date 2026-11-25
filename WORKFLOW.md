# Quy trình nghiệm thu Bài tập 13

Hệ thống phân loại `ID_Card` và `Passport` trên tập con MIDV-2020. Các nhãn chất lượng là `Clear`, `Blur`, `Glare` và `Occluded`. Nhóm dùng Label Studio theo phương án đã được giảng viên chấp thuận.

## Checklist nghiệm thu

- [ ] **Dự án gán nhãn.** Đề gốc yêu cầu bằng chứng tạo dự án trên Azure. Với phương án thay thế đã được duyệt, lưu xác nhận của giảng viên cùng ảnh chụp dự án Label Studio, tác vụ được gán nhãn tay và bản nhãn đã xuất.
- [ ] **Training Pipeline.** Chạy script PyTorch trên Azure ML Compute. Lưu job ID, trạng thái `Completed`, log và kết quả MLflow của lượt chạy nghiệm thu.
- [ ] **Inference API.** Gửi ảnh mô phỏng đến Azure Managed Online Endpoint. Lưu tên deployment, yêu cầu mẫu và phản hồi thực tế có `label`, `status` và `confidence`. Kèm kết quả đo độ trễ staging.
- [ ] **Báo cáo Human-in-the-loop.** Giải thích vì sao điểm mô hình không đủ để tạo nhãn mới, ai xác nhận loại và chất lượng giấy tờ, và nhóm phê duyệt phiên bản dữ liệu thế nào trước khi kích hoạt huấn luyện lại thủ công.

## 1. Chuẩn bị dữ liệu và gán nhãn

Giải nén MIDV-2020 vào `Data/MIDV2020_sample_5GB`. Nếu đã có `Data/splits/manifest.csv` từ cùng nguồn, dùng lại phiên bản đó. Lần đầu có thể tạo split trong thư mục ghi được `runs/assignment13/splits/`.

```console
docker compose run --rm app python -m data_pipeline.prepared_midv --source /app/Data/MIDV2020_sample_5GB --output /app/runs/assignment13/splits
```

Các lệnh tiếp theo dùng manifest sẵn có trong `Data/splits/`. Nếu vừa tạo split mới, thay đường dẫn thành `/app/runs/assignment13/splits/manifest.csv`. Tạo thư mục `runs/assignment13/labeling` trên máy và tạo danh sách ảnh gán nhãn.

```console
docker compose run --rm app python -m labeling.label_studio make-tasks --manifest /app/Data/splits/manifest.csv --dataset-root /app/Data/MIDV2020_sample_5GB --media-root /app/Data/MIDV2020_sample_5GB --per-group 3 --output /app/runs/assignment13/labeling/tasks.json
```

Danh sách lấy tối đa ba giấy tờ khác nhau theo mỗi tổ hợp split, loại và nguồn ảnh. Đặt `EKYC_REVIEW_MEDIA=./Data/MIDV2020_sample_5GB` trong `.env`, bật profile `review`, mở Label Studio ở cổng đã cấu hình và nhập `tasks.json` vào dự án dùng [quality.xml](labeling/quality.xml). Media của bộ này nằm trong các thư mục nguồn như `/label-studio/media/photo/images`. Tham khảo [quy trình Label Studio](labeling/README.md) để thiết lập Local Files. Nhập ảnh bổ sung nếu thiếu nhãn chất lượng trong một split. Người duyệt gán nhãn theo ảnh thực tế và lưu bản xuất JSON, ảnh chụp dự án và bằng chứng chấp thuận.

```console
docker compose run --rm app python -m labeling.label_studio apply-reviews --manifest /app/Data/splits/manifest.csv --review-export /app/runs/assignment13/labeling/reviews.json --output /app/runs/assignment13/labeling/reviewed-v1.csv
```

Giữ nguyên split theo mẫu giấy tờ. Sau khi chốt manifest, giữ nguyên nhãn và thành phần validation/test cho các vòng HITL của cùng phiên bản đánh giá. Dùng `reviewed-v1.csv` để chỉ tải các ảnh được tham chiếu lên Azure Blob, rồi đăng ký cả ảnh lẫn manifest thành data assets có phiên bản. [Lệnh Azure Blob và ML Compute](azure/README.md)

## 2. Huấn luyện và đánh giá

Azure ML Compute và Docker dùng chung lệnh `modeling.workflow train`, mặc định 3 epoch. MobileNetV3 dùng trọng số ImageNet, xoay nhẹ, dịch chuyển, thay đổi độ sáng và tương phản. Lưu Accuracy, Loss và Macro-F1 bằng MLflow. Có thể chạy thử trên máy trước khi gửi job Azure.

```console
docker compose run --rm app python -m modeling.workflow train --dataset-root /app/Data/MIDV2020_sample_5GB --manifest /app/runs/assignment13/labeling/reviewed-v1.csv --output /app/runs/assignment13/candidate-v1
```

Luồng chụp lại manifest, kiểm tra chia tập và nhãn đã duyệt, chọn checkpoint trên validation, khóa bundle rồi đánh giá một lần trên test. Xem `selected/held-out.json` để biết recall theo lớp, kết quả theo mẫu và nguồn ảnh, độ hiệu chuẩn và danh sách dự đoán sai có điểm cao. `workflow.json` ghi kết luận.

Mốc nội bộ hiện tại để đưa ứng viên lên staging là 85% cho accuracy, macro-F1 và recall từng lớp giấy tờ, 85% macro-F1 chất lượng và 85% quyết định tự chấp nhận đúng cả loại lẫn chất lượng trên các ảnh test đã gán nhãn. Cần đủ bốn nhãn chất lượng do người duyệt xác nhận trong cả ba split. Có thể chạy job với nhãn chưa đủ để lấy báo cáo. Output dành cho staging khi đó chỉ chứa `blocked.json`. Chế độ `--baseline` cũng luôn bị chặn khỏi staging.

## 3. Staging trên Managed Online Endpoint

Kiểm tra bundle trước khi đăng ký hoặc đưa lên staging.

```console
docker compose run --rm app python -m modeling.workflow check-release --run /app/runs/assignment13/candidate-v1
```

Với job Azure, tải thư mục `evidence` và kiểm tra kết quả tương tự. Triển khai deployment thử ở 0% lưu lượng và gọi đích danh deployment. Đo bằng ảnh mô phỏng đại diện, lưu JSON gồm p50, p95, max, lỗi và số yêu cầu vượt 3 giây. Chỉ chuyển lưu lượng khi benchmark qua và đã xem kết quả dự đoán mẫu. [Lệnh staging và benchmark](azure/README.md#3-triển-khai-và-kiểm-tra)

## 4. Giám sát và duyệt lại

Theo dõi tỷ lệ điểm phân loại dưới ngưỡng, lỗi và độ trễ. Sau khi endpoint mới gửi diagnostic logs, đặt tên endpoint đó trong `.env` và bỏ chế độ log lịch sử. Một điểm phân loại cao vẫn có thể đi kèm dự đoán sai. Người dùng có thể báo sai cả với ảnh điểm cao nếu ảnh thuộc allowlist tập train.

Khi đã bật dự án duyệt trực tiếp, ảnh mô phỏng đã xác minh và có trạng thái **Cần duyệt** tạo tác vụ Label Studio ngay sau khi suy luận. Dùng **Mở tác vụ Label Studio** để chọn `Passport` hoặc `ID_Card` và nhãn chất lượng. Có thể khai báo bộ ảnh đối chiếu riêng qua `EKYC_REVIEW_AUDIT_MANIFEST`. Ảnh trong bộ này chỉ được duyệt, không vào hàng đợi dùng để tạo manifest huấn luyện.

Tạo allowlist bằng `python -m labeling.label_studio make-allowlist` từ manifest đã chốt. Dùng [quy trình Label Studio](labeling/README.md) để tạo lô, xuất nhãn, phê duyệt và xuất bản manifest phiên bản mới. Chạy lại bước 2 bằng manifest đã phê duyệt, giữ nguyên validation/test. Người duyệt là nguồn nhãn đúng cho mẫu mới. Lấy dự đoán của mô hình làm nhãn huấn luyện tự động có thể lặp lại chính lỗi mô hình đang mắc.
