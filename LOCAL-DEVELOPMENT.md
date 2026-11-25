# Chạy và phát triển hệ thống eKYC

## Khởi động

Dùng Docker Compose v2 với Linux containers trên Windows, macOS hoặc Linux. Image dùng `linux/amd64` để cài bộ phụ thuộc PyTorch CPU đã khóa. Trên máy ARM, bật hỗ trợ giả lập amd64 của Docker (ví dụ Docker Desktop trên Apple Silicon).

```console
docker compose up --build -d --wait
docker compose ps
docker compose logs -f app
```

Mở http://127.0.0.1:8081. Demo và trang **Vận hành** mở được khi chưa có dataset. Để gửi ảnh dự đoán, nạp bundle hoặc cấu hình endpoint theo phần dưới. `docker compose down` dừng hệ thống. SQLite, log và kết quả huấn luyện nằm trong `runs/` trên máy. Docker giữ dữ liệu Label Studio trong volume `label-studio-data`.

Để tùy chỉnh, sao chép `.env.example` thành `.env` (PowerShell dùng `Copy-Item .env.example .env`). Nếu cổng 8081 đang bận, đổi `EKYC_APP_PORT` rồi mở cổng mới trong trình duyệt. Trên Linux, tạo `Data/` và `runs/` bằng tài khoản của bạn trước lần chạy đầu (`mkdir -p Data runs`). Nếu UID hoặc GID khác 1000, đặt `EKYC_CONTAINER_USER` theo kết quả `id -u` và `id -g`; kiểm tra tài khoản đó có quyền đọc `Data/` và ghi `runs/`. Trên Fedora hoặc máy bật SELinux, nếu bị từ chối quyền dù UID đúng, cấu hình nhãn container cho các thư mục bind mount, hoặc dùng Docker Desktop.

## Dữ liệu, huấn luyện và dự đoán

1. Giải nén tập con MIDV-2020 vào `Data/MIDV2020_sample_5GB`. Chuẩn bị các manifest theo template bằng lệnh dưới. Nếu `Data/splits` đã tồn tại, giữ lại để tái lập hoặc tạo output mới trong `runs/` và đưa ba manifest đã kiểm tra vào `Data/splits` từ máy host. [Nguồn MIDV-2020](https://l3i-share.univ-lr.fr/MIDV2020/midv2020.html)

   ```console
   docker compose run --rm -v ./Data:/output app python -m data_pipeline.prepared_midv --source /app/Data/MIDV2020_sample_5GB --output /output/splits
   ```

2. Dùng [luồng Bài tập 13](WORKFLOW.md) để huấn luyện và nghiệm thu ứng viên từ nhãn đã duyệt. Nút **Chạy baseline** trên trang Vận hành dùng để khảo sát manifest gốc. Transfer learning lần đầu cần tải trọng số ImageNet.
3. Đặt `EKYC_MODEL_BUNDLE` thành đường dẫn bundle trong container, ví dụ `/app/runs/assignment13/candidate-v1/selected/bundle`. Chạy `docker compose up -d --wait` để nạp mô hình. Dùng endpoint staging và quy trình nghiệm thu khi chuẩn bị phục vụ thật.
4. Thử ảnh JPEG hoặc PNG mô phỏng qua giao diện. Lệnh kiểm tra API bên dưới dùng ảnh do bạn chọn. Trên PowerShell dùng `curl.exe`.

   ```console
   curl --fail -H "Content-Type: image/jpeg" --data-binary "@sample.jpg" http://127.0.0.1:8081/predict
   ```

Để gọi Azure Managed Online Endpoint từ demo, để trống `EKYC_MODEL_BUNDLE` và đặt `EKYC_SCORING_URL`, `EKYC_SCORING_KEY`, cùng `EKYC_SCORING_DEPLOYMENT` nếu cần. Ngưỡng thống kê mặc định của demo là 0.7 khi dùng endpoint từ xa. Có thể đặt `--monitoring-threshold` trong lệnh chạy demo để khớp cấu hình model.

## Kết nối nguồn Azure

Điền `AZ_RESOURCE_GROUP` và `AZ_WORKSPACE` trong `.env`. Nếu có nhiều subscription, đặt thêm `AZ_SUBSCRIPTION`. Cấu hình bổ sung dựng image có Azure CLI và extension `ml`, dùng thư mục phiên đăng nhập `~/.azure` theo mặc định. Có thể đặt `EKYC_AZURE_CONFIG_DIR` trỏ tới một thư mục đăng nhập riêng. Chỉ dùng cấu hình này với stack cục bộ tin cậy vì CLI trong container có quyền dùng và làm mới phiên đăng nhập.

```console
docker compose -f compose.yaml -f compose.azure.yaml --profile review up --build -d --wait
```

Nếu phiên hết hạn hoặc thư mục cấu hình chưa có phiên đăng nhập, chạy `docker compose -f compose.yaml -f compose.azure.yaml exec app az login --use-device-code`. Dùng cùng hai tùy chọn `-f` khi khởi động lại stack để giữ kết nối Azure.

Để bật truy vấn Log Analytics, đặt `EKYC_LOG_ANALYTICS_WORKSPACE` bằng ID workspace và `AZ_ENDPOINT` bằng tên endpoint. Endpoint cần gửi cả log console và traffic vào workspace này. Nếu đang xem log còn lưu của một endpoint đã xóa, đặt `EKYC_AZURE_LOGS_ARCHIVED=1` để dashboard ghi rõ đây là dữ liệu lịch sử. Xem [hướng dẫn Azure](azure/README.md) để cấu hình diagnostic settings.

## Label Studio và nhãn chất lượng

Đặt `EKYC_REVIEW_MEDIA` thành thư mục dataset hoặc thư mục lô ảnh chứa `images/`, rồi chạy `docker compose --profile review up -d`. Mở http://127.0.0.1:8080, tạo tài khoản và dự án, sau đó nhập tác vụ theo [hướng dẫn Label Studio](labeling/README.md). Trong mạng Compose, demo gọi Label Studio qua `http://label-studio:8080`.

Để bật hàng đợi, tạo allowlist theo [hướng dẫn Label Studio](labeling/README.md) rồi điền cả ba biến `EKYC_REVIEW_QUEUE`, `EKYC_REVIEW_MANIFEST` và `EKYC_REVIEW_DATASET_ROOT` trong `.env`. Dùng đường dẫn trong container dưới `/app/Data/` hoặc `/app/runs/`. Thư mục hàng đợi cần ghi được, nên đặt dưới `/app/runs/`.

Sau khi đã tạo lô và nhập tác vụ vào dự án Label Studio, đặt `EKYC_REVIEW_BATCH` thành đường dẫn lô trong container, `EKYC_REVIEW_MEDIA` thành đường dẫn của cùng lô trên máy và `EKYC_LABEL_STUDIO_PROJECT` thành ID dự án. Chạy lại `docker compose --profile review up -d --wait` rồi mở **Vận hành → Duyệt ảnh**. Đăng nhập bằng tài khoản Label Studio. Để tùy chọn dùng token, xem phần **Duyệt trong webapp** của hướng dẫn gán nhãn và đặt các biến thông tin đăng nhập trong `.env` cục bộ.

Sau khi áp dụng nhãn hoặc xuất bản phiên bản đã phê duyệt, dùng manifest đó để huấn luyện thủ công. Ví dụ dưới đây giả định manifest là `runs/reviewed.csv` và các đường dẫn ảnh trong manifest tính từ `Data/MIDV2020_sample_5GB`. Mỗi lượt huấn luyện dùng một thư mục đầu ra mới.

```console
docker compose run --rm app python -m modeling.workflow train --dataset-root /app/Data/MIDV2020_sample_5GB --manifest /app/runs/reviewed.csv --output /app/runs/reviewed-model
```

## Vòng lặp phát triển

```console
docker compose -f compose.yaml -f compose.dev.yaml up --build -d --wait
docker compose -f compose.yaml -f compose.dev.yaml restart app
docker compose -f compose.yaml -f compose.dev.yaml exec app python -m demo.build
```

Sửa mã bằng editor trên máy. Tải lại trang sau khi sửa HTML, CSS hoặc JavaScript. Khởi động lại service sau khi sửa Python và build lại image sau khi đổi dependencies. Trước khi dừng hoặc khởi động lại container, kiểm tra các lượt huấn luyện đang chạy vì chúng dùng cùng container. Chạy script riêng bằng `docker compose run --rm app python -m <module>`. Dùng thêm `-f compose.yaml -f compose.dev.yaml` trước `run` để chạy mã đang sửa.

## Đối chiếu Bài tập 13

| Giai đoạn | Kết quả và bằng chứng cần lưu |
|---|---|
| Dữ liệu và gán nhãn | Lưu phiên bản dữ liệu mô phỏng trên Azure Blob. Dùng dự án Label Studio theo phương án đã được giảng viên chấp thuận. Lưu bằng chứng chấp thuận, ảnh chụp dự án và tác vụ đã duyệt, số ảnh theo lớp và manifest đã xuất. Microsoft công bố Azure ML Data Labeling ngừng từ 30/09/2026. [Thông báo của Microsoft](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-image-labeling-projects?view=azureml-api-2) |
| Huấn luyện | Theo [quy trình Azure](azure/README.md), lưu Azure ML job ID, trạng thái hoàn thành và metric Accuracy/Loss trong MLflow. Ghi lại tập con theo template, MobileNetV3, cấu hình augmentation và cách giữ tập test độc lập. Bản MIDV hiện phân loại hai loại giấy tờ. Head chất lượng cần nhãn `Clear`, `Blur`, `Glare` hoặc `Occluded`. |
| Triển khai | Lưu tên deployment, kết quả gọi Managed Online Endpoint và JSON benchmark. Ghi p50, p95 và số yêu cầu vượt 3 giây nếu có. Đo trên endpoint từ vị trí máy khách đã ghi trong báo cáo. |
| Monitoring và duyệt | Lưu tỷ lệ dự đoán dưới ngưỡng, ví dụ ảnh vào hàng đợi, nhãn do người duyệt xác nhận và phiên bản dữ liệu được phê duyệt. Sau đó mới kích hoạt huấn luyện thủ công. |

Trong báo cáo, giải thích rằng confidence thấp phát hiện mẫu cần xem lại nhưng không cung cấp nhãn đúng. Dùng dự đoán của mô hình làm nhãn cho mẫu giấy tờ mới có thể củng cố lỗi hiện có. Người duyệt xác định loại giấy tờ và chất lượng ảnh trước khi nhóm tạo phiên bản dữ liệu, đánh giá model mới trên tập test đã khóa và quyết định triển khai.
