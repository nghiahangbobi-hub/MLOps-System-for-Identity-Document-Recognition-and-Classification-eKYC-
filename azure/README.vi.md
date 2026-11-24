# Triển khai trên Azure ML

[English](README.md) · **Tiếng Việt**

Bắt đầu từ [thiết lập Blob Storage](blob-storage.md) để tạo workspace và data asset. Chạy từ thư mục gốc với Azure CLI đã đăng nhập và extension `ml`. Cần workspace, compute CPU, data asset thư mục ảnh, data asset manifest và SKU endpoint khả dụng. Đặt `AZ_SUBSCRIPTION`, `AZ_RESOURCE_GROUP`, `AZ_WORKSPACE`, `AZ_COMPUTE`, `AZ_DATASET_ASSET`, `AZ_MANIFEST_ASSET` và `AZ_ENDPOINT`. Định dạng asset là `azureml:name:version`. Đường dẫn `image_path` phải khớp cấu trúc thư mục ảnh. Đánh giá chất lượng cần nhãn đã duyệt.

## Chạy thử huấn luyện CPU

Sau khi kiểm tra Blob và đăng ký hai asset, kiểm tra quota đúng vùng. Ví dụ dùng Japan East và họ DSv3 khi đủ quota. Số node tối thiểu bằng 0 để compute thu nhỏ khi rảnh.

```bash
workspace=(--subscription "$AZ_SUBSCRIPTION" --resource-group "$AZ_RESOURCE_GROUP" --workspace-name "$AZ_WORKSPACE")
az ml compute list-usage "${workspace[@]}" --location japaneast -o table
export AZ_COMPUTE=ekyc-cpu
az ml compute create "${workspace[@]}" --name "$AZ_COMPUTE" --type amlcompute \
  --size Standard_D2s_v3 --min-instances 0 --max-instances 1 \
  --idle-time-before-scale-down 120
az ml compute show "${workspace[@]}" --name "$AZ_COMPUTE" --query provisioning_state -o tsv
```

Đợi `Succeeded`. Dùng compute sẵn có nếu phù hợp. Gửi một job kiểm thử cho cặp asset đã định phiên bản và lưu tên job.

```bash
work_dir=$(mktemp -d)
python azure/stage.py --training-only "$work_dir/aml"
spec="$work_dir/aml"
az ml environment create "${workspace[@]}" -f "$spec/environment-training.yml"
job=$(az ml job create "${workspace[@]}" -f "$spec/train.yml" \
  --set "compute=azureml:$AZ_COMPUTE" "inputs.dataset.path=$AZ_DATASET_ASSET" \
        "inputs.manifest.path=$AZ_MANIFEST_ASSET" inputs.epochs=1 inputs.batch_size=4 \
        limits.timeout=1800 --query name -o tsv)
printf '%s\n' "$job" > "$work_dir/job-name"
az ml job show "${workspace[@]}" --name "$job" --query status -o tsv
```

Kiểm tra trạng thái tới khi kết thúc. Nếu `job stream` báo `Incorrect padding`, dùng `job show` và mở **Outputs + logs** trong Azure ML Studio. Khi hoàn tất, tải output theo tên.

```bash
az ml job download "${workspace[@]}" --name "$job" \
  --output-name model_bundle --download-path "$work_dir/output"
```

Khi cần log build, tìm trong artifact datastore mặc định. Ví dụ dùng xác thực `key` do quyền Entra chỉ được cấp ở container đầu vào. CLI lấy khóa nội bộ.

```bash
mapfile -t artifact_store < <(
  az ml datastore show "${workspace[@]}" --name workspaceartifactstore \
    --query '[account_name, container_name]' -o tsv
)
artifact_account="${artifact_store[0]}"
artifact_container="${artifact_store[1]}"
artifact_prefix="ExperimentRun/dcid.$job/"
az storage blob list --auth-mode key --account-name "$artifact_account" \
  --container-name "$artifact_container" --prefix "$artifact_prefix" \
  --query '[].name' -o tsv
az storage blob download --auth-mode key --account-name "$artifact_account" \
  --container-name "$artifact_container" \
  --name "${artifact_prefix}azureml-logs/20_image_build_log.txt" \
  --file "$work_dir/image-build.log" -o none
```

Nếu tải output báo `BatchGetResolvedUrisDto`, xem đường dẫn trong **Outputs + logs**. Với vị trí mặc định, có thể tải hai tệp bằng Entra từ `workspaceblobstore`.

```bash
mapfile -t output_store < <(
  az ml datastore show "${workspace[@]}" --name workspaceblobstore \
    --query '[account_name, container_name]' -o tsv
)
bundle="$work_dir/model_bundle"
mkdir -p "$bundle"
for file in metadata.json weights.pt; do
  az storage blob download --auth-mode login --account-name "${output_store[0]}" \
    --container-name "${output_store[1]}" --name "azureml/$job/model_bundle/$file" \
    --file "$bundle/$file" -o none
done
AZUREML_MODEL_DIR="$bundle" python -c 'from modeling.score import init; init()'
```

Đối chiếu `manifest_sha256` trong Studio với `sha256sum` của tệp Blob đã tải. Đoạn kiểm tra CLI cần `mlflow-skinny` và `azureml-mlflow`.

```bash
tracking_uri=$(az ml workspace show --subscription "$AZ_SUBSCRIPTION" \
  --resource-group "$AZ_RESOURCE_GROUP" --name "$AZ_WORKSPACE" \
  --query mlflow_tracking_uri -o tsv)
MLFLOW_TRACKING_URI="$tracking_uri" python - "$job" <<'PYCODE'
import sys
from mlflow import MlflowClient

run = MlflowClient().get_run(sys.argv[1])
print(run.info.status, run.data.params["manifest_sha256"])
print({key: run.data.metrics[key] for key in ("train_samples", "val_samples", "test_samples")})
PYCODE
```

Trước khi gửi lại, kiểm tra job và experiment hiện tại, kể cả khi CLI trả lỗi chưa rõ trạng thái. Đến đây có thể xác minh chuỗi Blob → compute → output. Các mục tiếp theo dùng khi triển khai hoặc huấn luyện lại.

## Huấn luyện và triển khai

Khi chạy lại sau HITL, đăng ký manifest đã phê duyệt thành phiên bản `uri_file` mới và đặt `AZ_MANIFEST_ASSET` đúng phiên bản. Dùng lại thư mục ảnh gốc. Lưu lineage, phiên bản asset và tên job. Job CPU chạy `modeling.train`. [So sánh MobileNetV3](../modeling/QUALITY.vi.md)

```bash
work_dir=$(mktemp -d)
python azure/stage.py "$work_dir/aml"
spec="$work_dir/aml"
workspace=(--subscription "$AZ_SUBSCRIPTION" --resource-group "$AZ_RESOURCE_GROUP" --workspace-name "$AZ_WORKSPACE")
az ml environment create "${workspace[@]}" -f "$spec/environment-training.yml"
job=$(az ml job create "${workspace[@]}" -f "$spec/train.yml" \
  --set "compute=azureml:$AZ_COMPUTE" "inputs.dataset.path=$AZ_DATASET_ASSET" \
        "inputs.manifest.path=$AZ_MANIFEST_ASSET" --query name -o tsv)
az ml job stream "${workspace[@]}" --name "$job"
az ml job show "${workspace[@]}" --name "$job" --query status -o tsv
```

Dùng `--training-only` nếu chỉ chuẩn bị job train. Tạo thư mục staging mới khi triển khai. Có thể thêm `inputs.epochs=8 inputs.learning_rate=0.0005` vào `--set` để so sánh cấu hình. Giữ nguyên tập chia và lưu tham số, chỉ số MLflow, thời lượng.

Khi job đạt `Completed`, đăng ký output. Chọn `AZ_MODEL_VERSION` chưa dùng và `AZ_INSTANCE_TYPE` theo quota và SKU vùng. Xác minh endpoint và deployment trước khi chuyển lưu lượng.

```bash
az ml environment create "${workspace[@]}" -f "$spec/environment-inference.yml"
az ml model create "${workspace[@]}" --name ekyc-document-classifier \
  --version "$AZ_MODEL_VERSION" --type custom_model \
  --path "azureml://jobs/$job/outputs/model_bundle"
az ml online-endpoint create "${workspace[@]}" -f "$spec/endpoint.yml" --name "$AZ_ENDPOINT"
az ml online-deployment create "${workspace[@]}" -f "$spec/deployment.yml" \
  --endpoint-name "$AZ_ENDPOINT" \
  --set "model=azureml:ekyc-document-classifier:$AZ_MODEL_VERSION" "instance_type=$AZ_INSTANCE_TYPE"
az ml online-deployment show "${workspace[@]}" --endpoint-name "$AZ_ENDPOINT" \
  --name blue --query provisioning_state -o tsv
```

Đặt `EKYC_SCORING_URL` từ `scoring_uri`, `EKYC_SCORING_KEY` từ khóa endpoint trong môi trường của demo. Dùng `EKYC_SCORING_DEPLOYMENT=blue` để kiểm thử trực tiếp deployment. Chạy `python -m demo.server`, mở http://127.0.0.1:8080 và gửi ảnh qua demo. Sau khi kiểm tra thành công, chuyển lưu lượng và bỏ cấu hình chỉ định deployment.

```bash
az ml online-endpoint update "${workspace[@]}" --name "$AZ_ENDPOINT" --traffic "blue=100"
unset EKYC_SCORING_DEPLOYMENT
```

## Kiểm tra staging và dọn tài nguyên

Dùng một replica cho staging tạm và chỉ định deployment bằng header. Trước khi tạo, kiểm tra SKU Azure ML, quota lõi theo họ và tổng quota của vùng, cùng giá Linux. Đặt thời điểm dọn. Dùng ảnh MIDV-2020 mô phỏng thật từ tập dữ liệu. Khóa endpoint chỉ cần trong môi trường tiến trình proxy và benchmark.

Trước khi gửi request kiểm thử, orchestrator bật cả hai loại log của endpoint
và gửi về cùng một workspace Log Analytics hiện có. Đặt `AZ_LOG_ANALYTICS_ID`
là resource ID của workspace. Chế độ resource-specific cung cấp hai bảng
theo đúng tên dùng trong truy vấn. Deployment vẫn bật Application Insights.
[Giám sát endpoint Azure](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-monitor-online-endpoints?view=azureml-api-2)

```bash
endpoint_id=$(az ml online-endpoint show "${workspace[@]}" --name "$AZ_ENDPOINT" --query id -o tsv)
az monitor diagnostic-settings create --subscription "$AZ_SUBSCRIPTION" \
  --name ekyc-staging-logs --resource "$endpoint_id" --workspace "$AZ_LOG_ANALYTICS_ID" \
  --export-to-resource-specific true \
  --logs '[{"category":"AmlOnlineEndpointConsoleLog","enabled":true},{"category":"AmlOnlineEndpointTrafficLog","enabled":true}]'
```

```bash
export EKYC_SCORING_URL=$(az ml online-endpoint show "${workspace[@]}" \
  --name "$AZ_ENDPOINT" --query scoring_uri -o tsv)
export EKYC_SCORING_KEY=$(az ml online-endpoint get-credentials "${workspace[@]}" \
  --name "$AZ_ENDPOINT" --query primaryKey -o tsv)
export EKYC_SCORING_DEPLOYMENT=blue
python -m demo.server --port 8081
```

Mở http://127.0.0.1:8081, gửi ảnh và kiểm tra dự đoán cùng trạng thái HTTP. Đặt `MOCK_IMAGE` là đường dẫn ảnh, `CLIENT_LOCATION` là vị trí thực của máy đo. Đo riêng với lưu lượng trình duyệt và lưu JSON kết quả.

```bash
python -m azure.benchmark --runs 100 --client-location "$CLIENT_LOCATION" \
  --target-location japaneast scoring "$MOCK_IMAGE" --warmup 3
```

Để xác minh telemetry, dùng ảnh mô phỏng cho điểm tin cậy phân loại giấy tờ
thấp hơn ngưỡng trong bundle. Khi cả hai bảng đã nhận log kiểm thử, đặt tên
endpoint, deployment và khoảng thời gian UTC trong
[truy vấn tỷ lệ điểm tin cậy thấp](telemetry-low-confidence.kql), rồi chạy tại
workspace Log Analytics đã nhận log. Mỗi khung 15 phút UTC lấy số sự kiện
console dưới ngưỡng chia cho số request `POST /score` trả HTTP 200 trong bảng
lưu lượng. Kết quả ở dạng tỷ lệ từ phép chia và trả về null khi mẫu số bằng 0.
Khung 13:00 UTC ngày 23-11-2026 trên staging có 3 sự kiện và 108 request
thành công (2,78%). Mẫu số gồm toàn bộ lượt suy luận thành công trong khung,
kể cả warm-up và kiểm tra trình duyệt. Cờ yêu cầu duyệt cũng có thể xuất hiện
khi chất lượng ảnh kém hoặc chưa có dự đoán chất lượng. Dùng
[truy vấn lưu lượng](telemetry.kql) để xem số request, lỗi và độ trễ phía dịch vụ.

Sau bài kiểm tra, chỉ xóa endpoint staging dành riêng cho lần thử và deployment của nó. Ghi lại phản hồi Azure và thời gian UTC. Giữ bản đăng ký mô hình và minh chứng.

```bash
az ml online-endpoint delete "${workspace[@]}" --name "$AZ_ENDPOINT" --yes
az ml online-deployment show "${workspace[@]}" --endpoint-name "$AZ_ENDPOINT" \
  --name blue --query provisioning_state -o tsv
az ml online-endpoint show "${workspace[@]}" --name "$AZ_ENDPOINT" \
  --query provisioning_state -o tsv
date -u -Is
unset EKYC_SCORING_KEY EKYC_SCORING_URL EKYC_SCORING_DEPLOYMENT
```

Hai truy vấn trạng thái phải báo tài nguyên không tồn tại. Lỗi mạng hoặc xác thực cần kiểm tra lại.

## Thu thập minh chứng

Để chạy cục bộ, dùng môi trường có thư viện trong `conda-inference.yml`. Đặt `LOCAL_MODEL_DIR` thành đường dẫn tuyệt đối của bundle hoặc thư mục chứa bundle.

```bash
(cd "$spec/src/modeling" && azmlinfsrv --entry_script score.py --model_dir "$LOCAL_MODEL_DIR" --port 5001)
```

Ở shell thứ hai, đứng tại thư mục repo và đặt `EKYC_SCORING_URL=http://127.0.0.1:5001/score` để chạy demo hoặc benchmark cục bộ.

```bash
python -m azure.benchmark --runs 30 --client-location local --target-location local scoring sample.jpg
python -m azure.benchmark --runs 30 --client-location REGION_A --target-location REGION_B scoring sample.jpg
python -m azure.benchmark --runs 20 --client-location local --target-location local filesystem --directory /tmp
python -m azure.benchmark --runs 20 --client-location REGION_A --target-location REGION_B blob \
  --account-url "https://ACCOUNT.blob.core.windows.net" --container CONTAINER
```

Đổi URL suy luận theo nơi đo và bỏ khóa cloud khi chạy cục bộ.
`staging-benchmark.json` ghi phép đo từ máy học phần (Asia/Ho_Chi_Minh) đến
Japan East. Cả 3 lượt warm-up và 100 lượt đo đều thành công. Với request đã
mã hóa 317.512 byte, p50 là 1.033,41 ms và p95 là 1.467,94 ms. Hai phân vị
tính trên các lượt đo thành công, còn warm-up được ghi riêng. Độ trễ gồm
chuẩn bị request, truyền dữ liệu và giải mã phản hồi.

Để đối chứng theo vùng, dùng cùng một máy khách Azure được phê duyệt tại
Japan East để đo endpoint tại Japan East, sau đó đo endpoint ở một vùng
khác được phê duyệt và có cùng SKU `Standard_DS2_v2`. Giữ nguyên bundle,
môi trường suy luận, một replica, ảnh mô phỏng, timeout, 3 lượt warm-up và
100 request tuần tự cho mỗi đích. Lưu JSON thô, số lỗi và thời điểm của cả
hai phép đo. Lúc staging, Japan East có hạn mức 6 lõi dedicated cho họ DSv2
và 20 lõi dedicated tổng. Trước lần đo sau, kiểm tra quota còn trống ở từng
vùng, kể cả phần Azure dự trữ để nâng cấp nếu SKU yêu cầu. Nếu ngân sách
chỉ đủ một deployment staging mỗi lần, bố trí hai đợt đo nối tiếp. Nếu
vùng thứ hai thiếu quota hoặc không có SKU tương ứng, để phép đối chứng ở
trạng thái chờ.
[Hướng dẫn quota Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-quotas?view=azureml-api-2)

Chế độ Blob cần `azure-identity`, `azure-storage-blob` và quyền đọc, ghi, xóa trên container hiện có. Benchmark dùng dữ liệu tổng hợp tạm. Dùng cùng kích thước và số lượt khi so sánh. Đo filesystem có bộ đệm hệ điều hành. Đo Blob gồm gọi dịch vụ và truyền dữ liệu. Ghi vị trí máy khách, vùng, SKU, kích thước, số lượt và JSON thô.

## Kiểm thử bản phát hành

Chạy từ thư mục gốc. Benchmark nhận `--timeout` sau `scoring`. Mỗi mẫu đo một lượt gọi, gửi request mới để thử lại khi timeout.

```bash
.venv/bin/python -m unittest -v azure.test_release demo.test_release
EKYC_RELEASE_DATASET=/absolute/path/to/midv-slice .venv/bin/python -m unittest -v azure.test_release.LocalRuntimeReleaseTests
EKYC_CHECK_AZURE_SCHEMAS=1 .venv/bin/python -m unittest -v azure.test_release.AzureSchemaReleaseTests
```

Kiểm tra local cần thư viện huấn luyện, `azmlinfsrv` và một lô MIDV-2020 có manifest cùng các tập train và đánh giá. Kiểm tra huấn luyện bundle tạm và tạo xung đột cổng để kiểm tra phục hồi. Kiểm tra schema cần `jsonschema`, `PyYAML` và kết nối tới máy chủ schema của Microsoft.

## Hàng đợi duyệt ảnh mô phỏng

Bật cả ba tùy chọn hàng đợi với manifest MIDV-2020 và thư mục ảnh tương ứng. Hàng đợi nằm trên filesystem Linux cục bộ và chỉ chủ thư mục có quyền truy cập.

```bash
queue_dir=$(mktemp -d)
.venv/bin/python -m demo.server --port 8081 \
  --review-queue "$queue_dir" \
  --review-manifest "$MIDV_ROOT/manifest.csv" \
  --review-dataset-root "$MIDV_ROOT"
```

Mở http://127.0.0.1:8081 và gửi JPEG hoặc PNG có trong manifest. Khi khởi động, demo tính SHA-256 từng ảnh và dùng `image_id` làm mã nguồn. Dự đoán hợp lệ với `needs_review=true` tự vào hàng đợi. Nút báo sai có trên ảnh đủ điều kiện ở mọi mức confidence. Ảnh khác vẫn có thể suy luận. Trình nhập nhãn truy cập filesystem riêng có kiểm soát.

### API demo

- `POST /predict` nhận byte JPEG hoặc PNG. HTTP 200 trả `label`, `status`, `confidence` và `needs_review`. Nếu bật review, header `X-EKYC-Review-Status` là `queued`, `available` hoặc `not_allowlisted`. Phản hồi đủ điều kiện có `X-EKYC-Report-Token`.
- `POST /report` chỉ nhận JSON `{"token":"<X-EKYC-Report-Token>"}`. HTTP 200 trả `{"event_id":"<sha256>","status":"queued"}`. Demo ký digest và dự đoán. HTTP 403 báo mã sai. Suy luận lại sau khi khởi động lại để có mã mới. HTTP 400 hoặc 413 báo dữ liệu sai hoặc quá lớn.
- HTTP 503 báo hàng đợi lỗi. Gửi lại request gốc. Nhiều lượt gửi cùng digest tạo một sự kiện, mỗi lý do được thêm một lần. Nếu tắt review, `/predict` hoạt động bình thường và `/report` trả 404.

Trình nhập đọc `events/<sha256>/event.json` đã hoàn tất. Sự kiện gồm `event_id`, `image_sha256`, `image_file` (tệp `image.jpg` hoặc `image.png` bên cạnh), `source_image_id`, bốn trường `prediction` và `reasons` gồm `needs_review`, `reported_incorrect` hoặc cả hai. Dự đoán đầu tiên được giữ, lý do bổ sung được cập nhật nguyên tử. Ảnh và metadata được xuất bản cùng nhau, tệp tạm ở ngoài đường dẫn digest hoàn tất.

```bash
.venv/bin/python -m unittest -v demo.test_review_queue demo.test_release
```
