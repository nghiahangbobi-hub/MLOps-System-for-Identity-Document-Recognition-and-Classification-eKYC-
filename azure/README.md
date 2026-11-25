# Triển khai trên Azure ML

Chạy từ thư mục gốc bằng Bash, trong môi trường Python ở [README](../README.md).
Cần Azure CLI, workspace Azure ML sẵn có và quyền đọc/ghi Blob cho người tải
lên. Danh tính chạy job cần quyền đọc dữ liệu. Các lệnh minh họa dùng
datastore `workspaceblobstore` và ảnh giấy tờ mô phỏng.

## 1. Đăng ký dữ liệu

Dùng manifest đã duyệt theo [quy trình Bài tập 13](../WORKFLOW.md).
Chọn thư mục ảnh gốc và phiên bản Blob mới. Với các vòng HITL dùng cùng bộ ảnh,
có thể dùng lại asset ảnh và đăng ký manifest mới.

```bash
az login --use-device-code
az account list -o table
export AZ_SUBSCRIPTION='<subscription ID>'
export AZ_RESOURCE_GROUP='<resource group>'
export AZ_WORKSPACE='<workspace>'
az account set --subscription "$AZ_SUBSCRIPTION"
az extension add --name ml

export BLOB_ACCOUNT='<storage account backing workspaceblobstore>'
export BLOB_CONTAINER='<container backing workspaceblobstore>'
export DATASET_ROOT="Data/MIDV2020_sample_5GB"
export REVIEWED_MANIFEST="runs/assignment13/labeling/reviewed-v1.csv"
export DATA_VERSION="$(date -u +%Y%m%d%H%M%S)"
export BLOB_PREFIX="ekyc/midv/$DATA_VERSION"
local_run=$(mktemp -d)
az storage blob list --auth-mode login --account-name "$BLOB_ACCOUNT" \
  --container-name "$BLOB_CONTAINER" --num-results 1 -o none
python -m data_pipeline.stage --dataset-root "$DATASET_ROOT" \
  --manifest "$REVIEWED_MANIFEST" --output "$local_run/dataset"

az storage blob upload-batch --auth-mode login --account-name "$BLOB_ACCOUNT" \
  --destination "$BLOB_CONTAINER" --destination-path "$BLOB_PREFIX" \
  --source "$local_run/dataset"
az storage blob list --auth-mode login --account-name "$BLOB_ACCOUNT" \
  --container-name "$BLOB_CONTAINER" --prefix "$BLOB_PREFIX/" \
  --query '[].name' -o tsv
```

```bash
workspace=(--subscription "$AZ_SUBSCRIPTION" --resource-group "$AZ_RESOURCE_GROUP" --workspace-name "$AZ_WORKSPACE")
data_uri="azureml://datastores/workspaceblobstore/paths/$BLOB_PREFIX"
az ml data create "${workspace[@]}" --name ekyc-midv-frames --version "$DATA_VERSION" \
  --type uri_folder --path "$data_uri/"
az ml data create "${workspace[@]}" --name ekyc-midv-manifest --version "$DATA_VERSION" \
  --type uri_file --path "$data_uri/manifest.csv"
az ml data show "${workspace[@]}" --name ekyc-midv-frames --version "$DATA_VERSION" -o table
az ml data show "${workspace[@]}" --name ekyc-midv-manifest --version "$DATA_VERSION" -o table
export AZ_DATASET_ASSET="azureml:ekyc-midv-frames:$DATA_VERSION"
export AZ_MANIFEST_ASSET="azureml:ekyc-midv-manifest:$DATA_VERSION"
```

## 2. Gửi job huấn luyện

Dùng compute sẵn có hoặc tạo compute theo quota vùng của workspace.
Ví dụ sau dùng Japan East. Chờ compute ở trạng thái `Succeeded`.

```bash
workspace=(--subscription "$AZ_SUBSCRIPTION" --resource-group "$AZ_RESOURCE_GROUP" --workspace-name "$AZ_WORKSPACE")
az ml compute list-usage "${workspace[@]}" --location japaneast -o table
export AZ_COMPUTE=ekyc-cpu
az ml compute create "${workspace[@]}" --name "$AZ_COMPUTE" --type amlcompute \
  --size Standard_D2s_v3 --min-instances 0 --max-instances 1 \
  --idle-time-before-scale-down 120
az ml compute show "${workspace[@]}" --name "$AZ_COMPUTE" --query provisioning_state -o tsv
```

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

Sau một vòng duyệt, đăng ký manifest mới thành data asset `uri_file` với
phiên bản mới, cập nhật `AZ_MANIFEST_ASSET` rồi gửi job thủ công. Dùng lại
asset thư mục ảnh gốc. Job dùng MobileNetV3 pretrained, chọn trên validation
và đánh giá test sau khi khóa checkpoint. Kết quả gồm output `evidence` và
`model_bundle` cho ứng viên qua kiểm tra offline.

## 3. Triển khai và kiểm tra

Sau khi job `Completed`, tải output `evidence`, tìm thư mục chứa
`workflow.json` và kiểm tra hash cùng kết luận của ứng viên. [Lệnh tải output của Azure CLI](https://learn.microsoft.com/cli/azure/ml/job#az-ml-job-download)

```bash
az ml job download "${workspace[@]}" --name "$job" --output-name evidence \
  --download-path "$work_dir/evaluation"
evidence=$(python -c 'from pathlib import Path; import sys; print(next(Path(sys.argv[1]).rglob("workflow.json")).parent)' "$work_dir/evaluation")
python -m modeling.workflow check-release --run "$evidence"
```

Khi kiểm tra qua, đặt `AZ_MODEL_VERSION` thành phiên bản chưa dùng,
`AZ_ENDPOINT` thành tên endpoint và `AZ_INSTANCE_TYPE` thành SKU có quota.
Tạo deployment staging ở 0% lưu lượng. Nếu muốn dùng ứng viên huấn luyện cục bộ,
chạy cùng lệnh `check-release` và đăng ký thư mục bundle đã kiểm tra.

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

```bash
export EKYC_SCORING_URL=$(az ml online-endpoint show "${workspace[@]}" \
  --name "$AZ_ENDPOINT" --query scoring_uri -o tsv)
export EKYC_SCORING_KEY=$(az ml online-endpoint get-credentials "${workspace[@]}" \
  --name "$AZ_ENDPOINT" --query primaryKey -o tsv)
export EKYC_SCORING_DEPLOYMENT=blue
python -m demo.server --port 8081
```

Trong menu demo, chọn **Vận hành** để xem số liệu. Để bật số liệu Azure,
cấu hình diagnostic logs theo phần Monitoring bên dưới. Sau đó đặt
`EKYC_LOG_ANALYTICS_WORKSPACE` thành **Workspace ID** (GUID), export `AZ_ENDPOINT`
và `EKYC_SCORING_DEPLOYMENT` (mặc định `blue`) trước khi chạy demo.
Nút **Tải tỷ lệ độ tin cậy thấp** dùng Azure CLI đã đăng nhập và có quyền đọc workspace.

Mở http://127.0.0.1:8081, gửi ảnh và kiểm tra dự đoán cùng trạng thái HTTP. Đặt `MOCK_IMAGE` là đường dẫn ảnh, `CLIENT_LOCATION` là vị trí thực của máy đo. Đo riêng với lưu lượng trình duyệt và lưu JSON kết quả.

```bash
python -m azure.benchmark --runs 100 --client-location "$CLIENT_LOCATION" \
  --target-location japaneast scoring "$MOCK_IMAGE" --warmup 3 > staging-benchmark.json
```

Xem dự đoán mẫu và `staging-benchmark.json` trước khi chuyển lưu lượng.
Benchmark trả mã lỗi nếu bất kỳ lượt đo nào thất bại hoặc vượt 3 giây.
Mở lại demo với cấu hình mới nếu tiến trình trước đó còn chạy.

```bash
az ml online-endpoint update "${workspace[@]}" --name "$AZ_ENDPOINT" --traffic "blue=100"
unset EKYC_SCORING_DEPLOYMENT
```

Bật diagnostic settings cho console và traffic log khi cần theo dõi bằng
[telemetry.kql](telemetry.kql) và [telemetry-low-confidence.kql](telemetry-low-confidence.kql).
Dọn endpoint thử nghiệm sau khi sử dụng.

```bash
az ml online-endpoint delete "${workspace[@]}" --name "$AZ_ENDPOINT" --yes
unset EKYC_SCORING_URL EKYC_SCORING_KEY EKYC_SCORING_DEPLOYMENT
```
