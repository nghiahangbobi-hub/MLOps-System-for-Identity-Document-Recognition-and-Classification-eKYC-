# Azure Blob Storage từ đầu

[English](blob-storage.en.md) · **Tiếng Việt**

Dùng tài khoản **Azure for Students** của nhóm. Lần đầu hãy thử với một lô
MIDV-2020 nhỏ. Quy trình này lưu ảnh đã trích và manifest, rồi đăng ký hai
asset riêng cho Azure ML.

## 1. Tạo workspace và tìm container

1. Đăng nhập [Azure Portal](https://portal.azure.com/), mở **Subscriptions**
   và kiểm tra Azure for Students đang hoạt động.
2. Tìm **Azure Machine Learning**, chọn **Create**, chọn subscription trên,
   tạo resource group riêng cho đồ án và đặt tên workspace. Chọn vùng được
   subscription cho phép và kiểm tra quota compute. Nếu workspace không tạo
   được Storage account, tạo riêng account Standard, LRS trong cùng vùng,
   rồi chọn account đó khi tạo workspace. Với Azure for Students, lỗi
   `RequestDisallowedByAzure` kèm thông báo về vùng yêu cầu chọn một vùng
   được phép trong policy của subscription.
3. Mở **Launch studio → Data → Datastores → workspaceblobstore**. Ghi tên
   storage account và container từ datastore. Trong Portal, mở storage account
   tương ứng, chọn **Data storage → Containers** để xem container riêng tư.
4. Trong **Access control (IAM)** của storage account hoặc container, gán
   **Storage Blob Data Contributor** cho tài khoản thành viên phụ trách
   tải dữ liệu. Nếu không có quyền gán role, nhờ chủ subscription cấp quyền.
   Nếu role vừa cấp mà thao tác báo 403, chờ quyền được áp dụng rồi thử lại.

## 2. Tải ảnh và kiểm tra trên Blob

Nếu `az` chưa có trong PATH trên máy đồ án, chạy
`export PATH="$PWD/.venv/azure-cli/bin:$PATH"` từ thư mục repo.

Chạy lệnh trong Bash trên máy có `clips.tar`. Cloud Shell không nhìn thấy
đường dẫn trên máy bạn. Nếu dùng Portal để upload thủ công, giữ nguyên
cấu trúc thư mục và tên file như trong manifest.

```bash
az login --use-device-code
az account list -o table
export AZ_SUBSCRIPTION='<subscription ID>'
export AZ_RESOURCE_GROUP='<resource group>'
export AZ_WORKSPACE='<workspace>'
az account set --subscription "$AZ_SUBSCRIPTION"
az extension add --name ml
source .venv/bin/activate  # Hoặc kích hoạt venv Python của bạn
python -m pip install -r data_pipeline/requirements.txt

export BLOB_ACCOUNT='<storage account backing workspaceblobstore>'
export BLOB_CONTAINER='<container backing workspaceblobstore>'
export MIDV_ARCHIVE="$HOME/Documents/Datasets/MIDV2020/MIDV2020/MIDV2020/dataset/clips.tar"
export DATA_VERSION="$(date -u +%Y%m%d%H%M%S)"
export BLOB_PREFIX="ekyc/midv/$DATA_VERSION"
local_run=$(mktemp -d)
az storage blob list --auth-mode login --account-name "$BLOB_ACCOUNT" \
  --container-name "$BLOB_CONTAINER" --num-results 1 -o none
python -m data_pipeline.midv2020 --source "$MIDV_ARCHIVE" \
  --output "$local_run/dataset" --max-clips-per-template 2 --frames-per-clip 3 --min-frame-gap 10

az storage blob upload-batch --auth-mode login --account-name "$BLOB_ACCOUNT" \
  --destination "$BLOB_CONTAINER" --destination-path "$BLOB_PREFIX" \
  --source "$local_run/dataset"
az storage blob list --auth-mode login --account-name "$BLOB_ACCOUNT" \
  --container-name "$BLOB_CONTAINER" --prefix "$BLOB_PREFIX/" \
  --query '[].name' -o tsv
```

Trong Portal, mở prefix vừa tạo. Kiểm tra thấy JPEG và `manifest.csv`, mở
hoặc tải thử một JPEG và lưu ảnh chụp màn hình danh sách blob làm minh chứng.
Giữ riêng từng prefix phiên bản để các job cũ tiếp tục trỏ đến dữ liệu cũ.

### Nếu upload báo thiếu quyền

Tài khoản đang đăng nhập cần quyền truy cập dữ liệu Blob. Nếu bạn có quyền
gán role, chạy các lệnh sau để gán cho chính tài khoản đã đăng nhập, giới hạn
ở container dự án. Có thể dùng Portal IAM ở bước 1 thay cho các lệnh này.

```bash
account_id=$(az storage account show -g "$AZ_RESOURCE_GROUP" -n "$BLOB_ACCOUNT" --query id -o tsv)
object_id=$(az ad signed-in-user show --query id -o tsv)
scope="$account_id/blobServices/default/containers/$BLOB_CONTAINER"
az role assignment create --assignee-object-id "$object_id" \
  --assignee-principal-type User --role 'Storage Blob Data Contributor' \
  --scope "$scope"
```

Đợi Azure áp dụng role và chạy lại lệnh kiểm tra Blob trước khi upload.
Để xác minh manifest, tải xuống và so sánh với tệp gốc:

```bash
az storage blob download --auth-mode login --account-name "$BLOB_ACCOUNT" \
  --container-name "$BLOB_CONTAINER" --name "$BLOB_PREFIX/manifest.csv" \
  --file "$local_run/manifest-from-blob.csv" -o none
cmp "$local_run/dataset/manifest.csv" "$local_run/manifest-from-blob.csv"
```

## 3. Đăng ký dữ liệu cho Azure ML

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

Kiểm tra hai asset tại **Azure ML Studio → Data → Data assets**. Asset tồn
tại xác nhận đăng ký thành công. Job train đạt `Completed` mới xác nhận compute
đọc được dữ liệu. Nếu job báo lỗi quyền Blob, kiểm tra danh tính mà job dùng
để mount dữ liệu và cấp quyền đọc ở phạm vi container phù hợp. Tiếp theo dùng
[runbook huấn luyện và triển khai](README.md) trong cùng phiên Bash.

## 4. Sử dụng lại cho Label Studio và benchmark

Dùng [AzCopy với Microsoft Entra][azcopy] để tải lô ảnh cần gán nhãn. Giữ ảnh
và bản xuất JSON trong Blob riêng tư. Sau khi người duyệt phê duyệt, lưu bản
xuất và lineage rồi đăng ký manifest mới dưới một phiên bản `uri_file`.
Xem [luồng duyệt và huấn luyện thủ công](../docs/hitl.md). Prefix 4–5 GB dùng
cho ETL và benchmark có thể tạo sau khi lô nhỏ chạy thành công.

[azcopy]: https://learn.microsoft.com/azure/storage/common/storage-use-azcopy-authorize-user-identity

Tài liệu Microsoft: [Azure CLI Blob](https://learn.microsoft.com/azure/storage/blobs/blob-cli)
và [Azure ML data assets](https://learn.microsoft.com/cli/azure/ml/data).
