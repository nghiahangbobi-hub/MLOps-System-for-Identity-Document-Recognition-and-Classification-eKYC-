# Azure clean-room runbook / Quy trình Azure clean-room

## Gate and scope / Điều kiện và phạm vi

**EN.** For now, run section 1 only. Sections 2–5 require Root to confirm the 5 GB download is complete and approve the dataset, workspace, region, quota and spending window in the orchestrator checkpoint. Keep active downloads and existing jobs, endpoints and shared assets intact. Each attempt gets its own Blob prefix, experiment, job, data asset version and staging endpoint. Preserve the verification record and retain the existing workspace and resource group.

**VI.** Hiện chỉ thực hiện mục 1. Chỉ làm mục 2–5 sau khi Root xác nhận tải xong gói 5 GB và duyệt bộ dữ liệu, workspace, vùng, quota và khung chi phí trong checkpoint orchestrator. Giữ nguyên các lượt tải đang chạy, job, endpoint và tài nguyên dùng chung hiện có. Mỗi lần chạy có prefix Blob, experiment, job, phiên bản data asset và staging endpoint riêng. Lưu hồ sơ xác minh và giữ workspace, resource group hiện có.

## 1. Read-only inventory / Kiểm kê chỉ đọc

Run in Bash from the repo root / Chạy Bash từ thư mục gốc của repo.

```bash
export AZ_SUBSCRIPTION='<approved subscription ID>'
export AZ_RESOURCE_GROUP='<approved resource group>'
export AZ_WORKSPACE='<approved AML workspace>'
# Optional / Tùy chọn: export AZ_BIN=/path/to/project-local/bin/az
record=$(bash azure/clean-room-inventory.sh) || exit 1
az() { "${AZ_BIN:-az}" "$@"; }
workspace=(--subscription "$AZ_SUBSCRIPTION" -g "$AZ_RESOURCE_GROUP" -w "$AZ_WORKSPACE")
```

**EN.** Inventory records installed CLI versions, workspace and region, paginated current and archived jobs, compute, endpoints, datastore mappings, asset listings and regional quota. The output is a private directory outside the repository. Review `status.txt` and command stderr files before using it. Compare quota usage and limits for the training and endpoint SKU families, total dedicated cores, and online deployment upgrade reserve where applicable. Quota is a snapshot. Confirm capacity again before allocation.

**VI.** Kiểm kê ghi phiên bản CLI, workspace và vùng, danh sách phân trang các job hiện hành và đã archive, compute, endpoint, ánh xạ datastore, các asset và quota theo vùng. Đầu ra nằm trong thư mục riêng ngoài repo. Kiểm tra `status.txt` và các tệp stderr. Đối chiếu mức dùng với giới hạn theo họ SKU dùng để huấn luyện và triển khai, tổng số lõi dedicated và phần dự trữ nâng cấp của online deployment nếu áp dụng. Kiểm tra lại dung lượng khả dụng ngay trước khi cấp phát.

[Azure ML compute CLI](https://learn.microsoft.com/cli/azure/ml/compute) · [Azure ML quotas](https://learn.microsoft.com/azure/machine-learning/how-to-manage-quotas) · [Managed online endpoints](https://learn.microsoft.com/azure/machine-learning/concept-endpoints-online)

## 2. Prepare and publish / Chuẩn bị và đưa dữ liệu lên Blob

**EN.** After Root signs off, verify the finished 5 GB input against its trusted checksum when available, record SHA-256, byte count, source, license and archive layout, and check integrity with the matching archive tool. Review member paths before extraction. Use the verified layout and approved ETL to build a private dataset with `manifest.csv`. If the input is the MIDV-2020 clips tar, use the command below. Check template separation and sample counts per split/class, and record exact ETL options, Git revision and dirty changes. Keep any quality labels tied to approved review or tracked train-only augmentations. Upload the prepared dataset only. The 5 GB source remains private and read-only.

**VI.** Sau khi Root duyệt, so checksum tin cậy nếu có, ghi SHA-256, số byte, nguồn, giấy phép và cấu trúc gói 5 GB đã tải xong, rồi kiểm tra tính toàn vẹn bằng công cụ đúng định dạng. Kiểm tra đường dẫn trong gói trước khi giải nén. Dùng cấu trúc đã xác minh và ETL đã duyệt để tạo bộ dữ liệu riêng có `manifest.csv`. Nếu đầu vào là tar MIDV-2020 clips, dùng lệnh dưới đây. Kiểm tra tách biệt template và số mẫu theo split/lớp, ghi tham số ETL, Git revision và phần thay đổi chưa commit. Nhãn chất lượng phải có nguồn review đã duyệt hoặc augmentation chỉ dùng trong train có ghi vết. Chỉ tải lên bộ dữ liệu đã chuẩn bị. Gói nguồn 5 GB giữ riêng tư và chỉ đọc.

```bash
set -euo pipefail
umask 077
run="$(date -u +%Y%m%d%H%M%S)-$(openssl rand -hex 4)"
prefix="ekyc/clean-room/$run"
job="cr-$run"
endpoint="cr-$run"  # Confirm availability before create / Kiểm tra tên trước khi tạo
experiment="ekyc-clean-room-$run"
archive='<verified private path to completed MIDV clips tar>'
data="$record/dataset"
account='<approved storage account backing the datastore>'
container='<approved container>'
datastore='<approved workspace datastore>'
sha256sum "$archive" > "$record/input.sha256"
stat -c '%s' "$archive" > "$record/input-bytes.txt"
tar -tf "$archive" > "$record/archive-members.txt"
# Review member list and approve the ETL limits below before continuing.
# Duyệt danh mục trong gói và giới hạn ETL trước khi tiếp tục.
python -m data_pipeline.midv2020 --source "$archive" --output "$data" \
  --max-clips-per-template 2 --frames-per-clip 3 --min-frame-gap 10
```

**EN.** The command selects a bounded smoke slice. A full-scale run requires separately agreed sampling limits, runtime and budget. After local checks, confirm the datastore maps to the approved account/container, then publish into the empty prefix.

**VI.** Lệnh trên lấy một tập mẫu nhỏ để kiểm tra luồng. Chạy toàn bộ cần thống nhất riêng giới hạn lấy mẫu, thời gian và ngân sách. Sau các bước kiểm tra cục bộ, xác nhận datastore trỏ đến đúng account/container đã duyệt, rồi đưa dữ liệu lên prefix trống.

```bash
blob=(--subscription "$AZ_SUBSCRIPTION" --auth-mode login --account-name "$account")
[[ $(az storage blob list "${blob[@]}" --container-name "$container" \
  --prefix "$prefix/" --num-results 1 --query 'length(@)' -o tsv) == 0 ]]
az storage blob upload-batch "${blob[@]}" --destination "$container" \
  --destination-path "$prefix" --source "$data" --overwrite false -o none
az storage blob download "${blob[@]}" --container-name "$container" \
  --name "$prefix/manifest.csv" --file "$record/blob-manifest.csv" -o none
cmp "$data/manifest.csv" "$record/blob-manifest.csv"
sha256sum "$data/manifest.csv" "$record/blob-manifest.csv" > "$record/manifests.sha256"
uri="azureml://datastores/$datastore/paths/$prefix"
az ml data create "${workspace[@]}" --name ekyc-clean-room-frames --version "$run" \
  --type uri_folder --path "$uri/" > "$record/frames-asset.json"
az ml data create "${workspace[@]}" --name ekyc-clean-room-manifest --version "$run" \
  --type uri_file --path "$uri/manifest.csv" > "$record/manifest-asset.json"
```

## 3. Train / Huấn luyện

**EN.** Reuse approved AML compute if it already has min 0 and an approved cap. Otherwise use an approved new dedicated cluster with min 0, max 1 and a 120-second idle scale-down. Verify the configuration and provisioning state before submission. Stage from the verified source checkout. Reuse an exact verified training/inference environment version or register a new version from the staged specs as approved. Capture the submitted job, terminal status, MLflow metrics, split counts and verified output hashes. If submission times out, use `job show` on the recorded name before retrying. On terminal failure retain logs and use a fresh run ID for a separately approved retry.

**VI.** Dùng lại AML compute đã duyệt nếu có min 0 và giới hạn phù hợp. Nếu không, dùng cluster riêng mới đã duyệt với min 0, max 1 và 120 giây chờ scale-down. Kiểm tra cấu hình và trạng thái cấp phát trước khi submit. Stage từ mã nguồn đã xác minh. Dùng đúng phiên bản môi trường train/inference đã kiểm chứng hoặc đăng ký phiên bản mới từ cấu hình stage theo phê duyệt. Lưu job đã gửi, trạng thái kết thúc, chỉ số MLflow, số mẫu theo split và hash đầu ra đã xác minh. Nếu submit hết thời gian chờ, tra `job show` theo tên đã lưu trước khi thử lại. Nếu job kết thúc lỗi, giữ log và dùng run ID mới cho lần thử lại được duyệt.

```bash
compute='<approved compute name>'
train_env='azureml:<approved training environment>:<version>'
# Only for an approved NEW cluster / Chỉ khi tạo cluster MỚI đã duyệt:
# az ml compute create "${workspace[@]}" -n "$compute" --type amlcompute \
#   --size '<approved training SKU>' --min-instances 0 --max-instances 1 \
#   --idle-time-before-scale-down 120
az ml compute show "${workspace[@]}" -n "$compute" > "$record/training-compute.json"
python azure/stage.py "$record/aml"
printf '%s\n' "$run" "$prefix" "$experiment" "$job" "$endpoint" > "$record/run-targets.txt"
az ml job create "${workspace[@]}" -f "$record/aml/train.yml" \
  --set "name=$job" "experiment_name=$experiment" "compute=azureml:$compute" \
    "environment=$train_env" "inputs.dataset.path=azureml:ekyc-clean-room-frames:$run" \
    "inputs.manifest.path=azureml:ekyc-clean-room-manifest:$run" \
  > "$record/job-submit.json"
az ml job show "${workspace[@]}" -n "$job" > "$record/job-status.json"
```

Download evidence after `Completed`. If named-output download fails, use the authenticated fallback in the Azure README. Verify the bundle locally before registration. / Tải minh chứng khi job đạt `Completed`. Nếu tải named output lỗi, dùng cách xác thực dự phòng trong Azure README. Xác minh bundle cục bộ trước khi đăng ký. [English](README.md) · [Tiếng Việt](README.vi.md)

## 4. Stage, verify and clean up / Triển khai thử, xác minh và dọn dẹp

**EN.** Reserve a bounded deployment window and assign a cleanup owner before creation. Confirm the name is unused using a successful endpoint list and use the unique name exclusively for this run. Register the verified job output under a fresh version. Deploy `blue` with one replica and the approved regional SKU. Run the targeted mock-image smoke and, if approved, the documented traffic/telemetry and benchmark checks. Keep credentials only in process environment and avoid shell tracing. Save results, UTC timestamps, scoring configuration, errors and cost evidence privately.

**VI.** Chốt thời gian triển khai tối đa và người dọn dẹp trước khi tạo. Kiểm tra tên chưa được dùng bằng lệnh liệt kê endpoint thành công rồi dành riêng tên đó cho lượt chạy. Đăng ký output đã xác minh dưới một phiên bản mới. Triển khai `blue` với một replica theo SKU vùng đã duyệt. Kiểm thử có chỉ định deployment bằng ảnh giả lập và, nếu được duyệt, kiểm tra định tuyến, telemetry và benchmark theo README. Giữ credentials trong môi trường tiến trình và tắt shell tracing. Lưu riêng kết quả, mốc UTC, cấu hình scoring, lỗi và minh chứng chi phí.

```bash
inference_env='azureml:<approved inference environment>:<version>'
instance='<approved regional endpoint SKU>'
[[ $(az ml job show "${workspace[@]}" -n "$job" --query status -o tsv) == Completed ]]
matches=$(az ml online-endpoint list "${workspace[@]}" --query "[?name=='$endpoint'].name" -o tsv)
[[ -z $matches ]]
az ml model create "${workspace[@]}" --name ekyc-clean-room-model --version "$run" \
  --type custom_model --path "azureml://jobs/$job/outputs/model_bundle" > "$record/model.json"
az ml online-endpoint create "${workspace[@]}" -f "$record/aml/endpoint.yml" \
  --name "$endpoint" > "$record/endpoint.json"
az ml online-deployment create "${workspace[@]}" -f "$record/aml/deployment.yml" \
  --endpoint-name "$endpoint" \
  --set "model=azureml:ekyc-clean-room-model:$run" "environment=$inference_env" \
     "instance_type=$instance" instance_count=1 > "$record/deployment.json"
```

**EN.** Run cleanup from a separate shell if deployment or testing fails or the window expires. Reconstruct scope and endpoint from the private record, and confirm ownership against the create record or Azure activity log before deletion, including after a partial create. Capture a successful post-delete list with no matching endpoint, confirm the run's deployment is absent, and check training compute nodes scaled to zero when idle. Record follow-up if an API call fails or resources remain. Retain Blob inputs, registered assets and job artifacts for reproducibility under the approved retention policy.

**VI.** Chạy dọn dẹp từ shell riêng nếu triển khai hoặc kiểm thử lỗi hoặc hết khung thời gian. Khôi phục scope và tên endpoint từ hồ sơ riêng, xác nhận quyền sở hữu theo hồ sơ tạo hoặc Azure Activity Log trước khi xóa, kể cả khi tạo dở. Lưu kết quả lệnh liệt kê thành công sau khi xóa, xác nhận endpoint và deployment của lượt chạy đã biến mất, rồi kiểm tra training compute đã scale về 0 node khi rảnh. Ghi việc cần xử lý nếu API lỗi hoặc còn tài nguyên. Giữ Blob đầu vào, asset đã đăng ký và artifact job theo chính sách lưu giữ đã duyệt.

```bash
az ml online-endpoint delete "${workspace[@]}" --name "$endpoint" --yes
az ml online-endpoint list "${workspace[@]}" --query "[?name=='$endpoint']" \
  > "$record/endpoint-after-cleanup.json"
python -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])) != [])' \
  "$record/endpoint-after-cleanup.json"
az ml compute list-nodes "${workspace[@]}" --name "$compute" > "$record/compute-nodes-after.json"
date -u +%FT%TZ > "$record/cleanup-checked-utc.txt"
```

## 5. Retention and handoff / Lưu giữ và bàn giao

**EN.** Azure ML CLI job lifecycle supports archive and restore. Archive hides completed jobs from the default listing and preserves their usability. The CLI has no job delete command. Leave prior runs intact. Only archive an explicitly approved completed job after Root confirms its downloads and verification record are complete. Use `az ml job archive "${workspace[@]}" --name "$job"` after approval and `az ml job restore "${workspace[@]}" --name "$job"` to reverse it when authorized. Cancel requires a separate decision for the specific running job. Keep raw records, dataset, model, endpoint URLs and credentials off GitHub. Give Root the private record path, run status, pending checks and billed cost if available.

**VI.** Azure ML CLI hỗ trợ archive và restore job. Archive ẩn job đã hoàn thành khỏi danh sách mặc định và vẫn giữ khả năng tham chiếu. CLI không có lệnh xóa job. Giữ nguyên các lượt cũ. Chỉ archive job đã hoàn thành được chỉ định sau khi Root xác nhận đã tải đủ và lưu hồ sơ xác minh. Dùng `az ml job archive "${workspace[@]}" --name "$job"` sau khi duyệt và `az ml job restore "${workspace[@]}" --name "$job"` để khôi phục khi được phép. Cancel cần quyết định riêng cho đúng job đang chạy. Không đưa hồ sơ thô, dữ liệu, mô hình, URL endpoint hay credentials lên GitHub. Báo Root đường dẫn hồ sơ riêng, trạng thái, mục cần xác minh và chi phí đã tính nếu có.

[Azure ML job CLI](https://learn.microsoft.com/cli/azure/ml/job)
