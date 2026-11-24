# Set up Azure Blob Storage

**English** · [Tiếng Việt](blob-storage.md)

Use your team's Azure for Students subscription. Start with a small MIDV-2020 slice. Upload the extracted images and manifest, then register them as separate Azure ML data assets.

## 1. Create a workspace and find its container

1. Sign in to [Azure Portal](https://portal.azure.com/) and verify the subscription is active under **Subscriptions**.
2. Open **Azure Machine Learning**, select **Create**, choose the subscription, and create a resource group and workspace for the project. Select an allowed region with compute quota. If automatic storage creation fails, create a Standard, LRS storage account in the same region and select it in the workspace form. If `RequestDisallowedByAzure` reports a region policy, select a region allowed by the subscription.
3. In **Launch studio → Data → Datastores → workspaceblobstore**, find the storage account and container names. In Azure Portal, find the private container under the storage account's **Data storage → Containers**.
4. Use **Access control (IAM)** on the container or account to give the uploader **Storage Blob Data Contributor**. Ask the subscription owner if you lack permission to assign it. If a recent grant returns HTTP 403, wait for propagation and retry.

## 2. Upload and verify images in Blob Storage

If `az` is missing from PATH, activate your Azure CLI environment, for example `export PATH="$PWD/.venv/azure-cli/bin:$PATH"` from the repo root.

Run Bash on the machine containing `clips.tar`. Azure Cloud Shell cannot read your local filesystem. If uploading in the portal, preserve the folder layout and file names expected by the manifest.

```bash
az login --use-device-code
az account list -o table
export AZ_SUBSCRIPTION='<subscription ID>'
export AZ_RESOURCE_GROUP='<resource group>'
export AZ_WORKSPACE='<workspace>'
az account set --subscription "$AZ_SUBSCRIPTION"
az extension add --name ml
source .venv/bin/activate  # Or activate your Python virtual environment
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

Open the new prefix in the portal. Check for JPEGs and `manifest.csv`, open or download a sample JPEG, and save a private screenshot of the blob listing as evidence. Use a separate prefix per version so older jobs still resolve their inputs.

### If upload is denied

The signed-in user needs Blob data access. If allowed to assign roles, scope the assignment to the project's container. You can also use Portal IAM.

```bash
account_id=$(az storage account show -g "$AZ_RESOURCE_GROUP" -n "$BLOB_ACCOUNT" --query id -o tsv)
object_id=$(az ad signed-in-user show --query id -o tsv)
scope="$account_id/blobServices/default/containers/$BLOB_CONTAINER"
az role assignment create --assignee-object-id "$object_id" \
  --assignee-principal-type User --role 'Storage Blob Data Contributor' \
  --scope "$scope"
```

Wait for the grant to propagate and retry the Blob access check before uploading. Download the uploaded manifest and compare it with the original.

```bash
az storage blob download --auth-mode login --account-name "$BLOB_ACCOUNT" \
  --container-name "$BLOB_CONTAINER" --name "$BLOB_PREFIX/manifest.csv" \
  --file "$local_run/manifest-from-blob.csv" -o none
cmp "$local_run/dataset/manifest.csv" "$local_run/manifest-from-blob.csv"
```

## 3. Register Azure ML data assets

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

Check both entries under **Azure ML Studio → Data → Data assets**. A training job reaching `Completed` confirms compute was able to read the data. If access fails, check the identity that mounts the inputs and grant read access at the appropriate container scope. Continue from the same Bash session. [Azure training and deployment](README.md)

## 4. Reuse data for Label Studio and benchmarks

Download the labeling slice with Microsoft Entra authentication in AzCopy. Keep images and exported JSON in private storage. After the human approves the reviews, retain the export and lineage and register the published manifest as a new `uri_file` version. Create a larger, 4–5 GB ETL and benchmark prefix after the small slice succeeds. [AzCopy authentication](https://learn.microsoft.com/azure/storage/common/storage-use-azcopy-authorize-user-identity) · [Human-in-the-loop](../docs/hitl.md)

[Azure CLI Blob](https://learn.microsoft.com/azure/storage/blobs/blob-cli) · [Azure ML data assets](https://learn.microsoft.com/cli/azure/ml/data)
