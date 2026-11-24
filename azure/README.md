# Azure ML handoff

**English** · [Tiếng Việt](README.vi.md)

Start with the [Azure Blob Storage setup guide](blob-storage.en.md) if you still need a workspace or data assets.

Run commands from the repository root. The orchestrator owns cloud changes.
Use an authenticated Azure CLI with the `ml` extension, an existing workspace,
CPU training compute, dataset folder and manifest file assets, and an available
managed endpoint SKU. Set `AZ_SUBSCRIPTION`, `AZ_RESOURCE_GROUP`,
`AZ_WORKSPACE`, `AZ_COMPUTE`, `AZ_DATASET_ASSET`, `AZ_MANIFEST_ASSET` and
`AZ_ENDPOINT`. Asset variables use the form `azureml:name:version`.
The mounted dataset's layout must match the manifest's relative `image_path`
values. Use a reviewed manifest for quality evaluation.

## First CPU smoke run

After verifying Blob and registering both data assets, check quota in the
workspace region. In Japan East, the project uses the DSv3 family when quota
is available. A zero-node minimum scales down idle compute.

```bash
workspace=(--subscription "$AZ_SUBSCRIPTION" --resource-group "$AZ_RESOURCE_GROUP" --workspace-name "$AZ_WORKSPACE")
az ml compute list-usage "${workspace[@]}" --location japaneast -o table
export AZ_COMPUTE=ekyc-cpu
az ml compute create "${workspace[@]}" --name "$AZ_COMPUTE" --type amlcompute \
  --size Standard_D2s_v3 --min-instances 0 --max-instances 1 \
  --idle-time-before-scale-down 120
az ml compute show "${workspace[@]}" --name "$AZ_COMPUTE" --query provisioning_state -o tsv
```

Wait for `Succeeded` before submission. Use the workspace's existing compute
if it already meets these settings. Submit the smoke once for the versioned
Blob inputs and keep the job name for status checks and artifact retrieval.

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

Repeat the status query until the job reaches a terminal state. If
`job stream` reports `Incorrect padding`, continue with `job show`.
That CLI error can occur while the remote job is still preparing.
Use the job's **Outputs + logs** in Azure ML Studio or the workspace artifact
datastore to inspect progress. After completion, download the named output:

```bash
az ml job download "${workspace[@]}" --name "$job" \
  --output-name model_bundle --download-path "$work_dir/output"
```

For the default artifact datastore, the following commands recover the
image-build log when it exists. Use the existing Blob authentication mode for
that container. The smoke operator used `key` because their Entra data role
was scoped to the input container. The CLI retrieved the key internally.

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

If the named-output download reports a `BatchGetResolvedUrisDto` error,
check the run's **Outputs + logs** for its output path. For the default
placement used by this smoke, download the two files with Entra access to
`workspaceblobstore` as follows.

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

In Azure ML Studio, compare the run's `manifest_sha256` parameter with
`sha256sum` of the downloaded Blob manifest. For a CLI check, run the
following in an authenticated project environment with `mlflow-skinny` and
`azureml-mlflow` installed.

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

Check the existing job and experiment for an active run before resubmitting,
including after an ambiguous CLI error.
Stop here for Blob-to-train verification. The following sections cover
separate manual retraining and deployment workflows.

## Training and deployment

For a manual HITL retraining job, register the manifest published after review
approval as a new `uri_file` asset version and set `AZ_MANIFEST_ASSET` to that
exact version. Reuse the original dataset folder asset so the published
relative paths still resolve. Retain the publication's `lineage.json`, the
manifest asset version and the Azure job name together. This CPU job trains
with `modeling.train`. The validation-selected MobileNetV3 comparison and
held-out evaluation are documented in [modeling/QUALITY.md](../modeling/QUALITY.md).

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

Use `python azure/stage.py --training-only "$work_dir/aml"` when preparing only
the training assets. Use a fresh staging directory for deployment. To compare
training configurations, add `inputs.epochs=8 inputs.learning_rate=0.0005`
to the job creation command's existing `--set` list. Keep the dataset and split
fixed and capture each run's MLflow parameters, metrics and job duration.

After the job reaches `Completed`, register its named output. Set
`AZ_MODEL_VERSION` to an unused version and choose `AZ_INSTANCE_TYPE` based on
regional availability and quota. Verify endpoint and deployment status before
routing traffic.

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

For the demo, set `EKYC_SCORING_URL` from the endpoint's `scoring_uri` and
`EKYC_SCORING_KEY` from an endpoint key in the demo process's environment.
For smoke testing before traffic assignment, set `EKYC_SCORING_DEPLOYMENT=blue`.
Start `python -m demo.server` and open http://127.0.0.1:8080. The browser sends
the selected image to the loopback demo, which calls the scoring service.
After the targeted smoke test succeeds, route traffic and clear the demo's
deployment override before testing normal endpoint routing.

```bash
az ml online-endpoint update "${workspace[@]}" --name "$AZ_ENDPOINT" --traffic "blue=100"
unset EKYC_SCORING_DEPLOYMENT
```

## Managed staging verification and cleanup

For a temporary staging test, keep the deployment-targeted header and one
replica. Before provisioning, check the region's Azure ML VM sizes, family and
total dedicated-core quota, and Linux retail rate. Set a teardown deadline.
Use a genuine MIDV-2020 mock frame for the browser and serial benchmark.
Keep the endpoint key in the proxy and benchmark process environment.

Before sending test requests, have the orchestrator route both endpoint log
categories to the same existing Log Analytics workspace. Set
`AZ_LOG_ANALYTICS_ID` to the workspace resource ID. The resource-specific
destination supplies the two named tables used by the queries. Application
Insights remains enabled on the deployment.
[Azure endpoint monitoring](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-monitor-online-endpoints?view=azureml-api-2)

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

Open http://127.0.0.1:8081 in a browser, upload the mock image, and check the
rendered prediction and HTTP status. With the same environment, run scoring
separately from the browser traffic and save the benchmark JSON.

```bash
python -m azure.benchmark --runs 100 --client-location "$CLIENT_LOCATION" \
  --target-location japaneast scoring "$MOCK_IMAGE" --warmup 3
```

For low-confidence telemetry, use a mock fixture that scores strictly below
the bundle's configured document threshold. Once both tables have ingested
the test logs, set the endpoint name, deployment and UTC window in the
[low-confidence rate query](telemetry-low-confidence.kql) and run it in the
destination Log Analytics workspace. Each 15-minute UTC bucket divides the
number of structured console events below threshold by the number of
`POST /score` requests returning HTTP 200 in the traffic table. The result
is a fraction and is null when the denominator is zero. On staging, the
13:00 UTC bucket on 2026-11-23 had 3 events and 108 successful requests
(2.78%). The denominator covers all successful scoring in that window,
including warmups and browser checks. The review flag can also reflect missing
or degraded quality. Use [telemetry.kql](telemetry.kql) for traffic volumes,
failures and service-side latency.

After testing, delete the dedicated staging endpoint and its deployment.
Capture the Azure response confirming that the deployment and endpoint are
gone, along with the UTC time. Retain the registered model and evidence.

```bash
az ml online-endpoint delete "${workspace[@]}" --name "$AZ_ENDPOINT" --yes
az ml online-deployment show "${workspace[@]}" --endpoint-name "$AZ_ENDPOINT" \
  --name blue --query provisioning_state -o tsv
az ml online-endpoint show "${workspace[@]}" --name "$AZ_ENDPOINT" \
  --query provisioning_state -o tsv
date -u -Is
unset EKYC_SCORING_KEY EKYC_SCORING_URL EKYC_SCORING_DEPLOYMENT
```

The two status checks should return resource-not-found after cleanup.
A missing result caused by authentication or network failure needs a retry.

## Evidence collection

For local scoring, run the inference server from the staged scoring code
directory in an inference environment with the dependencies from
`conda-inference.yml`. Set `LOCAL_MODEL_DIR` to the absolute path of a trained
bundle or its containing directory.

```bash
(cd "$spec/src/modeling" && azmlinfsrv --entry_script score.py --model_dir "$LOCAL_MODEL_DIR" --port 5001)
```

Set `EKYC_SCORING_URL=http://127.0.0.1:5001/score` from a second shell at the
repository root for the local demo or benchmark.

```bash
python -m azure.benchmark --runs 30 --client-location local --target-location local scoring sample.jpg
python -m azure.benchmark --runs 30 --client-location REGION_A --target-location REGION_B scoring sample.jpg
python -m azure.benchmark --runs 20 --client-location local --target-location local filesystem --directory /tmp
python -m azure.benchmark --runs 20 --client-location REGION_A --target-location REGION_B blob \
  --account-url "https://ACCOUNT.blob.core.windows.net" --container CONTAINER
```

Change `EKYC_SCORING_URL` and remove any previously set cloud credential when
measuring local scoring. The archived `staging-benchmark.json` measures the
course workstation (Asia/Ho_Chi_Minh) to Japan East. All three warmups and all
100 measured requests succeeded. For the 317,512-byte encoded request,
measured p50 was 1,033.41 ms and p95 was 1,467.94 ms. These percentiles use
successful measured requests, with warmups reported separately. Scoring
latency includes request setup, transport and decoding.

For a regional control, use the same approved Azure client in Japan East to
measure Japan East, then a second approved region with the same available
`Standard_DS2_v2` SKU. Match model bundle, inference environment, one replica,
mock image, timeout, three warmups and 100 serial requests per target. Save
both raw results with failures and run times. The staging run recorded six
dedicated DSv2 cores and 20 dedicated cores total in Japan East. Before a
future run, confirm available quota in each region, including any upgrade
reserve for the SKU. Use sequential endpoint windows if only one staging
deployment fits the approved budget. If the second region cannot support the
matched SKU, keep the regional comparison pending.
[Azure ML quota guidance](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-quotas?view=azureml-api-2)

The Blob mode needs `azure-identity` and `azure-storage-blob` and a credential
with read, write and delete access to an existing container. It uses a
temporary synthetic payload. Compare the same transfer size and run count.
Filesystem timings include OS caching and measure buffered I/O. Blob timings
include service calls and transfer time. Report client locations, regions,
VM SKUs, payload sizes and counts with the raw JSON.

## Release checks

From the repository root, run the unit and CLI checks in the project environment.
The scoring benchmark accepts `--timeout` after `scoring`. Each measured call
records one attempt. Issue a new request to retry after a timeout.

```bash
.venv/bin/python -m unittest -v azure.test_release demo.test_release
EKYC_RELEASE_DATASET=/absolute/path/to/midv-slice .venv/bin/python -m unittest -v azure.test_release.LocalRuntimeReleaseTests
EKYC_CHECK_AZURE_SCHEMAS=1 .venv/bin/python -m unittest -v azure.test_release.AzureSchemaReleaseTests
```

The runtime check requires the training dependencies and `azmlinfsrv` in the
same environment, plus a MIDV-2020 slice with `manifest.csv` and training and
evaluation splits. It trains a disposable smoke bundle and forces one listener
collision to verify startup recovery. The schema check
requires `jsonschema`, `PyYAML` and access to Microsoft's schema host.

## Mock document review queue

To opt in, supply all three review flags. Use the MIDV-2020 mock manifest
and its matching image root. Keep the queue in a dedicated owner-only
directory on a local Linux filesystem.

```bash
queue_dir=$(mktemp -d)
.venv/bin/python -m demo.server --port 8081 \
  --review-queue "$queue_dir" \
  --review-manifest "$MIDV_ROOT/manifest.csv" \
  --review-dataset-root "$MIDV_ROOT"
```

Open http://127.0.0.1:8081 and upload a JPEG or PNG from the manifest.
The demo indexes each file's exact SHA-256 at startup and uses the manifest's
`image_id` as the source identifier. A successful prediction with
`needs_review=true` queues an eligible image automatically. The **Report
incorrect prediction** button is available for eligible predictions at any
confidence. Other images can still be scored. The demo has no HTTP route
for queue files. Give the local review importer controlled filesystem access
to the private queue.

### Demo API

- `POST /predict` accepts JPEG or PNG bytes. HTTP 200 returns the usual
  `label`, `status`, `confidence`, and `needs_review` JSON fields. With review
  enabled, the response includes `X-EKYC-Review-Status` with `queued`,
  `available`, or `not_allowlisted`. An eligible response also includes
  `X-EKYC-Report-Token`.
- `POST /report` accepts only `{"token":"<X-EKYC-Report-Token>"}` as JSON.
  HTTP 200 returns `{"event_id":"<sha256>","status":"queued"}`.
  The demo signs the digest and the scorer's response. HTTP 403 signals an
  invalid receipt. Score the image again after a server restart to obtain
  a current receipt. HTTP 400 or 413 signals an invalid or oversized report.
- HTTP 503 on either route indicates a queue failure. Retry the original
  request. Repeated submissions converge on one event per digest and add
  each reason once. Without review flags, `/predict` retains its original
  behavior and `/report` returns 404.

Importers read finalized `events/<sha256>/event.json`. Each event contains
`event_id`, `image_sha256`, `image_file` (the sibling `image.jpg` or
`image.png`), `source_image_id`, the four-field `prediction` object, and
`reasons` with `needs_review`, `reported_incorrect`, or both. The first
published prediction is retained. A later reason is merged atomically, so a
later read may observe additional reasons. The demo publishes a new event's
image and metadata together and leaves temporary work outside the finalized
digest paths.

```bash
.venv/bin/python -m unittest -v demo.test_review_queue demo.test_release
```
