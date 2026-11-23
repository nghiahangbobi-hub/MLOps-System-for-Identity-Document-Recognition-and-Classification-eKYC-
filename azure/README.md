# Azure ML handoff

Bắt đầu với [Azure Blob Storage từ đầu](blob-storage.md) nếu chưa tạo workspace hoặc data asset.

Run commands from the repository root. The orchestrator owns cloud changes.
Use an authenticated Azure CLI with the `ml` extension, an existing workspace,
CPU training compute, dataset folder and manifest file assets, and an available
managed endpoint SKU. Set `AZ_SUBSCRIPTION`, `AZ_RESOURCE_GROUP`,
`AZ_WORKSPACE`, `AZ_COMPUTE`, `AZ_DATASET_ASSET`, `AZ_MANIFEST_ASSET` and
`AZ_ENDPOINT`. Asset variables use the form `azureml:name:version`.
The mounted dataset's layout must match the manifest's relative `image_path`
values. Use a reviewed manifest for quality evaluation.

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

Application logs record HTTP status and elapsed time. Application Insights is
enabled on the deployment. To use [telemetry.kql](telemetry.kql), route
`AmlOnlineEndpointTrafficLog` from the endpoint diagnostic settings into Log
Analytics. [Azure endpoint monitoring](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-monitor-online-endpoints?view=azureml-api-2)

The orchestrator can enable traffic logs using an existing Log Analytics
workspace. Set `AZ_LOG_ANALYTICS_ID` to its Azure resource ID.

```bash
endpoint_id=$(az ml online-endpoint show "${workspace[@]}" --name "$AZ_ENDPOINT" --query id -o tsv)
az monitor diagnostic-settings create --subscription "$AZ_SUBSCRIPTION" \
  --name ekyc-traffic --resource "$endpoint_id" --workspace "$AZ_LOG_ANALYTICS_ID" \
  --export-to-resource-specific true \
  --logs '[{"category":"AmlOnlineEndpointTrafficLog","enabled":true}]'
```

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
measuring local scoring. For regional comparison run from the same client
location against both a same-region and a cross-region deployment with the
same model, SKU, image and request count. Keep the warm-up and failures with
each result. Scoring latency includes request setup, transport and decoding.

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
