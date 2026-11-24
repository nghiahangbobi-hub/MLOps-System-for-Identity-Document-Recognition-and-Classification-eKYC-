# Human-in-the-loop review

**English** · [Tiếng Việt](hitl.vi.md)

Use a trusted MIDV-2020 mock manifest and its dataset root. The queue indexes exact image bytes from that manifest. Only matching JPEG or PNG files are eligible for persistence. Keep the queue, batch, raw reviews, approval files, dataset versions, and model artifacts in private local storage. Keep the validation and test layouts fixed.

Set `work` to a private working directory and `dataset` to the prepared dataset directory containing the manifest and images.

## Queue predictions

Start the scorer using the [local runtime commands](../azure/README.md#evidence-collection). Set its URL in the demo process.

```bash
export EKYC_SCORING_URL=http://127.0.0.1:5001/score
python -m demo.server --port 8081 \
  --review-queue "$work/queue" \
  --review-manifest "$dataset/manifest.csv" \
  --review-dataset-root "$dataset"
```

Open the [local demo](http://127.0.0.1:8081), select a mock sample from the manifest and run inference. When the scorer returns `needs_review=true`, the server queues the original mock image and prediction. **Report incorrect prediction** queues the sample explicitly or adds the report reason to the existing event. Several actions on the same image produce one event keyed by SHA-256. A failed queue write can be retried. After a demo restart, make a fresh prediction to obtain a new report receipt. Previously committed queue events persist.

## Send the batch to Label Studio

```bash
python -m labeling.hitl batch --queue-root "$work/queue" \
  --base-manifest "$dataset/manifest.csv" --dataset-root "$dataset" \
  --output "$work/batch"
```

Create a Label Studio Community project with [hitl.xml](../labeling/hitl.xml). Set the local media root to the batch directory, or mount it read-only and use the path inside the container. Register its `images` subdirectory as a Local Files source in Tasks mode and save the source. Import `batch/tasks.json`. The reviewer opens each queued image and confirms or corrects the document type and quality. Choose `Outside_Scope` as the document type when the image should be excluded. Export raw JSON after finishing every task in that batch.

The batch snapshots finalized train events. Queue events from held-out templates can remain visible in the queue history. Publishing a new training version leaves validation and test rows unchanged.

## Explicitly approve and publish

Set `reviewer_id` to the approving human's identity. Keep the raw JSON export bytes intact. Review and approval are separate actions.

```bash
python -m labeling.hitl approve --batch "$work/batch" \
  --review-export "$work/reviews.json" --reviewer "$reviewer_id" \
  --output "$work/approval.json"
python -m labeling.hitl publish --batch "$work/batch" \
  --review-export "$work/reviews.json" --approval "$work/approval.json" \
  --base-manifest "$dataset/manifest.csv" --dataset-root "$dataset" \
  --versions-root "$work/versions"
```

Use the emitted version ID. Its directory contains `manifest.csv` and `lineage.json`. The approval records the reviewer and the exact export SHA-256. The lineage records input and output identity and before/after decisions, including exclusions. Publishing the same approved input again reuses the version.

## Submit training manually

After checking the version and lineage, run the trainer explicitly. Continue using the original dataset root for relative image paths.

```bash
python -m modeling.train --dataset-root "$dataset" \
  --manifest "$work/versions/$version_id/manifest.csv" \
  --output-dir "$work/manual-model" --device cpu --experiment ekyc-reviewed
```

Record the model run ID with the version ID, manifest digest, and observed quality sample counts. Quality model accuracy requires approved human labels in the evaluation split. Release still requires an actual human to finish and approve the real review batch and the operator to run managed Azure checks.

## Extracted sample review on this host

Open task 57 in [Label Studio project 3](http://127.0.0.1:8080/projects/3/data).
The task contains an actual queued MIDV-2020 mock prediction. After reviewing
the document type and quality, export raw JSON and retain the original bytes.
Run the following from the project root in the release environment after the
human reviewer has supplied the export and approval identity.
Use the annotation's `completed_by` account ID as `reviewer_id` when the
approving user has confirmed that review. This project serves images from a
batch-specific media directory. Approval accepts that exact batch URL and
records the SHA-256 of the original export.

```bash
run=runs/midv-5gb
dataset=Data/MIDV2020_sample_5GB
work=$(mktemp -d "$run/review-approval.XXXXXX")
python -m labeling.hitl approve --batch "$run/review-batch" \
  --review-export "$review_export" --reviewer "$reviewer_id" \
  --output "$work/approval.json"
python -m labeling.hitl publish --batch "$run/review-batch" \
  --review-export "$review_export" --approval "$work/approval.json" \
  --base-manifest "$run/inputs/manifest.csv" --dataset-root "$dataset" \
  --versions-root "$work/versions"
```

Use the emitted version with the manual training command above. Preserve
the approval, lineage and training run ID together.

## Reproduce the local technical test

Use the release environment and a new private work directory. The test uses a disposable Label Studio project, one fixture correction, and one fixture exclusion.

```bash
set -euo pipefail
py=.venv/release/bin/python
work=$(mktemp -d)
"$py" -m data_pipeline.midv2020 --source "$source_path" --output "$work/data" \
  --max-clips-per-template 2
"$py" -u -m ops.hitl_smoke --dataset-root "$work/data" --output "$work/smoke" \
  2>&1 | tee "$work/hitl.log"
```

