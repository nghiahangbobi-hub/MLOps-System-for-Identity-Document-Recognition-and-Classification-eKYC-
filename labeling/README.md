# MIDV-2020 quality review

**English** · [Tiếng Việt](README.vi.md)

Run from the project root. Keep the dataset, task files and annotation exports in
a local work directory outside the repository.
Install Pillow in the project environment with
`python -m pip install -r data_pipeline/requirements.txt`.

```bash
run=$(mktemp -d /tmp/ekyc-data.XXXXXX)
python -m data_pipeline.midv2020 --output "$run/dataset" \
  --max-clips-per-template 2 --frames-per-clip 3 --min-frame-gap 10
python -m labeling.label_studio make-tasks \
  --manifest "$run/dataset/manifest.csv" --dataset-root "$run/dataset" \
  --media-root "$run/dataset" --output "$run/tasks.json"
```

Use `--source /path/to/clips.tar` to override the default archive path. Omit the
clip limit to process the full eligible archive. `template_id` identifies the
document layout, such as `est_id`. `document_id` identifies a numbered mock
document and its clip. Each layout belongs to one split, and each split contains
both classes. Frame gaps are measured in source frame numbers. The build leaves
quality blank until review.

Set `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` and
`LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` to the media root before starting
Label Studio. With a container, mount the dataset read-only and set the
environment variable to the mounted path inside the container. Create a project,
and paste `labeling/quality.xml` into its labeling interface. In the project
storage settings, add a Local Files source pointing to the dataset's
`images` directory. Set its import method to Tasks and choose Save without
syncing. Import `tasks.json` from the project's Data Manager. The storage
connection lets Label Studio serve the images referenced in the JSON. See
[Label Studio local storage](https://labelstud.io/guide/storage_local).

Choose the dominant readability issue for each image. Select Clear when the
document is readable without a dominant issue. Select Outside_Scope for images
outside the supported document classes or with no target document. Export the
project in raw JSON format and save the file as `$run/reviews.json`.

```bash
python -m labeling.label_studio apply-reviews \
  --manifest "$run/dataset/manifest.csv" --review-export "$run/reviews.json" \
  --output "$run/dataset/reviewed.csv"
```

Reapply updated project exports to the original manifest. Skips leave quality
blank, matching completed annotations supply a quality label, and Outside_Scope
removes the row from the reviewed manifest. If reviewers disagree, resolve the
conflict in Label Studio and export again. Use the same dataset root with either
manifest.

Write to a separate output path. Apply exports as full snapshots of the original
manifest. Repeating the same export produces identical results. When separate
exports are applied concurrently to the same output, the last successful
publication determines the complete output.

## Disposable Community Edition smoke test

From the repository root, run the opt-in smoke command with the data
requirements installed and Docker or rootless Podman available.

```bash
set -o pipefail
python -u -m labeling.smoke 2>&1 | tee /tmp/ekyc-label-studio-smoke.log
```

Use `--source /path/to/clips.tar` for a different copy of the archive, or
`--runtime podman` to select the runtime explicitly. The command builds the
54-frame slice in a temporary directory, opens the pinned Community image on
a dynamic loopback port, imports every task, and verifies every served JPEG
against its source bytes. It posts one fixture annotation, downloads raw JSON,
and runs the review-ingestion CLI. It then removes the disposable container,
its project storage, and all fixture files. The log records HTTP statuses and
counts without credentials or review content. Fixture output is technical
verification only and provides no human label approval.

## Review queued predictions and publish a train manifest

Use a separate Label Studio project for queued predictions. Set `queue_root`
to the demo's review queue and `dataset_root` to the original mock MIDV
dataset. Set `work` to a private workspace outside the repository and create
it before the batch command. Here the base manifest is the dataset's original
manifest. For a subsequent correction batch, select the desired published
manifest as the base while using the same dataset root.

```bash
base_manifest="$dataset_root/manifest.csv"
python -m labeling.hitl batch \
  --queue-root "$queue_root" --base-manifest "$base_manifest" \
  --dataset-root "$dataset_root" --output "$work/batch"
```

Only finalized digest-named event directories are read. Queued image bytes
must match the event digest and the source image referenced by the base
manifest. A bad finalized event fails the batch. The snapshot records valid
validation and test events as held out and generates tasks only for train
events. An empty train batch fails with its held-out count. Keep the batch
directory and its evidence as produced.

Start Label Studio with local file serving enabled and set its document root
to `$work/batch` or to the equivalent read-only container mount. Add the
batch's `images` directory as a Local Files source, select Tasks mode, and
save without syncing. Create the project using `labeling/hitl.xml`, then
import `$work/batch/tasks.json`. Review the image and select the correct
document type and quality, or select Outside_Scope by itself to exclude
the train row. Export raw **JSON** to `$work/export.json`. A complete export
must contain an agreed completed decision for every task. Canceled, skipped,
and predicted results do not count as decisions.

Have the person approving that exact export run the separate approval command
with their local reviewer identifier. This records the supplied identity and
the SHA256 of the batch metadata, base manifest, and exact export bytes. Local
approval records rely on control of the workspace and the explicit approval
action.

```bash
python -m labeling.hitl approve --batch "$work/batch" \
  --review-export "$work/export.json" --reviewer "$reviewer_id" \
  --output "$work/approval.json"
python -m labeling.hitl publish --batch "$work/batch" \
  --review-export "$work/export.json" --approval "$work/approval.json" \
  --base-manifest "$base_manifest" --dataset-root "$dataset_root" \
  --versions-root "$work/versions"
```

Publication prints the new manifest path. Its sibling `lineage.json` records
the approval, manifest digest, affected event IDs, prediction and queue
reasons, before/after labels, and exclusions. Holdout row values stay the
same. The output includes quality provenance so a reviewed train correction
can differ from the source layout label while keeping its layout ID and split.

The version ID is the SHA256 of the approval file. Identical retries return
the same artifacts. An existing version with different bytes is an error.
Changing the export or base invalidates its previous approval. Save updated
exports and approvals under new paths. To train, manually submit a model
training job with the published manifest path and the original dataset root.
Retain the base, batch, export, approval and lineage with the training record.

Run the focused regressions with `python -m unittest -v labeling.test_hitl`.
