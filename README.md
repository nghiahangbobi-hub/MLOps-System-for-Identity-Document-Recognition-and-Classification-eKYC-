# MLOps eKYC

**English** · [Tiếng Việt](README.vi.md)

Classify MIDV-2020 clip frames as `ID_Card` or `Passport`, assess image quality when reviewed targets are available, and send uncertain or degraded images for review. The project covers the predictive-analysis branch of the course rubric.

## Local pipeline

Run from the project root. Use an isolated Python environment with compatible CPU PyTorch and torchvision wheels. Python 3.11 is a suitable starting point.

```bash
python -m pip install -r data_pipeline/requirements.txt
run_dir=$(mktemp -d)
python -m data_pipeline.midv2020 --output "$run_dir/dataset" \
  --max-clips-per-template 2 --frames-per-clip 3 --min-frame-gap 10
python -m labeling.label_studio make-tasks \
  --manifest "$run_dir/dataset/manifest.csv" \
  --dataset-root "$run_dir/dataset" --media-root "$run_dir" \
  --output "$run_dir/tasks.json"
```

Pass `--source /path/to/clips.tar` to override the archive path. Omit the clip cap when preparing the full dataset. Keep each group in a single split and use the same manifest and seed across experiment variants.

For quality training, run Label Studio Community locally, enable local file serving, and set its document root to the same media root used to generate tasks. Use [quality.xml](labeling/quality.xml) as the labeling interface, register the local image directory as a storage source, and import `tasks.json`. See the [labeling runbook](labeling/README.md) for the storage and export steps. Review each image as Clear, Blur, Glare, Occluded, or Outside_Scope. Export raw JSON and apply it to the manifest with the command below. Retain the raw export and reviewed manifest as evidence.

```bash
python -m labeling.label_studio apply-reviews \
  --manifest "$run_dir/dataset/manifest.csv" \
  --review-export "$run_dir/reviews.json" \
  --output "$run_dir/dataset/reviewed.csv"
```

Install the training dependencies into the project environment. The command below exercises the document-classification path. Use `reviewed.csv` for the quality path.

```bash
python -m pip install -r modeling/requirements.txt
export MLFLOW_ALLOW_FILE_STORE=true
python -m modeling.train \
  --dataset-root "$run_dir/dataset" \
  --manifest "$run_dir/dataset/manifest.csv" \
  --output-dir "$run_dir/model-baseline" \
  --device cpu --epochs 1 --batch-size 4 \
  --tracking-uri "file://$run_dir/mlruns" --experiment ekyc-smoke
```

The model bundle holds PyTorch weights and preprocessing and class metadata. Compare variants on validation metrics and evaluate the chosen configuration on the held-out test split. Evaluate quality on reviewed samples and report the quality sample count alongside metrics.

The scoring request is a JSON object with an `image_base64` field. The response includes `label`, `status`, `confidence`, and `needs_review`. Degraded image quality, missing quality evidence, or low document confidence can require manual review. A document-only bundle returns a null quality status.

If Label Studio is running on port 8080, start the demo with `python -m demo.server --port 8081` after setting `EKYC_SCORING_URL` according to the Azure runbook.

## Human-in-the-loop review

The [HITL runbook](docs/hitl.md) connects demo predictions and explicit reports for mock images to Label Studio, approval, dataset lineage, and operator-submitted training.

## Azure and tests

The [Azure Blob Storage setup guide](azure/blob-storage.md) covers workspace creation, image upload, and data asset registration (in Vietnamese).

The [Azure runbook](azure/README.md) covers deployment, demo configuration, and benchmark commands. Keep datasets, model artifacts, credentials, and evaluation outputs in private storage.

After installing dependencies in a virtual environment, run the component regression tests:

```bash
python -m unittest data_pipeline.test_release labeling.test_release labeling.test_hitl
python -m unittest modeling.test_release modeling.test_quality_compare
python -m unittest azure.test_release demo.test_release demo.test_review_queue
```
