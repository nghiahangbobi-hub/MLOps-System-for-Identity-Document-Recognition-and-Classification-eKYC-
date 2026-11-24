# Fixed-split MobileNetV3 comparison

**English** · [Tiếng Việt](QUALITY.vi.md)

Run from the project root in a Python 3.11 environment with the modeling
requirements. Use a CUDA-enabled Torch install only when an actual CUDA tensor
operation succeeds. The commands default to six Torch CPU threads, one interop
thread and two data workers. Run one comparison at a time unless resources have
been coordinated.

The comparison uses the seed-42 MIDV-2020 ETL layout assignment and a
train/validation-only CSV provided by the orchestrator. Both candidates start
from ImageNet MobileNetV3 Small and see the same RGB 224×224 preprocessing,
training augmentation and class order. The optional
`--crop-mode center_square` trial crops the central square before resizing
to retain the document's aspect ratio and reduce the background. The bundle
records the mode so scoring uses the same transform. The frozen candidate trains only the
classifier. The fine-tuned candidate updates the final four feature blocks at
3e-5 and the classifier at 1e-3. Both use AdamW and fixed pretrained
BatchNorm running statistics. Within and across candidates, select on validation
macro-F1, then accuracy, then negative log loss. The comparison fits temperature
and the review threshold using the selected checkpoint's validation logits.

## Development comparison

Use the fixed development CSV with image paths relative to the supplied
dataset root. Leave the held-out CSV with the orchestrator during this step.
If the orchestrator provides its SHA-256, supply `--held-out-sha256` to
bind the unseen held-out CSV to the selection.

```bash
set -o pipefail
runtime=.venv/model-release/bin/python
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
"$runtime" -u -m modeling.quality_compare compare \
  --dataset-root "$DATASET_ROOT" --development "$DEVELOPMENT_CSV" \
  --output "$RUN_DIR" --device cpu --threads 6 --epochs 8 \
  2>&1 | tee "$RUN_DIR.compare.log"
```

The output directory must be new. The bundle, exact training snapshot,
per-epoch JSONL, MLflow run, and selected configuration go there. Compare additional preprocessing trials using validation only. Then seal
the winner before asking for held-out access. Pass one `--comparison` per
completed trial. The seal command uses the same validation ranking and copies
only the winner's bundle and snapshot into a new final run directory.

```bash
"$runtime" -m modeling.quality_compare seal \
  --comparison "$FULL_FRAME_RUN" --comparison "$CENTER_SQUARE_RUN" \
  --output "$FINAL_RUN_DIR"
```

## Manual retraining from a human-approved version

For a HITL retraining run, explicitly supply the directory containing the
approved `manifest.csv` and `lineage.json`, the raw Label Studio export and
approval JSON, the published batch directory, and the original combined ETL
manifest. `DEVELOPMENT_CSV` remains the train/validation-only CSV from the
original ETL assignment. The comparison reconstructs Data's approved version
from the original inputs and verifies both published files byte for byte before
training. Only train rows may change or be excluded. The queue and approval
commands never launch training.

```bash
"$runtime" -u -m modeling.quality_compare compare \
  --dataset-root "$DATASET_ROOT" --development "$DEVELOPMENT_CSV" \
  --approved-version "$APPROVED_VERSION_DIR" --approval "$APPROVAL_JSON" \
  --review-export "$RAW_LABEL_STUDIO_EXPORT" --batch "$BATCH_DIR" \
  --base-manifest "$BASE_MANIFEST" --output "$RUN_DIR" \
  --device cpu --threads 6 --epochs 8 2>&1 | tee "$RUN_DIR.compare.log"
```

## One held-out pass

After selection, the orchestrator supplies the held-out CSV and its
authoritative combined ETL manifest. Use the sealed final run directory. The evaluator
verifies the bundle and development hashes, exact held-out roster, both classes,
and split isolation. The exclusive marker blocks repeated or concurrent
evaluation for this selection, including after an interrupted pass. An
interruption after claiming the pass requires an explicit protocol decision by
the orchestrator. Keep the marker and all raw logs.

```bash
"$runtime" -u -m modeling.quality_compare evaluate \
  --dataset-root "$DATASET_ROOT" --held-out "$HELD_OUT_CSV" \
  --base-manifest "$BASE_MANIFEST" --output "$FINAL_RUN_DIR" \
  --device cpu --threads 6 2>&1 | tee "$FINAL_RUN_DIR.evaluate.log"
```

`held-out.json` reports unfiltered accuracy, macro-F1, per-class recall,
confusion counts with true rows and predicted columns, support, per-template
results, uncalibrated and calibrated 15-bin top-label ECE, and the retained
coverage and accuracy at the validation-chosen document threshold. The
threshold maximizes validation coverage at the target retained accuracy
(default 85%), retaining confidence ties together. If no threshold attains
the target, the bundle sends all images to review. Coverage in this report
is for the document confidence policy. The API also sends missing or non-Clear
quality predictions to review. The report includes the resulting API coverage
separately.

Quality is reported separately and remains pending when reviewed holdout labels
or a trained quality head are unavailable. A quality result with incomplete
four-class coverage is marked partial. The 85% document acceptance gate uses
accuracy on all held-out frames, including those routed for review.

```bash
OMP_NUM_THREADS=6 "$runtime" -m unittest -v modeling.test_quality_compare modeling.test_release
```

## Additional capture domains

Evaluate a sealed model on one photo or scan per held-out mock document. Use
the original `split.json` from `ops.prepare_model_quality` and provide distinct
TAR files with matching document rosters. The evaluator reads the archives
in place and keeps the original crop, temperature, and threshold. Report
each capture domain separately and retain the JSON privately. Results for
different captures of the same mock document are paired observations.

```bash
"$runtime" -m ops.evaluate_capture_domains --bundle "$FINAL_RUN_DIR/bundle" \
  --split-record "$SPLIT_RECORD" --archive "$MIDV_DIR/photo.tar" \
  --archive "$MIDV_DIR/scan_rotated.tar" --archive "$MIDV_DIR/scan_upright.tar" \
  --archive "$MIDV_DIR/templates.tar" --output "$CAPTURE_RESULTS"
```

## Prepare a multi-domain fine-tuning dataset

Use the historical development manifest and both the saved split record and
its associated digest. This command creates a private development-only dataset
with a `domain` column and blank quality targets. Evaluate the frozen baseline
and each fine-tuned epoch on the same validation rows. Track per-domain
macro-F1 and require an explicit validation gate before publishing a bundle.

```bash
"$runtime" -m data_pipeline.multisource \
  --clips-manifest "$CLIPS_DEVELOPMENT_CSV" --clips-root "$CLIPS_DATASET_ROOT" \
  --split-record "$SPLIT_RECORD" --split-sha256 "$SPLIT_SHA256" \
  --archive-dir "$MIDV_DIR" --output "$PRIVATE_DATASET_ROOT"
```
