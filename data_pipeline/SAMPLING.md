# MIDV-2020 sampling under a JPEG byte limit

Run from the repository root after all five downloads complete. Use the approved
`clips.tar`, `photo.tar`, `scan_rotated.tar`, `scan_upright.tar`, and
`templates.tar` in one read-only source directory. Choose a new private output
directory under an existing parent outside the archive directory.

```bash
python -m data_pipeline.sample_midv \
  --archive-dir "$MIDV_ARCHIVE_DIR" \
  --split-record docs/model-quality-evidence/split.json \
  --split-sha256 b8ee591f0f05dc1161f489c0c7ead6ccb1acea667625c7468a2e3ba15d834054 \
  --output "$PRIVATE_DATASET_ROOT" \
  --max-image-bytes 5000000000
```

Seed 42 ranks mock documents deterministically. Every selected document contains
one image per still-image domain and three clips frames by default, spaced at
least ten frame numbers apart using the clips ETL sampler. The CLI reserves
coverage for all nine historical layouts, then fills the remaining budget
round-robin by layout with whole groups. A tight budget or unequal availability
can reduce template counts. The default domain mix is 3:1:1:1:1 by image count.
All captures of a document inherit the same historical train, validation, or
test split. Keep test rows out of training and selection.

The byte limit counts JPEG payloads. The CLI may leave unused bytes when no
remaining group fits. `manifest.csv` records split, domain, archive member,
JPEG bytes, per-image SHA-256 and seed. Quality labels are blank.
`provenance.json` records the manifest and split hashes, archive sizes and
coverage counts. The source archives are opened read-only.
