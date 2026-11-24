"""CSV contract shared by the dataset and review commands."""

import csv
from pathlib import Path, PurePosixPath


FIELDS = (
    "image_id", "image_path", "template_id", "document_id", "split",
    "label", "quality_status",
)
LABELS = ("ID_Card", "Passport")
QUALITY_LABELS = ("Clear", "Blur", "Glare", "Occluded")
REVIEW_LABELS = (*QUALITY_LABELS, "Outside_Scope")
SPLITS = {"train", "val", "validation", "test"}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return parse_manifest(csv.DictReader(stream, strict=True))


def parse_manifest(reader: csv.DictReader) -> list[dict[str, str]]:
    header = reader.fieldnames or ()
    fields = set(header)
    if len(fields) != len(header):
        raise ValueError("Manifest has duplicate columns")
    if not set(FIELDS[:-1]) <= fields:
        raise ValueError("Manifest is missing required columns")
    rows = []
    seen_ids = set()
    seen_paths = set()
    templates = {}
    documents = {}
    template_labels = {}
    for row in reader:
        if None in row or any(not (row.get(field) or "").strip() for field in FIELDS[:-1]):
            raise ValueError(f"Incomplete manifest row at line {reader.line_num}")
        image_path = PurePosixPath(row["image_path"])
        if image_path.is_absolute() or ".." in image_path.parts or not image_path.parts:
            raise ValueError(f"Unsafe image path: {image_path}")
        image_id = row["image_id"]
        if image_id.strip() in seen_ids:
            raise ValueError(f"Duplicate image_id: {image_id}")
        seen_ids.add(image_id.strip())
        if image_path in seen_paths:
            raise ValueError(f"Duplicate image path: {image_path}")
        seen_paths.add(image_path)
        label = row["label"]
        if label not in LABELS:
            raise ValueError(f"Invalid document label: {label}")
        split = row["split"]
        if split not in SPLITS:
            raise ValueError(f"Invalid split: {split}")
        group = "validation" if split == "val" else split
        template = row["template_id"].strip()
        if templates.setdefault(template, group) != group:
            raise ValueError(f"Inconsistent split for template: {template}")
        document = row["document_id"].strip()
        if documents.setdefault(document, group) != group:
            raise ValueError(f"Inconsistent split for document: {document}")
        quality = row.get("quality_status") or ""
        if quality and quality not in QUALITY_LABELS:
            raise ValueError(f"Invalid manifest quality_status: {quality}")
        normalized = {field: row[field] for field in FIELDS[:-1]}
        normalized["quality_status"] = quality
        if "quality_source" in fields:
            normalized["quality_source"] = row.get("quality_source") or ""
            source = normalized["quality_source"].strip().lower()
            if source not in ("", "unreviewed", "reviewed", "synthetic"):
                raise ValueError(f"Invalid quality provenance: {source}")
            if (source == "unreviewed" and quality) or (source == "reviewed" and not quality):
                raise ValueError(f"Quality status contradicts provenance: {source}")
            if source == "synthetic" and (split != "train" or not quality):
                raise ValueError("Synthetic rows require a quality target and the train split")
        # Reviewed train rows may explicitly correct a source-layout label.
        if not (split == "train" and quality and normalized.get(
            "quality_source", ""
        ).strip().lower() == "reviewed"):
            if template_labels.setdefault(template, label) != label:
                raise ValueError(f"Inconsistent split or label for template: {template}")
        rows.append(normalized)
    return rows


def write_manifest(path, rows, *, quality_source=False, fields=None):
    if fields is None:
        fields = (*FIELDS, "quality_source") if quality_source or any(
            "quality_source" in row for row in rows
        ) else FIELDS
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
