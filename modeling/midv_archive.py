"""Materialize model inputs for every supported image in the course MIDV TAR."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

from PIL import Image, ImageOps


IMAGE_PATH = re.compile(
    r"MIDV2020_sample_5GB/"
    r"(?P<domain>clips|photo|scan_rotated|scan_upright|templates)/images/"
    r"(?P<template>[a-z0-9_]+)/(?P<document>[0-9]+)"
    r"(?:/(?P<frame>[0-9]+))?\.jpe?g",
    re.IGNORECASE,
)
EXTRA_LAYOUTS = {"rus_internalpassport": ("train", "Passport")}


def identity(match, layout, path):
    domain, template, document, frame = match.group("domain", "template", "document", "frame")
    split, label = layout
    document_id = f"{template}/{int(document)}"
    image_id = f"{domain}/{document_id}"
    if frame is not None:
        image_id += f"/{int(frame)}"
    return {
        "image_id": image_id, "image_path": path, "template_id": template,
        "document_id": document_id, "split": split, "label": label, "domain": domain,
    }


def prepare(archive, reference, output, size):
    with reference.open(encoding="utf-8-sig", newline="") as stream:
        previous = {row["image_path"]: row for row in csv.DictReader(stream, strict=True)}
    layouts = dict(EXTRA_LAYOUTS)
    for row in previous.values():
        layout = row["split"], row["label"]
        if layouts.setdefault(row["template_id"], layout) != layout:
            raise ValueError("Conflicting template assignment in the reviewed manifest")
    output.mkdir()
    rows = []
    seen = set()
    with tarfile.open(archive, "r:") as source:
        for member in source:
            match = IMAGE_PATH.fullmatch(member.name)
            if not member.isfile() or match is None:
                continue
            if match["template"] not in layouts or (match["domain"] == "clips") != bool(match["frame"]):
                raise ValueError(f"Unknown archive image layout: {member.name}")
            relative = member.name.split("/", 1)[1]
            row = identity(match, layouts[match["template"]], relative)
            if row["image_id"] in seen:
                raise ValueError(f"Duplicate archive image ID: {row['image_id']}")
            seen.add(row["image_id"])
            with source.extractfile(member) as stream:
                raw = stream.read()
            prior = previous.pop(relative, None)
            if prior:
                if any(prior[field] != value for field, value in row.items()):
                    raise ValueError(f"Reviewed identity changed: {relative}")
                if hashlib.sha256(raw).hexdigest() != prior["sha256"]:
                    raise ValueError(f"Reviewed image bytes changed: {relative}")
            row.update(
                quality_status=prior["quality_status"] if prior else "",
                quality_source=prior["quality_source"] if prior else "",
            )
            row["image_path"] = Path(relative).with_suffix(".png").as_posix()
            target = output / row["image_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(io.BytesIO(raw), formats=["JPEG"]) as source_image:
                image = ImageOps.exif_transpose(source_image).convert("RGB")
                image.resize((size, size), Image.Resampling.BILINEAR).save(target, compress_level=1)
            with target.open("rb") as stream:
                row["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            row["size_bytes"] = target.stat().st_size
            rows.append(row)
    if previous or not rows:
        raise ValueError(f"Archive did not contain {len(previous)} source manifest images")

    # Preserve reviewed samples if resizing produces identical bytes across splits.
    rows.sort(key=lambda row: (not bool(row["quality_status"]), row["image_id"]))
    assignments = {}
    kept = []
    excluded = Counter()
    for row in rows:
        if assignments.setdefault(row["sha256"], row["split"]) != row["split"]:
            if row["quality_status"]:
                raise ValueError("Reviewed samples have identical pixels across splits")
            excluded[row["split"]] += 1
            continue
        kept.append(row)
    with (output / "manifest.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(kept[0]))
        writer.writeheader()
        writer.writerows(kept)
    summary = {
        "archive_bytes": archive.stat().st_size, "archive_images": len(rows),
        "splits": dict(Counter(row["split"] for row in kept)),
        "quality_labels_by_split": dict(Counter(row["split"] for row in kept if row["quality_status"])),
        "identical_across_splits_excluded": dict(excluded),
        "image_size": size,
    }
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    args = parser.parse_args()
    result = prepare(args.archive, args.reference, args.output, args.image_size)
    args.summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
