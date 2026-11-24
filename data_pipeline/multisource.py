"""Build a private train/validation dataset from historical clips and MIDV captures."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import tarfile
from tempfile import TemporaryDirectory

from data_pipeline.manifest import FIELDS, LABELS, parse_manifest, read_manifest
from data_pipeline.midv2020 import document_label


DOMAINS = ("photo", "scan_rotated", "scan_upright", "templates")
DEVELOPMENT_SPLITS = ("train", "validation")
HELD_OUT = {"alb_id", "lva_passport"}
CAPTURE = re.compile(r"images/([a-z0-9_]+)/([0-9]+)\.(jpg|jpeg)", re.IGNORECASE)
DOCUMENT = re.compile(r"([a-z0-9_]+)/([0-9]+)")


def historical_roster(split_record, split_sha256, clips_manifest):
    raw = split_record.read_bytes()
    if hashlib.sha256(raw).hexdigest() != split_sha256.lower():
        raise ValueError("Split record SHA-256 mismatch")
    history = json.loads(raw)
    layouts = history["layouts"]
    held_out = set(layouts["test"]) | HELD_OUT
    splits = {}
    for split in DEVELOPMENT_SPLITS:
        for template in layouts[split]:
            if template in splits or template in held_out or document_label(template) not in LABELS:
                raise ValueError(f"Invalid or overlapping historical layout: {template}")
            splits[template] = split

    raw = clips_manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != history["development.csv_sha256"]:
        raise ValueError("Historical clips manifest SHA-256 mismatch")
    rows = parse_manifest(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""),
                                         strict=True))
    roster = {}
    counts = Counter()
    for row in rows:
        template = row["template_id"]
        split = row["split"]
        label = row["label"]
        doc = row["document_id"]
        match = DOCUMENT.fullmatch(doc)
        if (splits.get(template) != split or match is None or match[1] != template
                or document_label(template) != label or row["quality_status"]):
            raise ValueError(f"Invalid historical clips row: {row['image_id']}")
        key = (template, int(match[2]))
        if roster.setdefault(key, doc) != doc:
            raise ValueError(f"Ambiguous historical document: {doc}")
        counts[split, label] += 1
    expected = {(split, label): history["splits"][split][label]
                for split in DEVELOPMENT_SPLITS for label in LABELS}
    if (counts != expected or min(expected.values()) <= 0
            or {template for template, _ in roster} != splits.keys()):
        raise ValueError("Historical clips roster does not match split counts and layouts")
    return rows, roster, splits


def copy_clips(rows, clips_root, staging):
    for row in rows:
        relative = Path(row["image_path"])
        source = (clips_root / relative).resolve(strict=True)
        if not source.is_relative_to(clips_root) or not source.is_file():
            raise ValueError(f"Clip image must be inside the clips root: {relative}")
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as stream, destination.open("xb") as target:
            shutil.copyfileobj(stream, target)


def archive_stat(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def extract_captures(path, domain, roster, splits, staging):
    before = archive_stat(path)
    seen = set()
    rows = []
    with tarfile.open(path, "r") as archive:
        for member in archive:
            match = CAPTURE.fullmatch(member.name)
            if not member.isfile() or match is None or match[1] not in splits:
                continue
            template, number, extension = match.groups()
            key = (template, int(number))
            if key in seen or key not in roster:
                raise ValueError(f"Unexpected or duplicate document in {domain}: {member.name}")
            seen.add(key)
            document = roster[key]
            relative = f"{domain}/{document}.{extension.lower()}"
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as stream, destination.open("xb") as target:
                header = stream.read(2)
                if header != b"\xff\xd8":
                    raise ValueError(f"Invalid JPEG header in {domain}: {member.name}")
                target.write(header)
                shutil.copyfileobj(stream, target)
            rows.append(dict(image_id=f"{domain}/{document}", image_path=relative,
                             template_id=template, document_id=document, split=splits[template],
                             label=document_label(template), quality_status="", domain=domain))
    if seen != roster.keys():
        raise ValueError(f"{domain} is missing {len(roster.keys() - seen)} roster documents")
    if archive_stat(path) != before:
        raise ValueError(f"Source archive changed while reading: {path}")
    return sorted(rows, key=lambda row: row["image_id"])


def build_dataset(clips_manifest, clips_root, split_record, split_sha256, archive_dir, output):
    if output.exists() or output.is_symlink():
        raise ValueError(f"Dataset output already exists: {output}")
    clips_root = clips_root.resolve(strict=True)
    archive_dir = archive_dir.resolve(strict=True)
    output = output.resolve()
    if output.is_relative_to(clips_root) or output.is_relative_to(archive_dir):
        raise ValueError("Output must be outside both source directories")
    rows, roster, splits = historical_roster(split_record, split_sha256, clips_manifest)
    with TemporaryDirectory(dir=output.parent, prefix=".multisource-") as temporary:
        staging = Path(temporary)
        copy_clips(rows, clips_root, staging)
        combined = [{**{field: row[field] for field in FIELDS}, "domain": "clips"}
                    for row in rows]
        for domain in DOMAINS:
            combined.extend(extract_captures(archive_dir / f"{domain}.tar", domain,
                                             roster, splits, staging))
        manifest = staging / "development.csv"
        with manifest.open("x", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=(*FIELDS, "domain"))
            writer.writeheader()
            writer.writerows(combined)
        read_manifest(manifest)
        if output.exists() or output.is_symlink():
            raise ValueError(f"Dataset output already exists: {output}")
        staging.rename(output)
    return Counter(row["split"] for row in combined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-manifest", required=True, type=Path,
                        help="Original historical clips development.csv")
    parser.add_argument("--clips-root", required=True, type=Path,
                        help="Root for the original clips manifest's relative image paths")
    parser.add_argument("--split-record", required=True, type=Path,
                        help="Historical split JSON with layouts, class counts and manifest SHA-256")
    parser.add_argument("--split-sha256", required=True,
                        help="Expected SHA-256 of the historical split JSON")
    parser.add_argument("--archive-dir", required=True, type=Path,
                        help="Read-only directory holding photo, scan_rotated, scan_upright and templates .tar files")
    parser.add_argument("--output", required=True, type=Path,
                        help="New private dataset root inside an existing parent directory")
    args = parser.parse_args()
    try:
        counts = build_dataset(**vars(args))
    except (OSError, ValueError, KeyError, TypeError, csv.Error, tarfile.TarError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Wrote {args.output / 'development.csv'} "
          f"(train={counts['train']}, validation={counts['validation']})")


if __name__ == "__main__":
    main()
