"""Generate Label Studio tasks and apply completed quality annotations."""

import argparse
import csv
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

from data_pipeline.images import validate_jpeg
from data_pipeline.manifest import REVIEW_LABELS, read_manifest, write_manifest


def make_tasks(manifest: Path, dataset_root: Path, media_root: Path, output: Path, *, per_group=None) -> int:
    dataset_root = dataset_root.resolve(strict=True)
    media_root = media_root.resolve(strict=True)
    dataset_root.relative_to(media_root)
    tasks = []
    seen_images = set()
    rows = read_manifest(manifest, preserve_extra=True)
    if per_group is not None:
        if per_group < 1:
            raise ValueError("Tasks per group must be positive")
        groups = defaultdict(list)
        for row in rows:
            group = row["split"], row["label"], row.get("domain", "")
            groups[group].append(row)
        representatives = []
        for group in sorted(groups):
            by_document = {}
            for row in sorted(groups[group], key=lambda row: sha256(row["image_id"].encode()).digest()):
                by_document.setdefault(row["document_id"], row)
            representatives.extend(list(by_document.values())[:per_group])
        rows = representatives
    for row in rows:
        image = (dataset_root / row["image_path"]).resolve(strict=True)
        image.relative_to(dataset_root)
        if not image.is_file():
            raise ValueError(f"Image path is not a file: {image}")
        if image in seen_images:
            raise ValueError(f"Duplicate resolved image path: {image}")
        seen_images.add(image)
        if Path(row["image_path"]).suffix.lower() in (".jpg", ".jpeg"):
            try:
                validate_jpeg(image)
            except (OSError, ValueError) as error:
                raise ValueError(f"Invalid JPEG: {image}") from error
        relative_path = image.relative_to(media_root).as_posix()
        tasks.append({"data": {
            "image": f"/data/local-files/?d={quote(relative_path)}",
            "image_id": row["image_id"],
            "image_path": row["image_path"],
        }})
    with TemporaryDirectory(dir=output.parent) as temp:
        staging = Path(temp) / "tasks.json"
        with staging.open("w", encoding="utf-8") as stream:
            json.dump(tasks, stream, indent=2)
            stream.write("\n")
        output.hardlink_to(staging)
    return len(tasks)


def completed_choices(task: dict) -> set[str]:
    annotations = task.get("annotations", [])
    if not isinstance(annotations, list):
        raise ValueError("Expected a list of annotations")
    choices = set()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError("Expected an annotation object")
        cancelled = annotation.get("was_cancelled", False)
        skipped = annotation.get("skipped", False)
        if not isinstance(cancelled, bool) or not isinstance(skipped, bool):
            raise ValueError("Expected boolean cancellation and skip flags")
        if cancelled or skipped:
            continue
        results = annotation.get("result", [])
        if not isinstance(results, list):
            raise ValueError("Expected a list of annotation results")
        if not results:
            continue
        if any(not isinstance(result, dict) for result in results):
            raise ValueError("Expected annotation result objects")
        quality = [result for result in results if result.get("from_name") == "quality"]
        if len(quality) != 1:
            raise ValueError("Expected one quality result per completed annotation")
        result = quality[0]
        value = result.get("value")
        values = value.get("choices") if isinstance(value, dict) else None
        if (result.get("to_name") != "image" or result.get("type") != "choices"
                or not isinstance(values, list) or len(values) != 1
                or values[0] not in REVIEW_LABELS):
            raise ValueError("Invalid quality choice")
        choices.add(values[0])
    return choices


def apply_reviews(manifest: Path, review_export: Path, output: Path) -> dict[str, int]:
    for source in (manifest, review_export):
        if output.resolve() == source.resolve() or (
            output.exists() and source.exists() and output.samefile(source)
        ):
            raise ValueError("Review output must be separate from its inputs")
    rows = read_manifest(manifest, preserve_extra=True)
    by_id = {row["image_id"]: row for row in rows}
    reviews = {}
    tasks = json.loads(review_export.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("Expected a Label Studio raw JSON task list")
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("data"), dict):
            raise ValueError("Each task must have a data object")
        data = task["data"]
        image_id = data.get("image_id")
        if not isinstance(image_id, str) or image_id not in by_id:
            raise ValueError(f"Unknown image_id: {image_id!r}")
        if data.get("image_path") != by_id[image_id]["image_path"]:
            raise ValueError(f"Image path mismatch for {image_id}")
        try:
            choices = completed_choices(task)
        except ValueError as error:
            raise ValueError(f"{image_id}: {error}") from error
        if image_id in reviews:
            choices.add(reviews[image_id])
        if len(choices) > 1:
            raise ValueError(f"Conflicting reviews for {image_id}: {sorted(choices)}")
        if choices:
            reviews[image_id] = choices.pop()

    kept = []
    excluded = 0
    for row in rows:
        status = reviews.get(row["image_id"], row["quality_status"])
        if status == "Outside_Scope":
            excluded += 1
        else:
            updated = {**row, "quality_status": status}
            if row["image_id"] in reviews:
                updated["quality_source"] = "reviewed"
            kept.append(updated)
    with TemporaryDirectory(dir=output.parent) as temp:
        staging = Path(temp) / "manifest.csv"
        fields = tuple(dict.fromkeys(key for row in (*rows, *kept) for key in row))
        write_manifest(staging, kept, fields=fields or None)
        staging.replace(output)
    labeled = sum(bool(row["quality_status"]) for row in kept)
    synthetic = sum(row.get("quality_source", "").strip().lower() == "synthetic"
                    for row in kept)
    counts = {"reviewed": labeled - synthetic, "unreviewed": len(kept) - labeled,
              "excluded": excluded}
    if synthetic:
        counts["synthetic"] = synthetic
    return counts


def write_allowlist(manifest, dataset_root, output):
    from demo.client import MAX_IMAGE_BYTES

    root = dataset_root.resolve(strict=True)
    entries = []
    for row in read_manifest(manifest):
        if row["split"] != "train":
            continue
        image = (root / row["image_path"]).resolve(strict=True)
        image.relative_to(root)
        if image.is_file() and 0 < image.stat().st_size <= MAX_IMAGE_BYTES:
            entries.append({key: row[key] for key in ("image_id", "image_path")})
    if not entries:
        raise ValueError("No eligible training images for the review queue")
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image_id", "image_path"))
        writer.writeheader()
        writer.writerows(entries)
    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tasks = commands.add_parser("make-tasks")
    tasks.add_argument("--manifest", type=Path, required=True)
    tasks.add_argument("--dataset-root", type=Path, required=True)
    tasks.add_argument("--media-root", type=Path, required=True)
    tasks.add_argument("--output", type=Path, required=True)
    tasks.add_argument("--per-group", type=int, help="Limit each split, document label and capture domain.")
    ingest = commands.add_parser("apply-reviews")
    ingest.add_argument("--manifest", type=Path, required=True)
    ingest.add_argument("--review-export", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)
    allowlist = commands.add_parser("make-allowlist")
    allowlist.add_argument("--manifest", type=Path, required=True)
    allowlist.add_argument("--dataset-root", type=Path, required=True)
    allowlist.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "make-tasks":
            count = make_tasks(args.manifest, args.dataset_root, args.media_root, args.output,
                               per_group=args.per_group)
            print(f"Wrote {count} tasks to {args.output}")
        elif args.command == "apply-reviews":
            counts = apply_reviews(args.manifest, args.review_export, args.output)
            summary = ", ".join(f"{name}={count}" for name, count in counts.items())
            print(f"Wrote {args.output} ({summary})")
        else:
            count = write_allowlist(args.manifest, args.dataset_root, args.output)
            print(f"Wrote {count} training images to {args.output}")
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
