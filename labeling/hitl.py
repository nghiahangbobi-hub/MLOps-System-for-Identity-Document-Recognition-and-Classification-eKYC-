"""Snapshot queued MIDV events, approve a raw review export, and publish a dataset version."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from PIL import Image

from data_pipeline.manifest import LABELS, QUALITY_LABELS, parse_manifest


DIGEST = re.compile(r"[0-9a-f]{64}")
MEDIA = {"image.jpg": "JPEG", "image.png": "PNG"}
REASONS = {"needs_review", "reported_incorrect"}
OUTSIDE = "Outside_Scope"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(raw):
    return json.loads(raw, object_pairs_hook=unique_keys)


def read_base(path):
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    rows = {row["image_id"]: row for row in parse_manifest(reader)}
    original = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    return digest(raw), rows, list(original.fieldnames), list(original)


def source_digest(root, row):
    path = (root / row["image_path"]).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Invalid base image path for {row['image_id']}")
    return digest(path.read_bytes())


def read_event(directory, rows, root):
    event_id = directory.name
    record_file = directory / "event.json"
    if record_file.is_symlink() or not record_file.is_file():
        raise ValueError(f"Invalid event record for {event_id}")
    event = load_json(record_file.read_bytes())
    if not isinstance(event, dict) or event.get("event_id") != event_id or event.get("image_sha256") != event_id:
        raise ValueError(f"Event identity mismatch for {event_id}")
    name = event.get("image_file")
    if not isinstance(name, str) or name not in MEDIA:
        raise ValueError(f"Invalid event image filename for {event_id}")
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Invalid event image file for {event_id}")
    image = path.read_bytes()
    if digest(image) != event_id:
        raise ValueError(f"Queued image digest mismatch for {event_id}")
    with Image.open(io.BytesIO(image), formats=[MEDIA[name]]) as decoded:
        decoded.load()
    source_id = event.get("source_image_id")
    if not isinstance(source_id, str) or source_id not in rows:
        raise ValueError(f"Unknown source_image_id in event {event_id}")
    row = rows[source_id]
    if source_digest(root, row) != event_id:
        raise ValueError(f"Base image digest mismatch for {source_id}")
    prediction = event.get("prediction")
    if not isinstance(prediction, dict) or set(prediction) != {"label", "status", "confidence", "needs_review"}:
        raise ValueError(f"Invalid prediction fields for {event_id}")
    confidence = prediction["confidence"]
    if (prediction["label"] not in LABELS or prediction["status"] not in (None, *QUALITY_LABELS)
            or type(confidence) not in (int, float) or not 0 <= confidence <= 1
            or type(prediction["needs_review"]) is not bool):
        raise ValueError(f"Invalid prediction values for {event_id}")
    reasons = event.get("reasons")
    if (not isinstance(reasons, list) or not reasons
            or any(not isinstance(reason, str) or reason not in REASONS for reason in reasons)):
        raise ValueError(f"Invalid queue reasons for {event_id}")
    return {
        "event_id": event_id,
        "source_image_id": source_id,
        "split": row["split"],
        "image_file": f"{event_id}{path.suffix}",
        "prediction": prediction,
        "reasons": sorted(set(reasons)),
    }, image


def task_data(batch_id, event):
    return {
        "batch_id": batch_id,
        "event_id": event["event_id"],
        "source_image_id": event["source_image_id"],
        "image": f"/data/local-files/?d=images/{event['image_file']}",
    }


def snapshot(queue_root, base_manifest, dataset_root, output):
    base_hash, rows, _, _ = read_base(base_manifest)
    root = dataset_root.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise ValueError("Batch output already exists")
    if output.resolve().is_relative_to(queue_root.resolve()):
        raise ValueError("Batch output must be outside the review queue")
    events_root = queue_root / "events"
    if events_root.is_symlink():
        raise ValueError("Queue events directory must be local to the queue")
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "batch"
        images = stage / "images"
        images.mkdir(parents=True)
        events, held_out = [], []
        for directory in sorted(events_root.iterdir()):
            if not DIGEST.fullmatch(directory.name):
                continue
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError(f"Invalid finalized event directory: {directory.name}")
            event, image = read_event(directory, rows, root)
            if event["split"] != "train":
                held_out.append(event)
                continue
            (images / event["image_file"]).write_bytes(image)
            events.append(event)
        if not events:
            raise ValueError(f"No eligible train events ({len(held_out)} held out)")
        content = {"schema": 1, "base_manifest_sha256": base_hash,
                   "events": events, "held_out": held_out}
        batch_id = digest(json_bytes(content))
        (stage / "batch.json").write_bytes(json_bytes({"batch_id": batch_id, **content}))
        tasks = [{"data": task_data(batch_id, event)} for event in events]
        (stage / "tasks.json").write_bytes(json_bytes(tasks))
        stage.rename(output)
    return batch_id, len(events), len(held_out)


def read_batch(path):
    raw = (path / "batch.json").read_bytes()
    batch = load_json(raw)
    if not isinstance(batch, dict):
        raise ValueError("Invalid batch metadata")
    batch_id = batch.pop("batch_id", None)
    base_hash = batch.get("base_manifest_sha256")
    if (batch.get("schema") != 1 or digest(json_bytes(batch)) != batch_id
            or not isinstance(base_hash, str) or not DIGEST.fullmatch(base_hash)):
        raise ValueError("Invalid batch hash")
    events = batch.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("Batch contains no train events")
    held_out = batch.get("held_out")
    if not isinstance(held_out, list) or any(
        not isinstance(event, dict) or event.get("split") not in ("val", "validation", "test")
        or not isinstance(event.get("event_id"), str) or not DIGEST.fullmatch(event["event_id"])
        for event in held_out
    ):
        raise ValueError("Invalid held-out event metadata")
    seen_ids, seen_sources = set(), set()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Invalid batch event")
        event_id = event.get("event_id")
        source_id = event.get("source_image_id")
        if (not isinstance(event_id, str) or not DIGEST.fullmatch(event_id) or event_id in seen_ids
                or not isinstance(source_id, str) or not source_id or source_id in seen_sources
                or event.get("split") != "train"
                or event.get("image_file") not in (f"{event_id}.jpg", f"{event_id}.png")):
            raise ValueError("Invalid or duplicate batch event")
        media = path / "images" / event["image_file"]
        if media.is_symlink() or not media.is_file() or digest(media.read_bytes()) != event_id:
            raise ValueError(f"Batch image digest mismatch for {event_id}")
        seen_ids.add(event_id)
        seen_sources.add(source_id)
    return {"batch_id": batch_id, **batch}, digest(raw)


def corrected_labels(results):
    controls = {"document_type": (*LABELS, OUTSIDE), "quality": QUALITY_LABELS}
    values = {}
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Invalid annotation result")
        name = result.get("from_name")
        value = result.get("value")
        choices = value.get("choices") if isinstance(value, dict) else None
        if (not isinstance(name, str) or name not in controls or name in values
                or result.get("to_name") != "image" or result.get("type") != "choices"
                or not isinstance(choices, list) or len(choices) != 1
                or choices[0] not in controls[name]):
            raise ValueError("Invalid or duplicate correction control")
        values[name] = choices[0]
    label, quality = values.get("document_type"), values.get("quality")
    if label == OUTSIDE and quality is None:
        return label, None
    if label in LABELS and quality in QUALITY_LABELS:
        return label, quality
    raise ValueError("Incomplete correction or quality supplied with Outside_Scope")


def read_reviews(raw, batch):
    tasks = load_json(raw)
    if not isinstance(tasks, list):
        raise ValueError("Expected a raw Label Studio JSON task list")
    expected = {event["event_id"]: task_data(batch["batch_id"], event) for event in batch["events"]}
    corrections = {}
    for task in tasks:
        data = task.get("data") if isinstance(task, dict) else None
        event_id = data.get("event_id") if isinstance(data, dict) else None
        if not isinstance(event_id, str) or event_id not in expected or any(
            data.get(key) != value for key, value in expected[event_id].items()
        ):
            raise ValueError("Export task does not match the batch")
        annotations = task.get("annotations", [])
        if not isinstance(annotations, list):
            raise ValueError("Expected a list of annotations")
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise ValueError("Invalid annotation object")
            cancelled = annotation.get("was_cancelled", False)
            skipped = annotation.get("skipped", False)
            if type(cancelled) is not bool or type(skipped) is not bool:
                raise ValueError("Invalid annotation flags")
            if cancelled or skipped:
                continue
            results = annotation.get("result", [])
            if not isinstance(results, list):
                raise ValueError("Expected a list of annotation results")
            if not results:
                continue
            correction = corrected_labels(results)
            if corrections.setdefault(event_id, correction) != correction:
                raise ValueError(f"Conflicting completed corrections for {event_id}")
    missing = expected.keys() - corrections.keys()
    if missing:
        raise ValueError(f"Incomplete export: {len(missing)} batch events lack a completed correction")
    return corrections


def approval_record(batch, batch_hash, export_hash, reviewer):
    if not isinstance(reviewer, str) or not reviewer.strip() or not reviewer.isprintable():
        raise ValueError("A nonblank reviewer identity is required")
    return {
        "schema": 1,
        "reviewer": reviewer.strip(),
        "batch_id": batch["batch_id"],
        "batch_sha256": batch_hash,
        "export_sha256": export_hash,
        "base_manifest_sha256": batch["base_manifest_sha256"],
    }


def approve(batch_path, review_export, reviewer, output):
    batch, batch_hash = read_batch(batch_path)
    raw = review_export.read_bytes()
    read_reviews(raw, batch)
    record = approval_record(batch, batch_hash, digest(raw), reviewer)
    payload = json_bytes(record)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "approval.json"
        stage.write_bytes(payload)
        try:
            output.hardlink_to(stage)
        except FileExistsError:
            if output.is_symlink() or not output.is_file() or output.read_bytes() != payload:
                raise ValueError("Existing approval differs; use a new approval path")
    return record


def publish(batch_path, review_export, approval_path, base_manifest, dataset_root, versions_root):
    batch, batch_hash = read_batch(batch_path)
    raw = review_export.read_bytes()
    approval_raw = approval_path.read_bytes()
    approval = load_json(approval_raw)
    if not isinstance(approval, dict) or approval != approval_record(
        batch, batch_hash, digest(raw), approval.get("reviewer")
    ):
        raise ValueError("Stale approval: batch or export hash does not match")
    if approval_raw != json_bytes(approval):
        raise ValueError("Approval file differs from its original representation")
    corrections = read_reviews(raw, batch)
    base_hash, base_rows, fields, original = read_base(base_manifest)
    if base_hash != batch["base_manifest_sha256"]:
        raise ValueError("Stale batch: base manifest hash does not match")

    root = dataset_root.resolve(strict=True)
    events = {event["source_image_id"]: event for event in batch["events"]}
    changes, output_rows = [], []
    for row in original:
        image_id = row["image_id"]
        event = events.get(image_id)
        if event is None:
            output_rows.append(row)
            continue
        if row["split"] != "train" or source_digest(root, base_rows[image_id]) != event["event_id"]:
            raise ValueError(f"Base row is not the approved train image: {image_id}")
        label, quality = corrections[event["event_id"]]
        excluded = label == OUTSIDE
        changes.append({
            "event_id": event["event_id"],
            "source_image_id": image_id,
            "split": row["split"],
            "prediction": event["prediction"],
            "reasons": event["reasons"],
            "before": {"label": row["label"], "quality_status": row.get("quality_status") or "",
                       "quality_source": row.get("quality_source") or ""},
            "after": {"label": label, "quality_status": quality,
                      "quality_source": None if excluded else "reviewed"},
            "action": "excluded" if excluded else "reviewed",
        })
        if not excluded:
            output_rows.append({**row, "label": label, "quality_status": quality,
                                "quality_source": "reviewed"})
    if len(changes) != len(events):
        raise ValueError("Batch contains a source absent from the base manifest")

    version_id = digest(approval_raw)
    destination = versions_root / version_id
    versions_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=versions_root) as temp:
        stage = Path(temp) / "version"
        stage.mkdir()
        output = stage / "manifest.csv"
        for field in ("quality_status", "quality_source"):
            if field not in fields:
                fields.append(field)
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)
        lineage = {
            "schema": 1,
            "version_id": version_id,
            "manifest_sha256": digest(output.read_bytes()),
            "approval_sha256": version_id,
            "approval": approval,
            "changes": changes,
            "held_out_event_ids": [event["event_id"] for event in batch["held_out"]],
        }
        (stage / "lineage.json").write_bytes(json_bytes(lineage))
        try:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            stage.rename(destination)
        except OSError:
            if not destination.is_dir() or destination.is_symlink():
                raise
            if any((destination / name).is_symlink()
                   or (destination / name).read_bytes() != (stage / name).read_bytes()
                   for name in ("manifest.csv", "lineage.json")):
                raise ValueError("Published version differs from the approved artifacts")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    batch = commands.add_parser("batch")
    approve_command = commands.add_parser("approve")
    publication = commands.add_parser("publish")
    for command, fields in (
        (batch, ("queue-root", "base-manifest", "dataset-root", "output")),
        (approve_command, ("batch", "review-export", "output")),
        (publication, ("batch", "review-export", "approval", "base-manifest",
                       "dataset-root", "versions-root")),
    ):
        for field in fields:
            command.add_argument(f"--{field}", type=Path, required=True)
    approve_command.add_argument("--reviewer", required=True)
    args = parser.parse_args()
    try:
        if args.command == "batch":
            batch_id, count, held_out = snapshot(
                args.queue_root, args.base_manifest, args.dataset_root, args.output
            )
            print(f"Batch {batch_id}: {count} train tasks, {held_out} held-out events skipped")
        elif args.command == "approve":
            record = approve(args.batch, args.review_export, args.reviewer, args.output)
            print(f"Approved batch {record['batch_id']} export SHA256 {record['export_sha256']}")
        else:
            destination = publish(args.batch, args.review_export, args.approval, args.base_manifest,
                                  args.dataset_root, args.versions_root)
            print(f"Published {destination / 'manifest.csv'}")
    except (OSError, ValueError, csv.Error, Image.DecompressionBombError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
