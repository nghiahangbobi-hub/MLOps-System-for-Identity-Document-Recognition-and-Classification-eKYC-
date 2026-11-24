"""Evaluate an existing classifier on new capture domains without tuning it."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

from PIL import Image, ImageOps
import torch

from modeling import score
from modeling.network import DOCUMENT_LABELS


TEMPLATE_CLASSES = {"_id": "ID_Card", "_passport": "Passport"}


def read_single_capture_archive(path, templates):
    """Index one capture per template and document number."""
    images = {}
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or len(parts) != 3 or parts[0] != "images":
                continue
            filename = PurePosixPath(parts[2])
            if parts[1] not in templates or filename.suffix.lower() not in {".jpg", ".jpeg", ".tif", ".tiff"}:
                continue
            key = parts[1], filename.stem
            if key in images:
                raise ValueError(f"Duplicate capture identity {key} in {path.name}")
            images[key] = member
    if {template for template, _ in images} != templates:
        raise ValueError(f"Missing requested template in {path.name}")
    return images


def evaluate(path, images, model, transform, metadata, batch_size):
    temperature = metadata.get("document_temperature")
    threshold = metadata["confidence_threshold"]
    rows = []
    keys = sorted(images, key=lambda key: images[key].offset_data)

    def decode(key):
        # Independent handles prevent concurrent seeks on a shared tar stream.
        with tarfile.open(path, "r:") as archive:
            with Image.open(archive.extractfile(images[key])) as original:
                return transform(ImageOps.exif_transpose(original).convert("RGB"))

    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for offset in range(0, len(keys), batch_size):
            batch_keys = keys[offset:offset + batch_size]
            logits = model(torch.stack(list(pool.map(decode, batch_keys))))[:, :len(DOCUMENT_LABELS)]
            if temperature is not None:
                logits = logits.double() / temperature
            confidences, predictions = logits.softmax(dim=1).max(dim=1)
            for (template, document), confidence, prediction in zip(batch_keys, confidences.tolist(), predictions.tolist()):
                label = TEMPLATE_CLASSES["_" + template.rsplit("_", 1)[-1]]
                rows.append({"template": template, "document": document, "label": label,
                             "prediction": DOCUMENT_LABELS[prediction], "confidence": confidence,
                             "below_document_threshold": confidence < threshold})
    return rows


def summarize(rows):
    counts = Counter((row["label"], row["prediction"]) for row in rows)
    matrix = [[counts[truth, prediction] for prediction in DOCUMENT_LABELS] for truth in DOCUMENT_LABELS]
    size = len(rows)
    per_class = {}
    for i, label in enumerate(DOCUMENT_LABELS):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predictions = sum(row[i] for row in matrix)
        per_class[label] = {"support": support, "recall": tp / support,
                            "precision": tp / predictions if predictions else 0,
                            "f1": 2 * tp / (support + predictions)}
    return {"samples": size, "correct": sum(matrix[i][i] for i in range(len(DOCUMENT_LABELS))),
            "accuracy": sum(matrix[i][i] for i in range(len(DOCUMENT_LABELS))) / size,
            "macro_f1": sum(result["f1"] for result in per_class.values()) / len(DOCUMENT_LABELS),
            "confusion_true_rows_predicted_columns": matrix, "per_class": per_class,
            "below_document_threshold": sum(row["below_document_threshold"] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--split-record", type=Path, required=True,
                        help="Original split JSON with train, validation and test under layouts")
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path")
    if len({path.name for path in args.archive}) != len(args.archive):
        parser.error("Archive basenames must be distinct")
    layouts = json.loads(args.split_record.read_text())["layouts"]
    templates = set(layouts["test"])
    if templates.intersection(set(layouts["train"]) | set(layouts["validation"])):
        parser.error("Held-out layouts overlap with development")
    suffixes = {"_" + name.rsplit("_", 1)[-1] for name in templates}
    if suffixes != TEMPLATE_CLASSES.keys() or "rus_internalpassport" in templates:
        parser.error("Held-out layouts must include both compatible document classes")

    os.environ["AZUREML_MODEL_DIR"] = str(args.bundle)
    torch.set_num_threads(4)
    score.init()
    model, transform, metadata = score._predictor
    results = {}
    reference_keys = None
    for archive in args.archive:
        images = read_single_capture_archive(archive, templates)
        if reference_keys is not None and reference_keys != images.keys():
            parser.error(f"Held-out document roster differs in {archive.name}")
        reference_keys = images.keys()
        rows = evaluate(archive, images, model, transform, metadata, batch_size=16)
        results[archive.name] = {"metrics": summarize(rows), "predictions": rows}
        print(archive.name, json.dumps(results[archive.name]["metrics"]), flush=True)
    report = {"protocol": "fixed model, original held-out layouts, matched mock document roster",
              "split_record_sha256": hashlib.sha256(args.split_record.read_bytes()).hexdigest(),
              "weights_sha256": metadata.get("weights_sha256"), "labels": DOCUMENT_LABELS,
              "confidence_threshold": metadata["confidence_threshold"], "results": results}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
