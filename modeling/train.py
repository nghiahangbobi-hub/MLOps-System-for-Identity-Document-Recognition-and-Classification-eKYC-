"""Train the document classifier and optional masked quality classifier."""

import argparse
import csv
import hashlib
import json
import os
import stat
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import mlflow
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights

from .network import (
    ARCHITECTURE, DOCUMENT_LABELS, IMAGE_SIZE, MAX_IMAGE_SIZE, MEAN, QUALITY_LABELS, STD,
    build_model, preprocessing,
)

IGNORE_QUALITY = -100
SPLITS = ("train", "val", "test")


def load_rows(manifest, dataset_root=None):
    splits = defaultdict(list)
    template_splits = {}
    document_splits = {}
    image_ids = set()
    image_paths = set()
    image_files = set()
    files_by_size = defaultdict(list)
    root = Path(dataset_root).resolve() if dataset_root is not None else None
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"image_id", "image_path", "template_id", "document_id", "split", "label"}
        if required.difference(reader.fieldnames or []):
            raise ValueError("Manifest is missing required columns")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Manifest contains duplicate columns")
        for row in reader:
            if None in row:
                raise ValueError(f"Unexpected manifest fields on line {reader.line_num}")
            if any(not (row[key] or "").strip() for key in required):
                raise ValueError(f"Empty manifest field on line {reader.line_num}")
            label = row["label"].strip()
            if label not in DOCUMENT_LABELS:
                raise ValueError(f"Invalid document label: {label}")
            split = row["split"].strip().lower()
            if split == "validation":
                split = "val"
            if split not in SPLITS:
                raise ValueError(f"Invalid split: {split}")
            image_id = row["image_id"].strip()
            if image_id in image_ids:
                raise ValueError(f"Duplicate image ID: {image_id}")
            image_ids.add(image_id)
            template = row["template_id"].strip()
            document = row["document_id"].strip()
            if template_splits.setdefault(template, split) != split:
                raise ValueError(f"Template appears in multiple splits: {template}")
            if document_splits.setdefault(document, split) != split:
                raise ValueError(f"Document appears in multiple splits: {document}")
            quality = (row.get("quality_status") or "").strip()
            source = (row.get("quality_source") or "").strip().lower()
            if source not in ("", "unreviewed", "reviewed", "synthetic"):
                raise ValueError(f"Invalid quality provenance: {source}")
            if (source == "unreviewed" and quality) or (source == "reviewed" and not quality):
                raise ValueError(f"Quality status contradicts provenance: {source}")
            if source == "synthetic" and (split != "train" or quality not in QUALITY_LABELS):
                raise ValueError("Synthetic rows require a valid quality target and the train split")
            if quality == "Outside_Scope":
                continue
            if quality and quality not in QUALITY_LABELS:
                raise ValueError(f"Invalid quality status: {quality}")
            relative_path = row["image_path"].strip()
            path = resolve_image_path(root, relative_path)
            if path in image_paths:
                raise ValueError(f"Duplicate image path: {relative_path}")
            image_paths.add(path)
            if root is not None:
                info = path.stat()
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError(f"Image path is not a regular file: {relative_path}")
                file_id = (info.st_dev, info.st_ino)
                if info.st_ino:
                    if file_id in image_files:
                        raise ValueError(f"Duplicate image file: {relative_path}")
                    image_files.add(file_id)
                files_by_size[info.st_size].append((path, split))
            target = QUALITY_LABELS.index(quality) if quality else IGNORE_QUALITY
            splits[split].append((relative_path, DOCUMENT_LABELS.index(label), target))
    if {label for _, label, _ in splits["train"]} != set(range(len(DOCUMENT_LABELS))):
        raise ValueError("Train split must contain both document classes")
    for files in files_by_size.values():
        if len({split for _, split in files}) < 2:
            continue
        digest_splits = {}
        for path, split in files:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").digest()
            if digest_splits.setdefault(digest, split) != split:
                raise ValueError(f"Identical image content appears in multiple splits: {path}")
    return splits


def resolve_image_path(root, relative_path):
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Image path must be relative to dataset root: {relative_path}")
    if root is None:
        return path
    path = (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path must be relative to dataset root: {relative_path}")
    return path


class Images(Dataset):
    def __init__(self, root, rows, size, transform=None):
        self.root = root.resolve()
        self.rows = rows
        self.transform = transform if transform is not None else preprocessing(size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        relative_path, label, quality = self.rows[index]
        path = resolve_image_path(self.root, relative_path)
        with Image.open(path) as image:
            tensor = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        return tensor, label, quality


def run_epoch(model, loader, device, has_quality_head, optimizer=None, *,
              frozen_feature_blocks=0, quality_weights=None):
    training = optimizer is not None
    model.train(training)
    if training and frozen_feature_blocks:
        for block in list(model.features.children())[:frozen_feature_blocks]:
            block.eval()
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
    count = correct = quality_count = quality_correct = 0
    document_loss = quality_loss = 0.0
    with torch.set_grad_enabled(training):
        for images, labels, quality in loader:
            images, labels, quality = images.to(device), labels.to(device), quality.to(device)
            if training and not frozen_feature_blocks and images.shape[-1] == 32:
                # The final spatial map is 1×1 at size 32. BatchNorm needs
                # running statistics when the last batch contains one image.
                for layer in model.modules():
                    if isinstance(layer, torch.nn.BatchNorm2d):
                        layer.train(images.shape[0] > 1)
            logits = model(images)
            document_logits = logits[:, :len(DOCUMENT_LABELS)]
            loss = F.cross_entropy(document_logits, labels)
            count += labels.numel()
            correct += (document_logits.argmax(1) == labels).sum().item()
            document_loss += loss.detach().item() * labels.numel()
            mask = quality != IGNORE_QUALITY
            if has_quality_head and mask.any():
                quality_logits = logits[mask, len(DOCUMENT_LABELS):]
                targets = quality[mask]
                auxiliary_loss = F.cross_entropy(quality_logits, targets, weight=quality_weights)
                loss = loss + auxiliary_loss
                quality_count += targets.numel()
                quality_correct += (quality_logits.argmax(1) == targets).sum().item()
                quality_loss += auxiliary_loss.detach().item() * targets.numel()
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    metrics = {"samples": count, "quality_samples": quality_count}
    if count:
        metrics.update(document_loss=document_loss / count, document_accuracy=correct / count)
    if quality_count:
        metrics.update(quality_loss=quality_loss / quality_count, quality_accuracy=quality_correct / quality_count)
    return metrics


def publish_bundle(output_dir, model, metadata):
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Keep the upload directory untouched until both files are written and tracked.
    # The final rename accepts a new or empty destination and refuses a populated one.
    with tempfile.TemporaryDirectory(prefix=".model-stage-", dir=output_dir.parent) as staging:
        bundle = Path(staging) / "bundle"
        bundle.mkdir()
        weights_path = bundle / "weights.pt"
        torch.save(model.cpu().state_dict(), weights_path)
        with weights_path.open("rb") as stream:
            metadata = {**metadata, "weights_sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
        metadata_path = bundle / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for path in (metadata_path, weights_path):
            mlflow.log_artifact(str(path), artifact_path="model")
        bundle.rename(output_dir)


def train(args):
    torch.manual_seed(args.seed)
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise ValueError("Model output directory must be new or empty")
    with args.manifest.open("rb") as stream:
        manifest_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    splits = load_rows(args.manifest, args.dataset_root)
    has_quality_head = any(target != IGNORE_QUALITY for _, _, target in splits["train"])
    device = torch.device(args.device)
    loaders = {
        split: DataLoader(Images(args.dataset_root, splits[split], args.image_size),
                          batch_size=args.batch_size, shuffle=split == "train",
                          num_workers=args.workers)
        for split in SPLITS if splits[split]
    }
    weights = MobileNet_V3_Small_Weights.DEFAULT if args.pretrained else None
    model = build_model(has_quality_head, weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    metadata = {
        "format_version": 1,
        "architecture": ARCHITECTURE,
        "document_labels": DOCUMENT_LABELS,
        "quality_labels": QUALITY_LABELS,
        "has_quality_head": has_quality_head,
        "preprocessing": {"image_size": args.image_size, "mean": MEAN, "std": STD},
        "confidence_threshold": args.confidence_threshold,
    }
    tracking_uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_uri = Path("mlruns").resolve().as_uri()
    if tracking_uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    if args.experiment:
        mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params({
            "manifest_sha256": manifest_sha256,
            "architecture": ARCHITECTURE, "has_quality_head": has_quality_head,
            "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
            "image_size": args.image_size, "learning_rate": args.learning_rate,
            "pretrained": args.pretrained, "device": str(device),
            "confidence_threshold": args.confidence_threshold,
        })
        # Record available labels independently of whether the model has a quality head.
        coverage = {}
        for split in SPLITS:
            rows = splits[split]
            documents = Counter(label for _, label, _ in rows)
            qualities = Counter(quality for _, _, quality in rows)
            coverage[split] = {
                "document": {label: documents[index] for index, label in enumerate(DOCUMENT_LABELS)},
                "quality": {label: qualities[index] for index, label in enumerate(QUALITY_LABELS)},
                "unlabeled_quality": qualities[IGNORE_QUALITY],
            }
        mlflow.log_dict(coverage, "manifest-coverage.json")
        best_loss = float("inf")
        best_state = None
        for epoch in range(1, args.epochs + 1):
            metrics = run_epoch(model, loaders["train"], device, has_quality_head, optimizer)
            mlflow.log_metrics({f"train_{key}": value for key, value in metrics.items()}, step=epoch)
            if "val" in loaders:
                metrics = run_epoch(model, loaders["val"], device, has_quality_head)
                mlflow.log_metrics({f"val_{key}": value for key, value in metrics.items()}, step=epoch)
                loss = metrics["document_loss"] + metrics.get("quality_loss", 0)
                if loss < best_loss:
                    best_loss = loss
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        if "test" in loaders:
            metrics = run_epoch(model, loaders["test"], device, has_quality_head)
            mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
        publish_bundle(args.output_dir, model, metadata)
        print(json.dumps({"run_id": run.info.run_id, "bundle": str(args.output_dir)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or not 32 <= args.image_size <= MAX_IMAGE_SIZE or args.workers < 0:
        parser.error(
            f"Use positive epochs and batch size, image size from 32 to {MAX_IMAGE_SIZE}, "
            "and nonnegative workers"
        )
    if not 0 <= args.confidence_threshold <= 1 or not 0 < args.learning_rate < float("inf"):
        parser.error("Invalid confidence threshold or learning rate")
    train(args)


if __name__ == "__main__":
    main()
