"""Manually compare pretrained 224px candidates and score one sealed holdout."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights

from . import score
from .network import ARCHITECTURE, DOCUMENT_LABELS, IMAGE_SIZE, MAX_IMAGE_SIZE, MEAN, QUALITY_LABELS, STD, build_model, preprocessing
from .quality_metrics import calibration, classification_metrics, coverage, fit_temperature, select_threshold
from .train import IGNORE_QUALITY, Images, load_rows, publish_bundle, resolve_image_path, run_epoch

SEED = 42
GOAL = 0.85
CANDIDATES = {"frozen": (0, 0.0), "fine_tuned": (4, 0.00003)}
HEAD_LR = 0.001
WEIGHT_DECAY = 0.01


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, strict=True))


def split_of(row):
    return "val" if row["split"] == "validation" else row["split"]


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, report):
    temporary = path.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.rename(path)


def require_classes(rows, split):
    if {label for _, label, _ in rows} != set(range(len(DOCUMENT_LABELS))):
        raise ValueError(f"{split} must contain both document classes")


def training_setup(model, trainable_blocks, backbone_lr):
    blocks = list(model.features.children())
    frozen = len(blocks) - trainable_blocks
    for block in blocks[:frozen]:
        block.requires_grad_(False)
    groups = [{"params": model.classifier.parameters(), "lr": HEAD_LR}]
    if trainable_blocks:
        groups.append({"params": (p for p in model.features.parameters() if p.requires_grad),
                       "lr": backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY), frozen


class DevelopmentImages(Dataset):
    """Decode and resize the development split once, applying augmentation per epoch."""

    def __init__(self, root, rows, training, workers, crop_mode, image_size):
        root = root.resolve()
        resize, *tensor_transforms = preprocessing(image_size, crop_mode=crop_mode).transforms

        def read(row):
            with Image.open(resolve_image_path(root, row[0])) as image:
                return resize(ImageOps.exif_transpose(image).convert("RGB"))

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            self.images = list(executor.map(read, rows))
        augmentation = [
            transforms.RandomAffine(degrees=7, translate=(0.04, 0.04), scale=(0.95, 1.05),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ] if training else []
        self.rows = rows
        self.transform = transforms.Compose([*augmentation, *tensor_transforms])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        _, label, quality = self.rows[index]
        return self.transform(self.images[index]), label, quality


def loader(dataset, args, training=False):
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=training,
        num_workers=args.workers, pin_memory=args.device.type == "cuda",
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(SEED),
    )


@torch.inference_mode()
def predict(model, batches, device):
    model.eval()
    outputs, labels, qualities = [], [], []
    for images, target, quality in batches:
        outputs.append(model(images.to(device, non_blocking=True)).cpu())
        labels.append(target)
        qualities.append(quality)
    return torch.cat(outputs), torch.cat(labels), torch.cat(qualities)


def document_metrics(outputs, targets, temperature=1.0):
    logits = outputs[:, :len(DOCUMENT_LABELS)]
    result = classification_metrics(targets, logits.argmax(dim=1), DOCUMENT_LABELS)
    scores, confidence, correct = calibration(logits, targets, temperature)
    return {**result, "calibration": scores}, confidence, correct


def api_coverage(outputs, confidence, correct, threshold, review_all):
    if outputs.shape[1] == len(DOCUMENT_LABELS):
        return {"retained": 0, "coverage": 0.0, "retained_accuracy": None}
    predicted_clear = outputs[:, len(DOCUMENT_LABELS):].argmax(1) == QUALITY_LABELS.index("Clear")
    # Document scores are in [0.5, 1], so -1 excludes non-Clear decisions.
    accepted_confidence = confidence.masked_fill(~predicted_clear, -1)
    return coverage(accepted_confidence, correct, threshold, review_all)


def quality_metrics(outputs, targets, train_support):
    mask = targets != IGNORE_QUALITY
    support = dict(zip(QUALITY_LABELS, torch.bincount(targets[mask], minlength=len(QUALITY_LABELS)).tolist()))
    if not any(train_support.values()) or not mask.any():
        return {"status": "pending", "reviewed_samples": mask.sum().item(),
                "reviewed_support": support, "training_support": train_support}
    result = classification_metrics(targets[mask], outputs[mask, len(DOCUMENT_LABELS):].argmax(1), QUALITY_LABELS)
    status = "measured" if all(support.values()) and all(train_support.values()) else "partial"
    return {"status": status, "training_support": train_support, **result}


def report_groups(raw_rows, targets, outputs, field):
    if len(raw_rows) != targets.numel() or len(raw_rows) != outputs.size(0):
        raise ValueError("Evaluation rows differ from manifest records")
    groups = {}
    for index, row in enumerate(raw_rows):
        groups.setdefault(row[field] or "", []).append(index)
    predicted = outputs[:, :len(DOCUMENT_LABELS)].argmax(1)
    return {group: classification_metrics(targets[indices], predicted[indices], DOCUMENT_LABELS)
            for group, indices in groups.items()}


def selection_rank(document, quality):
    document_f1 = document["macro_f1"]
    joint_f1 = min(document_f1, quality["macro_f1"]) if quality["status"] == "measured" else document_f1
    return joint_f1, document_f1, document["accuracy"], -document["calibration"]["nll"]


def fit_candidate(name, splits, development, args, log):
    torch.manual_seed(SEED)
    blocks, backbone_lr = CANDIDATES[name]
    support = Counter(target for _, _, target in splits["train"])
    quality_support = {label: support[i] for i, label in enumerate(QUALITY_LABELS)}
    has_quality = any(support[i] for i in range(len(QUALITY_LABELS)))
    model = build_model(has_quality, MobileNet_V3_Small_Weights.IMAGENET1K_V1).to(args.device)
    optimizer, frozen = training_setup(model, blocks, backbone_lr)
    quality_weights = None
    if has_quality:
        # Temper inverse-frequency weighting where reviewed defects are scarce.
        counts = torch.tensor(list(quality_support.values()), dtype=torch.float32)
        quality_weights = counts.clamp_min(1).rsqrt().to(args.device)
    train_loader = loader(development["train"], args, training=True)
    val_loader = loader(development["val"], args)
    best = None
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, train_loader, args.device, has_quality, optimizer,
                             frozen_feature_blocks=frozen, quality_weights=quality_weights)
        outputs, targets, qualities = predict(model, val_loader, args.device)
        validation, _, _ = document_metrics(outputs, targets)
        quality = quality_metrics(outputs, qualities, quality_support)
        row = {"candidate": name, "epoch": epoch, "train": training, "validation_document": validation,
               "validation_quality": quality}
        log.write(json.dumps(row, allow_nan=False) + "\n")
        log.flush()
        print(json.dumps({"candidate": name, "epoch": epoch, "train": training["document_accuracy"],
                          "val_accuracy": validation["accuracy"], "val_macro_f1": validation["macro_f1"],
                          "val_nll": validation["calibration"]["nll"],
                          "val_quality_macro_f1": quality.get("macro_f1")}), flush=True)
        mlflow.log_metrics({f"{name}_val_macro_f1": validation["macro_f1"],
                            f"{name}_val_accuracy": validation["accuracy"],
                            f"{name}_val_nll": validation["calibration"]["nll"],
                            **{f"{name}_train_{key}": value for key, value in training.items()},
                            **({f"{name}_val_quality_macro_f1": quality["macro_f1"]}
                               if "macro_f1" in quality else {})}, step=epoch)
        rank = selection_rank(validation, quality)
        if best is None or rank > best["rank"]:
            best = {"rank": rank, "epoch": epoch, "validation": validation, "outputs": outputs,
                    "targets": targets, "qualities": qualities,
                    "weights": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    "quality_support": quality_support}
    return best


def check_development(rows, reference):
    original = {row["image_id"]: row for row in reference}
    if len(original) != len(reference) or any(split_of(row) not in ("train", "val") for row in reference):
        raise ValueError("Reference must be the fixed seed-42 train/validation manifest")
    for row in rows:
        base = original.get(row["image_id"])
        if base is None or split_of(row) != split_of(base) or any(
            row[key] != base[key] for key in ("image_path", "template_id", "document_id")
        ):
            raise ValueError("Development row changed fixed ETL identity or split")
        if split_of(row) == "val" and any(row.get(key, "") != base.get(key, "")
                                          for key in ("label", "quality_status", "quality_source")):
            raise ValueError("Validation labels must match the fixed reference")
    for split in ("train", "val"):
        supplied = {row["image_id"] for row in rows if split_of(row) == split}
        expected = {row["image_id"] for row in reference if split_of(row) == split}
        if (split == "val" and supplied != expected) or not supplied <= expected:
            raise ValueError(f"Incomplete or altered fixed {split} assignment")


def start_tracking(output):
    tracking = os.getenv("MLFLOW_TRACKING_URI", "")
    if not tracking or tracking.startswith("file:"):
        tracking = (output / "mlruns").resolve().as_uri()
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking)
    run_id = os.getenv("MLFLOW_RUN_ID") or None
    if run_id is None:
        mlflow.set_experiment("fixed-split-mobilenetv3")
    return mlflow.start_run(run_id=run_id)


def compare(args):
    base = read_csv(args.development)
    rows = base
    lineage = None
    if args.approved_version:
        rows, lineage = approved_development(args, base)
    check_development(rows, base)
    args.output.mkdir(parents=True)
    snapshot = args.output / "development.csv"
    write_csv(snapshot, rows)
    splits = load_rows(snapshot, args.dataset_root)
    require_classes(splits["val"], "Validation")
    development = {
        split: DevelopmentImages(args.dataset_root, splits[split], split == "train",
                                 args.workers, args.crop_mode, args.image_size)
        for split in ("train", "val")
    }
    with start_tracking(args.output) as run, (args.output / "epochs.jsonl").open("x") as log:
        mlflow.log_params({"image_size": args.image_size, "seed": SEED, "epochs": args.epochs,
                           "batch_size": args.batch_size, "device": str(args.device),
                           "cpu_threads": args.threads, "workers": args.workers,
                           "torch_version": torch.__version__, "pretrained_weights": "IMAGENET1K_V1",
                           "crop_mode": args.crop_mode,
                           "head_lr": HEAD_LR, "quality_loss_weighting": "inverse_sqrt_frequency", "fine_tuned_backbone_lr": CANDIDATES["fine_tuned"][1],
                           "fine_tuned_blocks": CANDIDATES["fine_tuned"][0], "weight_decay": WEIGHT_DECAY,
                           "development_sha256": sha256(snapshot), "held_out_sha256": args.held_out_sha256})
        candidates = {name: fit_candidate(name, splits, development, args, log) for name in args.candidates}
        name = max(candidates, key=lambda candidate: candidates[candidate]["rank"])
        best = candidates[name]
        outputs, targets = best["outputs"], best["targets"]
        temperature = fit_temperature(outputs[:, :len(DOCUMENT_LABELS)], targets)
        validation, confidence, correct = document_metrics(outputs, targets, temperature)
        threshold, review_all = select_threshold(confidence, correct, args.retained_target)
        support = best["quality_support"]
        has_quality = any(support.values())
        model = build_model(has_quality)
        model.load_state_dict(best["weights"])
        metadata = {"format_version": 1, "architecture": ARCHITECTURE,
                    "document_labels": DOCUMENT_LABELS, "quality_labels": QUALITY_LABELS,
                    "has_quality_head": has_quality, "preprocessing": {
                        "image_size": args.image_size, "mean": MEAN, "std": STD, "crop_mode": args.crop_mode},
                    "confidence_threshold": threshold, "document_temperature": temperature,
                    "document_review_all": review_all}
        publish_bundle(args.output / "bundle", model, metadata)
        selection = {
            "selection_schema": 1, "mlflow_run_id": run.info.run_id, "seed": SEED,
            "runtime": {"torch_version": torch.__version__, "device": str(args.device),
                        "threads": args.threads, "workers": args.workers, "epochs": args.epochs,
                        "batch_size": args.batch_size, "crop_mode": args.crop_mode,
                        "image_size": args.image_size},
            "reference_development_sha256": sha256(args.development),
            "development_sha256": sha256(snapshot), "held_out_sha256": args.held_out_sha256,
            "weights_sha256": sha256(args.output / "bundle" / "weights.pt"),
            "metadata_sha256": sha256(args.output / "bundle" / "metadata.json"),
            "approval_lineage": lineage, "selected": name, "selected_epoch": best["epoch"],
            "candidates": {key: {"best_epoch": value["epoch"], "validation": value["validation"]}
                           for key, value in candidates.items()},
            "validation": validation, "temperature": temperature, "threshold": threshold,
            "review_all": review_all, "retained_target": args.retained_target,
            "validation_coverage": coverage(confidence, correct, threshold, review_all),
            "validation_api_coverage": api_coverage(outputs, confidence, correct, threshold, review_all),
            "validation_templates": report_groups(
                [row for row in rows if split_of(row) == "val"], targets, outputs, "template_id"),
            "quality_training_support": support,
            "validation_quality": quality_metrics(outputs, best["qualities"], support),
        }
        write_json(args.output / "selection.json", selection)
        mlflow.log_artifact(str(args.output / "selection.json"))
        mlflow.log_artifact(str(args.output / "epochs.jsonl"))
        print(f"Selected {name} at epoch {best['epoch']}. See {args.output / 'selection.json'}", flush=True)


def approved_development(args, reference):
    """Rebuild Data's approved version independently and verify both files."""
    from labeling.hitl import publish

    required = (args.batch, args.approval, args.review_export, args.base_manifest)
    if not all(required):
        raise ValueError("Approved version requires batch, approval, raw export and base manifest")
    base_rows = read_csv(args.base_manifest)
    if [row for row in base_rows if split_of(row) != "test"] != reference:
        raise ValueError("Base manifest development rows differ from the fixed reference")
    with tempfile.TemporaryDirectory(prefix="ekyc-approved-verify-") as temporary:
        rebuilt = publish(args.batch, args.review_export, args.approval, args.base_manifest,
                          args.dataset_root, Path(temporary))
        # Data's publication function returns the version directory.
        for name in ("manifest.csv", "lineage.json"):
            if sha256(rebuilt / name) != sha256(args.approved_version / name):
                raise ValueError(f"Approved version does not match verified {name}")
        lineage = json.loads((rebuilt / "lineage.json").read_text())
        rows = read_csv(rebuilt / "manifest.csv")
    return [row for row in rows if split_of(row) != "test"], {
        "version_id": lineage["version_id"], "manifest_sha256": lineage["manifest_sha256"],
        "approval_sha256": lineage["approval_sha256"], "export_sha256": lineage["approval"]["export_sha256"],
        "base_manifest_sha256": lineage["approval"]["base_manifest_sha256"],
    }


def seal(args):
    reports = [(path, json.loads((path / "selection.json").read_bytes())) for path in args.comparison]
    binding = ("reference_development_sha256", "development_sha256", "held_out_sha256", "approval_lineage")
    baseline = reports[0][1]
    for path, report in reports:
        if report["selection_schema"] != 1 or report["seed"] != SEED or any(
            report[key] != baseline[key] for key in binding
        ):
            raise ValueError(f"Comparison uses a different development roster or lineage: {path}")

    def validation_rank(entry):
        report = entry[1]
        validation = report["candidates"][report["selected"]]["validation"]
        return selection_rank(validation, report.get("validation_quality", {"status": "pending"}))

    winner, _ = max(reports, key=validation_rank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Final selection directory must be new")
    with tempfile.TemporaryDirectory(prefix=".seal-", dir=args.output.parent) as temporary:
        staged = Path(temporary) / "sealed"
        staged.mkdir()
        for name in ("selection.json", "development.csv"):
            shutil.copyfile(winner / name, staged / name)
        shutil.copytree(winner / "bundle", staged / "bundle")
        write_json(staged / "sealed.json", {
            "selected_comparison": str(winner.resolve()),
            "selection_sha256": sha256(staged / "selection.json"),
            "comparison_validation_rank": {
                str(path.resolve()): validation_rank((path, report)) for path, report in reports
            },
        })
        staged.rename(args.output)
    print(f"Sealed validation winner {winner} at {args.output}", flush=True)


def claim_evaluation(output, holdout_hash):
    with (output / "held-out.started").open("x", encoding="utf-8") as marker:
        json.dump({"held_out_sha256": holdout_hash, "started_utc": datetime.now(timezone.utc).isoformat()},
                  marker)
        marker.flush()
        os.fsync(marker.fileno())


def evaluate(args):
    output = args.output
    lock = json.loads((output / "sealed.json").read_text())
    selection_path = output / "selection.json"
    if sha256(selection_path) != lock["selection_sha256"]:
        raise ValueError("Sealed selection changed")
    selection = json.loads(selection_path.read_text())
    if (output / "held-out.started").exists() or (output / "held-out.json").exists():
        raise ValueError("Held-out evaluation already started for this selection")
    if selection["selection_schema"] != 1 or selection["seed"] != SEED:
        raise ValueError("Invalid selection record")
    holdout_hash = sha256(args.held_out)
    if selection["held_out_sha256"] is not None and holdout_hash != selection["held_out_sha256"]:
        raise ValueError("Held-out manifest differs from the sealed checksum")
    development = output / "development.csv"
    bundle = output / "bundle"
    if (sha256(development) != selection["development_sha256"]
            or sha256(bundle / "metadata.json") != selection["metadata_sha256"]
            or sha256(bundle / "weights.pt") != selection["weights_sha256"]):
        raise ValueError("Selection inputs or bundle changed after validation")
    approved = selection["approval_lineage"]
    if approved and sha256(args.base_manifest) != approved["base_manifest_sha256"]:
        raise ValueError("Held-out reference does not match the approved base manifest")
    held_out = read_csv(args.held_out)
    base = read_csv(args.base_manifest)
    if held_out != [row for row in base if split_of(row) == "test"]:
        raise ValueError("Held-out rows differ from the authoritative fixed test split")
    dev_rows = read_csv(development)
    check_development(dev_rows, [row for row in base if split_of(row) != "test"])
    if not approved and dev_rows != [row for row in base if split_of(row) != "test"]:
        raise ValueError("Development snapshot differs from the authoritative fixed split")
    with tempfile.TemporaryDirectory(prefix="ekyc-holdout-check-") as temporary:
        combined = Path(temporary) / "combined.csv"
        fields = dict.fromkeys(key for row in (*dev_rows, *held_out) for key in row)
        write_csv(combined, [{key: row.get(key, "") for key in fields} for row in (*dev_rows, *held_out)])
        splits = load_rows(combined, args.dataset_root)
    require_classes(splits["test"], "Held-out")
    # Match load_rows' Outside_Scope exclusion and verify positional identity
    # before claiming the one evaluation. The evaluation loader does not shuffle.
    evaluated_rows = [row for row in held_out if (row.get("quality_status") or "").strip() != "Outside_Scope"]
    expected_order = [(row["image_path"].strip(), DOCUMENT_LABELS.index(row["label"].strip()))
                      for row in evaluated_rows]
    if expected_order != [(path, label) for path, label, _ in splits["test"]]:
        raise ValueError("Held-out dataset order differs from manifest records")
    os.environ["AZUREML_MODEL_DIR"] = str(bundle)
    score.init()
    # The exclusive marker also blocks concurrent or interrupted evaluations.
    claim_evaluation(output, holdout_hash)
    model, transform, metadata = score._predictor
    test_images = Images(args.dataset_root, splits["test"], metadata["preprocessing"]["image_size"], transform=transform)
    outputs, targets, qualities = predict(model.to(args.device), loader(test_images, args), args.device)
    if targets.tolist() != [label for _, label in expected_order]:
        raise ValueError("Held-out prediction targets differ from manifest order")
    temperature = metadata["document_temperature"]
    result, confidence, correct = document_metrics(outputs, targets, temperature)
    raw_calibration, _, _ = calibration(outputs[:, :len(DOCUMENT_LABELS)], targets, 1.0)
    high_confidence = 0.90
    errors = [
        {"image_id": row["image_id"], "template_id": row["template_id"], "domain": row.get("domain"),
         "expected": row["label"], "predicted": DOCUMENT_LABELS[prediction], "score": value}
        for row, prediction, value, matched in zip(
            evaluated_rows, outputs[:, :len(DOCUMENT_LABELS)].argmax(1).tolist(),
            confidence.tolist(), correct.tolist(), strict=True
        ) if not matched
    ]
    evaluated_quality = qualities != IGNORE_QUALITY
    accepted_quality = 0
    incorrect_accepted = 0
    if outputs.shape[1] > len(DOCUMENT_LABELS) and not metadata["document_review_all"]:
        clear = QUALITY_LABELS.index("Clear")
        accepted = (evaluated_quality & (confidence >= metadata["confidence_threshold"])
                    & (outputs[:, len(DOCUMENT_LABELS):].argmax(1) == clear))
        accepted_quality = accepted.sum().item()
        incorrect_accepted = (accepted & (~correct | (qualities != clear))).sum().item()
    report = {
        "held_out_sha256": holdout_hash, "selection_sha256": sha256(output / "selection.json"),
        "selected": selection["selected"], "selected_epoch": selection["selected_epoch"],
        "minimum_accuracy": GOAL, "acceptance": "pass" if result["accuracy"] >= GOAL else "fail",
        "document": result, "uncalibrated": raw_calibration,
        "error_audit": {"confidence_threshold": high_confidence,
                        "high_confidence_errors": sum(error["score"] >= high_confidence for error in errors),
                        "errors": sorted(errors, key=lambda error: error["score"], reverse=True)},
        "document_review_policy": {"threshold": metadata["confidence_threshold"],
                                   "temperature": temperature, "review_all": metadata["document_review_all"],
                                   **coverage(confidence, correct, metadata["confidence_threshold"],
                                              metadata["document_review_all"])},
        "api_coverage": api_coverage(outputs, confidence, correct,
                                     metadata["confidence_threshold"], metadata["document_review_all"]),
        "reviewed_acceptance": {"accepted": accepted_quality, "incorrect": incorrect_accepted,
                                "accuracy": (1 - incorrect_accepted / accepted_quality
                                             if accepted_quality else None)},
        "quality": quality_metrics(outputs, qualities, selection["quality_training_support"]),
        "templates": report_groups(evaluated_rows, targets, outputs, "template_id"),
    }
    if "domain" in held_out[0]:
        report["domains"] = report_groups(evaluated_rows, targets, outputs, "domain")
    write_json(output / "held-out.json", report)
    print(json.dumps({"accuracy": result["accuracy"], "macro_f1": result["macro_f1"],
                      "acceptance": report["acceptance"], "report": str(output / "held-out.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("compare", "evaluate"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--dataset-root", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--threads", type=int, default=6)
        sub.add_argument("--batch-size", type=int, default=32)
        sub.add_argument("--workers", type=int, default=2)
        sub.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
        sub.add_argument("--base-manifest", type=Path, required=name == "evaluate")
        if name == "compare":
            sub.add_argument("--development", type=Path, required=True)
            sub.add_argument("--epochs", type=int, default=3)
            sub.add_argument("--image-size", type=int, choices=range(32, MAX_IMAGE_SIZE + 1),
                             default=IMAGE_SIZE, metavar="[32-1024]")
            sub.add_argument("--candidates", nargs="+", choices=tuple(CANDIDATES), default=tuple(CANDIDATES))
            sub.add_argument("--crop-mode", choices=("full_frame", "center_square"), default="full_frame")
            sub.add_argument("--retained-target", type=float, default=GOAL)
            sub.add_argument("--held-out-sha256")
            sub.add_argument("--approved-version", type=Path)
            sub.add_argument("--approval", type=Path)
            sub.add_argument("--review-export", type=Path)
            sub.add_argument("--batch", type=Path)
        else:
            sub.add_argument("--held-out", type=Path, required=True)
    sealing = subparsers.add_parser("seal")
    sealing.add_argument("--comparison", type=Path, action="append", required=True)
    sealing.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seal":
        seal(args)
        return
    if min(args.threads, args.batch_size) < 1 or args.workers < 0:
        parser.error("Threads and batch size must be positive, and workers nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Torch runtime")
    args.device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.command == "compare":
        if args.epochs < 1 or not 0 < args.retained_target <= 1:
            parser.error("Epochs must be positive and retained target must be in (0, 1]")
        if args.held_out_sha256 and (
            len(args.held_out_sha256) != 64 or set(args.held_out_sha256) - set("0123456789abcdef")
        ):
            parser.error("Expected a lowercase hexadecimal held-out SHA-256")
        supplied = (args.approved_version, args.approval, args.review_export, args.batch, args.base_manifest)
        if any(supplied) and not all(supplied):
            parser.error("HITL runs require approved version, approval, review export, batch and base manifest")
        compare(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
