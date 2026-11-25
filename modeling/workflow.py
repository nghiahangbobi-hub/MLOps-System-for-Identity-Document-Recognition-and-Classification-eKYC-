"""Train, evaluate and qualify one eKYC candidate from a versioned manifest."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .quality_compare import GOAL, QUALITY_LABELS, read_csv, require_classes, sha256, split_of, write_csv, write_json
from .train import load_rows


def quality_support(rows):
    counts = Counter((split_of(row), row["quality_status"]) for row in rows
                     if row.get("quality_source", "").strip().lower() == "reviewed")
    return {split: {label: counts[split, label] for label in QUALITY_LABELS}
            for split in ("train", "val", "test")}


def release_issues(report, baseline):
    issues = []
    if baseline:
        issues.append("Diagnostic baseline")
    document = report["document"]
    if min(document["accuracy"], document["macro_f1"], *document["per_class_recall"].values()) < GOAL:
        issues.append(f"Document accuracy, macro-F1 or class recall below {GOAL:.0%}")
    quality = report["quality"]
    if quality["status"] != "measured" or quality["macro_f1"] < GOAL:
        issues.append(f"Quality evaluation requires all four labels and at least {GOAL:.0%} macro-F1")
    reviewed = report["reviewed_acceptance"]
    if not reviewed["accepted"] or reviewed["accuracy"] < GOAL:
        issues.append(f"Auto-accepted reviewed images require at least {GOAL:.0%} correct document and quality")
    return issues


def train(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new or empty run directory")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.csv"
    with args.manifest.open("rb") as source, manifest.open("xb") as target:
        shutil.copyfileobj(source, target)
    rows = read_csv(manifest)
    counts = quality_support(rows)
    missing = [f"{split}/{label}" for split, support in counts.items()
               for label, count in support.items() if not count]
    if missing and not args.baseline:
        print("Incomplete quality coverage: " + ", ".join(missing), flush=True)
    if not args.baseline and any(row.get("quality_status") and row.get("quality_source") != "reviewed"
                                for row in rows):
        raise ValueError("The release workflow requires Label Studio review provenance")
    verified = load_rows(manifest, args.dataset_root)
    require_classes(verified["val"], "Validation")
    require_classes(verified["test"], "Held-out")

    development = args.output / "development.csv"
    holdout = args.output / "test.csv"
    for path, split_rows in ((development, [row for row in rows if split_of(row) in ("train", "val")]),
                             (holdout, [row for row in rows if split_of(row) == "test"])):
        if not split_rows:
            raise ValueError(f"Missing rows for {path.name}")
        write_csv(path, split_rows)

    comparison = args.output / "comparison"
    selection = args.output / "selected"
    runtime = ["--dataset-root", args.dataset_root, "--device", args.device,
               "--threads", args.threads, "--batch-size", args.batch_size, "--workers", args.workers]
    for options in (
        ["compare", *runtime, "--development", development, "--held-out-sha256", sha256(holdout),
         "--epochs", args.epochs, "--output", comparison],
        ["seal", "--comparison", comparison, "--output", selection],
        ["evaluate", *runtime, "--base-manifest", manifest, "--held-out", holdout, "--output", selection],
    ):
        subprocess.run([sys.executable, "-m", "modeling.quality_compare", *map(str, options)], check=True)
    report = selection / "held-out.json"
    issues = release_issues(json.loads(report.read_text()), args.baseline)
    if missing and not args.baseline:
        issues.append("Missing human-reviewed labels: " + ", ".join(missing))
    result = {"state": "blocked" if issues else "ready_for_staging", "baseline": args.baseline,
              "issues": issues, "reviewed_quality_support": counts,
              "manifest_sha256": sha256(manifest), "holdout_report_sha256": sha256(report)}
    write_json(args.output / "workflow.json", result)
    print(json.dumps(result), flush=True)
    if args.bundle_output:
        if args.bundle_output.exists() and any(args.bundle_output.iterdir()):
            raise ValueError("Choose a new or empty bundle output directory")
        if issues:
            args.bundle_output.mkdir(parents=True, exist_ok=True)
            write_json(args.bundle_output / "blocked.json", result)
        else:
            bundle = check_release(args.output)
            shutil.copytree(bundle, args.bundle_output, dirs_exist_ok=True)


def check_release(run):
    workflow = json.loads((run / "workflow.json").read_text())
    selected = run / "selected"
    report_path = selected / "held-out.json"
    report = json.loads(report_path.read_text())
    issues = release_issues(report, workflow["baseline"])
    if issues or workflow["state"] != "ready_for_staging":
        raise ValueError("Candidate blocked: " + "; ".join(workflow.get("issues") or issues))
    selection_path = selected / "selection.json"
    selection = json.loads(selection_path.read_text())
    bundle = selected / "bundle"
    for path, expected in (
        (run / "manifest.csv", workflow["manifest_sha256"]),
        (run / "test.csv", report["held_out_sha256"]),
        (report_path, workflow["holdout_report_sha256"]),
        (selection_path, report["selection_sha256"]),
        (selected / "development.csv", selection["development_sha256"]),
        (bundle / "metadata.json", selection["metadata_sha256"]),
        (bundle / "weights.pt", selection["weights_sha256"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"Release evidence changed: {path.name}")
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    for name in ("manifest", "dataset-root", "output"):
        training.add_argument(f"--{name}", type=Path, required=True)
    training.add_argument("--bundle-output", type=Path)
    training.add_argument("--baseline", action="store_true", help="Allow partial labeling for a diagnostic run.")
    training.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--threads", type=int, default=6)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--workers", type=int, default=0)
    check = commands.add_parser("check-release")
    check.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "train":
            train(args)
        else:
            print(check_release(args.run))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
