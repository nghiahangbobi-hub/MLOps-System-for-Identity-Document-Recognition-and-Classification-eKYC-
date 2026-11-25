"""Stage only Azure definitions and Model-owned Python source into a new directory."""

import argparse
from pathlib import Path
import shutil

AZURE = Path(__file__).resolve().parent
MODELING = AZURE.parent / "modeling"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--training-only", action="store_true",
                        help="Stage the training job and its environment only.")
    args = parser.parse_args()
    required_files = ("workflow.py", "train.py", "score.py")
    for filename in required_files:
        entrypoint = MODELING / filename
        if not entrypoint.is_file() or entrypoint.is_symlink():
            parser.error(f"Missing Model-owned entrypoint: modeling/{filename}")
    try:
        args.destination.mkdir()
    except OSError as exc:
        parser.error(str(exc))
    try:
        stage(args.destination, args.training_only)
    except OSError as exc:
        shutil.rmtree(args.destination)
        parser.error(str(exc))


def stage(destination, training_only):
    source = destination / "src" / "modeling"
    source.mkdir(parents=True)
    for file in MODELING.rglob("*.py"):
        relative = file.relative_to(MODELING)
        if (any(part.startswith(".") for part in relative.parts) or file.match("test_*.py")
                or not file.is_file() or file.is_symlink()):
            continue
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    definitions = ["train.yml", "environment-training.yml", "conda-training.yml"]
    if not training_only:
        definitions.extend(("endpoint.yml", "deployment.yml", "environment-inference.yml",
                            "conda-inference.yml"))
    for filename in definitions:
        shutil.copyfile(AZURE / filename, destination / filename)


if __name__ == "__main__":
    main()
