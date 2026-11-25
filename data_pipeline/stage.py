"""Materialize only the MIDV images referenced by a release manifest."""

import argparse
import hashlib
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from .manifest import read_manifest


def materialize(dataset_root, manifest, output):
    root = dataset_root.resolve(strict=True)
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(root):
        raise ValueError("Choose a new output directory outside the dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".stage-midv-") as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        shutil.copyfile(manifest, staging / "manifest.csv")
        rows = read_manifest(staging / "manifest.csv", preserve_extra=True)
        if not rows:
            raise ValueError("Manifest contains no images")
        for row in rows:
            relative = Path(row["image_path"])
            source = (root / relative).resolve(strict=True)
            target = staging / relative
            if not source.is_relative_to(root) or not source.is_file() or target == staging / "manifest.csv":
                raise ValueError(f"Invalid dataset image: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as image, target.open("xb") as copy:
                shutil.copyfileobj(image, copy)
            if expected := row.get("sha256"):
                with target.open("rb") as copy:
                    if hashlib.file_digest(copy, "sha256").hexdigest() != expected:
                        raise ValueError(f"Dataset image checksum mismatch: {relative}")
        staging.rename(output)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "manifest", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    try:
        count = materialize(args.dataset_root, args.manifest, args.output)
        print(f"Staged {count} images in {args.output}")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
