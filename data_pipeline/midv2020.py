"""Build an image dataset from the MIDV-2020 clips tar archive."""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from tempfile import TemporaryDirectory

from data_pipeline.images import validate_jpeg
from data_pipeline.manifest import LABELS, write_manifest


DEFAULT_SOURCE = Path.home() / "Documents/Datasets/MIDV2020/MIDV2020/MIDV2020/dataset/clips.tar"
IMAGE_MEMBER = re.compile(r"images/([a-z0-9_]+)/([0-9]+)/([0-9]+)\.jpg")


def document_label(document_type: str) -> str | None:
    if document_type.endswith("_id"):
        return "ID_Card"
    if document_type.endswith("_passport"):
        return "Passport"
    return None


def index_clips(archive: tarfile.TarFile) -> dict:
    clips = defaultdict(dict)
    for member in archive:
        match = IMAGE_MEMBER.fullmatch(member.name)
        if not member.isfile() or match is None:
            continue
        document_type, number, frame = match.groups()
        if document_label(document_type) is None:
            continue
        frames = clips[document_type, number]
        if int(frame) in frames:
            raise ValueError(f"Duplicate frame in archive: {member.name}")
        frames[int(frame)] = member
    return clips


def assign_splits(clips: dict, seed: int, limit: int | None) -> dict[str, str]:
    by_label = defaultdict(list)
    for template in sorted({document_type for document_type, _ in clips}):
        by_label[document_label(template)].append(template)
    splits = {}
    for label in LABELS:
        templates = sorted(
            by_label[label],
            key=lambda template: hashlib.sha256(f"{seed}/{template}".encode()).digest(),
        )[:limit]
        count = len(templates)
        if count < 3:
            raise ValueError(f"Need at least three templates for {label}; found {count}")
        held_out = max(1, round(count * 0.15))
        for index, template in enumerate(templates):
            if index < held_out:
                split = "validation"
            elif index < 2 * held_out:
                split = "test"
            else:
                split = "train"
            splits[template] = split
    return splits


def sample_frames(frames: dict, count: int, min_gap: int) -> list[int]:
    separated = []
    for number in sorted(frames):
        if not separated or number - separated[-1] >= min_gap:
            separated.append(number)
    if len(separated) <= count:
        return separated
    if count == 1:
        return [separated[len(separated) // 2]]
    return [separated[round(i * (len(separated) - 1) / (count - 1))] for i in range(count)]


def build_dataset(source: Path, output: Path, seed: int, limit: int | None,
                  frames_per_clip: int, min_gap: int,
                  max_clips_per_template: int | None = None) -> int:
    if output.exists() or output.is_symlink():
        raise ValueError(f"Dataset output already exists: {output}")
    if frames_per_clip < 1 or min_gap < 1 or (limit is not None and limit < 3):
        raise ValueError("Frame count and gap must be positive, and template limit at least 3")
    if max_clips_per_template is not None and max_clips_per_template < 1:
        raise ValueError("Clip limit must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, "r") as archive:
        clips = index_clips(archive)
        splits = assign_splits(clips, seed, limit)
        rows = []
        selected = []
        clip_counts = defaultdict(int)
        for document_type, number in sorted(clips):
            if document_type not in splits:
                continue
            if (max_clips_per_template is not None
                    and clip_counts[document_type] >= max_clips_per_template):
                continue
            clip_counts[document_type] += 1
            frames = clips[document_type, number]
            document_id = f"{document_type}/{number}"
            for frame in sample_frames(frames, frames_per_clip, min_gap):
                member = frames[frame]
                stem = PurePosixPath(member.name).stem
                selected.append(member)
                rows.append({
                    "image_id": f"{document_id}/{stem}",
                    "image_path": member.name,
                    "template_id": document_type,
                    "document_id": document_id,
                    "split": splits[document_type],
                    "label": document_label(document_type),
                    "quality_status": "",
                })
        with TemporaryDirectory(dir=output.parent) as temp:
            staging = Path(temp) / "dataset"
            staging.mkdir()
            for member in sorted(selected, key=lambda item: item.offset_data):
                destination = staging / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as image, destination.open("wb") as stream:
                    shutil.copyfileobj(image, stream)
                try:
                    validate_jpeg(destination)
                except (OSError, ValueError) as error:
                    raise ValueError(f"Invalid JPEG in archive: {member.name}") from error
            write_manifest(staging / "manifest.csv", rows)
            staging.rename(output)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-templates-per-label", type=int)
    parser.add_argument("--max-clips-per-template", type=int)
    parser.add_argument("--frames-per-clip", type=int, default=3)
    parser.add_argument("--min-frame-gap", type=int, default=10)
    args = parser.parse_args()
    try:
        count = build_dataset(args.source, args.output, args.seed,
                              args.max_templates_per_label, args.frames_per_clip,
                              args.min_frame_gap, args.max_clips_per_template)
    except (ValueError, OSError, tarfile.TarError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Wrote {count} images and {args.output / 'manifest.csv'}")


if __name__ == "__main__":
    main()
