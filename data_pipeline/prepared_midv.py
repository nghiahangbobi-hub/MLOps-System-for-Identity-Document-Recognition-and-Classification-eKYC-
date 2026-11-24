"""Index the extracted five-domain MIDV-2020 course sample in place."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from data_pipeline.images import validate_jpeg
from data_pipeline.manifest import FIELDS, read_manifest, write_manifest
from data_pipeline.midv2020 import document_label, sample_frames
from data_pipeline.multisource import CAPTURE
from data_pipeline.sample_midv import CLIP, SOURCES, SPLITS, historical_splits


DEFAULT_SOURCE = Path("Data/MIDV2020_sample_5GB")
SPLIT_RECORD = Path(__file__).with_name("midv-split.json")
SPLIT_SHA256 = "b8ee591f0f05dc1161f489c0c7ead6ccb1acea667625c7468a2e3ba15d834054"


def index_domain(source, domain, splits):
    root = source / domain
    pattern = CLIP if domain == "clips" else CAPTURE
    index = defaultdict(dict)
    for path in sorted((root / "images").rglob("*")):
        match = pattern.fullmatch(path.relative_to(root).as_posix())
        if match is None or match[1] not in splits:
            continue
        if not path.is_file() or path.resolve() != path:
            raise ValueError(f"Expected regular source image: {path}")
        document = match[1], int(match[2])
        frame = int(match[3]) if domain == "clips" else 0
        if frame in index[document]:
            raise ValueError(f"Duplicate capture: {path}")
        index[document][frame] = path
    return index


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(source, output, frames_per_clip=3, min_frame_gap=10):
    if frames_per_clip < 1 or min_frame_gap < 1:
        raise ValueError("Frame count and gap must be positive")
    source = source.resolve(strict=True)
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(source):
        raise ValueError("Choose a new output directory outside the source")
    splits, split_sha256 = historical_splits(SPLIT_RECORD, SPLIT_SHA256)
    indexes = {domain: index_domain(source, domain, splits) for domain in SOURCES}
    clips = indexes["clips"]
    if {template for template, _ in clips} != set(splits):
        raise ValueError("The extracted sample must cover all nine templates")
    for domain, index in indexes.items():
        if index.keys() != clips.keys():
            raise ValueError(f"Document roster differs from clips: {domain}")

    rows = []
    seen = {}
    for (template, number), frames in sorted(clips.items()):
        selected = sample_frames(frames, frames_per_clip, min_frame_gap)
        document = f"{template}/{number}"
        if len(selected) != frames_per_clip:
            raise ValueError(f"Insufficient separated frames: {document}")
        for domain, index in indexes.items():
            for frame in selected if domain == "clips" else (0,):
                path = index[template, number][frame]
                validate_jpeg(path)
                sha256 = digest(path)
                split = splits[template]
                if sha256 in seen:
                    raise ValueError(f"Duplicate JPEG payload: {seen[sha256]} and {path}")
                seen[sha256] = path
                image_id = f"{domain}/{document}"
                if domain == "clips":
                    image_id += f"/{frame}"
                rows.append(dict(image_id=image_id, image_path=path.relative_to(source).as_posix(),
                                 template_id=template, document_id=document, split=split,
                                 label=document_label(template), quality_status="", domain=domain,
                                 size_bytes=path.stat().st_size, sha256=sha256))

    fields = (*FIELDS, "domain", "size_bytes", "sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".midv-prepared-") as temporary:
        staging = Path(temporary) / "manifests"
        staging.mkdir(mode=0o700)
        by_split = {split: [row for row in rows if row["split"] == split] for split in SPLITS}
        partitions = {
            "manifest.csv": rows,
            "development.csv": [row for row in rows if row["split"] != "test"],
            "held-out.csv": by_split["test"],
            **{f"{split}.csv": partition for split, partition in by_split.items()},
        }
        hashes = {}
        for name, partition in partitions.items():
            path = staging / name
            write_manifest(path, partition, fields=fields)
            read_manifest(path)
            hashes[name] = digest(path)
        report = dict(split_sha256=split_sha256, sha256=hashes, frames_per_clip=frames_per_clip,
                      min_frame_gap=min_frame_gap, documents=len(clips),
                      image_bytes=sum(row["size_bytes"] for row in rows),
                      images_by_domain=Counter(row["domain"] for row in rows),
                      images_by_split={split: Counter(row["label"] for row in rows
                                                       if row["split"] == split)
                                       for split in SPLITS})
        (staging / "provenance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise ValueError(f"Output already exists: {output}")
        staging.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-clip", type=int, default=3)
    parser.add_argument("--min-frame-gap", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.frames_per_clip, args.min_frame_gap), indent=2))


if __name__ == "__main__":
    main()
