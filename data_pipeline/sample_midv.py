"""Sample complete MIDV mock-document groups under an image-byte budget."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
import tarfile
from tempfile import TemporaryDirectory

from data_pipeline.images import validate_jpeg
from data_pipeline.manifest import FIELDS, LABELS, read_manifest, write_manifest
from data_pipeline.midv2020 import document_label, sample_frames
from data_pipeline.multisource import CAPTURE, DOMAINS, HELD_OUT, archive_stat


SOURCES = ("clips", *DOMAINS)
SPLITS = ("train", "validation", "test")
MAX_BYTES = 5_000_000_000
SEED = 42
CLIP = re.compile(r"images/([a-z0-9_]+)/([0-9]+)/([0-9]+)\.(jpg|jpeg)", re.IGNORECASE)


@dataclass
class Group:
    template: str
    number: int
    members: dict[str, tuple[tarfile.TarInfo, ...]]

    @property
    def size(self):
        return sum(member.size for members in self.members.values() for member in members)


def historical_splits(record, expected):
    raw = record.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected.lower():
        raise ValueError("Historical split SHA-256 mismatch")
    layouts = json.loads(raw)["layouts"]
    splits = {}
    for split in SPLITS:
        labels = set()
        for template in layouts[split]:
            label = document_label(template)
            if (template in splits or template == "rus_internalpassport"
                    or not re.fullmatch(r"[a-z0-9_]+", template) or label not in LABELS):
                raise ValueError(f"Invalid or duplicated historical template: {template}")
            splits[template] = split
            labels.add(label)
        if labels != set(LABELS):
            raise ValueError(f"Historical split must contain both classes: {split}")
    if len(splits) != 9 or {key for key, value in splits.items() if value == "test"} != HELD_OUT:
        raise ValueError("Expected nine historical templates with the original test pair")
    return splits, digest


def index_archive(archive, domain, splits):
    pattern = CLIP if domain == "clips" else CAPTURE
    index = defaultdict(dict)
    for member in archive:
        match = pattern.fullmatch(member.name)
        if match is None or match[1] not in splits:
            continue
        if not member.isfile() or member.size <= 0 or member.issparse():
            raise ValueError(f"Invalid image member in {domain}: {member.name}")
        key = match[1], int(match[2])
        frame = int(match[3]) if domain == "clips" else 0
        if frame in index[key]:
            raise ValueError(f"Duplicate capture in {domain}: {member.name}")
        index[key][frame] = member
    archive.fileobj.seek(archive.offset)
    if archive.fileobj.read(1024) != bytes(1024):
        raise ValueError(f"Source tar is incomplete or missing its end marker: {domain}")
    return index


def collect_groups(indexes, splits, frames_per_clip, min_frame_gap):
    groups = {template: [] for template in splits}
    for (template, number), frames in indexes["clips"].items():
        key = template, number
        numbers = sample_frames(frames, frames_per_clip, min_frame_gap)
        if len(numbers) != frames_per_clip or any(key not in indexes[domain] for domain in DOMAINS):
            continue
        members = {"clips": tuple(frames[frame] for frame in numbers)}
        members.update({domain: (indexes[domain][key][0],) for domain in DOMAINS})
        groups[template].append(Group(template, number, members))
    for template, candidates in groups.items():
        if not candidates:
            raise ValueError(f"No complete five-source document groups for {template}")
        candidates.sort(key=lambda group: hashlib.sha256(
            f"{SEED}/{group.template}/{group.number}".encode()
        ).digest())
    return groups


def select_groups(groups, budget):
    # Reserve enough bytes to represent every layout before filling round-robin.
    templates = sorted(groups, key=lambda template: hashlib.sha256(
        f"{SEED}/{template}".encode()
    ).digest())
    minimum = {template: min(group.size for group in groups[template]) for template in templates}
    reserved = sum(minimum.values())
    if reserved > budget:
        raise ValueError(f"Need at least {reserved} image bytes to cover all nine templates")
    selected = []
    for template in templates:
        reserved -= minimum[template]
        candidates = groups[template]
        index = next(i for i, group in enumerate(candidates) if group.size <= budget - reserved)
        group = candidates.pop(index)
        selected.append(group)
        budget -= group.size
    for candidates in zip_longest(*(groups[template] for template in templates)):
        for group in candidates:
            if group is not None and group.size <= budget:
                selected.append(group)
                budget -= group.size
    return selected


def extract_group_member(archive, domain, member, group, split, staging):
    document = f"{group.template}/{group.number}"
    frame = f"/{int(Path(member.name).stem)}" if domain == "clips" else ""
    image_id = f"{domain}/{document}{frame}"
    relative = f"images/{image_id}.jpg"
    target = staging / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with archive.extractfile(member) as source, target.open("xb") as stream:
        while chunk := source.read(1024 * 1024):
            stream.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if size != member.size:
        raise ValueError(f"Incorrect extracted byte count: {member.name}")
    validate_jpeg(target)
    return dict(image_id=image_id, image_path=relative, template_id=group.template,
                document_id=document, split=split, label=document_label(group.template),
                quality_status="", domain=domain, source_archive=f"{domain}.tar",
                source_member=member.name, size_bytes=size, sha256=digest.hexdigest(), seed=SEED)


def build_dataset(archive_dir, split_record, split_sha256, output, max_image_bytes,
                  frames_per_clip, min_frame_gap):
    if not 1 <= max_image_bytes <= MAX_BYTES:
        raise ValueError(f"Image budget must be between 1 and {MAX_BYTES} bytes")
    if frames_per_clip < 1 or min_frame_gap < 1:
        raise ValueError("Frame count and gap must be positive")
    if output.exists() or output.is_symlink():
        raise ValueError(f"Dataset output already exists: {output}")
    archive_dir = archive_dir.resolve(strict=True)
    output = output.resolve()
    if output.is_relative_to(archive_dir):
        raise ValueError("Dataset output must be outside the source directory")
    splits, split_digest = historical_splits(split_record, split_sha256)
    paths = {domain: archive_dir / f"{domain}.tar" for domain in SOURCES}
    stats = {domain: archive_stat(path) for domain, path in paths.items()}

    with ExitStack() as stack:
        archives = {domain: stack.enter_context(tarfile.open(path, "r:"))
                    for domain, path in paths.items()}
        indexes = {domain: index_archive(archive, domain, splits)
                   for domain, archive in archives.items()}
        groups = collect_groups(indexes, splits, frames_per_clip, min_frame_gap)
        available = {template: len(candidates) for template, candidates in groups.items()}
        selected = select_groups(groups, max_image_bytes)

        temp = Path(stack.enter_context(TemporaryDirectory(dir=output.parent, prefix=".midv-sample-")))
        staging = temp / "dataset"
        staging.mkdir(mode=0o700)
        rows = []
        for domain, archive in archives.items():
            entries = [(member, group) for group in selected for member in group.members[domain]]
            for member, group in sorted(entries, key=lambda entry: entry[0].offset_data):
                rows.append(extract_group_member(archive, domain, member, group,
                                                 splits[group.template], staging))
        if any(archive_stat(path) != stats[domain] for domain, path in paths.items()):
            raise ValueError("A source archive changed during the build")

        manifest = staging / "manifest.csv"
        fields = (*FIELDS, "domain", "source_archive", "source_member", "size_bytes", "sha256", "seed")
        write_manifest(manifest, sorted(rows, key=lambda row: row["image_id"]), fields=fields)
        read_manifest(manifest)
        with manifest.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        provenance = dict(seed=SEED, split_sha256=split_digest, manifest_sha256=digest,
                          max_image_bytes=max_image_bytes,
                          image_bytes=sum(row["size_bytes"] for row in rows),
                          frames_per_clip=frames_per_clip, min_frame_gap=min_frame_gap,
                          archive_sizes={f"{domain}.tar": stats[domain][0] for domain in SOURCES},
                          eligible_groups_by_template=available,
                          selected_groups_by_template=dict(Counter(group.template for group in selected)),
                          images_by_domain=dict(Counter(row["domain"] for row in rows)),
                          images_by_split={split: dict(Counter(row["label"] for row in rows
                                                               if row["split"] == split))
                                           for split in SPLITS})
        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise ValueError(f"Dataset output already exists: {output}")
        staging.rename(output)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", required=True, type=Path,
                        help="Read-only directory containing all five completed, uncompressed .tar files")
    parser.add_argument("--split-record", required=True, type=Path, help="Historical nine-layout split JSON")
    parser.add_argument("--split-sha256", required=True, help="Expected SHA-256 of the split record")
    parser.add_argument("--output", required=True, type=Path,
                        help="New private dataset root under an existing parent outside the source")
    parser.add_argument("--max-image-bytes", type=int, default=MAX_BYTES,
                        help="Maximum sum of extracted JPEG sizes, at most 5000000000")
    parser.add_argument("--frames-per-clip", type=int, default=3)
    parser.add_argument("--min-frame-gap", type=int, default=10)
    try:
        result = build_dataset(**vars(parser.parse_args()))
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Wrote {sum(result['images_by_domain'].values())} JPEGs ({result['image_bytes']} bytes)")


if __name__ == "__main__":
    main()
