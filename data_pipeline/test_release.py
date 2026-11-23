"""Focused archive and manifest release regressions."""

from collections import defaultdict
import io
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from data_pipeline.manifest import LABELS, read_manifest, write_manifest
from data_pipeline.midv2020 import build_dataset


class DatasetReleaseTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.archive = self.root / "clips.tar"
        self.output = self.root / "dataset"
        image = io.BytesIO()
        Image.new("RGB", (16, 16)).save(image, "JPEG")
        self.jpeg = image.getvalue()
        self.templates = [f"{name}_{kind}" for name in "abc" for kind in ("id", "passport")]

    def archive_entries(self):
        return [(f"images/{template}/{clip}/{frame}.jpg", self.jpeg)
                for template in self.templates + ["rus_internalpassport"]
                for clip in ("00", "01") for frame in (1, 2, 11, 21)]

    def write_archive(self, entries):
        with tarfile.open(self.archive, "w") as archive:
            for name, contents in entries:
                member = tarfile.TarInfo(name)
                member.size = len(contents)
                archive.addfile(member, io.BytesIO(contents))

    def build(self, output=None):
        return build_dataset(self.archive, output or self.output, 42, None, 3, 10)

    def test_grouped_splits_and_separated_frames_are_deterministic(self):
        entries = self.archive_entries()
        self.write_archive(entries)
        before = self.archive.read_bytes()
        self.assertEqual(self.build(), 36)
        self.assertEqual(self.archive.read_bytes(), before)
        manifest = self.output / "manifest.csv"
        self.write_archive(reversed(entries))
        self.build(self.root / "again")
        self.assertEqual(manifest.read_bytes(), (self.root / "again/manifest.csv").read_bytes())
        groups, classes, frames = defaultdict(set), defaultdict(set), defaultdict(list)
        for row in read_manifest(manifest):
            self.assertEqual(row["quality_status"], "")
            groups[row["template_id"]].add(row["split"])
            classes[row["split"]].add(row["label"])
            frames[row["document_id"]].append(int(Path(row["image_path"]).stem))
            self.assertEqual((self.output / row["image_path"]).read_bytes(), self.jpeg)
        self.assertEqual(set(groups), set(self.templates))
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual(set(classes), {"train", "validation", "test"})
        self.assertTrue(all(labels == set(LABELS) for labels in classes.values()))
        self.assertTrue(all(numbers == [1, 11, 21] for numbers in frames.values()))

    def test_missing_class_or_templates(self):
        entries = self.archive_entries()
        for excluded in (self.templates[0:6:2], self.templates[:1]):
            with self.subTest(excluded=excluded):
                self.write_archive((name, image) for name, image in entries
                                   if name.split("/")[1] not in excluded)
                with self.assertRaisesRegex(ValueError, "at least three templates for ID_Card"):
                    self.build()
                self.assertFalse(self.output.exists())

    def test_duplicate_frame_number(self):
        entries = self.archive_entries()
        for alias in ("1.jpg", "0001.jpg"):
            with self.subTest(alias=alias):
                path = str(Path(entries[0][0]).with_name(alias))
                self.write_archive(entries + [(path, self.jpeg)])
                with self.assertRaisesRegex(ValueError, "Duplicate frame"):
                    self.build()

    def test_corrupt_archive_and_jpeg_can_be_retried(self):
        self.archive.write_bytes(b"not a tar")
        with self.assertRaises(tarfile.TarError):
            self.build()
        entries = self.archive_entries()
        for corrupt in (b"not a jpeg", self.jpeg[:-10]):
            with self.subTest(corrupt_len=len(corrupt)):
                self.write_archive([(entries[0][0], corrupt), *entries[1:]])
                with self.assertRaisesRegex(ValueError, "Invalid JPEG"):
                    self.build()
                self.assertFalse(self.output.exists())
        self.write_archive(entries)
        self.assertEqual(self.build(), 36)

    def test_storage_failure_leaves_no_dataset_and_retry_succeeds(self):
        self.write_archive(self.archive_entries())
        with patch("data_pipeline.midv2020.shutil.copyfileobj", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.build()
        self.assertEqual(list(self.root.iterdir()), [self.archive])
        self.assertEqual(self.build(), 36)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.build()

    def test_corrupt_manifest_columns_labels_splits_and_duplicates(self):
        valid = dict(image_id="a", image_path="a.jpg", label="ID_Card", template_id="a",
                     document_id="a/1", split="train")
        second = dict(valid, image_id="b", image_path="b.jpg")
        path = self.root / "manifest.csv"
        cases = [
            ([dict(valid, label=" ")], "Incomplete"),
            ([dict(valid, label="Driver_License")], "Invalid document label"),
            ([dict(valid, split="holdout")], "Invalid split"),
            ([valid, dict(second, split="test")], "Inconsistent split"),
            ([valid, dict(second, template_id="b", split="test")], "split for document"),
            ([valid, dict(second, label="Passport")], "Inconsistent split or label"),
            ([valid, dict(second, image_id="a ")], "Duplicate image_id"),
            ([valid, dict(second, image_path="./a.jpg")], "Duplicate image path"),
            ([dict(valid, split="test", quality_status="Blur", quality_source="synthetic")],
             "require a quality target and the train split"),
            ([dict(valid, quality_status="Clear", quality_source="unknown")],
             "Invalid quality provenance"),
            ([dict(valid, quality_status="Clear", quality_source="unreviewed")],
             "contradicts provenance"),
            ([dict(valid, quality_source="reviewed")], "contradicts provenance"),
        ]
        for rows, message in cases:
            with self.subTest(message=message):
                write_manifest(path, rows)
                with self.assertRaisesRegex(ValueError, message):
                    read_manifest(path)
        header = "image_id,image_path,template_id,document_id,split,label"
        path.write_text(header + ",label\n")
        with self.assertRaisesRegex(ValueError, "duplicate columns"):
            read_manifest(path)
        path.write_text(header.rsplit(",", 1)[0] + "\n")
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            read_manifest(path)
