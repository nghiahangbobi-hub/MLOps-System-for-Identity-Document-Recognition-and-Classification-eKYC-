"""Regression checks for the byte-budgeted MIDV sampler."""

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from data_pipeline.manifest import LABELS
from data_pipeline.sample_midv import SOURCES, build_dataset


class SampleMidvTest(unittest.TestCase):
    def test_replay_budget_grouping_and_incomplete_archive(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_dir = root / "sources"
            archive_dir.mkdir()
            record = Path(__file__).resolve().parents[1] / "docs/model-quality-evidence/split.json"
            raw = record.read_bytes()
            layouts = json.loads(raw)["layouts"]
            templates = [template for split in layouts.values() for template in split]
            image = io.BytesIO()
            Image.new("RGB", (8, 8), "blue").save(image, "JPEG")
            jpeg = image.getvalue()
            for domain in SOURCES:
                with tarfile.open(archive_dir / f"{domain}.tar", "w") as archive:
                    for template in [*templates, "rus_internalpassport"]:
                        for number in (1, 2, 3):
                            if domain == "templates" and number == 3:
                                continue
                            names = ([f"{number:02}/{frame}.jpg" for frame in (0, 1, 10, 20)]
                                     if domain == "clips" else [f"{number:03}.jpeg"])
                            for name in names:
                                header = tarfile.TarInfo(f"images/{template}/{name}")
                                header.size = len(jpeg)
                                archive.addfile(header, io.BytesIO(jpeg))

            digest = hashlib.sha256(raw).hexdigest()
            group_size = len(jpeg) * 7
            budget = group_size * 11 + 1
            output = root / "sample"
            stats = build_dataset(archive_dir, record, digest, output, budget, 3, 10)
            self.assertEqual(stats["image_bytes"], 11 * group_size)
            self.assertEqual(set(stats["eligible_groups_by_template"].values()), {2})
            self.assertEqual(set(stats["selected_groups_by_template"].values()), {1, 2})
            for counts in stats["images_by_split"].values():
                self.assertEqual(set(counts), set(LABELS))

            manifest = output / "manifest.csv"
            self.assertEqual(stats["manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
            with manifest.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 77)
            for row in rows:
                self.assertEqual(row["quality_status"], "")
                self.assertEqual(row["seed"], "42")
                self.assertIn(row["template_id"], layouts[row["split"]])
                self.assertNotEqual(row["document_id"].split("/")[-1], "3")
                data = (output / row["image_path"]).read_bytes()
                self.assertEqual(int(row["size_bytes"]), len(data))
                self.assertEqual(row["sha256"], hashlib.sha256(data).hexdigest())
                if row["domain"] == "clips":
                    self.assertIn(Path(row["source_member"]).stem, ("0", "10", "20"))

            replay = root / "replay"
            subprocess.run([sys.executable, "-m", "data_pipeline.sample_midv",
                            "--archive-dir", str(archive_dir), "--split-record", str(record),
                            "--split-sha256", digest, "--output", str(replay),
                            "--max-image-bytes", str(budget)], check=True, capture_output=True)
            for name in ("manifest.csv", "provenance.json"):
                self.assertEqual((output / name).read_bytes(), (replay / name).read_bytes())

            rejected = root / "rejected"
            with self.assertRaisesRegex(ValueError, "split SHA-256 mismatch"):
                build_dataset(archive_dir, record, "0" * 64, rejected, budget, 3, 10)
            with self.assertRaisesRegex(ValueError, "cover all nine templates"):
                build_dataset(archive_dir, record, digest, rejected, group_size * 9 - 1, 3, 10)
            with self.assertRaisesRegex(ValueError, "between 1 and 5000000000"):
                build_dataset(archive_dir, record, digest, rejected, 5_000_000_001, 3, 10)
            with (archive_dir / "photo.tar").open("r+b") as stream:
                stream.truncate(stream.seek(0, 2) // 2)
            with self.assertRaises((ValueError, tarfile.ReadError)):
                build_dataset(archive_dir, record, digest, rejected, budget, 3, 10)
            self.assertFalse(rejected.exists())
