import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from data_pipeline.prepared_midv import SOURCES, SPLITS, SPLIT_RECORD, build
from data_pipeline.manifest import read_manifest


class PreparedMidvTests(unittest.TestCase):
    def setUp(self):
        workspace = TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name)
        self.source = self.root / "source"
        layouts = json.loads(SPLIT_RECORD.read_text())["layouts"]
        templates = [template for split in layouts.values() for template in split]
        for template_index, template in enumerate(templates):
            for domain_index, domain in enumerate(SOURCES):
                folder = self.source / domain / "images" / template
                paths = ([folder / "01" / f"{frame}.jpg" for frame in range(0, 21, 10)]
                         if domain == "clips" else [folder / "01.jpg"])
                for index, path in enumerate(paths):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (16, 16), (24 * template_index, 40 * domain_index, 75 * index)).save(path)

    def test_replay_preserves_manifest_and_partitions(self):
        first, second = self.root / "first", self.root / "second"
        self.assertEqual(build(self.source, first), build(self.source, second))
        rows = read_manifest(first / "manifest.csv")
        self.assertEqual(len(rows), 63)
        self.assertTrue(all(not row["quality_status"] for row in rows))
        development = read_manifest(first / "development.csv")
        held_out = read_manifest(first / "held-out.csv")
        self.assertEqual(held_out, [row for row in rows if row["split"] == "test"])
        self.assertEqual(development, [row for row in rows if row["split"] != "test"])
        for split in SPLITS:
            self.assertEqual(read_manifest(first / f"{split}.csv"), [
                row for row in rows if row["split"] == split
            ])

    def test_rejects_incomplete_roster_and_duplicate_payload(self):
        photo = self.source / "photo/images/alb_id/01.jpg"
        original = photo.read_bytes()
        photo.unlink()
        output = self.root / "output"
        with self.assertRaisesRegex(ValueError, "roster"):
            build(self.source, output)
        photo.write_bytes(original)
        scan = self.source / "scan_rotated/images/alb_id/01.jpg"
        scan.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "Duplicate JPEG"):
            build(self.source, output)
        self.assertFalse(output.exists())

    def test_rejects_corrupt_selected_image(self):
        (self.source / "clips/images/alb_id/01/10.jpg").write_bytes(b"invalid")
        with self.assertRaises(OSError):
            build(self.source, self.root / "output")
        self.assertFalse((self.root / "output").exists())
