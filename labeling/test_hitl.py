"""Regression tests for queue snapshot, explicit approval, and publication."""

from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from data_pipeline.manifest import read_manifest
from labeling.hitl import approve, digest, json_bytes, load_json, publish, snapshot


def choice(control, value):
    return dict(from_name=control, to_name="image", type="choices", value={"choices": [value]})


class HITLTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.base = self.dataset / "manifest.csv"
        self.queue = self.root / "queue"
        self.events = self.queue / "events"
        self.events.mkdir(parents=True)
        self.batch = self.root / "batch"
        self.export = self.root / "export.json"
        self.approval = self.root / "approval.json"
        self.versions = self.root / "versions"
        self.rows = []
        self.event_dirs = []
        data = (
            ("train", "a_id", "ID_Card"),
            ("train", "a_id", "ID_Card"),
            ("train", "b_passport", "Passport"),
            ("train", "c_id", "ID_Card"),
            ("validation", "d_id", "ID_Card"),
            ("test", "e_passport", "Passport"),
        )
        for i, (split, template, label) in enumerate(data):
            name = f"{i}.png" if i == 2 else f"{i}.jpg"
            row = dict(image_id=str(i), image_path=name, template_id=template,
                       document_id=f"{template}/{i}", split=split, label=label,
                       quality_status="", quality_source="", custom=f"original-{i}")
            self.rows.append(row)
            Image.new("RGB", (8, 8), (i * 40, 0, 0)).save(self.dataset / name)
            if i not in (1, 3):
                self.event_dirs.append(self.enqueue(row))
        with self.base.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def enqueue(self, row):
        image = (self.dataset / row["image_path"]).read_bytes()
        event_id = digest(image)
        directory = self.events / event_id
        directory.mkdir()
        image_name = "image" + Path(row["image_path"]).suffix
        record = dict(event_id=event_id, image_sha256=event_id, image_file=image_name,
                      source_image_id=row["image_id"],
                      prediction=dict(label=row["label"], status=None, confidence=0.5,
                                      needs_review=True),
                      reasons=["needs_review", "reported_incorrect", "needs_review"])
        (directory / "event.json").write_bytes(json_bytes(record))
        (directory / image_name).write_bytes(image)
        return directory

    def create_batch(self):
        return snapshot(self.queue, self.base, self.dataset, self.batch)

    def prepare_export(self):
        self.create_batch()
        tasks = load_json((self.batch / "tasks.json").read_bytes())
        by_source = {task["data"]["source_image_id"]: task for task in tasks}
        by_source["0"]["annotations"] = [{"result": [
            choice("document_type", "Passport"), choice("quality", "Glare"),
        ]}]
        by_source["2"]["annotations"] = [{"result": [choice("document_type", "Outside_Scope")]}]
        self.export.write_bytes(json_bytes(tasks))
        return tasks

    def approve(self, reviewer="reviewer-local"):
        return approve(self.batch, self.export, reviewer, self.approval)

    def publish(self):
        return publish(self.batch, self.export, self.approval,
                       self.base, self.dataset, self.versions)

    def test_snapshot_approval_publication_and_heldout_rows(self):
        (self.events / ".stage-unfinished").mkdir()
        before = self.base.read_bytes()
        tasks = self.prepare_export()
        batch = load_json((self.batch / "batch.json").read_bytes())
        self.assertEqual({task["data"]["source_image_id"] for task in tasks}, {"0", "2"})
        self.assertEqual({event["source_image_id"] for event in batch["held_out"]}, {"4", "5"})
        self.assertEqual(len(list((self.batch / "images").iterdir())), 2)
        for file in (self.batch / "images").iterdir():
            self.assertEqual(file.stem, digest(file.read_bytes()))
        with self.assertRaises(FileNotFoundError):
            self.publish()
        approval = self.approve()
        self.assertEqual(approval["reviewer"], "reviewer-local")
        self.assertEqual(approval["export_sha256"], digest(self.export.read_bytes()))
        self.assertEqual(approval["batch_id"], batch["batch_id"])
        version = self.publish()
        self.assertEqual(version.name, digest(self.approval.read_bytes()))
        published = version / "manifest.csv"
        parsed = {row["image_id"]: row for row in read_manifest(published)}
        self.assertEqual(parsed["0"]["label"], "Passport")
        self.assertEqual(parsed["0"]["quality_status"], "Glare")
        self.assertEqual(parsed["0"]["quality_source"], "reviewed")
        self.assertNotIn("2", parsed)
        self.assertEqual(self.base.read_bytes(), before)
        with published.open(newline="") as stream:
            by_id = {row["image_id"]: row for row in csv.DictReader(stream)}
        for i in (1, 3, 4, 5):
            self.assertEqual(by_id[str(i)], self.rows[i])
        lineage = load_json((version / "lineage.json").read_bytes())
        self.assertEqual(lineage["manifest_sha256"], digest(published.read_bytes()))
        self.assertEqual(lineage["approval"], approval)
        changes = {row["source_image_id"]: row for row in lineage["changes"]}
        self.assertEqual(changes["0"]["before"]["label"], "ID_Card")
        self.assertEqual(changes["0"]["after"]["label"], "Passport")
        self.assertEqual(changes["2"]["action"], "excluded")

    def test_snapshot_rejects_corruption_or_wrong_source(self):
        event = self.event_dirs[0]
        path = event / "event.json"
        original = path.read_bytes()
        corrupt = load_json(original)
        corrupt["source_image_id"] = "missing"
        path.write_bytes(json_bytes(corrupt))
        with self.assertRaisesRegex(ValueError, "Unknown source_image_id"):
            self.create_batch()
        corrupt["source_image_id"] = "1"
        path.write_bytes(json_bytes(corrupt))
        with self.assertRaisesRegex(ValueError, "Base image digest mismatch"):
            self.create_batch()
        path.write_bytes(original)
        media = event / "image.jpg"
        media.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Queued image digest mismatch"):
            self.create_batch()
        self.assertFalse(self.batch.exists())

    def test_incomplete_cancelled_conflicting_and_foreign_exports(self):
        tasks = self.prepare_export()
        first, second = tasks
        original = first["annotations"]
        cases = [
            ([second], "Incomplete export"),
            ([dict(first, annotations=[dict(original[0], was_cancelled=True)]), second],
             "Incomplete export"),
            ([dict(first, annotations=[], predictions=original), second], "Incomplete export"),
            ([dict(first, annotations=[{"result": [choice("quality", "Clear")]}]), second],
             "Incomplete correction"),
            ([first, second, dict(first, annotations=[{"result": [
                choice("document_type", "ID_Card"), choice("quality", "Blur")
            ]}])], "Conflicting completed corrections"),
            ([dict(first, data={**first["data"], "batch_id": "foreign"}), second],
             "does not match the batch"),
        ]
        for export, message in cases:
            with self.subTest(message=message):
                self.export.write_bytes(json_bytes(export))
                with self.assertRaisesRegex(ValueError, message):
                    self.approve()
                self.assertFalse(self.approval.exists())
        self.export.write_bytes(json_bytes(tasks + tasks + [
            dict(first, annotations=[dict(original[0], skipped=True)])
        ]))
        self.assertEqual(self.approve(), self.approve())
        self.publish()

    def test_stale_hashes_and_changed_source_do_not_publish(self):
        self.prepare_export()
        raw = self.export.read_bytes()
        self.approve()
        self.export.write_bytes(raw + b"\n")
        with self.assertRaisesRegex(ValueError, "Stale approval"):
            self.publish()
        with self.assertRaisesRegex(ValueError, "Existing approval differs"):
            self.approve()
        self.export.write_bytes(raw)
        base = self.base.read_bytes()
        self.base.write_bytes(base + b"\n")
        with self.assertRaisesRegex(ValueError, "Stale batch"):
            self.publish()
        self.base.write_bytes(base)
        (self.dataset / "0.jpg").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "not the approved train image"):
            self.publish()
        self.assertFalse(self.versions.exists())

    def test_storage_failure_retry_concurrency_and_immutable_publication(self):
        self.prepare_export()
        with patch("labeling.hitl.Path.hardlink_to", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.approve()
        self.assertFalse(self.approval.exists())
        self.approve()
        with patch("labeling.hitl.Path.rename", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.publish()
        self.assertEqual(list(self.versions.iterdir()), [])
        with ThreadPoolExecutor(max_workers=4) as executor:
            versions = list(executor.map(lambda _: self.publish(), range(8)))
        version = versions[0]
        self.assertEqual(set(versions), {version})
        self.assertEqual(list(self.versions.iterdir()), [version])
        (version / "manifest.csv").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Published version differs"):
            self.publish()
        self.assertEqual((version / "manifest.csv").read_bytes(), b"tampered")
