"""Release regressions using disposable manifests, exports and media."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from PIL import Image

from data_pipeline.manifest import read_manifest, write_manifest
from labeling.label_studio import apply_reviews, make_tasks


def annotation(choice, **flags):
    return {"result": [{"from_name": "quality", "to_name": "image", "type": "choices",
                        "value": {"choices": [choice]}}], **flags}


class ReviewReleaseTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.media = self.root / "media"
        self.media.mkdir()
        self.input = self.root / "manifest.csv"
        self.output = self.root / "reviewed.csv"
        self.export = self.root / "reviews.json"
        self.rows = [dict(image_id=str(i), image_path=f"picture {i}.jpg", template_id="a_id",
                          document_id="a_id/1", split="train", label="ID_Card",
                          quality_status="", quality_source="") for i in range(3)]
        write_manifest(self.input, self.rows)

    def task(self, index, *annotations, **extra):
        row = self.rows[index]
        return dict(data={key: row[key] for key in ("image_id", "image_path")},
                    annotations=list(annotations), **extra)

    def write_export(self, *tasks, path=None):
        path = path or self.export
        path.write_text(json.dumps(tasks))
        return path

    def apply(self, export=None):
        return apply_reviews(self.input, export or self.export, self.output)

    def test_canceled_predicted_and_duplicate_reviews(self):
        self.rows[1].update(quality_status="Glare", quality_source="synthetic")
        write_manifest(self.input, self.rows)
        before = self.input.read_bytes()
        self.write_export(self.task(0, annotation("Clear"), annotation("Clear")),
                          self.task(0, annotation("Blur", was_cancelled=True)),
                          self.task(0, annotation("Clear")),
                          self.task(1, annotation("Blur", skipped=True),
                                    predictions=[annotation("Blur")]),
                          self.task(2, annotation("Outside_Scope")))
        self.assertEqual(self.apply(), {"reviewed": 1, "unreviewed": 0, "excluded": 1,
                                        "synthetic": 1})
        self.assertEqual(self.input.read_bytes(), before)
        self.assertEqual([(row["quality_status"], row["quality_source"])
                          for row in read_manifest(self.output)],
                         [("Clear", "reviewed"), ("Glare", "synthetic")])
        first = self.output.read_bytes()
        self.apply()
        self.assertEqual(self.output.read_bytes(), first)
        self.write_export(self.task(1, annotation("Blur")))
        self.apply()
        self.assertEqual(read_manifest(self.output)[1]["quality_source"], "reviewed")

    def test_conflicts_corruption_and_missing_choices_leave_output_intact(self):
        self.output.write_bytes(b"prior publication")
        cases = [
            ([self.task(0, annotation("Clear"), annotation("Blur"))], "Conflicting reviews"),
            ([self.task(0, annotation("Clear")), self.task(0, annotation("Blur"))],
             "Conflicting reviews"),
            ([self.task(0, None)], "annotation object"),
            ([self.task(0, annotation("Clear", was_cancelled="false"))], "boolean cancellation"),
            ([self.task(0, annotation(""))], "Invalid quality choice"),
            ([self.task(0, {"result": [{"from_name": "unrelated"}]})], "one quality result"),
            ([{**self.task(0), "data": {"image_id": "0", "image_path": "wrong.jpg"}}],
             "Image path mismatch"),
            ([{**self.task(0), "data": {"image_id": "missing"}}], "Unknown image_id"),
        ]
        for tasks, message in cases:
            with self.subTest(message=message):
                self.write_export(*tasks)
                with self.assertRaisesRegex(ValueError, message):
                    self.apply()
                self.assertEqual(self.output.read_bytes(), b"prior publication")
        self.export.write_text("[{")
        with self.assertRaises(json.JSONDecodeError):
            self.apply()
        self.assertEqual(self.output.read_bytes(), b"prior publication")
        self.write_export(self.task(1, annotation("Occluded")))
        self.assertEqual(self.apply()["reviewed"], 1)

    def test_retry_after_failed_review_publication(self):
        self.write_export(self.task(0, annotation("Glare")))
        self.output.write_bytes(b"prior publication")
        before = set(self.root.iterdir())
        with patch("labeling.label_studio.Path.replace", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.apply()
        self.assertEqual(self.output.read_bytes(), b"prior publication")
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertEqual(self.apply()["reviewed"], 1)

    def test_concurrent_snapshots_publish_complete_output(self):
        first = self.write_export(self.task(0, annotation("Clear")))
        second = self.write_export(self.task(1, annotation("Blur")), path=self.root / "other.json")
        start = Barrier(4, timeout=5)

        def apply_simultaneously(export):
            start.wait()
            return self.apply(export)

        with ThreadPoolExecutor(4) as executor:
            futures = [executor.submit(apply_simultaneously, path)
                       for path in (first, second, first, second)]
            self.assertEqual([result.result()["reviewed"] for result in futures], [1] * 4)
        final = [row["quality_status"] for row in read_manifest(self.output)]
        self.assertIn(final, (["Clear", "", ""], ["", "Blur", ""]))

    def test_input_aliases_are_protected(self):
        self.write_export(self.task(0, annotation("Clear")))
        for source in (self.input, self.export):
            before = source.read_bytes()
            self.output.hardlink_to(source)
            with self.assertRaisesRegex(ValueError, "separate from its inputs"):
                self.apply()
            self.assertEqual(source.read_bytes(), before)
            self.output.unlink()

    def test_task_storage_failures_and_retry(self):
        output = self.root / "tasks.json"
        before = set(self.root.iterdir())

        def generate():
            return make_tasks(self.input, self.media, self.root, output)

        with self.assertRaises(FileNotFoundError):
            generate()
        for row in self.rows:
            Image.new("RGB", (8, 8)).save(self.media / row["image_path"], "JPEG")
        broken = self.media / self.rows[0]["image_path"]
        image = broken.read_bytes()
        broken.write_bytes(image[:20])
        with self.assertRaisesRegex(ValueError, "Invalid JPEG"):
            generate()
        broken.write_bytes(image)
        with patch("labeling.label_studio.json.dump", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                generate()
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertEqual(generate(), 3)
        task_json = output.read_bytes()
        paths = [parse_qs(urlsplit(task["data"]["image"]).query)["d"][0]
                 for task in json.loads(task_json)]
        self.assertEqual(paths, [f"media/{row['image_path']}" for row in self.rows])
        with self.assertRaises(FileExistsError):
            generate()
        self.assertEqual(output.read_bytes(), task_json)
