"""Focused model release checks. Run with python -m unittest -v modeling.test_release."""

import base64
import csv
import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from . import score, train
from .network import ARCHITECTURE, DOCUMENT_LABELS, MEAN, QUALITY_LABELS, STD, build_model


def metadata(has_quality_head=True, threshold=0.5):
    return {
        "format_version": 1,
        "architecture": ARCHITECTURE,
        "document_labels": list(DOCUMENT_LABELS),
        "quality_labels": list(QUALITY_LABELS),
        "has_quality_head": has_quality_head,
        "preprocessing": {"image_size": 32, "mean": list(MEAN), "std": list(STD)},
        "confidence_threshold": threshold,
    }


def image_request(size=(8, 8)):
    stream = io.BytesIO()
    Image.new("RGB", size).save(stream, format="PNG")
    return {"image_base64": base64.b64encode(stream.getvalue()).decode("ascii")}


class TemporaryFiles(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))


class ManifestTests(TemporaryFiles):
    def setUp(self):
        super().setUp()
        self.rows = [
            self.row("a", "layout-a", "train", "ID_Card"),
            self.row("b", "layout-b", "train", "Passport"),
        ]

    @staticmethod
    def row(identifier, template, split, label, quality="", source=""):
        return dict(image_id=identifier, image_path=f"{identifier}.jpg", template_id=template,
                    document_id=f"document-{identifier}", split=split, label=label,
                    quality_status=quality, quality_source=source)

    def load(self, rows, *, root=None, drop=()):
        manifest = self.root / "rows.csv"
        fields = [field for field in self.rows[0] if field not in drop]
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return train.load_rows(manifest, root)

    def test_split_leakage_and_duplicate_identity(self):
        held_out = self.row("c", "layout-c", "test", "ID_Card")
        cases = [
            ("Template appears", {"template_id": "layout-a"}),
            ("Document appears", {"document_id": "document-a"}),
            ("Duplicate image ID", {"image_id": "a"}),
            ("Duplicate image path", {"image_path": "./a.jpg"}),
        ]
        for message, updates in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.load([*self.rows, {**held_out, **updates}])

    def test_resolved_aliases_and_missing_files(self):
        for row in self.rows:
            (self.root / row["image_path"]).touch()
        held_out = self.row("c", "layout-c", "test", "ID_Card")
        alias = self.root / "c.jpg"
        with self.assertRaises(FileNotFoundError):
            self.load([*self.rows, held_out], root=self.root)
        alias.symlink_to("a.jpg")
        with self.assertRaisesRegex(ValueError, "Duplicate image path"):
            self.load([*self.rows, held_out], root=self.root)
        alias.unlink()
        os.link(self.root / "a.jpg", alias)
        with self.assertRaisesRegex(ValueError, "Duplicate image file"):
            self.load([*self.rows, held_out], root=self.root)
        alias.unlink()
        (self.root / "a.jpg").write_bytes(b"same JPEG bytes")
        alias.write_bytes(b"same JPEG bytes")
        with self.assertRaisesRegex(ValueError, "Identical image content"):
            self.load([*self.rows, held_out], root=self.root)
        alias.write_bytes(b"other JPEG data")
        self.assertEqual(len(self.load([*self.rows, held_out], root=self.root)["test"]), 1)

    def test_missing_labels_and_classes(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            self.load(self.rows, drop=("label",))
        for label, message in (("", "Empty manifest field"), ("Unknown", "Invalid document label")):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, message):
                self.load([self.rows[0], {**self.rows[1], "label": label}])
        with self.assertRaisesRegex(ValueError, "both document classes"):
            self.load([self.rows[0], {**self.rows[1], "quality_status": "Outside_Scope"}])

    def test_optional_and_valid_quality_labels(self):
        self.assertTrue(all(row[2] == train.IGNORE_QUALITY
                            for row in self.load(self.rows, drop=("quality_status",))["train"]))
        rows = [
            {**self.rows[0], "quality_source": "unreviewed"},
            {**self.rows[1], "quality_status": "Blur", "quality_source": "synthetic"},
            self.row("c", "layout-c", "validation", "Passport", "Clear"),
            self.row("d", "layout-d", "test", "ID_Card", "Occluded", "reviewed"),
            self.row("e", "layout-e", "test", "ID_Card", "Outside_Scope", "reviewed"),
        ]
        splits = self.load(rows)
        self.assertEqual([item[2] for item in splits["train"]], [train.IGNORE_QUALITY, 1])
        self.assertEqual([item[2] for item in splits["val"]], [0])
        self.assertEqual([item[2] for item in splits["test"]], [3])

    def test_invalid_quality_and_provenance(self):
        cases = [
            ("Invalid quality provenance", "train", "Blur", "unknown"),
            ("contradicts provenance", "train", "Blur", "unreviewed"),
            ("contradicts provenance", "train", "", "reviewed"),
            ("Synthetic rows", "train", "", "synthetic"),
            ("Synthetic rows", "test", "Blur", "synthetic"),
            ("Invalid quality status", "train", "Unknown", "reviewed"),
        ]
        for message, split, quality, source in cases:
            row = self.row("c", "layout-c", split, "ID_Card", quality, source)
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, message):
                self.load([*self.rows, row])

    def test_corrupt_dataset_image(self):
        (self.root / "a.jpg").write_bytes(b"corrupt")
        dataset = train.Images(self.root, [("a.jpg", 0, train.IGNORE_QUALITY)], 32)
        with self.assertRaises(OSError):
            dataset[0]


class ScoringTests(TemporaryFiles):
    def setUp(self):
        super().setUp()
        self.enterContext(patch.object(score, "_predictor", None))
        self.enterContext(patch.dict(os.environ, {"AZUREML_MODEL_DIR": str(self.root)}))

    def test_bad_request_structure_and_corrupt_image(self):
        cases = [b"\xff", "{", "[" * 1500, [], {}, {"image_base64": 1},
                 {"image_base64": ""}, {"image_base64": "%%%%"},
                 {"image_base64": "bm90IGFuIGltYWdl"}]
        for request in cases:
            with self.subTest(kind=type(request)), self.assertRaises(ValueError):
                score.request_image(request)
        with patch.object(score.Image, "open", side_effect=SyntaxError):
            with self.assertRaisesRegex(ValueError, "readable image"):
                score.request_image(image_request())

    def test_oversized_encoded_decoded_body_and_pixels(self):
        for request in (
            {"image_base64": "A" * (score.MAX_BASE64_LENGTH + 1)},
            {"image_base64": base64.b64encode(bytes(score.MAX_IMAGE_BYTES + 1)).decode()},
            " " * (score.MAX_REQUEST_LENGTH + 1),
        ):
            with self.assertRaisesRegex(ValueError, "up to 8 MiB"):
                score.request_image(request)
        with self.assertRaisesRegex(ValueError, "25 megapixels"):
            score.request_image(image_request((5000, 5001)))

    def test_confidence_boundary_with_repeated_requests(self):
        tensor = torch.zeros(3, 32, 32)
        for threshold, expected in (
            (math.nextafter(0.5, 0), False),
            (0.5, False),
            (math.nextafter(0.5, 1), True),
            (0, False),
            (1, True),
        ):
            settings = metadata(threshold=threshold)
            score._predictor = (lambda _: torch.zeros(1, 6), lambda _: tensor, settings)
            with self.subTest(threshold=threshold):
                request = {**image_request(), "image_id": "private-test-identity"}
                with redirect_stdout(io.StringIO()) as stdout:
                    result = score.run(request)
                    self.assertEqual(result, score.run(json.dumps(request).encode()))
                self.assertEqual([json.loads(line) for line in stdout.getvalue().splitlines()], [{
                    "event": "ekyc.low_document_confidence",
                    "confidence": 0.5,
                    "threshold": threshold,
                }] * (2 if expected else 0))
                self.assertEqual(result["confidence"], 0.5)
                self.assertEqual(result["needs_review"], expected)
        score._predictor = (lambda _: torch.tensor([[0., 0., 0., 1., 0., 0.]]),
                            lambda _: tensor, metadata(threshold=0))
        with redirect_stdout(io.StringIO()) as stdout:
            self.assertTrue(score.run(image_request())["needs_review"])
        self.assertEqual(stdout.getvalue(), "")
        score._predictor = (lambda _: torch.zeros(1, 2), lambda _: tensor, metadata(False, 0))
        with redirect_stdout(io.StringIO()) as stdout:
            result = score.run(image_request())
            self.assertIsNone(result["status"])
            self.assertTrue(result["needs_review"])
        self.assertEqual(stdout.getvalue(), "")
        settings = {**metadata(threshold=0.5), "document_review_all": True}
        score._predictor = (lambda _: torch.zeros(1, 6), lambda _: tensor, settings)
        with redirect_stdout(io.StringIO()) as stdout:
            self.assertTrue(score.run(image_request())["needs_review"])
        self.assertEqual(stdout.getvalue(), "")

    def test_rejects_missing_ambiguous_and_malformed_bundles(self):
        with self.assertRaisesRegex(RuntimeError, "found 0"):
            score.init()
        for name in ("one", "two"):
            bundle = self.root / name
            bundle.mkdir()
            (bundle / "weights.pt").touch()
            (bundle / "metadata.json").touch()
        with self.assertRaisesRegex(RuntimeError, "found 2"):
            score.init()
        (self.root / "two" / "weights.pt").unlink()
        path = self.root / "one" / "metadata.json"
        for raw, message in (
            (b"{", "Unreadable model metadata"),
            (b" " * (score.MAX_METADATA_BYTES + 1), "size limit"),
            (b"[]", "must be an object"),
            (json.dumps({**metadata(), "confidence_threshold": True}).encode(), "Unsupported"),
            (json.dumps({**metadata(), "confidence_threshold": float("nan")}).encode(), "Unsupported"),
            (json.dumps({**metadata(), "document_labels": []}).encode(), "Unsupported"),
            (json.dumps({**metadata(), "preprocessing": {"image_size": 100000}}).encode(), "image size"),
            (json.dumps({**metadata(), "preprocessing": {**metadata()["preprocessing"],
                                                         "std": [1, 0, 1]}}).encode(), "normalization"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                path.write_bytes(raw)
                score.init()

    def test_publication_failure_retry_and_scorer_restart(self):
        bundle = self.root / "registered" / "1"
        bundle.mkdir(parents=True)
        model = build_model(True)
        with patch.object(train.mlflow, "log_artifact", side_effect=[None, OSError("storage")]):
            with self.assertRaisesRegex(OSError, "storage"):
                train.publish_bundle(bundle, model, metadata())
        self.assertEqual(list(bundle.iterdir()), [])
        self.assertEqual(list(bundle.parent.iterdir()), [bundle])
        with patch.object(train.mlflow, "log_artifact"):
            train.publish_bundle(bundle, model, metadata())
            with self.assertRaises(OSError):
                train.publish_bundle(bundle, build_model(False), metadata(False))

        score.init()
        request = image_request()
        expected = score.run(request)
        weights_path = bundle / "weights.pt"
        original = weights_path.read_bytes()
        weights_path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            score.init()
        with self.assertRaisesRegex(RuntimeError, "Call init"):
            score.run(request)
        weights_path.write_bytes(original)
        score.init()
        self.assertEqual(score.run(request), expected)

    def test_invalid_legacy_weights(self):
        (self.root / "metadata.json").write_text(json.dumps(metadata(False)))
        path = self.root / "weights.pt"
        with path.open("wb") as stream:
            stream.truncate(score.MAX_WEIGHTS_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "size limit"):
            score.init()
        path.write_bytes(b"invalid checkpoint")
        with self.assertRaisesRegex(ValueError, "Unreadable"):
            score.init()
        torch.save({"invalid.key": torch.zeros(1)}, path)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            score.init()
        weights = build_model(False).state_dict()
        weights["classifier.3.bias"][0] = float("nan")
        torch.save(weights, path)
        with self.assertRaisesRegex(ValueError, "finite tensors"):
            score.init()


if __name__ == "__main__":
    unittest.main()
