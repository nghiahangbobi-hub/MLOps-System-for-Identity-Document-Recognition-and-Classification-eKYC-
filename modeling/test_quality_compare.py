"""Tests for validation-only selection, calibration, and approved dataset binding."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from . import quality_compare as qc
from . import score
from .network import DOCUMENT_LABELS, QUALITY_LABELS, build_model
from .quality_metrics import calibration, classification_metrics, coverage, fit_temperature, select_threshold
from .test_release import image_request, metadata


def row(identifier, split, label="ID_Card"):
    return {"image_id": identifier, "image_path": f"{identifier}.jpg", "template_id": identifier,
            "document_id": identifier, "split": split, "label": label, "quality_status": ""}


class MetricsTests(unittest.TestCase):
    def test_confusion_macro_recall_and_empty_quality(self):
        actual = torch.tensor([0, 0, 1, 1])
        predicted = torch.tensor([0, 0, 0, 1])
        result = classification_metrics(actual, predicted, DOCUMENT_LABELS)
        self.assertEqual(result["confusion_true_rows_predicted_columns"], [[2, 0], [1, 1]])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (0.8 + 2 / 3) / 2)
        self.assertEqual(result["per_class_recall"], {"ID_Card": 1.0, "Passport": 0.5})
        pending = qc.quality_metrics(torch.zeros(4, 2), torch.full((4,), qc.IGNORE_QUALITY),
                                     dict.fromkeys(QUALITY_LABELS, 0))
        self.assertEqual(pending["status"], "pending")
        self.assertEqual(pending["reviewed_samples"], 0)

    def test_calibration_nll_ece_and_fixed_policy(self):
        logits = torch.full((4, 2), 0.0)
        targets = torch.tensor([0, 0, 0, 1])
        scores, confidence, correct = calibration(logits, targets, 1)
        self.assertAlmostEqual(scores["ece"], 0.25)
        self.assertEqual(fit_temperature(logits, targets), 1)
        self.assertEqual(select_threshold(confidence, correct, 0.85), (1, True))
        self.assertEqual(coverage(torch.ones(4), correct, 1.0, True)["retained"], 0)
        logits[:, 0] = 8
        calibrated = fit_temperature(logits, targets)
        self.assertGreater(calibrated, 1)
        self.assertLess(calibration(logits, targets, calibrated)[0]["nll"],
                        calibration(logits, targets, 1)[0]["nll"])

    def test_ties_at_threshold_and_no_test_retuning(self):
        confidence = torch.tensor([0.9, 0.8, 0.8, 0.5], dtype=torch.float64)
        correct = torch.tensor([True, True, False, False])
        self.assertEqual(select_threshold(confidence, correct, 0.8), (0.9, False))
        policy = select_threshold(confidence, correct, 0.66)
        self.assertEqual(policy, (0.8, False))
        self.assertEqual(coverage(confidence, ~correct, *policy),
                         {"retained": 3, "coverage": 0.75, "retained_accuracy": 1 / 3})

    def test_inference_uses_calibration_and_explicit_review_all(self):
        settings = {**metadata(), "document_temperature": 2.0, "document_review_all": False}
        logits = torch.tensor([[2., 0., 3., 0., 0., 0.]])
        with patch.object(score, "_predictor", (lambda _: logits, lambda _: torch.zeros(3, 32, 32), settings)):
            result = score.run(image_request())
            self.assertAlmostEqual(result["confidence"], torch.sigmoid(torch.tensor(1.)).item())
            self.assertFalse(result["needs_review"])
            settings["document_review_all"] = True
            self.assertTrue(score.run(image_request())["needs_review"])


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_configuration_freezes_backbone_or_unfreezes_last_four_blocks(self):
        for name, (blocks, rate) in qc.CANDIDATES.items():
            with self.subTest(candidate=name):
                model = build_model(False)
                optimizer, frozen = qc.training_setup(model, blocks, rate)
                self.assertEqual(len(model.features) - frozen, blocks)
                self.assertTrue(all(not p.requires_grad for p in model.features[:frozen].parameters()))
                self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))
                if blocks:
                    self.assertTrue(all(p.requires_grad for p in model.features[frozen:].parameters()))
                    self.assertEqual([group["lr"] for group in optimizer.param_groups], [qc.HEAD_LR, rate])
                else:
                    self.assertEqual(len(optimizer.param_groups), 1)

    def test_development_rejects_test_rows_or_validation_changes(self):
        training, validation = row("train", "train"), row("validation", "validation")
        reference = [training, validation]
        qc.check_development([{**training, "label": "Passport"}, validation], reference)
        with self.assertRaisesRegex(ValueError, "Validation labels"):
            qc.check_development([training, {**validation, "label": "Passport"}], reference)
        with self.assertRaisesRegex(ValueError, "fixed ETL identity"):
            qc.check_development([training, validation, row("test", "test")], reference)

    def test_cached_validation_preprocessing_matches_serving_transforms(self):
        from PIL import Image
        from .network import preprocessing

        path = self.root / "test.jpg"
        Image.new("RGB", (29, 61), "red").save(path)
        for crop_mode in ("full_frame", "center_square"):
            with self.subTest(crop_mode=crop_mode):
                item = qc.DevelopmentImages(self.root, [(path.name, 0, -100)],
                                            False, 0, crop_mode)[0]
                with Image.open(path) as image:
                    expected = preprocessing(224, crop_mode=crop_mode)(image.convert("RGB"))
                self.assertTrue(torch.equal(item[0], expected))

    def test_seal_selects_only_using_validation_and_keeps_a_bound_snapshot(self):
        comparisons = []
        for name, macro_f1 in (("first", 0.7), ("second", 0.8)):
            trial = self.root / name
            (trial / "bundle").mkdir(parents=True)
            (trial / "development.csv").write_text("development")
            (trial / "bundle" / "weights.pt").write_text(name)
            (trial / "selection.json").write_text(json.dumps({
                "selection_schema": 1, "seed": 42, "selected": "frozen",
                "reference_development_sha256": "fixture", "development_sha256": "snapshot",
                "held_out_sha256": None, "approval_lineage": None,
                "candidates": {"frozen": {"validation": {
                    "accuracy": macro_f1, "macro_f1": macro_f1, "calibration": {"nll": 1}}}},
            }))
            comparisons.append(trial)
        final = self.root / "final"
        qc.seal(SimpleNamespace(comparison=comparisons, output=final))
        self.assertEqual((final / "bundle" / "weights.pt").read_text(), "second")
        lock = json.loads((final / "sealed.json").read_text())
        self.assertEqual(lock["selection_sha256"], qc.sha256(final / "selection.json"))

    def test_evaluation_claim_is_exclusive(self):
        qc.claim_evaluation(self.root, "first")
        with self.assertRaises(FileExistsError):
            qc.claim_evaluation(self.root, "second")
        self.assertEqual(json.loads((self.root / "held-out.started").read_text())["held_out_sha256"], "first")

    def test_approved_version_rebuild_rejects_tampered_lineage(self):
        base = self.root / "base.csv"
        dev = [row("training", "train"), row("validation", "validation")]
        test = row("test", "test")
        qc.write_csv(base, [*dev, test])
        published = self.root / "published"
        published.mkdir()
        qc.write_csv(published / "manifest.csv", [*dev, test])
        lineage = {"version_id": "fixture", "manifest_sha256": qc.sha256(published / "manifest.csv"),
                   "approval_sha256": "approval", "approval": {
                       "export_sha256": "export", "base_manifest_sha256": qc.sha256(base)}}
        (published / "lineage.json").write_text(json.dumps(lineage))
        selected = self.root / "selected"
        selected.mkdir()
        for name in ("lineage.json", "manifest.csv"):
            (selected / name).write_bytes((published / name).read_bytes())
        options = SimpleNamespace(approved_version=selected, batch=self.root / "batch",
                                  approval=self.root / "approval", review_export=self.root / "export",
                                  base_manifest=base, dataset_root=self.root)
        with patch("labeling.hitl.publish", return_value=published) as verify:
            rows, verified = qc.approved_development(options, dev)
            self.assertEqual(rows, dev)
            self.assertEqual(verified["version_id"], "fixture")
            verify.assert_called_once()
            (selected / "lineage.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "verified lineage.json"):
                qc.approved_development(options, dev)


if __name__ == "__main__":
    unittest.main()
