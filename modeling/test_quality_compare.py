"""Tests for validation-only selection, calibration, and approved dataset binding."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

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

    def test_evaluate_groups_interleaved_domains_in_one_pass(self):
        for with_domain in (True, False):
            with self.subTest(with_domain=with_domain):
                output = self.root / str(with_domain)
                bundle = output / "bundle"
                bundle.mkdir(parents=True)
                development = [row("train-a", "train"), row("train-b", "train", "Passport"),
                               row("validation", "val")]
                held_out = [row("z", "test"), row("a", "test", "Passport"),
                            row("excluded", "test"), row("y", "test", "Passport"),
                            row("b", "test")]
                held_out[2]["quality_status"] = "Outside_Scope"
                if with_domain:
                    for record, domain in zip([*development, *held_out],
                                              ["clips", "clips", "clips",
                                               "photo", "clips", "excluded", "photo", "clips"]):
                        record["domain"] = domain
                for index, record in enumerate([*development, *held_out]):
                    if record["quality_status"] != "Outside_Scope":
                        predicted_passport = record["image_id"] == "y"
                        Image.new("RGB", (3, 3), (int(predicted_passport), index, 0)).save(
                            output / record["image_path"], format="PNG")
                qc.write_csv(output / "development.csv", development)
                qc.write_csv(output / "holdout.csv", held_out)
                qc.write_csv(output / "base.csv", [*development, *held_out])
                (bundle / "metadata.json").touch()
                (bundle / "weights.pt").touch()
                selection = output / "selection.json"
                qc.write_json(selection, {
                    "selection_schema": 1, "seed": qc.SEED, "selected": "frozen", "selected_epoch": 1,
                    "approval_lineage": None, "quality_training_support": dict.fromkeys(QUALITY_LABELS, 0),
                    "held_out_sha256": qc.sha256(output / "holdout.csv"),
                    "development_sha256": qc.sha256(output / "development.csv"),
                    "metadata_sha256": qc.sha256(bundle / "metadata.json"),
                    "weights_sha256": qc.sha256(bundle / "weights.pt"),
                })
                qc.write_json(output / "sealed.json", {"selection_sha256": qc.sha256(selection)})
                settings = {"document_temperature": 1.0, "confidence_threshold": 0.5,
                            "document_review_all": False}
                model = torch.nn.Identity()

                def transform(image):
                    passport = image.getpixel((0, 0))[0]
                    return torch.tensor([1.0 - passport, float(passport)])

                options = SimpleNamespace(output=output, dataset_root=output,
                                          held_out=output / "holdout.csv", base_manifest=output / "base.csv",
                                          device=torch.device("cpu"), batch_size=2, workers=0)
                with patch.dict("os.environ"), patch.object(score, "init"), \
                        patch.object(score, "_predictor", (model, transform, settings)), \
                        patch.object(qc, "predict", wraps=qc.predict) as predict:
                    qc.evaluate(options)
                    predict.assert_called_once()
                result = json.loads((output / "held-out.json").read_text())
                self.assertEqual(result["document"]["samples"], 4)
                self.assertNotIn("excluded", result["templates"])
                if with_domain:
                    self.assertEqual(set(result["domains"]), {"photo", "clips"})
                    photo, clips = (result["domains"][name] for name in ("photo", "clips"))
                    self.assertEqual(photo["confusion_true_rows_predicted_columns"], [[1, 0], [0, 1]])
                    self.assertEqual(clips["confusion_true_rows_predicted_columns"], [[1, 0], [1, 0]])
                    self.assertEqual(photo["macro_f1"], 1.0)
                    self.assertEqual(clips["per_class_support"], {"ID_Card": 1, "Passport": 1})
                else:
                    self.assertNotIn("domains", result)

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
