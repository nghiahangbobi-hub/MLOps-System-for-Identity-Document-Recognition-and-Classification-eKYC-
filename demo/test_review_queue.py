import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import http.client
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from demo import client, review_queue, server
from demo.test_release import serving


MOCK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j9v8AAAAASUVORK5CYII="
)
NEEDS_REVIEW = dict(label="ID_Card", status="Blur", confidence=0.7, needs_review=True)
CLEAR = dict(label="Passport", status="Clear", confidence=0.9, needs_review=False)


def call(port, route, body, content_type):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
    try:
        connection.request("POST", route, body, {"Content-Type": content_type})
        response = connection.getresponse()
        return response.status, json.load(response), response.headers
    finally:
        connection.close()


class ReviewQueueTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(TemporaryDirectory())
        self.root = Path(directory)
        self.manifest = self.root / "manifest.csv"
        self.source = self.root / "sample.png"
        self.source.write_bytes(MOCK_PNG)
        self.manifest.write_text("image_id,image_path\nmock/one,sample.png\n")
        self.queue = review_queue.ReviewQueue(self.root / "queue", self.manifest, self.root)
        self.digest = hashlib.sha256(MOCK_PNG).hexdigest()
        self.event_dir = self.queue.events / self.digest

    def predict(self, port, image=MOCK_PNG):
        return call(port, "/predict", image, "image/png")

    def report(self, port, token, **extra):
        return call(port, "/report", json.dumps(dict(token=token, **extra)), "application/json")

    def event(self):
        return json.loads((self.event_dir / "event.json").read_bytes())

    def test_automatic_queue_concurrent_duplicates_then_report(self):
        with serving(server.Handler, review_queue=self.queue) as port:
            with patch.object(server, "score", return_value=NEEDS_REVIEW):
                with ThreadPoolExecutor(6) as pool:
                    results = list(pool.map(lambda _: self.predict(port), range(12)))
            for status, prediction, headers in results:
                self.assertEqual((status, prediction), (200, NEEDS_REVIEW))
                self.assertEqual(headers["X-EKYC-Review-Status"], "queued")
            token = results[0][2]["X-EKYC-Report-Token"]
            with ThreadPoolExecutor(6) as pool:
                reports = list(pool.map(lambda _: self.report(port, token), range(12)))
            self.assertTrue(all(status == 200 for status, _, _ in reports))
        self.assertEqual(list(self.queue.events.iterdir()), [self.event_dir])
        self.assertEqual({path.name for path in self.event_dir.iterdir()}, {"event.json", "image.png"})
        self.assertEqual((self.event_dir / "image.png").read_bytes(), MOCK_PNG)
        self.assertEqual(self.event(), {
            "event_id": self.digest, "image_sha256": self.digest, "image_file": "image.png",
            "source_image_id": "mock/one", "prediction": NEEDS_REVIEW,
            "reasons": ["needs_review", "reported_incorrect"],
        })

    def test_report_uses_exact_prior_prediction_and_survives_restart(self):
        with serving(server.Handler, review_queue=self.queue) as port:
            with patch.object(server, "score", return_value=CLEAR):
                status, prediction, headers = self.predict(port)
            self.assertEqual((status, prediction), (200, CLEAR))
            self.assertEqual(headers["X-EKYC-Review-Status"], "available")
            self.assertFalse(self.event_dir.exists())
            token = headers["X-EKYC-Report-Token"]
            with patch.object(server, "score") as score:
                self.assertEqual(self.report(port, token)[0], 200)
                score.assert_not_called()
        restarted = review_queue.ReviewQueue(self.queue.root, self.manifest, self.root)
        with serving(server.Handler, review_queue=restarted) as port:
            self.assertEqual(self.report(port, token)[0], 403)
            with patch.object(server, "score", return_value=NEEDS_REVIEW):
                self.assertEqual(self.predict(port)[0], 200)
        self.assertEqual(self.event()["prediction"], CLEAR)
        self.assertEqual(self.event()["reasons"], ["reported_incorrect", "needs_review"])

    def test_invalid_unallowlisted_unconfigured_and_no_image_route(self):
        with serving(server.Handler, review_queue=self.queue) as port:
            with patch.object(server, "score", return_value=NEEDS_REVIEW):
                status, _, headers = self.predict(port, MOCK_PNG + b"changed")
                self.assertEqual(status, 200)
                self.assertEqual(headers["X-EKYC-Review-Status"], "not_allowlisted")
                self.assertIsNone(headers["X-EKYC-Report-Token"])
                self.assertEqual(list(self.queue.events.iterdir()), [])
                _, _, headers = self.predict(port)
            token = headers["X-EKYC-Report-Token"]
            forged = bytearray(base64.urlsafe_b64decode(token))
            forged[-3] ^= 1
            self.assertEqual(self.report(port, base64.urlsafe_b64encode(forged).decode())[0], 403)
            self.assertEqual(self.report(port, token, prediction=CLEAR)[0], 400)
            for invalid in (b"{", b"[]", b"x" * 4097):
                self.assertIn(call(port, "/report", invalid, "application/json")[0], (400, 413))
            with patch.object(server, "score", side_effect=client.ScoringError("invalid image", status=400)):
                self.assertEqual(self.predict(port, b"invalid")[0], 400)
            with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=2)) as connection:
                connection.request("GET", f"/events/{self.digest}/image.png")
                self.assertEqual(connection.getresponse().status, 404)
            self.assertEqual(self.event()["reasons"], ["needs_review"])
        with serving(server.Handler) as port, patch.object(server, "score", return_value=CLEAR):
            status, prediction, headers = self.predict(port)
            self.assertEqual((status, prediction), (200, CLEAR))
            self.assertIsNone(headers["X-EKYC-Review-Status"])
            self.assertEqual(self.report(port, token)[0], 404)

    def test_publish_and_reason_storage_failures_are_retryable(self):
        with serving(server.Handler, review_queue=self.queue) as port:
            with patch.object(server, "score", return_value=NEEDS_REVIEW):
                with patch.object(review_queue, "write_event", side_effect=OSError("disk full")):
                    with self.assertLogs(server.LOGGER, "WARNING"):
                        self.assertEqual(self.predict(port)[0], 503)
                self.assertEqual(list(self.queue.events.iterdir()), [])
                status, _, headers = self.predict(port)
            self.assertEqual(status, 200)
            token = headers["X-EKYC-Report-Token"]
            before = (self.event_dir / "event.json").read_bytes()
            with patch.object(review_queue.os, "replace", side_effect=OSError("disk full")):
                with self.assertLogs(server.LOGGER, "WARNING"):
                    self.assertEqual(self.report(port, token)[0], 503)
            self.assertEqual((self.event_dir / "event.json").read_bytes(), before)
            self.assertEqual(self.report(port, token)[0], 200)
        self.assertEqual(self.event()["reasons"], ["needs_review", "reported_incorrect"])
        self.assertEqual(len(list(self.event_dir.iterdir())), 2)

    def test_report_refuses_changed_source_and_accepts_retry(self):
        with serving(server.Handler, review_queue=self.queue) as port:
            with patch.object(server, "score", return_value=CLEAR):
                _, _, headers = self.predict(port)
            token = headers["X-EKYC-Report-Token"]
            self.source.write_bytes(MOCK_PNG + b"changed after indexing")
            with self.assertLogs(server.LOGGER, "WARNING"):
                self.assertEqual(self.report(port, token)[0], 503)
            self.assertFalse(self.event_dir.exists())
            self.source.write_bytes(MOCK_PNG)
            self.assertEqual(self.report(port, token)[0], 200)

    def test_directory_sync_failure_after_publication_is_retryable(self):
        sync = review_queue.sync_directory

        def fail_events(path):
            if path == self.queue.events:
                raise OSError("sync failed")
            sync(path)

        with serving(server.Handler, review_queue=self.queue) as port:
            with patch.object(server, "score", return_value=NEEDS_REVIEW):
                with patch.object(review_queue, "sync_directory", side_effect=fail_events):
                    with self.assertLogs(server.LOGGER, "WARNING"):
                        self.assertEqual(self.predict(port)[0], 503)
                self.assertEqual(self.event()["reasons"], ["needs_review"])
                self.assertEqual(self.predict(port)[0], 200)
        self.assertEqual(list(self.queue.events.iterdir()), [self.event_dir])
        self.assertEqual(self.event()["reasons"], ["needs_review"])

    def test_duplicate_surfaces_corrupt_persisted_image(self):
        with serving(server.Handler, review_queue=self.queue) as port:
            with patch.object(server, "score", return_value=NEEDS_REVIEW):
                self.assertEqual(self.predict(port)[0], 200)
                (self.event_dir / "image.png").write_bytes(b"damaged")
                with self.assertLogs(server.LOGGER, "WARNING"):
                    self.assertEqual(self.predict(port)[0], 503)
        self.assertEqual(self.event()["reasons"], ["needs_review"])

    def test_manifest_rejects_ambiguous_paths_and_shared_queue(self):
        image_alias = self.root / "alias.png"
        image_alias.symlink_to(self.source)
        self.manifest.write_text("image_id,image_path\none,sample.png\ntwo,alias.png\n")
        with self.assertRaisesRegex(ValueError, "Ambiguous image digest"):
            review_queue.load_allowlist(self.manifest, self.root)
        with TemporaryDirectory() as outside:
            outside_image = Path(outside) / "outside.png"
            outside_image.write_bytes(MOCK_PNG)
            image_alias.unlink()
            image_alias.symlink_to(outside_image)
            self.manifest.write_text("image_id,image_path\none,alias.png\n")
            with self.assertRaisesRegex(ValueError, "Invalid mock manifest entry"):
                review_queue.load_allowlist(self.manifest, self.root)
        self.manifest.write_text("image_id,image_path\nmock/one,sample.png\n")
        self.queue.root.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "owner-only"):
            review_queue.ReviewQueue(self.queue.root, self.manifest, self.root)
