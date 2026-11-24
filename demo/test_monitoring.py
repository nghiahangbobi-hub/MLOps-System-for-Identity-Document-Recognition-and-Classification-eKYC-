import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from demo.monitoring import Monitoring, azure_snapshot


class MonitoringTests(unittest.TestCase):
    def test_snapshot_filters_window_and_counts_threshold_and_sla_boundaries(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "metrics.sqlite3"
            metrics = Monitoring(path, 0.7)
            with patch("demo.monitoring.time", return_value=1000):
                metrics.record(200, 10, 0.1)
            with patch("demo.monitoring.time", return_value=2000):
                metrics.record(200, 3000, 0.69)
                metrics.record(200, 3001, 0.7)
                metrics.record(502, 10, None)
                snapshot = Monitoring(path, 0.7).snapshot("15m", None, "http://localhost:8080")
                self.assertEqual(snapshot["total"], 3)
                self.assertEqual(snapshot["scored"], 2)
                self.assertEqual(snapshot["low"], 1)
                self.assertEqual(snapshot["within_sla"], 1)
                self.assertEqual(snapshot["errors"], 1)
                self.assertEqual(snapshot["latest"], (502, 10))
                self.assertIsNone(snapshot["queued"])

    def test_counts_only_committed_queue_directories(self):
        with TemporaryDirectory() as root:
            events = Path(root) / "events"
            events.mkdir()
            for name in ("a" * 64, "b" * 64, ".staging"):
                (events / name).mkdir()
            for name in ("a" * 64, ".staging"):
                (events / name / "event.json").touch()
            store = Monitoring(Path(root) / "metrics.sqlite3", 0.7)
            result = store.snapshot("1h", SimpleNamespace(events=events), "http://localhost:8080")
            self.assertEqual(result["queued"], 1)
            self.assertEqual(result["total"], 0)
            self.assertIsNone(result["latest"])

    @patch.dict("os.environ", {
        "EKYC_LOG_ANALYTICS_WORKSPACE": "8247bd35-ba59-4ed5-a404-302d964b325a",
        "AZ_ENDPOINT": "ekyc-test",
        "EKYC_SCORING_DEPLOYMENT": "green",
    })
    @patch("demo.monitoring.subprocess.run")
    def test_azure_aggregates_buckets_for_configured_deployment(self, run):
        run.return_value.stdout = json.dumps([
            {"SuccessfulPredictions": 10, "LowConfidenceEvents": 2},
            {"SuccessfulPredictions": 5, "LowConfidenceEvents": 1},
        ])
        self.assertEqual(azure_snapshot("15m"), {"scored": 15, "low": 3})
        command = run.call_args.args[0]
        query = command[command.index("--analytics-query") + 1]
        self.assertIn('let endpoint = "ekyc-test";', query)
        self.assertIn('let deployment = "green";', query)
        self.assertIn("windowEnd - 15m", query)
