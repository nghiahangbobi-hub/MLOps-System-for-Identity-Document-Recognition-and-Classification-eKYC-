import json
import subprocess
import sys
import unittest
from io import BytesIO
from unittest.mock import patch

from demo.operations import azure_jobs, azure_log_command, github_runs, stream_logs


class OperationsTests(unittest.TestCase):
    @patch.dict("os.environ", {"AZ_RESOURCE_GROUP": "test-rg", "AZ_WORKSPACE": "test-ml"})
    @patch("demo.operations.subprocess.run")
    def test_azure_includes_archived_and_distinguishes_active_jobs(self, run):
        run.return_value.stdout = json.dumps([
            {"name": "prior", "status": "Completed", "creation_context": {"created_at": "2026-01-01"}},
            {"name": "active", "status": "Running", "creation_context": {"created_at": "2026-01-02"}},
        ])
        self.assertEqual([(job["id"], job["active"]) for job in azure_jobs()],
                         [("active", True), ("prior", False)])
        self.assertIn("--include-archived", run.call_args.args[0])
        self.assertIn("active", azure_log_command("active"))
        with self.assertRaises(ValueError):
            azure_log_command("--debug")

    @patch.dict("os.environ", {"EKYC_GITHUB_REPO": "example/ekyc"})
    @patch("demo.operations.urlopen")
    def test_github_uses_conclusion_for_completed_runs(self, urlopen):
        urlopen.return_value.__enter__.return_value = BytesIO(json.dumps({"workflow_runs": [
            {"id": 5, "name": "CI", "status": "completed", "conclusion": "failure",
             "html_url": "https://github.com/example/ekyc/actions/runs/5"},
        ]}).encode())
        run = github_runs()[0]
        self.assertEqual(run["status"], "failure")
        self.assertFalse(run["active"])

    def test_log_stream_emits_utf8_and_reports_exit(self):
        events = []
        with subprocess.Popen([sys.executable, "-c", "print('Đang chạy')"],
                              stdout=subprocess.PIPE) as process:
            message = stream_logs(process, lambda event, data: events.append((event, data)))
        self.assertEqual(events, [("line", "Đang chạy\n")])
        self.assertEqual(message, "Đã kết thúc luồng log.")
