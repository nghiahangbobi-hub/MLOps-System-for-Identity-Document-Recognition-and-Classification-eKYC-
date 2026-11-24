import unittest
from threading import Lock
from unittest.mock import Mock
from urllib.error import URLError

from demo.review import Review, ReviewConflict


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = Review.__new__(Review)
        self.review.batch = {"batch_id": "test"}
        self.review.lock = Lock()
        self.review.tasks = Mock(return_value={"sample": {"id": 7}})
        self.records = []

        def request(path, body=None):
            self.assertEqual(path, "/api/tasks/7/annotations/")
            if body is None:
                return self.records
            record = {"id": 8, "completed_by": 12, "result": body["result"]}
            self.records.append(record)
            return record

        self.review.request = Mock(side_effect=request)

    def test_summary_distinguishes_pending_from_previously_reviewed_and_missing(self):
        self.review.list = Mock(return_value=[
            {"available": False, "annotation": None},
            {"available": True, "annotation": None},
            {"available": True, "annotation": {"id": 8}},
            {"available": True, "annotation": {"conflict": True}},
        ])
        self.assertEqual(self.review.summary(), dict(pending=1, reviewed=1, missing=1, conflicts=1))

    def test_retry_and_conflict(self):
        first = self.review.save("sample", "Passport", "Blur")
        self.assertEqual(first, {"id": 8, "reviewer": 12, "label": "Passport", "quality": "Blur"})
        self.assertEqual(self.review.save("sample", "Passport", "Blur"), first)
        with self.assertRaises(ReviewConflict):
            self.review.save("sample", "Passport", "Glare")
        self.assertEqual(len(self.records), 1)

    def test_protect_outside_scope_and_ignore_canceled_annotation(self):
        self.records.append({"id": 3, "was_cancelled": True, "result": []})
        with self.assertRaises(ValueError):
            self.review.save("sample", "Outside_Scope", "Glare")
        self.assertEqual(self.review.save("sample", "Outside_Scope", None)["quality"], None)
        result = self.records[-1]["result"]
        self.assertEqual([entry["from_name"] for entry in result], ["document_type"])

    def test_connection_failure_after_submission_reconciles_by_reading(self):
        request = self.review.request.side_effect

        def fail_after_commit(path, body=None):
            response = request(path, body)
            if body is not None:
                raise URLError("connection lost after commit")
            return response

        self.review.request.side_effect = fail_after_commit
        self.assertEqual(self.review.save("sample", "Passport", "Blur")["id"], 8)
        self.assertEqual(len(self.records), 1)
