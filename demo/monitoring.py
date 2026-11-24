"""Aggregate demo prediction metrics without storing images or predictions."""

from contextlib import closing
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
from time import time
from uuid import UUID

WINDOWS = {"15m": 15, "1h": 60, "24h": 1440}


class Monitoring:
    def __init__(self, path, threshold):
        self.path = path
        self.threshold = threshold
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS requests "
                "(timestamp REAL, status INTEGER, latency_ms REAL, confidence REAL)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS requests_time ON requests (timestamp)")

    def record(self, status, latency_ms, confidence):
        now = time()
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DELETE FROM requests WHERE timestamp < ?", (now - 86400,))
            db.execute("INSERT INTO requests VALUES (?, ?, ?, ?)", (now, status, latency_ms, confidence))

    def snapshot(self, window, queue, label_studio_url):
        since = time() - WINDOWS[window] * 60
        with closing(sqlite3.connect(self.path)) as db, db:
            total, successes, scored, low, within_sla = db.execute("""
                SELECT count(*),
                       count(CASE WHEN status = 200 THEN 1 END),
                       count(CASE WHEN status = 200 THEN confidence END),
                       count(CASE WHEN status = 200 AND confidence < ? THEN 1 END),
                       count(CASE WHEN status = 200 AND latency_ms <= 3000 THEN 1 END)
                FROM requests WHERE timestamp >= ?
            """, (self.threshold, since)).fetchone()
            latest = db.execute(
                "SELECT status, latency_ms FROM requests WHERE timestamp >= ? "
                "ORDER BY timestamp DESC, rowid DESC LIMIT 1", (since,)
            ).fetchone()
            recent = db.execute(
                "SELECT timestamp, status, latency_ms, confidence FROM requests "
                "WHERE timestamp >= ? ORDER BY timestamp DESC, rowid DESC LIMIT 20", (since,)
            ).fetchall()
        queued = None
        if queue is not None:
            queued = sum(
                entry.is_dir() and not entry.is_symlink() and (entry / "event.json").is_file()
                for entry in queue.events.iterdir() if re.fullmatch("[a-f0-9]{64}", entry.name)
            )
        return {
            "total": total, "errors": total - successes, "scored": scored,
            "low": low, "threshold": self.threshold, "within_sla": within_sla,
            "latest": latest, "recent": recent, "queued": queued, "label_studio_url": label_studio_url,
            "azure_configured": bool(os.getenv("EKYC_LOG_ANALYTICS_WORKSPACE")),
        }


def azure_snapshot(window):
    workspace = str(UUID(os.environ["EKYC_LOG_ANALYTICS_WORKSPACE"]))
    endpoint = os.getenv("AZ_ENDPOINT", "")
    deployment = os.getenv("EKYC_SCORING_DEPLOYMENT", "blue")
    if not all(re.fullmatch(r"[a-zA-Z0-9-]{1,63}", name) for name in (endpoint, deployment)):
        raise ValueError("Invalid Azure endpoint or deployment name")
    query = (Path(__file__).resolve().parents[1] / "azure" / "telemetry-low-confidence.kql").read_text()
    query = (query.replace("<endpoint-name>", endpoint)
             .replace('let deployment = "blue";', f'let deployment = "{deployment}";')
             .replace("windowEnd - 24h", f"windowEnd - {WINDOWS[window]}m"))
    response = subprocess.run(
        [os.getenv("EKYC_AZ_CLI", "az"), "monitor", "log-analytics", "query", "--workspace", workspace,
         "--analytics-query", query, "--output", "json", "--only-show-errors"],
        capture_output=True, check=True, timeout=15,
    )
    rows = json.loads(response.stdout)
    return {
        "scored": sum(row["SuccessfulPredictions"] for row in rows),
        "low": sum(row["LowConfidenceEvents"] for row in rows),
    }
