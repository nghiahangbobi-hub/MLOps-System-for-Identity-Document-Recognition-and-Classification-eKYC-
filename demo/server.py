"""Run from the repository root with python -m demo.server."""

import argparse
from base64 import b64decode, b64encode
from binascii import Error as Base64Error
from hmac import compare_digest
import json
import logging
import os
import sqlite3
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter, sleep
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

from demo.client import MAX_IMAGE_BYTES, ScoringError, encode_image, score
from demo.monitoring import Monitoring, WINDOWS, azure_snapshot
from demo.operations import (
    LOG_SLOTS, azure_jobs, azure_log_command, github_runs, local_build_log, local_builds, stream_logs,
)
from demo.review_queue import ReviewQueue

LOGGER = logging.getLogger("ekyc.demo")
UPLOAD_TIMEOUT_SECONDS = 15


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def reply(self, code, body, content_type="application/json", *, headers=None):
        monitoring = getattr(self.server, "monitoring", None)
        start = getattr(self, "prediction_start", None)
        if monitoring is not None and start is not None:
            try:
                monitoring.record(code, (perf_counter() - start) * 1000,
                                  getattr(self, "prediction_confidence", None))
            except sqlite3.Error:
                LOGGER.exception("Could not record prediction metrics")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            route = urlsplit(self.path)
        except ValueError:
            self.reply(400, b'{"error":"Invalid URL."}')
            return
        if route.path == "/review" or route.path.startswith("/api/review/"):
            self.review_request(route.path)
            return
        pages = {"/": "index.html", "/monitoring": "monitoring.html"}
        if route.path in pages:
            html = Path(__file__).with_name(pages[route.path]).read_bytes()
            self.reply(200, html, "text/html; charset=utf-8")
        elif route.path in {"/demo.css", "/demo/blocks.js"}:
            asset = Path(__file__).with_name(route.path.rsplit("/", 1)[-1])
            mime = "text/css" if asset.suffix == ".css" else "text/javascript"
            self.reply(200, asset.read_bytes(), f"{mime}; charset=utf-8")
        elif route.path in {"/api/operations/azure", "/api/operations/github", "/api/operations/builds"}:
            source = route.path.rsplit("/", 1)[-1]
            self.operation_list({"azure": azure_jobs, "github": github_runs, "builds": local_builds}[source])
        elif route.path == "/api/operations/build-log":
            self.build_log(route)
        elif route.path == "/api/operations/log":
            self.operation_log(route)
        elif route.path == "/api/monitoring/review":
            self.review_summary()
        elif route.path in {"/api/monitoring", "/api/monitoring/azure", "/api/monitoring/live"}:
            self.monitoring_snapshot(route)
        else:
            self.reply(404, b'{"error":"Unknown route."}')

    def review_request(self, path, *, write=False):
        review = getattr(self.server, "review", None)
        if review is None:
            status = 503
            error = "Chưa cấu hình lô duyệt và dự án Label Studio."
        else:
            from demo.review import ReviewConflict

            try:
                review = self.authenticate_reviewer(review)
            except HTTPError as exc:
                if exc.code in (403, 404):
                    self.reply(403, b'{"error":"Label Studio denied access to this review project."}')
                    return
                if exc.code != 401:
                    self.reply(502, b'{"error":"Label Studio login unavailable."}')
                    return
                review = None
            except (OSError, RuntimeError):
                self.reply(502, b'{"error":"Label Studio login unavailable."}')
                return
            if review is None:
                self.reply(401, b'{"error":"Reviewer login required."}', headers={
                    "WWW-Authenticate": 'Basic realm="eKYC Label Studio", charset="UTF-8"'
                })
                return

            try:
                if path == "/review" and not write:
                    self.reply(200, Path(__file__).with_name("review.html").read_bytes(), "text/html; charset=utf-8")
                    return
                if path == "/api/review/tasks":
                    if write:
                        if (self.headers.get_content_type() != "application/json"
                                or self.headers.get("X-EKYC-Review") != "1"):
                            self.reply(415, b'{"error":"Expected a review request."}')
                            return
                        body = self.read_payload(4096, b'{"error":"Review exceeds the size limit."}')
                        if body is None:
                            return
                        data = json.loads(body)
                        if not isinstance(data, dict) or set(data) != {"event_id", "label", "quality"} or any(
                            not isinstance(data[key], str) for key in ("event_id", "label")
                        ) or data["quality"] is not None and not isinstance(data["quality"], str):
                            raise ValueError("Nhãn không hợp lệ.")
                        result = review.save(**data)
                    else:
                        result = {"events": review.list()}
                    self.reply(200, json.dumps(result).encode())
                    return
                if path.startswith("/api/review/image/") and not write:
                    event_id = path.rsplit("/", 1)[-1]
                    if event_id in review.events:
                        image, mime = review.image(event_id)
                        self.reply(200, image, mime)
                        return
                status, error = 404, "Không tìm thấy tác vụ."
            except ReviewConflict as exc:
                status, error = 409, str(exc)
            except (ValueError, RecursionError) as exc:
                status, error = (400, str(exc)) if write else (502, "Dữ liệu Label Studio không hợp lệ.")
            except (OSError, KeyError, TypeError):
                status, error = 502, "Không đọc hoặc lưu được tác vụ trên Label Studio. Có thể thử lại."
        self.reply(status, json.dumps({"error": error}).encode())

    def authenticate_reviewer(self, review):
        authorization = self.headers.get("Authorization", "")
        if self.server.review_auth is not None:
            return review if compare_digest(authorization.encode(), self.server.review_auth) else None
        scheme, _, payload = authorization.partition(" ")
        if scheme.lower() != "basic" or len(payload) > 4096:
            return None
        try:
            email, separator, password = b64decode(payload, validate=True).decode().partition(":")
        except (Base64Error, ValueError):
            return None
        if separator and email and password:
            return review.for_reviewer(email.strip(), password)
        return None

    def monitoring_snapshot(self, route):
        windows = parse_qs(route.query, keep_blank_values=True).get("window", ["1h"])
        if len(windows) != 1 or windows[0] not in WINDOWS:
            self.reply(400, b'{"error":"Unknown monitoring window."}')
            return
        if route.path.endswith("/live"):
            self.live_metrics(windows[0])
            return
        try:
            if route.path.endswith("/azure"):
                metrics = azure_snapshot(windows[0])
            else:
                metrics = self.metrics(windows[0])
        except (OSError, sqlite3.Error, subprocess.SubprocessError, ValueError, KeyError, TypeError):
            LOGGER.warning("Monitoring data unavailable for %s", route.path)
            self.reply(503, b'{"error":"Monitoring data unavailable."}')
            return
        self.reply(200, json.dumps(metrics).encode())

    def metrics(self, window):
        metrics = self.server.monitoring.snapshot(
            window, self.server.review_queue, self.server.label_studio_url
        )
        review = getattr(self.server, "review", None)
        metrics["label_studio_project_url"] = f"{review.base}/projects/{review.project}/data" if review else None
        return metrics

    def review_summary(self):
        review = getattr(self.server, "review", None)
        summary = None
        if review is not None:
            try:
                summary = review.summary()
            except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                self.reply(503, b'{"error":"Label Studio status unavailable."}')
                return
        self.reply(200, json.dumps({"summary": summary}).encode())

    def operation_list(self, load):
        try:
            runs = load()
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
            LOGGER.warning("Could not load %s", load.__name__)
            self.reply(503, b'{"error":"Source unavailable."}')
            return
        self.reply(200, json.dumps({"runs": runs}).encode())

    def build_log(self, route):
        names = parse_qs(route.query).get("run", [])
        try:
            if len(names) != 1:
                raise ValueError("Expected one build ID")
            result = local_build_log(names[0])
        except ValueError:
            self.reply(400, b'{"error":"Invalid build ID."}')
            return
        except OSError:
            self.reply(404, b'{"error":"Build log unavailable."}')
            return
        self.reply(200, json.dumps(result).encode())

    def start_events(self):
        self.connection.settimeout(8)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

    def emit(self, event, data):
        self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

    def live_metrics(self, window):
        self.start_events()
        previous = None
        until = perf_counter() + 90
        try:
            while perf_counter() < until:
                try:
                    metrics = self.metrics(window)
                except (sqlite3.Error, OSError):
                    self.emit("unavailable", None)
                    return
                if metrics != previous:
                    self.emit("snapshot", metrics)
                    previous = metrics
                else:
                    self.emit("heartbeat", None)
                sleep(1)
        except OSError:
            pass

    def operation_log(self, route):
        names = parse_qs(route.query).get("job", [])
        try:
            command = azure_log_command(names[0]) if len(names) == 1 else None
        except ValueError:
            command = None
        if command is None:
            self.reply(400, b'{"error":"Invalid job name or workspace configuration."}')
            return
        if not LOG_SLOTS.acquire(blocking=False):
            self.reply(429, b'{"error":"Too many open log streams."}')
            return
        try:
            self.start_events()
            try:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL) as process:
                    try:
                        message = stream_logs(process, self.emit)
                    finally:
                        if process.poll() is None:
                            process.kill()
                self.emit("end", message)
            except (FileNotFoundError, subprocess.SubprocessError):
                self.emit("end", "Không đọc được luồng log Azure CLI.")
        except OSError:
            pass
        finally:
            LOG_SLOTS.release()

    def read_payload(self, limit, size_error):
        lengths = self.headers.get_all("Content-Length", [])
        if (self.headers.get_all("Transfer-Encoding") or len(lengths) != 1
                or not lengths[0].isascii() or not lengths[0].isdecimal()):
            self.reply(400, b'{"error":"Expected one valid Content-Length header."}')
            return
        length = lengths[0].lstrip("0") or "0"
        size = int(length) if len(length) <= len(str(limit)) else limit + 1
        if not 0 < size <= limit:
            self.reply(413, size_error)
            return
        try:
            self.connection.settimeout(UPLOAD_TIMEOUT_SECONDS)
            payload = self.rfile.read(size)
            if len(payload) != size:
                self.reply(400, b'{"error":"Image upload was incomplete."}')
                return
            return payload
        except TimeoutError:
            self.reply(408, b'{"error":"Image upload timed out."}')

    def do_POST(self):
        queue = getattr(self.server, "review_queue", None)
        if self.path == "/api/review/tasks":
            self.review_request(self.path, write=True)
        elif self.path == "/report" and queue is not None:
            self.report(queue)
        elif self.path == "/predict":
            self.prediction_start = perf_counter()
            self.prediction_confidence = None
            self.predict(queue)
        else:
            self.reply(404, b'{"error":"Unknown route."}')

    def predict(self, queue):
        if self.headers.get_content_type() not in {"image/jpeg", "image/png"}:
            self.reply(415, b'{"error":"Choose a JPEG or PNG image."}')
            return
        start = perf_counter()
        image = self.read_payload(MAX_IMAGE_BYTES, b'{"error":"Choose an image between 1 byte and 8 MiB."}')
        if image is None:
            return
        headers = {}
        try:
            result = score(encode_image(image))
            self.prediction_confidence = result.get("confidence")
            status = 200
            if queue is not None:
                review_status, token = queue.record_prediction(image, result)
                headers["X-EKYC-Review-Status"] = review_status
                if token is not None:
                    headers["X-EKYC-Report-Token"] = token
        except ScoringError as exc:
            status = exc.status
            result = {"error": str(exc)}
        except OSError:
            self.queue_failure()
            return
        LOGGER.info("predict status=%d duration_ms=%.1f", status, (perf_counter() - start) * 1000)
        self.reply(status, json.dumps(result).encode(), headers=headers)

    def report(self, queue):
        if self.headers.get_content_type() != "application/json":
            self.reply(415, b'{"error":"Expected a JSON report."}')
            return
        body = self.read_payload(4096, b'{"error":"Report body exceeds the size limit."}')
        if body is None:
            return
        try:
            report = json.loads(body)
            if not isinstance(report, dict) or set(report) != {"token"} or not isinstance(report["token"], str):
                raise ValueError("Expected a token")
        except (ValueError, RecursionError):
            self.reply(400, b'{"error":"Expected one report token."}')
            return
        receipt = queue.open_receipt(report["token"])
        if receipt is None or receipt[0] not in queue.allowlist:
            self.reply(403, b'{"error":"Report token is invalid. Score the image again."}')
            return
        digest, prediction = receipt
        try:
            queue.enqueue(digest, prediction, "reported_incorrect")
        except OSError:
            self.queue_failure()
            return
        self.reply(200, json.dumps({"event_id": digest, "status": "queued"}).encode())

    def queue_failure(self):
        LOGGER.warning("review_queue status=503")
        self.reply(503, b'{"error":"Review queue is unavailable. Retry the request."}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--review-queue", type=Path)
    parser.add_argument("--review-manifest", type=Path)
    parser.add_argument("--review-dataset-root", type=Path)
    parser.add_argument("--monitoring-db", type=Path, default=Path("runs/monitoring.sqlite3"))
    parser.add_argument("--monitoring-threshold", type=float, default=0.7)
    parser.add_argument("--label-studio-url", default=os.getenv(
        "EKYC_LABEL_STUDIO_URL", "http://127.0.0.1:8080"
    ))
    parser.add_argument("--review-batch", type=Path)
    parser.add_argument("--label-studio-project", type=int)
    args = parser.parse_args()
    if not 0 <= args.monitoring_threshold <= 1:
        parser.error("--monitoring-threshold must be between 0 and 1.")
    try:
        label_studio = urlsplit(args.label_studio_url)
        if label_studio.scheme not in {"https", "http"} or not label_studio.hostname:
            raise ValueError("Use an HTTP(S) URL for Label Studio.")
    except ValueError as exc:
        parser.error(str(exc))
    config = (args.review_queue, args.review_manifest, args.review_dataset_root)
    if any(config) and not all(config):
        parser.error("Configure all three review options together.")
    review = None
    auth = None
    if args.review_batch is not None or args.label_studio_project is not None:
        token = os.getenv("EKYC_LABEL_STUDIO_TOKEN")
        password = os.getenv("EKYC_REVIEW_PASSWORD")
        studio_account = (os.getenv("EKYC_LABEL_STUDIO_EMAIL"), os.getenv("EKYC_LABEL_STUDIO_PASSWORD"))
        credentials = studio_account if not token and all(studio_account) else None
        if not (args.review_batch and args.label_studio_project):
            parser.error("In-app review needs a batch and Label Studio project.")
        if token:
            if not password:
                parser.error("Token-based review needs EKYC_REVIEW_PASSWORD.")
            reviewer = os.getenv("EKYC_REVIEW_USER", "reviewer")
            auth = b"Basic " + b64encode(f"{reviewer}:{password}".encode())
    try:
        if args.review_batch is not None:
            from demo.review import Review

            review = Review(args.review_batch, args.label_studio_url, args.label_studio_project, token,
                            personal=os.getenv("EKYC_LABEL_STUDIO_TOKEN_TYPE") == "personal", credentials=credentials)
        queue = ReviewQueue(*config) if all(config) else None
        monitoring = Monitoring(args.monitoring_db, args.monitoring_threshold)
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO)
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.review_queue = queue
        server.review = review
        server.review_auth = auth
        server.monitoring = monitoring
        server.label_studio_url = args.label_studio_url
        LOGGER.info("Demo listening on http://127.0.0.1:%d", server.server_port)
        server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
