"""Run from the repository root with python -m demo.server."""

import argparse
import hashlib
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter

from demo.client import MAX_IMAGE_BYTES, ScoringError, encode_image, score
from demo.review_queue import ReviewQueue

LOGGER = logging.getLogger("ekyc.demo")
UPLOAD_TIMEOUT_SECONDS = 15


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def reply(self, code, body, content_type="application/json", *, headers=None):
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
        if self.path == "/":
            html = Path(__file__).with_name("index.html").read_bytes()
            self.reply(200, html, "text/html; charset=utf-8")
        else:
            self.reply(404, b'{"error":"Unknown route."}')

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
        if self.path == "/report" and queue is not None:
            self.report(queue)
        elif self.path == "/predict":
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
            status = 200
            if queue is not None:
                digest = hashlib.sha256(image).hexdigest()
                review_status = "not_allowlisted"
                if digest in queue.allowlist:
                    review_status = "available"
                    if result["needs_review"]:
                        queue.enqueue(digest, result, "needs_review", image=image)
                        review_status = "queued"
                    headers["X-EKYC-Report-Token"] = queue.receipt(digest, result)
                headers["X-EKYC-Review-Status"] = review_status
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
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--review-queue", type=Path)
    parser.add_argument("--review-manifest", type=Path)
    parser.add_argument("--review-dataset-root", type=Path)
    args = parser.parse_args()
    config = (args.review_queue, args.review_manifest, args.review_dataset_root)
    if any(config) and not all(config):
        parser.error("Configure all three review options together.")
    try:
        queue = ReviewQueue(*config) if all(config) else None
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO)
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.review_queue = queue
        LOGGER.info("Demo listening on http://127.0.0.1:%d", server.server_port)
        server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
