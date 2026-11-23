import concurrent.futures
from contextlib import contextmanager
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import socket
import threading
import unittest
from unittest.mock import patch

from demo import client, server


PREDICTION = dict(label="Passport", status=None, confidence=0.5, needs_review=True)


@contextmanager
def serving(handler, *, review_queue=None):
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as httpd:
        httpd.review_queue = review_queue
        worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01})
        worker.start()
        try:
            yield httpd.server_port
        finally:
            httpd.shutdown()
            worker.join()


def request(port, body=b"image", **headers):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("POST", "/predict", body, headers)
        response = connection.getresponse()
        return response.status, json.load(response)
    finally:
        connection.close()


class DemoReleaseTests(unittest.TestCase):
    def test_upload_framing_and_limits(self):
        with serving(server.Handler) as port, patch.object(server, "score") as score:
            for framing, body, status in (
                (b"", b"", 400),
                (b"Content-Length: nope\r\n", b"", 400),
                (b"Content-Length: 1\r\nContent-Length: 2\r\n", b"", 400),
                (b"Transfer-Encoding: chunked\r\nContent-Length: 1\r\n", b"", 400),
                (b"Content-Length: 0\r\n", b"", 413),
                (b"Content-Length: 8388609\r\n", b"", 413),
                (b"Content-Length: " + b"9" * 5000 + b"\r\n", b"", 413),
                (b"Content-Length: 4\r\n", b"a", 400),
            ):
                with self.subTest(status=status, framing=framing[:45]):
                    with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
                        sock.sendall(b"POST /predict HTTP/1.1\r\nHost: localhost\r\n"
                                     b"Content-Type: image/jpeg\r\n" + framing + b"\r\n" + body)
                        sock.shutdown(socket.SHUT_WR)
                        response = http.client.HTTPResponse(sock)
                        response.begin()
                        self.assertEqual(response.status, status)
                        response.close()
            self.assertEqual(request(port, **{"Content-Type": "text/plain"})[0], 415)
            score.assert_not_called()

    def test_upload_timeout_and_next_request(self):
        with serving(server.Handler) as port, patch.object(server, "UPLOAD_TIMEOUT_SECONDS", 0.05):
            with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
                sock.sendall(b"POST /predict HTTP/1.1\r\nHost: localhost\r\n"
                             b"Content-Type: image/jpeg\r\nContent-Length: 2\r\n\r\na")
                response = http.client.HTTPResponse(sock)
                response.begin()
                self.assertEqual(response.status, 408)
                response.close()
            with patch.object(server, "score", return_value=PREDICTION):
                self.assertEqual(request(port, **{"Content-Type": "image/jpeg"}), (200, PREDICTION))

    def test_concurrent_requests_keep_predictions_separate(self):
        import base64

        barrier = threading.Barrier(4, timeout=2)

        def predict(body):
            raw = base64.b64decode(json.loads(body)["image_base64"])
            barrier.wait()
            return dict(PREDICTION, confidence=int(raw) / 10)

        def upload(index):
            return request(port, str(index).encode(), **{"Content-Type": "image/jpeg"})

        with serving(server.Handler) as port, patch.object(server, "score", side_effect=predict):
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                results = list(pool.map(upload, range(4)))
        self.assertEqual(results, [(200, dict(PREDICTION, confidence=i / 10)) for i in range(4)])

    def test_maps_scorer_status(self):
        with serving(server.Handler) as port:
            for status in (400, 502, 504):
                with patch.object(server, "score", side_effect=client.ScoringError("safe", status=status)):
                    self.assertEqual(request(port, **{"Content-Type": "image/png"}), (status, {"error": "safe"}))


class ScoringReleaseTests(unittest.TestCase):
    def test_network_failures_and_explicit_retry(self):
        timed_out = threading.Event()
        release_timeout = threading.Event()
        calls = []

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                command = self.rfile.read(int(self.headers["Content-Length"]))
                calls.append(command)
                if command == b"disconnect":
                    return
                if command == b"timeout":
                    timed_out.set()
                    release_timeout.wait(2)
                    return
                if command == b"bad-image":
                    self.send_error(400)
                    return
                if command in (b"azure-invalid-image", b"azure-model-failure"):
                    self.send_response(424)
                    self.send_header("ms-azureml-model-error-statuscode",
                                     "400" if command == b"azure-invalid-image" else "500")
                    self.end_headers()
                    return
                if command == b"azure-timeout":
                    self.send_error(408)
                    return
                if command == b"redirect":
                    self.send_response(307)
                    self.send_header("Location", "/must-not-follow")
                    self.end_headers()
                    return
                payload = {
                    b"oversized": b" " * (64 * 1024) + b"{}",
                    b"malformed": b"[",
                    b"deep": b"[" * 2000,
                }.get(command, json.dumps(json.dumps(PREDICTION)).encode())
                self.send_response(200)
                extra = 10 if command == b"truncated" else 0
                self.send_header("Content-Length", str(len(payload) + extra))
                self.end_headers()
                self.wfile.write(payload)

        with serving(Upstream) as port, patch.dict(os.environ, {
            "EKYC_SCORING_URL": f"http://127.0.0.1:{port}/score",
            "EKYC_SCORING_KEY": "", "EKYC_SCORING_DEPLOYMENT": "",
        }):
            try:
                for command, status in ((b"disconnect", 502), (b"truncated", 502),
                                        (b"oversized", 502), (b"malformed", 502), (b"deep", 502),
                                        (b"redirect", 502), (b"bad-image", 400),
                                        (b"azure-invalid-image", 400), (b"azure-model-failure", 502),
                                        (b"azure-timeout", 504), (b"timeout", 504)):
                    with self.subTest(command=command):
                        with self.assertRaises(client.ScoringError) as raised:
                            client.score(command, timeout=0.2)
                        self.assertEqual(raised.exception.status, status)
                self.assertTrue(timed_out.is_set())
                self.assertEqual(client.score(b"retry"), PREDICTION)
            finally:
                release_timeout.set()
        self.assertEqual(calls, [b"disconnect", b"truncated", b"oversized", b"malformed",
                                 b"deep", b"redirect", b"bad-image", b"azure-invalid-image",
                                 b"azure-model-failure", b"azure-timeout", b"timeout", b"retry"])

    def test_url_validation_and_image_cap(self):
        for url in ("", "http://127.0.0.1:invalid/score", "http://example.org",
                    "https://user:password@example.org"):
            with self.subTest(url=url), patch.dict(os.environ, EKYC_SCORING_URL=url):
                with self.assertRaises(client.ScoringError):
                    client.score(b"body")
        for size in (0, client.MAX_IMAGE_BYTES + 1):
            with self.assertRaises(client.ScoringError):
                client.encode_image(bytes(size))
        self.assertIn(b"image_base64", client.encode_image(bytes(client.MAX_IMAGE_BYTES)))
