from contextlib import redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from azure import benchmark, stage


ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)


class AzureReleaseTests(unittest.TestCase):
    def test_clean_entry_point_help(self):
        for command, option in (
            (("azure/stage.py", "--help"), "--training-only"),
            (("-m", "azure.benchmark", "--help"), "--client-location"),
            (("-m", "demo.server", "--help"), "--port"),
        ):
            with self.subTest(command=command):
                result = run(*command)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(option, result.stdout)

    def test_stage_clean_cli_and_repeated_destination(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "full"
            command = ("azure/stage.py", str(destination))
            first = run(*command)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertTrue((destination / "src/modeling/score.py").is_file())
            self.assertEqual(len(list(destination.glob("*.yml"))), 7)
            self.assertTrue(all(path.suffix in {".py", ".yml"} for path in destination.rglob("*")
                                if path.is_file()))
            snapshot = (destination / "deployment.yml").read_bytes()
            self.assertEqual(run(*command).returncode, 2)
            self.assertEqual((destination / "deployment.yml").read_bytes(), snapshot)
            training = Path(directory) / "training"
            self.assertEqual(run("azure/stage.py", "--training-only", str(training)).returncode, 0)
            self.assertFalse((training / "deployment.yml").exists())

    def test_failed_copy_rolls_back_and_can_retry(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "stage"
            with patch.object(sys, "argv", ["stage", str(destination)]):
                with patch.object(stage.shutil, "copyfile", side_effect=OSError("disk full")):
                    with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
                        stage.main()
                    self.assertEqual(raised.exception.code, 2)
                self.assertFalse(destination.exists())
                stage.main()
                self.assertTrue((destination / "train.yml").exists())

    def test_filesystem_cli_failure_then_retry_and_cleanup(self):
        base = ("-m", "azure.benchmark", "--runs", "2", "--client-location", "local",
                "--target-location", "local", "filesystem", "--payload-mib", "1", "--directory")
        with TemporaryDirectory() as directory:
            target = Path(directory) / "storage"
            failure = run(*base, str(target))
            self.assertEqual(failure.returncode, 1)
            self.assertIn("FileNotFoundError", failure.stderr)
            self.assertEqual(failure.stdout, "")
            target.mkdir()
            success = run(*base, str(target))
            self.assertEqual(success.returncode, 0, success.stderr)
            for metric in json.loads(success.stdout)["results"].values():
                self.assertEqual(metric["successes"], 2)
            self.assertEqual(list(target.iterdir()), [])

    def test_scoring_cli_reports_independent_attempts_on_connection_failure(self):
        import socket

        with TemporaryDirectory() as directory, socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            sample = Path(directory) / "image"
            sample.write_bytes(b"test-image")
            url = f"http://127.0.0.1:{listener.getsockname()[1]}/score"
            with patch.dict(os.environ, EKYC_SCORING_URL=url):
                failure = run("-m", "azure.benchmark", "--runs", "2", "--client-location", "local",
                              "--target-location", "local", "scoring", "--warmup", "0",
                              "--timeout", "1", str(sample))
            self.assertEqual(failure.returncode, 1, failure.stderr)
            result = json.loads(failure.stdout)["results"]["scoring"]
            self.assertEqual(result["attempts"], 2)
            self.assertEqual(result["successes"], 0)
            self.assertEqual(sum(result["failures"].values()), 2)

    def test_blob_failures_disable_retries_and_clean_up_owned_object(self):
        class ResourceExistsError(Exception):
            pass

        class ResourceNotFoundError(Exception):
            pass

        for failure in (None, ResourceExistsError("collision"), TimeoutError("signed URL")):
            with self.subTest(failure=failure):
                identity = ModuleType("azure.identity")
                identity.DefaultAzureCredential = Mock(return_value=TemporaryDirectory())
                storage = ModuleType("azure.storage.blob")
                service = Mock()
                storage.BlobServiceClient = Mock(return_value=service)
                service.__enter__ = Mock(return_value=service)
                service.__exit__ = Mock(return_value=False)
                blob = service.get_blob_client.return_value
                blob.upload_blob.side_effect = [failure, TimeoutError("signed URL"), None]
                blob.download_blob.return_value.readall.side_effect = [b"", b"x"]
                if isinstance(failure, TimeoutError):
                    blob.delete_blob.side_effect = ResourceNotFoundError()
                exceptions = ModuleType("azure.core.exceptions")
                exceptions.ResourceExistsError = ResourceExistsError
                exceptions.ResourceNotFoundError = ResourceNotFoundError
                args = SimpleNamespace(mode="blob", account_url="https://example.invalid",
                                       container="release", runs=2)
                with patch.dict(sys.modules, {"azure.identity": identity, "azure.storage.blob": storage,
                                              "azure.core.exceptions": exceptions}):
                    if failure is None:
                        results = benchmark.transfer(args, b"x")
                        self.assertEqual(results["write"]["failures"], {"TimeoutError": 1})
                        self.assertEqual(results["read"]["failures"], {"OSError": 1})
                        self.assertEqual(results["write"]["successes"], 1)
                        self.assertEqual(results["read"]["successes"], 1)
                    else:
                        with self.assertRaises(type(failure)):
                            benchmark.transfer(args, b"x")
                self.assertEqual(storage.BlobServiceClient.call_args.kwargs["retry_total"], 0)
                self.assertEqual(blob.delete_blob.call_count, int(not isinstance(failure, ResourceExistsError)))


class AzureSchemaReleaseTests(unittest.TestCase):
    @unittest.skipUnless(os.getenv("EKYC_CHECK_AZURE_SCHEMAS"), "opt-in schema download")
    def test_definitions_match_published_schemas(self):
        import urllib.request

        import jsonschema
        import yaml

        for path in sorted((ROOT / "azure").glob("*.yml")):
            with self.subTest(path=path.name):
                spec = yaml.safe_load(path.read_text())
                if "$schema" in spec:
                    with urllib.request.urlopen(spec["$schema"], timeout=20) as response:
                        schema = json.load(response)
                    jsonschema.validate(spec, schema)
                else:
                    self.assertTrue(spec["dependencies"])
                print(f"{path.name}: validated")


class LocalRuntimeReleaseTests(unittest.TestCase):
    @staticmethod
    def select_inference_port():
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            return listener.getsockname()[1]

    def start_inference(self, executable, model_dir, code, env, log):
        import errno
        from time import monotonic, sleep
        from urllib.error import URLError
        from urllib.request import urlopen

        # azmlinfsrv rejects port 0 and passes an explicit Gunicorn --bind.
        # Retry only when that bind collides after releasing the probe socket.
        for attempt in range(3):
            port = self.select_inference_port()
            base = f"http://127.0.0.1:{port}"
            with log.open("w") as output:
                process = subprocess.Popen(
                    [str(executable), "--entry_script", "score.py", "--model_dir", str(model_dir),
                     "--port", str(port)], env=env, cwd=code,
                    stdout=output, stderr=subprocess.STDOUT,
                )
            ready = False
            try:
                deadline = monotonic() + 20
                while process.poll() is None and monotonic() < deadline:
                    if f"Listening at: http://0.0.0.0:{port} " in log.read_text():
                        try:
                            with urlopen(base, timeout=1) as response:
                                self.assertEqual(response.status, 200)
                            ready = True
                            print(f"Actual inference listener: {base}")
                            return process, base
                        except (URLError, TimeoutError):
                            pass
                    sleep(0.1)
                details = log.read_text()
                collision = (
                    process.poll() is not None
                    and f"connection to ('0.0.0.0', {port}) failed: [Errno {errno.EADDRINUSE}]" in details
                )
                if not collision or attempt == 2:
                    self.fail(details)
                print(f"Gunicorn port {port} was claimed before bind. Retrying after EADDRINUSE.")
            finally:
                if not ready:
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=10)

    @unittest.skipUnless(os.getenv("EKYC_RELEASE_DATASET"), "opt-in MIDV-2020 runtime")
    def test_staged_training_and_inference_runtime(self):
        from concurrent.futures import ThreadPoolExecutor
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        from demo import client, server
        from demo.test_release import request, serving

        dataset = Path(os.environ["EKYC_RELEASE_DATASET"]).resolve()
        sample = sorted(dataset.glob("images/**/*.jpg"))[0]
        executable = Path(sys.executable).with_name("azmlinfsrv")
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MLFLOW_DISABLE_AGENT_HINT="1")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "aml"
            staged = run("azure/stage.py", str(source))
            self.assertEqual(staged.returncode, 0, staged.stderr)
            bundle = root / "models" / "ekyc-document-classifier" / "1"
            training = subprocess.run(
                [sys.executable, "-m", "modeling.train", "--dataset-root", str(dataset),
                 "--manifest", str(dataset / "manifest.csv"), "--output-dir", str(bundle),
                 "--epochs", "1", "--batch-size", "4", "--image-size", "64", "--device", "cpu",
                 "--tracking-uri", (root / "mlruns").as_uri()],
                cwd=source / "src", env=env, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(training.returncode, 0, training.stderr)
            print(training.stdout.strip())
            select_port = self.select_inference_port
            with socket.socket() as occupied:
                occupied.bind(("0.0.0.0", 0))
                occupied.listen()
                occupied_port = occupied.getsockname()[1]
                attempts = 0

                def collide_once():
                    nonlocal attempts
                    attempts += 1
                    return occupied_port if attempts == 1 else select_port()

                with patch.object(self, "select_inference_port", side_effect=collide_once):
                    process, base = self.start_inference(
                        executable, root / "models", source / "src" / "modeling", env, root / "inference.log",
                    )
            with process:
                try:
                    self.assertGreaterEqual(attempts, 2)
                    for body in (b"{", b'{"image_base64":"!!!"}', client.encode_image(b"not a JPEG"),
                                 json.dumps({"image_base64": "A" * (4 * ((client.MAX_IMAGE_BYTES + 2) // 3) + 4)}).encode()):
                        with self.assertRaises(HTTPError) as raised:
                            urlopen(Request(base + "/score", body, {"Content-Type": "application/json"}), timeout=5)
                        with raised.exception as error:
                            self.assertEqual(error.code, 400, error.read())
                    with serving(server.Handler) as demo_port, patch.dict(os.environ, {
                        "EKYC_SCORING_URL": base + "/score",
                        "EKYC_SCORING_KEY": "",
                        "EKYC_SCORING_DEPLOYMENT": "",
                    }):
                        with urlopen(f"http://127.0.0.1:{demo_port}/", timeout=2) as response:
                            self.assertIn(b"/predict", response.read())
                        image = sample.read_bytes()

                        def upload(_):
                            return request(demo_port, image, **{"Content-Type": "image/jpeg"})

                        with ThreadPoolExecutor(4) as pool:
                            results = list(pool.map(upload, range(8)))
                        self.assertTrue(all(result == results[0] for result in results))
                        status, prediction = results[0]
                        self.assertEqual(status, 200)
                        self.assertIsNone(prediction["status"])
                        self.assertTrue(prediction["needs_review"])
                        invalid_status, _ = request(demo_port, b"invalid", **{"Content-Type": "image/jpeg"})
                        self.assertEqual(invalid_status, 400)
                        benchmark_run = run("-m", "azure.benchmark", "--runs", "3", "--client-location",
                                            "local", "--target-location", "local-inference-server", "scoring",
                                            "--timeout", "5", str(sample))
                        self.assertEqual(benchmark_run.returncode, 0, benchmark_run.stderr)
                        self.assertEqual(json.loads(benchmark_run.stdout)["results"]["scoring"]["successes"], 3)
                        print(benchmark_run.stdout.strip())
                    print("Actual inference HTTP status: 4 invalid scores=400, 8 demo uploads=200, invalid demo upload=400")
                finally:
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=10)
