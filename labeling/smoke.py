"""Exercise MIDV import, media, and export in a disposable Label Studio instance."""

import argparse
from http.cookiejar import CookieJar
import json
from pathlib import Path
import re
from secrets import token_hex, token_urlsafe
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, ProxyHandler, Request, build_opener

from data_pipeline.manifest import read_manifest
from data_pipeline.midv2020 import DEFAULT_SOURCE, build_dataset
from labeling.label_studio import make_tasks


IMAGE = (
    "docker.io/heartexlabs/label-studio@"
    "sha256:844f82c5b08de3f77a57c66f480e0036d3c76c6eff53699e8d47686613faa922"
)


def container_command(runtime, *args, check=True):
    # Runtime diagnostics can include environment and filesystem details.
    result = subprocess.run([runtime, *args], capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(f"{runtime} {args[0]} failed (exit {result.returncode})")
    return result


def choose_runtime():
    if shutil.which("docker") and container_command("docker", "info", check=False).returncode == 0:
        return "docker"
    if shutil.which("podman") and container_command("podman", "info", check=False).returncode == 0:
        return "podman"
    raise RuntimeError("No usable Docker or Podman runtime")


class Studio:
    def __init__(self, base):
        self.base = base
        self.cookies = CookieJar()
        self.client = build_opener(ProxyHandler({}), HTTPCookieProcessor(self.cookies))

    def request(self, path, data=None, *, form=False):
        headers = {"Referer": self.base + path}
        if data is not None:
            csrf = next((cookie.value for cookie in self.cookies if cookie.name == "csrftoken"), None)
            if csrf is None:
                raise RuntimeError("Label Studio did not issue a CSRF cookie")
            headers["X-CSRFToken"] = csrf
            headers["Content-Type"] = "application/x-www-form-urlencoded" if form else "application/json"
            data = (urlencode(data) if form else json.dumps(data)).encode()
        with self.client.open(Request(self.base + path, data=data, headers=headers), timeout=10) as response:
            return response.status, response.read()

    def healthy(self):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                status, body = self.request("/health")
                if status == 200 and json.loads(body) == {"status": "UP"}:
                    print("GET /health HTTP 200 " + body.decode())
                    return
            except (OSError, URLError, ValueError):
                pass
            time.sleep(1)
        raise RuntimeError("Disposable Label Studio health timeout")


def verify(studio, work, tasks, dataset):
    studio.healthy()
    studio.request("/user/signup/")
    studio.request("/user/signup/", {
        "email": "fixture@mlops-ekyc.invalid",
        "password": token_urlsafe(32),
    }, form=True)
    config = Path(__file__).with_name("quality.xml")
    status, body = studio.request("/api/projects/", {
        "title": "Disposable eKYC fixture smoke",
        "label_config": config.read_text(encoding="utf-8"),
    })
    project = json.loads(body)["id"]
    print(f"POST project HTTP {status}")
    status, _ = studio.request("/api/storages/localfiles/", {
        "project": project,
        "title": "Disposable MIDV media",
        "path": "/label-studio/midv/images",
        "use_blob_urls": False,
    })
    print(f"POST local storage HTTP {status}")
    status, body = studio.request(f"/api/projects/{project}/import?return_task_ids=true", tasks)
    imported = json.loads(body)
    task_ids = imported["task_ids"]
    if imported["task_count"] != len(tasks) or len(set(task_ids)) != len(tasks):
        raise RuntimeError("Imported task count did not match the fixture")
    print(f"POST task import HTTP {status} task_count={imported['task_count']}")
    remaining = {task["data"]["image_id"]: task["data"] for task in tasks}
    first_task_image_id = None
    for task_id in task_ids:
        _, body = studio.request(f"/api/tasks/{task_id}/")
        task = json.loads(body)
        data = task["data"]
        if remaining.pop(data["image_id"], None) != data or task["annotations"] or task["predictions"]:
            raise RuntimeError("Imported task differs from the clean fixture")
        status, jpeg = studio.request(data["image"])
        if status != 200 or not jpeg.startswith(b"\xff\xd8") or jpeg != (dataset / data["image_path"]).read_bytes():
            raise RuntimeError("Label Studio served unexpected JPEG bytes")
        if first_task_image_id is None:
            first_task_image_id = data["image_id"]
    if remaining:
        raise RuntimeError("Label Studio did not return every imported task")
    print(f"GET local media HTTP 200 jpeg_byte_matches={len(tasks)}")
    status, _ = studio.request(f"/api/tasks/{task_ids[0]}/annotations/", {
        "result": [{
            "from_name": "quality", "to_name": "image", "type": "choices",
            "value": {"choices": ["Clear"]},
        }],
    })
    print(f"POST fixture annotation HTTP {status} fixture_only=true")
    status, exported = studio.request(f"/api/projects/{project}/export?exportType=JSON")
    print(f"GET raw JSON export HTTP {status} bytes={len(exported)}")
    reviews, output = work / "reviews.json", work / "reviewed.csv"
    reviews.write_bytes(exported)
    subprocess.run([
        sys.executable, "-m", "labeling.label_studio", "apply-reviews",
        "--manifest", str(dataset / "manifest.csv"), "--review-export", str(reviews),
        "--output", str(output),
    ], check=True)
    rows = read_manifest(output)
    labels = {row["image_id"]: row["quality_status"] for row in rows if row["quality_status"]}
    if labels != {first_task_image_id: "Clear"} or len(rows) != len(tasks):
        raise RuntimeError("Ingested manifest does not match the fixture")
    print("PASS raw export ingestion verified; fixture labels are not human evidence")


def run(runtime, source):
    name = f"mlops-ekyc-fixture-{token_hex(6)}"
    print(f"Runtime: {container_command(runtime, '--version').stdout.strip()}")
    print(f"Image: {IMAGE}")
    with TemporaryDirectory(prefix="ekyc-label-studio-smoke-") as temp:
        work = Path(temp)
        dataset = work / "dataset"
        frames = build_dataset(source, dataset, 42, None, 3, 10, max_clips_per_template=2)
        if frames != 54:
            raise RuntimeError(f"Expected the 54-frame slice, found {frames}")
        task_file = work / "tasks.json"
        make_tasks(dataset / "manifest.csv", dataset, dataset, task_file)
        tasks = json.loads(task_file.read_bytes())
        try:
            container_command(
                runtime, "run", "--detach", "--name", name,
                "--publish", "127.0.0.1::8080",
                "--volume", f"{dataset}:/label-studio/midv:ro,Z",
                "--env", "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true",
                "--env", "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/midv",
                IMAGE,
            )
            port = container_command(runtime, "port", name, "8080/tcp").stdout.strip()
            if not re.fullmatch(r"127\.0\.0\.1:\d+", port):
                raise RuntimeError("Disposable container did not bind exclusively to loopback")
            print(f"Disposable loopback: {port}")
            verify(Studio(f"http://{port}"), work, tasks, dataset)
        finally:
            if container_command(runtime, "inspect", name, check=False).returncode == 0:
                container_command(runtime, "rm", "--force", "--volumes", name)
                print("Removed disposable container and anonymous project storage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--runtime", choices=("docker", "podman"))
    args = parser.parse_args()
    try:
        run(args.runtime or choose_runtime(), args.source)
    except HTTPError as error:
        parser.exit(1, f"Label Studio returned HTTP {error.code}\n")
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Smoke failed: {error}\n")


if __name__ == "__main__":
    main()
