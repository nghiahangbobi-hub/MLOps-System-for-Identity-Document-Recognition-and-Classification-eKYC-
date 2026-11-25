"""Read-only integrations for the local operations console."""

import codecs
import json
import os
import re
import selectors
import subprocess
from threading import BoundedSemaphore
from time import monotonic
from urllib.request import Request, urlopen

from demo.build import build_root

LOG_SLOTS = BoundedSemaphore(3)
ACTIVE_AZURE = {
    "notstarted", "starting", "provisioning", "preparing", "queued",
    "running", "finalizing", "cancelrequested", "canceling", "cancelling", "paused",
}


def azure_command():
    group = os.getenv("AZ_RESOURCE_GROUP")
    workspace = os.getenv("AZ_WORKSPACE")
    if not group or not workspace:
        return None
    command = [os.getenv("EKYC_AZ_CLI", "az"), "ml", "job"]
    scope = ["--resource-group", group, "--workspace-name", workspace, "--only-show-errors"]
    if subscription := os.getenv("AZ_SUBSCRIPTION"):
        scope.extend(["--subscription", subscription])
    return command, scope


def azure_jobs():
    config = azure_command()
    if config is None:
        return None
    command, scope = config
    response = subprocess.run(
        [*command, "list", *scope, "--include-archived", "--max-results", "50", "-o", "json"],
        capture_output=True, check=True, timeout=18,
    )
    jobs = []
    for job in json.loads(response.stdout):
        name = job["name"]
        status = job.get("status") or "Unknown"
        jobs.append({
            "id": name, "title": job.get("display_name") or name,
            "status": status, "active": status.lower() in ACTIVE_AZURE,
            "detail": job.get("experiment_name") or "",
            "date": (job.get("creation_context") or {}).get("created_at"),
            "url": ((job.get("services") or {}).get("Studio") or {}).get("endpoint")
                   or job.get("studio_url"),
            "inline_log": True,
        })
    return sorted(jobs, key=lambda job: job["date"] or "", reverse=True)


def github_repo():
    repo = os.getenv("EKYC_GITHUB_REPO")
    if repo and not re.fullmatch(r"[a-zA-Z0-9-]+/[a-zA-Z0-9_.-]+", repo):
        raise ValueError("Invalid GitHub repository")
    return repo


def github_runs():
    repo = github_repo()
    if not repo:
        return None
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "ekyc-monitoring"}
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(f"https://api.github.com/repos/{repo}/actions/runs?per_page=30", headers=headers)
    with urlopen(request, timeout=10) as response:
        runs = json.load(response)["workflow_runs"]
    return [{
        "id": str(run["id"]), "title": run.get("name") or run.get("display_title"),
        "status": run["conclusion"] or run["status"],
        "active": run["status"] != "completed", "detail": run.get("head_branch"),
        "date": run.get("created_at"), "url": run["html_url"], "inline_log": False,
    } for run in runs]


def local_builds():
    return local_runs(build_root())


def local_runs(root):
    runs = [json.loads(path.read_text(encoding="utf-8"))
            for path in root.glob("*/status.json") if re.fullmatch("[a-f0-9]{32}", path.parent.name)]
    return sorted(runs, key=lambda run: run["date"], reverse=True)[:30]


def local_build_log(run):
    return local_run_log(build_root(), run, "build.log")


def local_run_log(root, run, filename):
    if not re.fullmatch("[a-f0-9]{32}", run):
        raise ValueError("Invalid run ID")
    folder = root / run
    metadata = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    with (folder / filename).open("rb") as log:
        size = log.seek(0, os.SEEK_END)
        log.seek(max(0, size - 256 * 1024))
        content = log.read(256 * 1024).decode("utf-8", errors="replace")
    return {"content": content, "active": metadata["active"], "status": metadata["status"]}


def azure_log_command(job):
    config = azure_command()
    if config is None or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,254}", job):
        raise ValueError("Invalid Azure job or workspace configuration")
    command, scope = config
    return [*command, "stream", "--name", job, *scope]


def stream_logs(process, emit, seconds=120, byte_limit=256 * 1024):
    """Read bounded chunks and emit heartbeats to detect browser disconnects."""
    deadline = monotonic() + seconds
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        while monotonic() < deadline and byte_limit > 0:
            if not selector.select(timeout=min(5, max(0, deadline - monotonic()))):
                emit("heartbeat", None)
                continue
            chunk = os.read(process.stdout.fileno(), min(4096, byte_limit))
            if not chunk:
                if tail := decoder.decode(b"", final=True):
                    emit("line", tail)
                code = process.wait(timeout=2)
                return "Đã kết thúc luồng log." if not code else f"Azure CLI kết thúc với mã {code}."
            byte_limit -= len(chunk)
            if content := decoder.decode(chunk):
                emit("line", content)
    return "Đã đạt giới hạn xem log. Mở lại luồng hoặc xem toàn bộ log trên Azure ML Studio."
