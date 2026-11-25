"""Submit a versioned training run after an operator approves its inputs."""

from hashlib import sha256
import os
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from threading import Lock

from azure.stage import AZURE, modeling_sources, stage
from demo.operations import azure_command


LOCK = Lock()
ASSET = re.compile(r"azureml:[A-Za-z0-9_-]+:[A-Za-z0-9_.-]+")


def inputs():
    values = {name: os.getenv(f"EKYC_AZURE_{name.upper()}", "")
              for name in ("dataset", "manifest", "compute")}
    if (azure_command() is None or not all(ASSET.fullmatch(values[key]) for key in ("dataset", "manifest"))
            or not re.fullmatch(r"[A-Za-z0-9_-]+", values["compute"])):
        return None
    return values


def revision(config):
    digest = sha256()
    for key, value in sorted(config.items()):
        digest.update(f"{key}={value}\0".encode())
    for source, relative in [(AZURE / "train.yml", Path("train.yml")), *modeling_sources()]:
        digest.update(relative.as_posix().encode() + b"\0")
        digest.update(sha256(source.read_bytes()).digest())
    return digest.hexdigest()


def submit(config):
    command, scope = azure_command()
    name = f"ekyc-reviewed-{revision(config)[:20]}"
    with LOCK:
        query = [*command, "show", *scope, "--name", name, "--query", "name", "-o", "tsv"]
        previous = subprocess.run(query, capture_output=True, text=True, timeout=20)
        if previous.returncode == 0 and previous.stdout.strip() == name:
            return {"name": name, "existing": True}
        if previous.returncode == 0 or not re.search(r"not[ _-]?found|does not exist", previous.stderr, re.I):
            raise RuntimeError("Không xác minh được job hiện có trên Azure.")
        with TemporaryDirectory(prefix="ekyc-azure-job-") as directory:
            spec = Path(directory)
            stage(spec, training_only=True)
            create = [*command, "create", *scope, "--file", str(spec / "train.yml"), "--name", name,
                      "--set", f"compute=azureml:{config['compute']}",
                      f"inputs.dataset.path={config['dataset']}", f"inputs.manifest.path={config['manifest']}",
                      "inputs.epochs=3", "--query", "name", "-o", "tsv"]
            try:
                result = subprocess.run(create, capture_output=True, text=True, timeout=150)
                if result.returncode == 0 and result.stdout.strip() == name:
                    return {"name": name, "existing": False}
            except subprocess.TimeoutExpired:
                pass
            current = subprocess.run(query, capture_output=True, text=True, timeout=20)
            if current.returncode == 0 and current.stdout.strip() == name:
                return {"name": name, "existing": True}
            raise RuntimeError("Chưa xác nhận được job. Kiểm tra Azure hoặc thử lại với cùng manifest.")
