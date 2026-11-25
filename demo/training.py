"""Run a fresh local comparison and sealed evaluation on the project dataset."""

import fcntl
from datetime import datetime, timezone
from hashlib import file_digest
from importlib.util import find_spec
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from threading import Thread
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[1]
INPUTS = ("manifest.csv", "development.csv", "test.csv")


class TrainingUnavailable(Exception):
    pass


class TrainingBusy(Exception):
    pass


def project_path(*parts):
    path = PROJECT.joinpath(*parts).resolve()
    if not path.is_relative_to(PROJECT):
        raise TrainingUnavailable("Đường dẫn huấn luyện phải nằm trong project.")
    return path


def training_root():
    return project_path("runs", "training")


def save(folder, run, status, detail):
    run.update(status=status, detail=detail, active=status == "running")
    pending = folder / "status.tmp"
    pending.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
    pending.replace(folder / "status.json")


def launch():
    dataset = project_path("Data", "MIDV2020_sample_5GB")
    inputs = [project_path("Data", "splits", name) for name in INPUTS]
    if not dataset.is_dir() or not all(path.is_file() for path in inputs):
        raise TrainingUnavailable("Thiếu dataset hoặc các manifest trong Data/.")
    if not all(find_spec(name) for name in ("torch", "torchvision", "mlflow")):
        raise TrainingUnavailable("Hãy chạy webapp trong môi trường đã cài modeling/requirements.txt.")

    root = training_root()
    root.mkdir(parents=True, exist_ok=True)
    with project_path("runs", "training", ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TrainingBusy("Một lượt huấn luyện đang chạy.") from exc
        # Any active row without a worker holding the lock was interrupted.
        mark_interrupted(root)
        run_id = uuid4().hex
        folder = root / run_id
        folder.mkdir()
        for name, source in zip(INPUTS, inputs):
            shutil.copyfile(source, folder / name)
        temp = folder / "tmp"
        temp.mkdir()
        run = {
            "id": run_id, "title": f"MobileNetV3 · {run_id[:8]}",
            "date": datetime.now(timezone.utc).isoformat(), "inline_log": True,
        }
        cache = project_path("runs", "torch-cache")
        env = dict(os.environ, TMPDIR=str(temp), TORCH_HOME=str(cache),
                   XDG_CACHE_HOME=str(temp), PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPYCACHEPREFIX", None)
        save(folder, run, "running", "Đang khởi động")
        with (folder / "training.log").open("wb") as log:
            try:
                process = subprocess.Popen(
                    [sys.executable, "-u", "-m", "demo.training", run_id, str(lock.fileno())],
                    cwd=PROJECT, env=env, pass_fds=(lock.fileno(),), start_new_session=True,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                )
            except OSError:
                save(folder, run, "failed", "Không khởi động được tiến trình")
                raise
    Thread(target=process.wait, daemon=True).start()
    # The worker inherits the lock. Closing the parent's copy leaves it held.
    return run


def mark_interrupted(root):
    for path in root.glob("*/status.json"):
        if re.fullmatch("[a-f0-9]{32}", path.parent.name):
            run = json.loads(path.read_text(encoding="utf-8"))
            if run["active"]:
                save(path.parent, run, "interrupted", "Tiến trình đã dừng trước khi hoàn tất")


def reconcile():
    """Recover interrupted runs while excluding concurrent launches."""
    root = training_root()
    if not root.is_dir():
        return
    with project_path("runs", "training", ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        mark_interrupted(root)


def train(folder, lock_fd):
    run = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    try:
        with (folder / "test.csv").open("rb") as test:
            digest = file_digest(test, "sha256").hexdigest()
        comparison = folder / "comparison"
        selected = folder / "selected"
        dataset_args = ["--dataset-root", project_path("Data", "MIDV2020_sample_5GB"), "--workers", "0"]
        for detail, args in (
            ("Huấn luyện và chọn trên validation",
             ["compare", *dataset_args, "--development", folder / "development.csv",
              "--held-out-sha256", digest, "--output", comparison]),
            ("Khóa mô hình đã chọn",
             ["seal", "--comparison", comparison, "--output", selected]),
            ("Đánh giá trên test",
             ["evaluate", *dataset_args, "--base-manifest", folder / "manifest.csv",
              "--held-out", folder / "test.csv", "--output", selected]),
        ):
            save(folder, run, "running", detail)
            print(detail, flush=True)
            subprocess.run([sys.executable, "-u", "-m", "modeling.quality_compare", *args],
                           cwd=PROJECT, pass_fds=(lock_fd,), check=True)
        report = json.loads((selected / "held-out.json").read_text(encoding="utf-8"))
        run["metrics"] = {key: report["document"][key] for key in ("accuracy", "macro_f1")}
        run["metrics"]["accepted"] = report["acceptance"] == "pass"
        save(folder, run, "success", "Đã đánh giá trên test")
    except Exception:
        save(folder, run, "failed", "Huấn luyện hoặc đánh giá thất bại. Xem log.")
        raise


if __name__ == "__main__":
    run_id, lock = sys.argv[1:]
    if not re.fullmatch("[a-f0-9]{32}", run_id):
        raise ValueError("Invalid training run ID")
    train(training_root() / run_id, int(lock))
