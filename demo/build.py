"""Run the CI checks locally and publish their status and log to the console."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


def build_root():
    return Path(os.getenv("EKYC_BUILD_DIR", "runs/builds"))


def main():
    folder = build_root() / uuid4().hex
    folder.mkdir(parents=True, mode=0o700)
    metadata = {
        "id": folder.name, "title": "Kiểm tra CI cục bộ", "status": "running", "active": True,
        "inline_log": True, "date": datetime.now(timezone.utc).isoformat(), "detail": "compileall",
    }

    def save():
        temporary = folder / "status.tmp"
        temporary.write_text(json.dumps(metadata), encoding="utf-8")
        temporary.replace(folder / "status.json")

    with (folder / "build.log").open("wb", buffering=0) as log:
        save()
        code = 1
        try:
            arguments = ["-m", "compileall", "-q", "azure", "data_pipeline", "demo", "labeling", "modeling"]
            log.write(("$ python " + " ".join(arguments) + "\n").encode())
            code = subprocess.run([sys.executable, *arguments], stdout=log, stderr=subprocess.STDOUT).returncode
        except KeyboardInterrupt:
            code = 130
        finally:
            metadata.update(status="success" if code == 0 else "cancelled" if code == 130 else "failure", active=False)
            save()
    print(folder)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
