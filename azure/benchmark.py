"""Measure serial scoring latency or filesystem/Blob transfer time as JSON."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4
from urllib.parse import urlsplit

from demo.client import MAX_IMAGE_BYTES, ScoringError, encode_image, score

SCORING_SLA_MS = 3000


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def measure(operation, runs):
    durations = []
    failures = {}
    for _ in range(runs):
        start = perf_counter()
        try:
            operation()
        except Exception as exc:
            # SDK exceptions can contain signed URLs or other request details.
            error = str(exc) if isinstance(exc, ScoringError) else type(exc).__name__
            failures[error] = failures.get(error, 0) + 1
        else:
            durations.append((perf_counter() - start) * 1000)
    summary = {"attempts": runs, "successes": len(durations), "failures": failures,
               "successful_attempt_ms": durations}
    if durations:
        summary["p50_ms"] = median(durations)
        summary["p95_ms"] = sorted(durations)[(95 * len(durations) + 99) // 100 - 1]
    return summary


def storage_results(read, write, runs, payload_bytes):
    results = {}
    for name, operation in (("write", write), ("read", read)):
        metrics = measure(operation, runs)
        if metrics.get("p50_ms", 0) > 0:
            metrics["median_mib_per_second"] = payload_bytes / (1024**2) / (metrics["p50_ms"] / 1000)
        results[name] = metrics
    return results


def verify_size(size, expected):
    if size != expected:
        raise OSError("Incomplete transfer")


def transfer(args, payload):
    if args.mode == "filesystem":
        with TemporaryDirectory(dir=args.directory, prefix="ekyc-bench-") as directory:
            path = Path(directory) / "payload"
            verify_size(path.write_bytes(payload), len(payload))
            return storage_results(
                lambda: verify_size(len(path.read_bytes()), len(payload)),
                lambda: verify_size(path.write_bytes(payload), len(payload)),
                args.runs, len(payload),
            )

    from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    with DefaultAzureCredential() as credential:
        with BlobServiceClient(args.account_url, credential, retry_total=0) as service:
            blob = service.get_blob_client(args.container, f"ekyc-bench/{uuid4().hex}")
            cleanup = True
            try:
                try:
                    blob.upload_blob(payload, overwrite=False)
                except ResourceExistsError:
                    cleanup = False
                    raise
                return storage_results(
                    lambda: verify_size(len(blob.download_blob().readall()), len(payload)),
                    lambda: blob.upload_blob(payload, overwrite=True),
                    args.runs, len(payload),
                )
            finally:
                if cleanup:
                    try:
                        blob.delete_blob()
                    except ResourceNotFoundError:
                        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=positive, default=20)
    parser.add_argument("--client-location", required=True)
    parser.add_argument("--target-location", required=True)
    sub = parser.add_subparsers(dest="mode", required=True)
    scoring = sub.add_parser("scoring")
    scoring.add_argument("image", type=Path)
    scoring.add_argument("--warmup", type=int, choices=range(11), default=2)
    scoring.add_argument("--timeout", type=positive, default=30, help="Socket timeout in seconds.")
    for mode in ("filesystem", "blob"):
        storage = sub.add_parser(mode)
        storage.add_argument("--payload-mib", type=positive, default=16)
    sub.choices["filesystem"].add_argument("--directory", type=Path)
    sub.choices["blob"].add_argument("--account-url", required=True)
    sub.choices["blob"].add_argument("--container", required=True)
    args = parser.parse_args()
    report = {
        "mode": args.mode,
        "client_location": args.client_location,
        "target_location": args.target_location,
        "platform": platform.platform(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        if args.mode == "scoring":
            with args.image.open("rb") as stream:
                payload = stream.read(MAX_IMAGE_BYTES + 1)
            body = encode_image(payload)
            operation = lambda: score(body, timeout=args.timeout)
            warmup = measure(operation, args.warmup)
            metrics = measure(operation, args.runs)
            durations = metrics["successful_attempt_ms"]
            metrics.update(sla_ms=SCORING_SLA_MS, max_ms=max(durations, default=None),
                           over_sla=sum(value > SCORING_SLA_MS for value in durations))
            metrics["sla_pass"] = metrics["successes"] == args.runs and not metrics["over_sla"]
            report.update(payload_bytes=len(body), image_sha256=hashlib.sha256(payload).hexdigest(),
                          endpoint_host=urlsplit(os.getenv("EKYC_SCORING_URL", "")).hostname,
                          deployment=os.getenv("EKYC_SCORING_DEPLOYMENT") or None,
                          warmup=warmup, results={"scoring": metrics})
        else:
            payload = b"\x00" * (args.payload_mib * 1024**2)
            report.update(payload_bytes=len(payload), results=transfer(args, payload))
    except Exception as exc:
        parser.exit(1, f"Benchmark setup or cleanup failed ({type(exc).__name__}).\n")
    print(json.dumps(report, indent=2))
    if args.mode == "scoring":
        return int(not report["results"]["scoring"]["sla_pass"])
    return int(any(entry["failures"] for entry in report["results"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
