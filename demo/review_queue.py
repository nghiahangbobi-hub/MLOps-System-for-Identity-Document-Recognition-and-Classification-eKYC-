"""Private, content-addressed queue for explicitly allowlisted mock documents."""

import base64
import csv
from dataclasses import dataclass
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
from tempfile import NamedTemporaryFile, mkdtemp

from demo.client import MAX_IMAGE_BYTES, RESULT_FIELDS


@dataclass(frozen=True)
class Source:
    image_id: str
    path: Path
    image_file: str


def read_image(path):
    with path.open("rb") as stream:
        data = stream.read(MAX_IMAGE_BYTES + 1)
    if not 0 < len(data) <= MAX_IMAGE_BYTES:
        raise ValueError("Allowlisted images must be between 1 byte and 8 MiB.")
    return data


def load_allowlist(manifest, dataset_root):
    root = dataset_root.resolve(strict=True)
    allowlist = {}
    image_ids = {}
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image_id", "image_path"} <= set(reader.fieldnames or ()):
            raise ValueError("The mock manifest must contain image_id and image_path.")
        for row in reader:
            image_id = row.get("image_id")
            relative = Path(row.get("image_path") or "")
            path = (root / relative).resolve(strict=True)
            if (not image_id or relative.is_absolute() or ".." in relative.parts
                    or not path.is_relative_to(root) or not path.is_file()):
                raise ValueError(f"Invalid mock manifest entry on line {reader.line_num}.")
            data = read_image(path)
            if data.startswith(b"\xff\xd8\xff") and relative.suffix.lower() in (".jpeg", ".jpg"):
                image_file = "image.jpg"
            elif data.startswith(b"\x89PNG\r\n\x1a\n") and relative.suffix.lower() == ".png":
                image_file = "image.png"
            else:
                raise ValueError(f"Expected a JPEG or PNG on mock manifest line {reader.line_num}.")
            digest = hashlib.sha256(data).hexdigest()
            source = Source(image_id, path, image_file)
            if digest in allowlist and allowlist[digest].image_id != image_id:
                raise ValueError(f"Ambiguous image digest on mock manifest line {reader.line_num}.")
            if image_id in image_ids and image_ids[image_id] != digest:
                raise ValueError(f"Conflicting image ID on mock manifest line {reader.line_num}.")
            allowlist[digest] = source
            image_ids[image_id] = digest
    if not allowlist:
        raise ValueError("The mock manifest contains no images.")
    return allowlist


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_private(directory):
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError:
        pass
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Queue directories must be owner-only directories.")


def write_event(stream, event):
    stream.write(json.dumps(event, ensure_ascii=True, allow_nan=False).encode() + b"\n")
    stream.flush()
    os.fsync(stream.fileno())


class ReviewQueue:
    def __init__(self, root, manifest, dataset_root):
        self.allowlist = load_allowlist(manifest, dataset_root)
        self.root = root.absolute()
        self.events = self.root / "events"
        ensure_private(self.root)
        ensure_private(self.events)
        if self.root.stat().st_dev != self.events.stat().st_dev:
            raise ValueError("Queue staging and events must share one filesystem.")
        sync_directory(self.root)
        sync_directory(self.root.parent)
        self.receipt_key = secrets.token_bytes(32)

    def record_prediction(self, image, prediction):
        digest = hashlib.sha256(image).hexdigest()
        if digest not in self.allowlist:
            return "not_allowlisted", None
        if prediction["needs_review"]:
            self.enqueue(digest, prediction, "needs_review", image=image)
            return "queued", self.receipt(digest, prediction)
        return "available", self.receipt(digest, prediction)

    def receipt(self, digest, prediction):
        body = json.dumps([digest, prediction], allow_nan=False).encode()
        signature = hmac.digest(self.receipt_key, body, "sha256")
        return base64.urlsafe_b64encode(signature + body).decode("ascii")

    def open_receipt(self, token):
        try:
            signed = base64.b64decode(token, altchars=b"-_", validate=True)
        except (ValueError, TypeError):
            return None
        signature, body = signed[:32], signed[32:]
        if not hmac.compare_digest(signature, hmac.digest(self.receipt_key, body, "sha256")):
            return None
        digest, prediction = json.loads(body)
        return digest, prediction

    def enqueue(self, digest, prediction, reason, *, image=None):
        source = self.allowlist[digest]
        final = self.events / digest
        fd = os.open(self.root / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if final.is_symlink():
                raise OSError("Invalid review event directory")
            if final.exists():
                self.merge_reason(final, source, digest, reason)
            else:
                self.publish(final, source, digest, prediction, reason, image)
            sync_directory(final)
            sync_directory(self.events)
            sync_directory(self.root)

    def publish(self, final, source, digest, prediction, reason, image):
        if image is None:
            try:
                image = read_image(source.path)
            except ValueError as exc:
                raise OSError("Allowlist image changed") from exc
        if hashlib.sha256(image).hexdigest() != digest:
            raise OSError("Allowlist image changed")
        event = {
            "event_id": digest,
            "image_sha256": digest,
            "image_file": source.image_file,
            "source_image_id": source.image_id,
            "prediction": {field: prediction[field] for field in RESULT_FIELDS},
            "reasons": [reason],
        }
        draft = Path(mkdtemp(prefix=".event-", dir=self.root))
        try:
            fd = os.open(draft / source.image_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(image)
                stream.flush()
                os.fsync(stream.fileno())
            fd = os.open(draft / "event.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as stream:
                write_event(stream, event)
            sync_directory(draft)
            draft.rename(final)
        finally:
            if draft.exists():
                shutil.rmtree(draft)

    def merge_reason(self, final, source, digest, reason):
        path = final / "event.json"
        image = final / source.image_file
        if path.is_symlink() or image.is_symlink() or not image.is_file():
            raise OSError("Invalid persisted review event")
        try:
            event = json.loads(path.read_bytes())
            if (event["image_sha256"], event["event_id"], event["source_image_id"], event["image_file"]) != (
                digest, digest, source.image_id, source.image_file,
            ):
                raise ValueError("Queue event does not match its allowlist entry")
            if hashlib.sha256(read_image(image)).hexdigest() != digest:
                raise ValueError("Queue image does not match its event digest")
            reasons = event["reasons"]
            if not isinstance(reasons, list):
                raise ValueError("Queue event reasons are invalid")
            if reason in reasons:
                return
            reasons.append(reason)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise OSError("Invalid persisted review event") from exc
        with NamedTemporaryFile(dir=final, prefix=".event-", delete=False) as stream:
            draft = Path(stream.name)
            try:
                write_event(stream, event)
                os.replace(draft, path)
            finally:
                draft.unlink(missing_ok=True)
