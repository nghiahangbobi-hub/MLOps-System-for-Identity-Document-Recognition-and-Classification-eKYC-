"""Review a verified HITL batch through the Label Studio API."""

import json
from copy import copy
from hashlib import sha256
from threading import Lock
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import NAMESPACE_URL, uuid5

from labeling.hitl import (
    completed_corrections, corrected_labels, matches_task, read_batch, task_data,
)


class ReviewConflict(ValueError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Review:
    def __init__(self, path, base, project, token, *, personal=False, credentials=None):
        self.batch, _ = read_batch(path)
        self.path = path
        self.base = base.rstrip("/")
        self.project = project
        self.token = token
        self.personal = personal
        self.client = build_opener(NoRedirect())
        self.lock = Lock()
        self.credentials = credentials
        self.session = None
        self.session_lock = Lock()
        self.events = {event["event_id"]: event for event in self.batch["events"]}

    def request(self, path, body=None):
        if self.credentials is not None:
            return self.session_request(path, body)
        def send(route, payload, authorization=None):
            headers = {"Accept": "application/json", "Content-Type": "application/json"}
            if authorization:
                headers["Authorization"] = authorization
            request = Request(self.base + route, json.dumps(payload).encode() if payload is not None else None,
                              headers)
            with self.client.open(request, timeout=10) as response:
                return json.load(response)

        if self.personal:
            access = send("/api/token/refresh", {"refresh": self.token})["access"]
            return send(path, body, f"Bearer {access}")
        return send(path, body, f"Token {self.token}")

    def session_request(self, path, body):
        from labeling.studio import Studio

        with self.session_lock:
            if self.session is None:
                email, password = self.credentials
                studio = Studio(self.base)
                studio.request("/user/login/")
                studio.request("/user/login/", {"email": email, "password": password}, form=True)
                self.session = studio
            try:
                _, raw = self.session.request(path, body)
            except HTTPError as error:
                if error.code in (401, 403):
                    self.session = None
                raise
            return json.loads(raw)

    def for_reviewer(self, email, password):
        from labeling.studio import Studio

        studio = Studio(self.base)
        studio.request("/user/login/")
        studio.request("/user/login/", {"email": email, "password": password}, form=True)
        studio.request("/api/current-user/whoami")
        studio.request(f"/api/projects/{self.project}/")
        review = copy(self)

        def request(path, body=None):
            _, raw = studio.request(path, body)
            return json.loads(raw)

        review.request = request
        return review

    def tasks(self):
        tasks = {}
        page = 1
        while True:
            result = self.request(f"/api/tasks/?project={self.project}&fields=all&page_size=100&page={page}")
            entries = result["tasks"]
            for task in entries:
                event_id = task["data"].get("event_id")
                if event_id not in self.events:
                    continue
                expected = task_data(self.batch["batch_id"], self.events[event_id])
                if (event_id in tasks or task.get("project") != self.project
                        or not matches_task(task["data"], expected)):
                    raise ReviewConflict("Tác vụ Label Studio bị trùng hoặc không khớp lô.")
                tasks[event_id] = task
            if page * 100 >= result["total"]:
                break
            if not entries:
                raise ReviewConflict("Danh sách tác vụ Label Studio không đầy đủ.")
            page += 1
        return tasks

    def list(self):
        tasks = self.tasks()
        return [{**event, "available": event_id in tasks,
                 "annotation": self.annotation(tasks[event_id].get("annotations", [])) if event_id in tasks else None}
                for event_id, event in self.events.items()]

    def summary(self):
        counts = dict(pending=0, reviewed=0, missing=0, conflicts=0)
        for event in self.list():
            annotation = event["annotation"]
            if not event["available"]:
                counts["missing"] += 1
            elif annotation is None:
                counts["pending"] += 1
            elif annotation.get("conflict"):
                counts["conflicts"] += 1
            else:
                counts["reviewed"] += 1
        return counts

    @staticmethod
    def annotation(annotations):
        records = list(completed_corrections(annotations))
        if not records:
            return None
        first, (label, quality) = records[0]
        if any(correction != (label, quality) for _, correction in records):
            return {"conflict": True}
        return {"id": first["id"], "reviewer": first.get("completed_by"), "label": label, "quality": quality}

    def image(self, event_id):
        path = self.path / "images" / self.events[event_id]["image_file"]
        content = path.read_bytes()
        if sha256(content).hexdigest() != event_id:
            raise ReviewConflict("Ảnh không còn khớp lô duyệt.")
        return content, "image/png" if path.suffix == ".png" else "image/jpeg"

    def save(self, event_id, label, quality):
        results = [{"from_name": field, "to_name": "image", "type": "choices",
                    "value": {"choices": [value]}}
                   for field, value in (("document_type", label), ("quality", quality))
                   if field != "quality" or quality is not None]
        corrected_labels(results)
        with self.lock:
            task = self.tasks().get(event_id)
            if task is None:
                raise ReviewConflict("Chưa có tác vụ tương ứng trong Label Studio.")
            route = f"/api/tasks/{int(task['id'])}/annotations/"

            def existing():
                record = self.annotation(self.request(route))
                if record and (record.get("label"), record.get("quality")) != (label, quality):
                    raise ReviewConflict("Tác vụ đã có nhãn khác. Kiểm tra trong Label Studio.")
                return record

            if record := existing():
                return record
            unique_id = str(uuid5(NAMESPACE_URL, f"{self.batch['batch_id']}/{task['id']}/{label}/{quality}"))
            try:
                annotation = self.request(route, {"result": results, "unique_id": unique_id})
            except OSError:
                if record := existing():
                    return record
                raise
            return self.annotation([annotation])
