"""Send allowlisted mock images directly to a Label Studio review project."""

import json
from threading import Lock
from urllib.error import HTTPError
from urllib.parse import quote

from demo.review_queue import load_allowlist
from labeling.studio import Studio


class LiveReview:
    def __init__(self, base, public_base, project, root, credentials, audit_manifest=None):
        self.base = base.rstrip("/")
        self.public_base = public_base.rstrip("/")
        self.project = project
        self.root = root.resolve(strict=True)
        self.credentials = credentials
        self.audit = load_allowlist(audit_manifest, root) if audit_manifest else {}
        self.session = None
        self.lock = Lock()

    def request(self, path, body=None):
        if self.session is None:
            email, password = self.credentials
            session = Studio(self.base)
            session.request("/user/login/")
            session.request("/user/login/", {"email": email, "password": password}, form=True)
            self.session = session
        try:
            _, raw = self.session.request(path, body)
        except HTTPError as exc:
            if exc.code in (401, 403):
                self.session = None
            raise
        return json.loads(raw)

    def send(self, event_id, training_sources):
        source = self.audit.get(event_id) or training_sources.get(event_id)
        if source is None:
            return None
        media = quote(source.path.relative_to(self.root).as_posix())
        data = {"event_id": event_id, "source_image_id": source.image_id,
                "review_only": event_id in self.audit, "image": f"/data/local-files/?d={media}"}
        with self.lock:
            page = 1
            while True:
                result = self.request(f"/api/tasks/?project={self.project}&page_size=100&page={page}")
                for task in result["tasks"]:
                    if task["data"].get("event_id") == event_id:
                        if task["data"] != data:
                            raise ValueError("The Label Studio task does not match the reviewed source")
                        return self.url(int(task["id"]))
                if page * 100 >= result["total"]:
                    break
                page += 1
            result = self.request(f"/api/projects/{self.project}/import?return_task_ids=true", [{"data": data}])
            task_id, = result["task_ids"]
            return self.url(int(task_id))

    def url(self, task_id):
        return f"{self.public_base}/projects/{self.project}/data?task={task_id}"
