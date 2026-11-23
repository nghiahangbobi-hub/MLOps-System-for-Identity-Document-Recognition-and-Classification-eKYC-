"""JSON client shared by the browser demo and latency benchmark."""

import base64
from http.client import HTTPException
import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit

MAX_IMAGE_BYTES = 8 * 1024 * 1024
RESULT_FIELDS = ("label", "status", "confidence", "needs_review")


class ScoringError(Exception):
    """A safe message about a failed scoring request."""

    def __init__(self, message, *, status=502):
        super().__init__(message)
        self.status = status


def encode_image(image):
    if not 0 < len(image) <= MAX_IMAGE_BYTES:
        raise ScoringError("Choose an image between 1 byte and 8 MiB.")
    return json.dumps({"image_base64": base64.b64encode(image).decode("ascii")}).encode()


def score(body, timeout=30):
    url = os.getenv("EKYC_SCORING_URL", "")
    try:
        address = urlsplit(url)
        address.port
    except ValueError:
        raise ScoringError("Configure a valid EKYC_SCORING_URL.") from None
    if address.scheme != "https" and (
        address.scheme != "http" or address.hostname not in {"localhost", "127.0.0.1", "::1"}
    ):
        raise ScoringError("Set EKYC_SCORING_URL to an HTTPS or loopback scoring URL.")
    if not address.hostname or address.username or address.password:
        raise ScoringError("Configure a valid EKYC_SCORING_URL.")
    headers = {"Content-Type": "application/json"}
    key = os.getenv("EKYC_SCORING_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    deployment = os.getenv("EKYC_SCORING_DEPLOYMENT")
    if deployment:
        headers["azureml-model-deployment"] = deployment
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    try:
        request = urllib.request.Request(url, body, headers, method="POST")
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            raw = response.read(64 * 1024 + 1)
            if len(raw) > 64 * 1024:
                raise ValueError("Response too large")
            if response.length:
                raise HTTPException("Incomplete scoring response")
            result = json.loads(raw)
            if isinstance(result, str):
                result = json.loads(result)
        prediction = {field: result[field] for field in RESULT_FIELDS}
        confidence = prediction["confidence"]
        if (
            prediction["label"] not in ("ID_Card", "Passport")
            or prediction["status"] not in (None, "Clear", "Blur", "Glare", "Occluded")
            or type(confidence) not in (int, float)
            or not 0 <= confidence <= 1
            or not isinstance(prediction["needs_review"], bool)
        ):
            raise ValueError("Unexpected prediction values")
        return prediction
    except urllib.error.HTTPError as exc:
        model_status = exc.headers.get("ms-azureml-model-error-statuscode")
        exc.close()
        if exc.code == 424 and model_status == "400":
            raise ScoringError("The scoring model rejected the image.", status=400) from None
        if exc.code in (408, 504):
            raise ScoringError("Scoring endpoint timed out.", status=504) from None
        status = exc.code if exc.code in (400, 413) else 502
        raise ScoringError(f"Scoring endpoint returned HTTP {exc.code}.", status=status) from None
    except (ValueError, TypeError, KeyError, RecursionError):
        raise ScoringError("Scoring endpoint returned an invalid response.") from None
    except (OSError, HTTPException) as exc:
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        if isinstance(reason, TimeoutError):
            raise ScoringError("Scoring endpoint timed out.", status=504) from None
        raise ScoringError("Could not reach the scoring endpoint.") from None
