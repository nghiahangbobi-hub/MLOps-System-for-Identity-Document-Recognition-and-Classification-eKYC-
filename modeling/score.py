"""Azure ML scoring entrypoint for the custom model bundle."""

import base64
import hashlib
import io
import json
import os
import pickle
import re
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps

if __package__:
    from .network import ARCHITECTURE, DOCUMENT_LABELS, MAX_IMAGE_SIZE, QUALITY_LABELS, build_model, preprocessing
else:
    # Azure ML loads the entrypoint by filename outside the Python package.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from network import ARCHITECTURE, DOCUMENT_LABELS, MAX_IMAGE_SIZE, QUALITY_LABELS, build_model, preprocessing

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_BASE64_LENGTH = 4 * ((MAX_IMAGE_BYTES + 2) // 3)
MAX_REQUEST_LENGTH = MAX_BASE64_LENGTH + 4096
MAX_METADATA_BYTES = 16 * 1024
MAX_WEIGHTS_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
_predictor = None


def validate_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("Model metadata must be an object")
    config = metadata.get("preprocessing")
    threshold = metadata.get("confidence_threshold")
    if (
        type(metadata.get("format_version")) is not int
        or metadata["format_version"] != 1
        or metadata.get("architecture") != ARCHITECTURE
        or metadata.get("document_labels") != list(DOCUMENT_LABELS)
        or metadata.get("quality_labels") != list(QUALITY_LABELS)
        or type(metadata.get("has_quality_head")) is not bool
        or type(threshold) not in (int, float)
        or not 0 <= threshold <= 1
        or not isinstance(config, dict)
    ):
        raise ValueError("Unsupported model metadata")
    temperature = metadata.get("document_temperature", 1)
    if type(temperature) not in (float, int) or not 0.1 <= temperature <= 10:
        raise ValueError("Invalid document temperature")
    if type(metadata.get("document_review_all", False)) is not bool:
        raise ValueError("Invalid document review policy")
    if config.get("crop_mode", "full_frame") not in ("full_frame", "center_square"):
        raise ValueError("Invalid model crop mode")
    size = config.get("image_size")
    if type(size) is not int or not 32 <= size <= MAX_IMAGE_SIZE:
        raise ValueError("Invalid model image size")
    for name, minimum in (("mean", 0), ("std", 1e-6)):
        values = config.get(name)
        if not isinstance(values, list) or len(values) != 3 or not all(
            type(value) in (int, float) and minimum <= value <= 1 for value in values
        ):
            raise ValueError(f"Invalid model normalization {name}")
    if "weights_sha256" in metadata:
        digest = metadata["weights_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid model weights checksum")


def init():
    load_bundle(os.getenv("AZUREML_MODEL_DIR") or os.getenv("MODEL_DIR"))


def load_bundle(configured_root):
    global _predictor
    _predictor = None
    if not configured_root:
        raise RuntimeError("Set AZUREML_MODEL_DIR or MODEL_DIR to the model bundle location")
    root = Path(configured_root)
    bundles = [
        path.parent for path in sorted(root.rglob("metadata.json"))
        if (path.parent / "weights.pt").is_file()
    ]
    if len(bundles) != 1:
        raise RuntimeError(f"Expected one model bundle under {root}, found {len(bundles)}")
    bundle = bundles[0]
    with (bundle / "metadata.json").open("rb") as stream:
        raw_metadata = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw_metadata) > MAX_METADATA_BYTES:
        raise ValueError("Model metadata exceeds the size limit")
    try:
        metadata = json.loads(raw_metadata)
    except (ValueError, RecursionError) as exc:
        raise ValueError("Unreadable model metadata") from exc
    validate_metadata(metadata)
    model = build_model(metadata["has_quality_head"])
    with (bundle / "weights.pt").open("rb") as stream:
        if not 0 < os.fstat(stream.fileno()).st_size <= MAX_WEIGHTS_BYTES:
            raise ValueError("Model weights are empty or exceed the size limit")
        if "weights_sha256" in metadata:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != metadata["weights_sha256"]:
                raise ValueError("Model weights checksum mismatch")
            stream.seek(0)
        try:
            weights = torch.load(stream, map_location="cpu", weights_only=True)
            if not isinstance(weights, dict) or any(
                not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                for value in weights.values()
            ):
                raise ValueError("Model weights must contain finite tensors")
            model.load_state_dict(weights)
        except (RuntimeError, EOFError, pickle.UnpicklingError, TypeError, IndexError) as exc:
            raise ValueError("Unreadable or incompatible model weights") from exc
    config = metadata["preprocessing"]
    transform = preprocessing(config["image_size"], config["mean"], config["std"],
                              config.get("crop_mode", "full_frame"))
    _predictor = model.eval(), transform, metadata
    return metadata


def request_image(raw_data):
    try:
        payload = raw_data
        if isinstance(raw_data, (str, bytes, bytearray)):
            if len(raw_data) > MAX_REQUEST_LENGTH:
                raise ValueError
            payload = json.loads(raw_data)
        if not isinstance(payload, dict):
            raise ValueError
        encoded = payload["image_base64"]
        if not isinstance(encoded, str) or len(encoded) > MAX_BASE64_LENGTH:
            raise ValueError
        content = base64.b64decode(encoded, validate=True)
        if not 0 < len(content) <= MAX_IMAGE_BYTES:
            raise ValueError
    except (KeyError, ValueError, RecursionError):
        raise ValueError("Expected a JSON object with a valid image_base64 up to 8 MiB") from None

    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError
            return ImageOps.exif_transpose(image).convert("RGB")
    except (Image.DecompressionBombError, OSError, ValueError, SyntaxError, OverflowError):
        raise ValueError("Expected a readable image up to 25 megapixels") from None


@torch.inference_mode()
def run(raw_data, request_headers=None):
    predictor = _predictor
    if predictor is None:
        raise RuntimeError("Call init() before scoring")
    model, transform, metadata = predictor
    try:
        image = request_image(raw_data)
    except ValueError as exc:
        if request_headers is None:
            raise
        from azureml_inference_server_http.api.aml_response import AMLResponse

        return AMLResponse({"error": str(exc)}, 400, json_str=True)
    logits = model(transform(image).unsqueeze(0))[0]
    document_count = len(DOCUMENT_LABELS)
    temperature = metadata.get("document_temperature")
    document_logits = logits[:document_count]
    if temperature is not None:
        document_logits = document_logits.double() / temperature
    confidence, index = document_logits.softmax(dim=0).max(dim=0)
    confidence = confidence.item()
    threshold = metadata["confidence_threshold"]
    low_document_confidence = confidence < threshold
    if low_document_confidence:
        # The inference server routes stdout prints to its App Insights logger.
        print(json.dumps({
            "event": "ekyc.low_document_confidence",
            "confidence": confidence,
            "threshold": threshold,
        }))
    status = None
    if metadata["has_quality_head"]:
        status = QUALITY_LABELS[logits[document_count:].argmax().item()]
    return {
        "label": DOCUMENT_LABELS[index.item()],
        "status": status,
        "confidence": confidence,
        "needs_review": (status != "Clear" or metadata.get("document_review_all", False)
                         or low_document_confidence),
    }
