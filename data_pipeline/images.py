"""Validate image bytes before publishing dataset or review artifacts."""

from pathlib import Path

from PIL import Image


def validate_jpeg(path: Path) -> None:
    try:
        with Image.open(path, formats=["JPEG"]) as image:
            image.load()
    except Image.DecompressionBombError as error:
        raise ValueError("JPEG exceeds the decoder's pixel limit") from error
