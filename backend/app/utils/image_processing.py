import base64
from typing import Optional

import cv2
import numpy as np
from fastapi import UploadFile


ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MB


class InvalidImageError(Exception):
    pass


async def decode_upload_to_bgr(file: UploadFile) -> np.ndarray:
    # Note: we intentionally do NOT gate on file.content_type here.
    # Clients (Flutter's http.MultipartFile.fromPath in particular) often
    # omit an explicit contentType and default to "application/octet-stream",
    # which would fail a strict header check even though the bytes are a
    # perfectly valid image. The real, trustworthy check is whether the
    # bytes actually decode as an image -- so we do that instead, and only
    # use ALLOWED_CONTENT_TYPES as an early filter for headers we DO trust
    # (i.e. only reject when a content_type was provided AND is clearly
    # not an image at all, e.g. "text/html" or "application/json").
    if file.content_type and not file.content_type.startswith("image/") and file.content_type != "application/octet-stream":
        raise InvalidImageError(f"Unsupported content type: {file.content_type}")

    raw = await file.read()
    if not raw:
        raise InvalidImageError("Uploaded file is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise InvalidImageError("Image exceeds maximum allowed size (15MB)")

    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("Could not decode image -- file may be corrupted")
    return image


def encode_bgr_to_base64_jpeg(image_bgr: np.ndarray) -> Optional[str]:
    if image_bgr is None or image_bgr.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", image_bgr)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("utf-8")
