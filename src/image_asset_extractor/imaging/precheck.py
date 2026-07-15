from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageOps, UnidentifiedImageError

from ..errors import ExtractorError
from ..schemas.config import ExtractionConfig

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
FORMAT_EXTENSIONS = {"PNG": {".png"}, "JPEG": {".jpg", ".jpeg"}, "WEBP": {".webp"}}
FORMAT_MIMES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


@dataclass(slots=True)
class PrecheckedImage:
    rgba: np.ndarray
    width: int
    height: int
    source_format: str
    mime_type: str
    has_alpha: bool
    alpha_valid: bool
    is_single_color: bool
    file_size: int


def _transparent_canvas(alpha: np.ndarray) -> bool:
    """Distinguish a transparent canvas from sparse translucent artwork."""
    low = alpha <= 8
    low_count = int(np.count_nonzero(low))
    if low_count == 0:
        return False
    image_area = alpha.size
    if low_count < max(16, round(image_area * 0.002)):
        return False
    count, labels, stats, _ = cv2.connectedComponentsWithStats(low.astype(np.uint8), 8)
    edge_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    edge_area = max(
        (int(stats[label, cv2.CC_STAT_AREA]) for label in edge_labels if label > 0),
        default=0,
    )
    edge_alpha = np.concatenate((alpha[0], alpha[-1], alpha[:, 0], alpha[:, -1]))
    return (
        edge_area >= max(16, round(image_area * 0.002))
        and float(np.mean(edge_alpha <= 8)) >= 0.05
    )


def load_and_precheck(source: str | Path | bytes, config: ExtractionConfig, source_name: str | None = None) -> PrecheckedImage:
    config.validate()
    if isinstance(source, bytes):
        raw = source
        suffix = Path(source_name).suffix.lower() if source_name else ""
    else:
        path = Path(source)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ExtractorError("IMAGE_UNREADABLE", "Image file cannot be read", {"reason": str(exc)}) from exc
        suffix = path.suffix.lower()
    if len(raw) > config.max_file_bytes:
        raise ExtractorError("FILE_TOO_LARGE", "Image file exceeds the configured size limit", {"bytes": len(raw)})
    try:
        with Image.open(BytesIO(raw)) as image:
            source_format = (image.format or "").upper()
            if source_format not in ALLOWED_FORMATS:
                raise ExtractorError("UNSUPPORTED_FORMAT", "Only PNG, JPG, JPEG and WebP are supported", {"format": source_format})
            if suffix and suffix not in FORMAT_EXTENSIONS[source_format]:
                raise ExtractorError("FORMAT_MISMATCH", "File extension does not match decoded image format", {"extension": suffix, "format": source_format})
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            if width < 1 or height < 1:
                raise ExtractorError("EMPTY_IMAGE", "Image dimensions are empty")
            if width > config.max_dimension or height > config.max_dimension or width * height > config.max_pixels:
                raise ExtractorError("IMAGE_TOO_LARGE", "Image dimensions exceed configured limits", {"width": width, "height": height})
            has_alpha = "A" in image.getbands() or "transparency" in image.info
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    except ExtractorError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ExtractorError("IMAGE_DECODE_FAILED", "File is not a valid supported image", {"reason": str(exc)}) from exc

    alpha = rgba[:, :, 3]
    if has_alpha and not np.any(alpha > 0):
        raise ExtractorError("FULLY_TRANSPARENT", "Image is fully transparent")
    alpha_valid = bool(has_alpha and _transparent_canvas(alpha))
    rgb = rgba[:, :, :3]
    is_single_color = bool(np.all(rgb == rgb[0, 0]) and (not has_alpha or np.all(alpha == alpha[0, 0])))
    if is_single_color:
        raise ExtractorError("EMPTY_IMAGE", "Image contains no distinguishable foreground")
    return PrecheckedImage(
        rgba=rgba,
        width=width,
        height=height,
        source_format=source_format,
        mime_type=FORMAT_MIMES[source_format],
        has_alpha=has_alpha,
        alpha_valid=alpha_valid,
        is_single_color=is_single_color,
        file_size=len(raw),
    )
