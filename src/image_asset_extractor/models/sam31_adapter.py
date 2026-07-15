from __future__ import annotations

import json
import os
from io import BytesIO
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from ..errors import ExtractorError
from .base import ModelAdapter, ModelCapability


class SAM31Adapter(ModelAdapter):
    name = "sam31"

    def capability(self) -> ModelCapability:
        available, mode, reason = self._availability()
        return ModelCapability(self.name, self.enabled, available, mode, reason, ["box-prompt", "point-prompt", "low-confidence-fallback"])

    def refine(self, rgba: np.ndarray, **hints) -> np.ndarray:
        available, mode, reason = self._availability()
        if not available:
            raise ExtractorError("SAM31_UNAVAILABLE", "SAM 3.1 adapter is unavailable", {"reason": reason})
        if mode != "service":
            raise ExtractorError("SAM31_SERVICE_REQUIRED", "SAM 3.1 must run in its isolated service environment")
        buffer = BytesIO()
        Image.fromarray(rgba, "RGBA").save(buffer, "PNG")
        boundary = "----image-asset-extractor-sam31"
        parts: list[bytes] = []

        def add_field(name: str, value: str) -> None:
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
            )

        prompt = hints.get("text_prompt")
        if prompt:
            add_field("text_prompt", str(prompt))
        box = hints.get("box", [0.01, 0.01, 0.98, 0.98])
        add_field("box", json.dumps(box))
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"roi.png\"\r\n"
            "Content-Type: image/png\r\n\r\n".encode()
            + buffer.getvalue()
            + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode())
        endpoint = os.getenv("SAM31_SEGMENT_PATH", "/segment")
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        request = Request(
            f"{self.service_url}{endpoint}",
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urlopen(request, timeout=180) as response:
                mask = np.asarray(Image.open(BytesIO(response.read())).convert("L"))
        except Exception as exc:
            raise ExtractorError("SAM31_REQUEST_FAILED", "SAM 3.1 service request failed", {"reason": str(exc)}) from exc
        return mask
