from __future__ import annotations

import os
from io import BytesIO
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from ..errors import ExtractorError
from .base import ModelAdapter, ModelCapability


class RMBGAdapter(ModelAdapter):
    name = "rmbg"

    def capability(self) -> ModelCapability:
        available, mode, reason = self._availability()
        return ModelCapability(self.name, self.enabled, available, mode, reason, ["roi-soft-alpha", "edge-refinement"])

    def refine(self, rgba: np.ndarray, **hints) -> np.ndarray:
        available, mode, reason = self._availability()
        if not available:
            raise ExtractorError("RMBG_UNAVAILABLE", "RMBG adapter is unavailable", {"reason": reason})
        if mode != "service":
            raise ExtractorError("RMBG_LOCAL_NOT_LOADED", "Local RMBG inference must run in its isolated environment")
        buffer = BytesIO()
        Image.fromarray(rgba, "RGBA").save(buffer, "PNG")
        boundary = "----image-asset-extractor"
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"roi.png\"\r\n"
            "Content-Type: image/png\r\n\r\n"
        ).encode() + buffer.getvalue() + f"\r\n--{boundary}--\r\n".encode()
        endpoint = os.getenv("RMBG_REMOVE_BG_PATH", "/remove-bg")
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        request = Request(f"{self.service_url}{endpoint}", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            with urlopen(request, timeout=60) as response:
                result = np.asarray(Image.open(BytesIO(response.read())).convert("RGBA"))[:, :, 3]
        except Exception as exc:
            raise ExtractorError("RMBG_REQUEST_FAILED", "RMBG service request failed", {"reason": str(exc)}) from exc
        return result
