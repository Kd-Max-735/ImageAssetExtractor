from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
import json
from urllib.request import urlopen
from urllib.error import HTTPError

import numpy as np


@dataclass(slots=True)
class ModelCapability:
    name: str
    enabled: bool
    available: bool
    mode: str | None
    reason: str
    capabilities: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


class ModelAdapter(ABC):
    name: str

    def __init__(self, enabled: bool, model_path: str = "", service_url: str = ""):
        self.enabled = enabled
        self.model_path = model_path.strip()
        self.service_url = service_url.strip().rstrip("/")
        self._availability_cache: tuple[bool, str | None, str] | None = None

    @abstractmethod
    def capability(self) -> ModelCapability: ...

    @abstractmethod
    def refine(self, rgba: np.ndarray, **hints) -> np.ndarray: ...

    def _availability(self) -> tuple[bool, str | None, str]:
        if self._availability_cache is not None:
            return self._availability_cache
        result = self._probe_availability()
        self._availability_cache = result
        return result

    def _probe_availability(self) -> tuple[bool, str | None, str]:
        if not self.enabled:
            return False, None, "disabled by configuration"
        if self.service_url:
            try:
                with urlopen(f"{self.service_url}/health", timeout=1.0) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if response.status == 200 and payload.get("status") == "ok":
                    return True, "service", "service health check passed"
                return False, "service", "service health check returned an unhealthy response"
            except HTTPError as exc:
                if exc.code == 404:
                    try:
                        with urlopen(f"{self.service_url}/openapi.json", timeout=1.0) as response:
                            schema = json.loads(response.read().decode("utf-8"))
                        if response.status == 200 and schema.get("paths"):
                            return True, "service", "service OpenAPI probe passed; /health is not exposed"
                    except Exception as probe_exc:
                        return False, "service", f"health and OpenAPI probes failed: {probe_exc}"
                return False, "service", f"service health check failed: HTTP {exc.code}"
            except Exception as exc:
                return False, "service", f"service health check failed: {exc}"
        if self.model_path:
            if Path(self.model_path).exists():
                return False, "local-external", "model path exists but an isolated service URL is required"
            return False, "local", "configured model path does not exist"
        return False, None, "no model path or service URL configured"
