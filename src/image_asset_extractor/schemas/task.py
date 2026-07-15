from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class TaskRecord:
    id: str
    status: str = "pending"
    progress: int = 0
    stage: str = "pending"
    source_file: str = ""
    output_dir: str = ""
    assets: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None
    revision: int = 0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    cancelled: bool = False
    background_type: str | None = None
    background_confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.id,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "sourceFile": self.source_file,
            "assetCount": len([a for a in self.assets if a.get("status") != "deleted"]),
            "error": self.error,
            "revision": self.revision,
            "backgroundType": self.background_type,
            "backgroundConfidence": self.background_confidence,
            "backgroundStatus": "needs-review" if self.background_confidence is not None and self.background_confidence < 0.6 else ("auto-confirmed" if self.background_confidence is not None else None),
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }
