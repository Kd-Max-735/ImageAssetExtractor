from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class BBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


@dataclass(slots=True)
class BackgroundInfo:
    type: str
    confidence: float
    color: tuple[int, int, int] | None
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ConnectedRegion:
    id: str
    area: int
    bbox: BBox
    centroid: tuple[float, float]
    perimeter: float
    aspect_ratio: float
    average_color: tuple[float, float, float]
    average_alpha: float
    contour_complexity: float
    min_distance: float
    touches_edge: bool
    has_holes: bool
    suspected_noise: bool
    suspected_shadow: bool
    label: int = field(repr=False)


@dataclass(slots=True)
class Asset:
    id: str
    index: int
    file: str
    bbox: BBox
    mask_area: int
    confidence: float
    confidence_reasons: list[str]
    region_ids: list[str]
    flags: list[str]
    status: str
    detection_mask: np.ndarray = field(repr=False)
    alpha_mask: np.ndarray = field(repr=False)

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "file": self.file,
            "index": self.index,
            "x": self.bbox.x,
            "y": self.bbox.y,
            "width": self.bbox.width,
            "height": self.bbox.height,
            "maskArea": self.mask_area,
            "confidence": round(self.confidence, 4),
            "confidenceReasons": self.confidence_reasons,
            "regionIds": self.region_ids,
            "flags": self.flags,
            "status": self.status,
        }


@dataclass(slots=True)
class ExtractionResult:
    source_file: str
    source_width: int
    source_height: int
    background: BackgroundInfo
    assets: list[Asset]
    output_dir: str | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "sourceFile": self.source_file,
            "sourceWidth": self.source_width,
            "sourceHeight": self.source_height,
            "backgroundType": self.background.type,
            "backgroundConfidence": round(self.background.confidence, 4),
            "backgroundStatus": "needs-review" if self.background.confidence < 0.6 else "auto-confirmed",
            "backgroundReasons": self.background.reasons,
            "assetCount": len(self.assets),
            "assets": [asset.metadata() for asset in self.assets],
        }
