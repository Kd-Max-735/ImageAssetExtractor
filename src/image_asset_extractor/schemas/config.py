from __future__ import annotations

from dataclasses import dataclass

from ..errors import ExtractorError


BACKGROUND_TYPES = {"auto", "transparent", "white", "solid", "near-solid", "gradient", "complex", "unknown"}
ASSET_MODES = {"auto", "icon", "person", "product", "decoration", "general"}
SHADOW_MODES = {"auto", "preserve", "remove"}
TEXT_MODES = {"auto", "preserve", "remove"}
OUTPUT_CANVASES = {"tight", "square"}
OUTPUT_FORMATS = {"png"}


@dataclass(slots=True)
class ExtractionConfig:
    background_type: str = "auto"
    asset_mode: str = "icon"
    shadow_mode: str = "preserve"
    text_mode: str = "auto"
    min_area_ratio: float = 0.0001
    merge_distance_ratio: float = 0.01
    padding_ratio: float = 0.05
    background_tolerance: float = 15.0
    output_prefix: str = "icon"
    number_digits: int = 3
    output_canvas: str = "tight"
    output_format: str = "png"
    canvas_size: int | None = None
    edge_feather: float = 0.0
    split_strength: float = 0.5
    include_zip: bool = True
    use_rmbg: bool = False
    use_sam31: bool = False
    max_file_bytes: int = 30 * 1024 * 1024
    max_dimension: int = 10_000
    max_pixels: int = 100_000_000
    max_assets: int = 500

    def validate(self) -> None:
        enum_fields = {
            "background_type": (self.background_type, BACKGROUND_TYPES),
            "asset_mode": (self.asset_mode, ASSET_MODES),
            "shadow_mode": (self.shadow_mode, SHADOW_MODES),
            "text_mode": (self.text_mode, TEXT_MODES),
            "output_canvas": (self.output_canvas, OUTPUT_CANVASES),
            "output_format": (self.output_format, OUTPUT_FORMATS),
        }
        for name, (value, allowed) in enum_fields.items():
            if value not in allowed:
                raise ExtractorError("INVALID_PARAMETER", f"{name} has an unsupported value", {"value": value, "allowed": sorted(allowed)})
        # Keep accepting the legacy field, but shadow removal is no longer a
        # product behavior. All accepted values have preserve semantics.
        self.shadow_mode = "preserve"
        if not 0 <= self.min_area_ratio < 1:
            raise ExtractorError("INVALID_PARAMETER", "min_area_ratio must be in [0, 1)")
        if not 0 <= self.merge_distance_ratio <= 0.25:
            raise ExtractorError("INVALID_PARAMETER", "merge_distance_ratio must be in [0, 0.25]")
        if not 0 <= self.padding_ratio <= 1:
            raise ExtractorError("INVALID_PARAMETER", "padding_ratio must be in [0, 1]")
        if not 0 <= self.background_tolerance <= 255:
            raise ExtractorError("INVALID_PARAMETER", "background_tolerance must be in [0, 255]")
        if self.number_digits < 1 or self.number_digits > 9:
            raise ExtractorError("INVALID_PARAMETER", "number_digits must be in [1, 9]")
        if self.canvas_size is not None and not 16 <= self.canvas_size <= 10_000:
            raise ExtractorError("INVALID_PARAMETER", "canvas_size must be in [16, 10000]")
        if not 0 <= self.edge_feather <= 20:
            raise ExtractorError("INVALID_PARAMETER", "edge_feather must be in [0, 20]")
        if not 0 <= self.split_strength <= 1:
            raise ExtractorError("INVALID_PARAMETER", "split_strength must be in [0, 1]")
        if not self.output_prefix or len(self.output_prefix) > 64 or any(c in self.output_prefix for c in '<>:"/\\|?*'):
            raise ExtractorError("INVALID_PARAMETER", "output_prefix is not a safe file prefix")
