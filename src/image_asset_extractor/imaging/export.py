from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .. import __algorithm_version__, __version__
from ..errors import ExtractorError
from ..schemas.asset import ExtractionResult
from ..schemas.config import ExtractionConfig

CSV_FIELDS = ["id", "file", "index", "x", "y", "width", "height", "maskArea", "confidence", "status", "flags"]


def _edge_connected(mask: np.ndarray) -> np.ndarray:
    """Keep mask components that can reach the candidate crop boundary."""
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros(mask.shape, dtype=bool)
    boundary = np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    external_labels = np.unique(boundary[boundary > 0])
    return np.isin(labels, external_labels) if len(external_labels) else np.zeros(mask.shape, dtype=bool)


def _recover_flattened_effects(
    observed: np.ndarray,
    detection: np.ndarray,
    alpha: np.ndarray,
    background: np.ndarray,
    source_alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover only bbox-connected effects while preserving the detected subject."""
    observed_f = observed.astype(np.float32)
    delta = observed_f - background
    limits = np.full(delta.shape, np.inf, np.float32)
    positive = delta > 0.25
    negative = delta < -0.25
    limits[positive] = np.broadcast_to(255.0 - background, delta.shape)[positive] / delta[positive]
    limits[negative] = np.broadcast_to(-background, delta.shape)[negative] / delta[negative]
    scale = np.min(limits, axis=2)
    scale[~np.isfinite(scale)] = np.inf
    best_alpha = np.where(np.isfinite(scale), 1.0 / np.maximum(1.0, scale), 0.0)
    smoothed_alpha = cv2.GaussianBlur(best_alpha.astype(np.float32), (0, 0), sigmaX=2.5, sigmaY=2.5)
    allowed = alpha > 0
    structure = detection > 0
    source_alpha_evidence = (source_alpha > 0) & (source_alpha < 255)
    outside = allowed & ~structure & ~source_alpha_evidence
    external = _edge_connected(~structure)
    effect = allowed & external & (best_alpha >= 1.0 / 255.0) & (smoothed_alpha >= 0.012)
    effect_alpha = np.maximum(best_alpha, smoothed_alpha)
    recovered_rgb = np.clip(
        background + delta / np.maximum(effect_alpha[:, :, None], 1.0 / 255.0),
        0,
        255,
    )
    output_rgb = observed.copy()
    output_alpha = alpha.copy()
    # A detected pixel belongs to the opaque structure of a flattened image.
    # Use actual source alpha when present so sparse source-alpha evidence is
    # never promoted to 255.
    output_alpha[structure] = source_alpha[structure]
    output_alpha[outside] = 0
    output_rgb[effect] = np.clip(recovered_rgb[effect], 0, 255).astype(np.uint8)
    output_alpha[effect] = np.clip(effect_alpha[effect] * 255.0, 1, 255).astype(np.uint8)
    return output_rgb, output_alpha


def render_asset(result: ExtractionResult, asset, source_rgba: np.ndarray, config: ExtractionConfig) -> Image.Image:
    box = asset.bbox
    rgba = source_rgba[box.y:box.y2, box.x:box.x2].copy()
    alpha = asset.alpha_mask.copy()
    if config.edge_feather > 0:
        sigma = max(0.1, float(config.edge_feather))
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
    rgba[:, :, 3] = alpha

    # A flattened opaque image has no unique original RGBA solution. Recover a
    # gamut-boundary explanation whose recomposition stays close to the source.
    if (
        result.background.type in {"solid", "near-solid"}
        and result.background.color is not None
        and result.background.confidence >= 0.75
    ):
        recovered_rgb, alpha = _recover_flattened_effects(
            rgba[:, :, :3], asset.detection_mask, alpha,
            np.asarray(result.background.color, dtype=np.float32),
            source_rgba[box.y:box.y2, box.x:box.x2, 3],
        )
        rgba[:, :, :3] = recovered_rgb
        rgba[:, :, 3] = alpha
    rgba[alpha == 0, :3] = 0
    image = Image.fromarray(rgba, "RGBA")
    if config.output_canvas == "square":
        side = max(image.width, image.height, config.canvas_size or 0)
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.alpha_composite(image, ((side - image.width) // 2, (side - image.height) // 2))
        image = canvas
    return image


def export_result(
    result: ExtractionResult,
    source_rgba: np.ndarray,
    output_root: str | Path,
    include_zip: bool = True,
    *,
    config: ExtractionConfig | None = None,
    overwrite: bool = False,
) -> Path:
    config = config or ExtractionConfig(include_zip=include_zip)
    root = Path(output_root)
    icons = root / "icons"
    result_dir = root / "result"
    metadata_path = result_dir / "metadata.json"
    if metadata_path.exists() and not overwrite:
        raise ExtractorError("OUTPUT_EXISTS", "Output directory already contains an extraction result", {"path": str(root)})
    icons.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    wanted = {asset.file for asset in result.assets}
    if overwrite and metadata_path.is_file():
        try:
            old_files = [item.get("file", "") for item in json.loads(metadata_path.read_text("utf-8")).get("assets", [])]
        except (OSError, ValueError, TypeError):
            old_files = []
        for old_file in old_files:
            old_path = root / old_file
            if old_file and old_file not in wanted and old_path.is_file():
                old_path.unlink()

    for asset in result.assets:
        render_asset(result, asset, source_rgba, config).save(root / asset.file)

    metadata = result.metadata()
    metadata["processingConfig"] = {
        "backgroundType": config.background_type,
        "assetMode": config.asset_mode,
        "minAreaRatio": config.min_area_ratio,
        "mergeDistanceRatio": config.merge_distance_ratio,
        "paddingRatio": config.padding_ratio,
        "backgroundTolerance": config.background_tolerance,
        "edgeFeather": config.edge_feather,
        "splitStrength": config.split_strength,
        "useRmbg": config.use_rmbg,
        "useSam31": config.use_sam31,
    }
    metadata["algorithmVersion"] = __algorithm_version__
    metadata["packageVersion"] = __version__
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    with (result_dir / "metadata.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for asset in metadata["assets"]:
            row = {key: asset[key] for key in CSV_FIELDS if key != "flags"}
            row["flags"] = ";".join(asset["flags"])
            writer.writerow(row)

    annotated = Image.fromarray(source_rgba[:, :, :3], "RGB")
    draw = ImageDraw.Draw(annotated)
    for asset in result.assets:
        b = asset.bbox
        color = (220, 40, 40) if asset.status == "needs-review" else (20, 190, 80)
        draw.rectangle((b.x, b.y, b.x2 - 1, b.y2 - 1), outline=color, width=max(1, min(source_rgba.shape[:2]) // 300))
        draw.text((b.x + 2, b.y + 2), str(asset.index), fill=color)
    annotated.save(result_dir / "annotated.png")

    if include_zip:
        zip_path = result_dir / "assets.zip"
        if zip_path.is_file():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(icons.glob("*.png")):
                archive.write(path, path.relative_to(root).as_posix())
            for name in ("metadata.json", "metadata.csv", "annotated.png"):
                path = result_dir / name
                archive.write(path, path.relative_to(root).as_posix())
    else:
        zip_path = result_dir / "assets.zip"
        if overwrite and zip_path.is_file():
            zip_path.unlink()
    result.output_dir = str(root)
    return root
