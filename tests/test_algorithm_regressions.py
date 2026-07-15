from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageDraw

from image_asset_extractor.imaging.clustering import cluster_regions
from image_asset_extractor.imaging.components import analyze_components
from image_asset_extractor.imaging.grid import projection_cells
from image_asset_extractor.schemas.asset import BBox, BackgroundInfo, ConnectedRegion
from image_asset_extractor.schemas.config import ExtractionConfig
from image_asset_extractor.services.extraction import ExtractionEngine


def _bytes(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, "PNG")
    return stream.getvalue()


def _exported(result, root, index: int = 0) -> np.ndarray:
    return np.asarray(Image.open(root / result.assets[index].file).convert("RGBA"))


def _region(
    label: int,
    area: int,
    bbox: tuple[int, int, int, int],
    centroid: tuple[float, float],
    color: tuple[float, float, float],
) -> ConnectedRegion:
    box = BBox(*bbox)
    return ConnectedRegion(
        id=f"region-{label}", area=area, bbox=box, centroid=centroid,
        perimeter=float((box.width + box.height) * 2), aspect_ratio=box.width / box.height,
        average_color=color, average_alpha=255.0, contour_complexity=1.0,
        min_distance=1.0, touches_edge=False, has_holes=False,
        suspected_noise=False, suspected_shadow=False, label=label,
    )


def test_flattened_small_opaque_icon_keeps_opaque_core(tmp_path):
    image = Image.new("RGB", (40, 40), "white")
    ImageDraw.Draw(image).rectangle((15, 15, 24, 24), fill=(38, 188, 112))
    root = tmp_path / "small-core"
    result = ExtractionEngine().extract(_bytes(image), root, ExtractionConfig(include_zip=False))
    asset = result.assets[0]
    rendered = _exported(result, root)
    core = cv2.erode((asset.detection_mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert np.any(core)
    assert float(np.median(rendered[:, :, 3][core])) >= 250


def test_flattened_thin_line_core_is_not_globally_low_alpha(tmp_path):
    image = Image.new("RGB", (100, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.line((24, 55, 48, 22, 74, 55), fill=(90, 90, 90), width=2, joint="curve")
    draw.line((32, 47, 65, 47), fill=(90, 90, 90), width=2)
    root = tmp_path / "thin-line"
    result = ExtractionEngine().extract(_bytes(image), root, ExtractionConfig(include_zip=False))
    asset = result.assets[0]
    rendered = _exported(result, root)
    core_alpha = rendered[:, :, 3][asset.detection_mask > 0]
    assert len(core_alpha) > 20
    assert float(np.median(core_alpha)) >= 250


def test_flattened_closed_hole_stays_transparent(tmp_path):
    image = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((25, 20, 95, 100), fill=(107, 132, 45))
    draw.ellipse((52, 52, 68, 68), fill=(236, 244, 251))
    root = tmp_path / "closed-hole"
    result = ExtractionEngine().extract(_bytes(image), root, ExtractionConfig(include_zip=False))
    asset = result.assets[0]
    rendered = _exported(result, root)
    assert rendered[60 - asset.bbox.y, 60 - asset.bbox.x, 3] == 0


def test_flattened_outer_glow_recomposition_remains_exact(tmp_path):
    height = width = 120
    core = np.zeros((height, width), np.uint8)
    cv2.rectangle(core, (45, 45), (74, 74), 255, -1)
    glow = cv2.GaussianBlur(core, (0, 0), 3.0).astype(np.float32) / 255.0 * 0.55
    alpha = np.maximum(glow, core.astype(np.float32) / 255.0)
    color = np.asarray((20, 120, 240), np.float32)
    observed = np.clip(color * alpha[:, :, None] + 255.0 * (1.0 - alpha[:, :, None]), 0, 255).astype(np.uint8)
    root = tmp_path / "outer-glow"
    result = ExtractionEngine().extract(_bytes(Image.fromarray(observed, "RGB")), root, ExtractionConfig(include_zip=False))
    asset = result.assets[0]
    rendered = _exported(result, root).astype(np.float32)
    a = rendered[:, :, 3:4] / 255.0
    recomposed = rendered[:, :, :3] * a + 255.0 * (1.0 - a)
    original = observed[asset.bbox.y:asset.bbox.y2, asset.bbox.x:asset.bbox.x2].astype(np.float32)
    assert float(np.mean(np.abs(recomposed - original))) < 1.0
    fringe = (rendered[:, :, 3] > 5) & (rendered[:, :, 3] < 230)
    assert not np.any(fringe) or float(np.mean(np.ptp(rendered[:, :, :3][fringe], axis=1) < 35)) < 0.05


def test_gradient_sparse_source_alpha_100_is_preserved(tmp_path):
    height, width = 160, 240
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    left = np.asarray((30, 190, 210), np.float32)
    right = np.asarray((225, 145, 130), np.float32)
    rgb = np.ascontiguousarray(np.clip(left * (1 - x) + right * x + y * np.asarray((5, 8, 12)), 0, 255).astype(np.uint8))
    cv2.circle(rgb, (80, 80), 30, (255, 220, 40), -1)
    cv2.circle(rgb, (80, 80), 12, (250, 250, 250), -1)
    source_alpha = np.full((height, width), 255, np.uint8)
    cv2.circle(source_alpha, (80, 80), 12, 100, -1)
    rgba = np.dstack((rgb, source_alpha))
    root = tmp_path / "source-alpha"
    result = ExtractionEngine().extract(_bytes(Image.fromarray(rgba, "RGBA")), root, ExtractionConfig(include_zip=False))
    asset = next(item for item in result.assets if item.bbox.x <= 80 < item.bbox.x2 and item.bbox.y <= 80 < item.bbox.y2)
    rendered = np.asarray(Image.open(root / asset.file).convert("RGBA"))
    assert rendered[80 - asset.bbox.y, 80 - asset.bbox.x, 3] == 100
    contours, _ = cv2.findContours(asset.detection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    envelope = np.zeros(asset.detection_mask.shape, np.uint8)
    cv2.drawContours(envelope, contours, -1, 255, cv2.FILLED)
    assert not np.any((rendered[:, :, 3] > 0) & (envelope == 0))


def test_cross_cell_glass_satellites_do_not_merge_distant_icons():
    regions = [
        _region(1, 54036, (1170, 150, 261, 255), (1306.3, 274.5), (209.2, 215.3, 245.6)),
        _region(2, 4504, (136, 156, 249, 249), (260.0, 280.0), (223.6, 235.8, 246.7)),
        _region(3, 4504, (656, 156, 249, 249), (780.0, 280.0), (230.5, 235.8, 248.5)),
        _region(4, 6626, (165, 175, 91, 156), (208.3, 233.8), (164.3, 198.6, 231.1)),
        _region(5, 66333, (130, 670, 261, 261), (260.0, 800.0), (188.5, 226.3, 240.5)),
        _region(6, 59712, (650, 670, 261, 261), (785.7, 805.1), (207.5, 232.4, 246.3)),
        _region(7, 59712, (1170, 670, 261, 261), (1305.7, 805.1), (220.0, 236.5, 250.0)),
    ]
    cells = [
        (130, 150, 261, 261), (650, 150, 261, 261), (1170, 150, 261, 261),
        (130, 670, 261, 261), (650, 670, 261, 261), (1170, 670, 261, 261),
    ]
    clusters = cluster_regions(regions, (1200, 1600), 0.01, grid_cells=cells)
    assert len(clusters) == 6
    assert any({item.id for item in cluster} == {"region-2", "region-4"} for cluster in clusters)


def test_cross_cell_guard_keeps_nearby_legal_satellite():
    regions = [
        _region(1, 1000, (40, 60, 40, 40), (60.0, 80.0), (60.0, 100.0, 180.0)),
        _region(2, 120, (84, 72, 10, 10), (89.0, 77.0), (62.0, 102.0, 178.0)),
    ]
    cells = [(35, 50, 48, 60), (83, 50, 48, 60)]
    clusters = cluster_regions(regions, (200, 200), 0.01, grid_cells=cells)
    assert len(clusters) == 1


def test_regular_grid_does_not_merge_neighbors(tmp_path):
    image = Image.new("RGBA", (260, 260), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for row in range(5):
        for column in range(5):
            x, y = 15 + column * 48, 15 + row * 48
            draw.rectangle((x, y, x + 18, y + 18), fill=(40, 130, 220, 255))
    result = ExtractionEngine().extract(_bytes(image), tmp_path / "regular-grid", ExtractionConfig(include_zip=False))
    assert len(result.assets) == 25


def test_macro_grid_owns_split_strokes_by_repeated_cell():
    mask = np.zeros((300, 260), np.uint8)
    split_cells = {(1, 1), (4, 3), (7, 6)}
    for row in range(10):
        for column in range(8):
            x, y = 10 + column * 31, 10 + row * 28
            if (column, row) in split_cells:
                mask[y:y + 12, x:x + 5] = 255
                mask[y:y + 12, x + 7:x + 12] = 255
            else:
                mask[y:y + 12, x:x + 12] = 255
    rgba = np.zeros((300, 260, 4), np.uint8)
    rgba[mask > 0] = (90, 90, 90, 255)
    regions, _ = analyze_components(mask, rgba, 1)
    cells = projection_cells(mask)
    clusters = cluster_regions(regions, mask.shape, 0.01, mask, cells)
    assert len(cells) == 80
    assert len(clusters) == 80


def test_photo_primary_background_and_coarse_subject_envelope(tmp_path, monkeypatch):
    image = Image.new("RGB", (300, 300), "white")
    draw = ImageDraw.Draw(image)
    centers = [(x, y) for y in (0, 150, 300) for x in (0, 150, 300)]
    for cx, cy in centers:
        draw.ellipse((cx - 65, cy - 65, cx + 65, cy + 65), fill=(245, 220, 170), outline=(220, 120, 50), width=4)
        draw.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), fill=(251, 241, 216))

    monkeypatch.setattr(
        "image_asset_extractor.services.extraction.classify_background",
        lambda checked, requested="auto": BackgroundInfo("near-solid", 0.55, (255, 255, 255), ["synthetic-primary"]),
    )
    monkeypatch.setattr(
        "image_asset_extractor.imaging.foreground.background_color_candidates",
        lambda rgb: [(255, 255, 255), (251, 241, 216)],
    )
    root = tmp_path / "photo-envelope"
    result = ExtractionEngine().extract(_bytes(image), root, ExtractionConfig(include_zip=False))
    assert result.background.color == (255, 255, 255)
    assert len(result.assets) == len(centers)
    assert all(any(reason.startswith("stable-coarse-instance-envelope:") for reason in asset.confidence_reasons) for asset in result.assets)
    assert not any("page-like-candidate" in asset.flags for asset in result.assets)
    boxes = [asset.bbox for asset in result.assets]
    for i, first in enumerate(boxes):
        for second in boxes[i + 1:]:
            x1, y1 = max(first.x, second.x), max(first.y, second.y)
            x2, y2 = min(first.x2, second.x2), min(first.y2, second.y2)
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            union = first.width * first.height + second.width * second.height - intersection
            assert intersection / max(1, union) < 0.5
    center_asset = next(asset for asset in result.assets if asset.bbox.x <= 150 < asset.bbox.x2 and asset.bbox.y <= 150 < asset.bbox.y2)
    rendered = np.asarray(Image.open(root / center_asset.file).convert("RGBA"))
    assert rendered[150 - center_asset.bbox.y, 150 - center_asset.bbox.x, 3] >= 250
