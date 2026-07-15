from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_asset_extractor.imaging.background import classify_background
from image_asset_extractor.imaging.clustering import cluster_regions
from image_asset_extractor.imaging.components import analyze_components
from image_asset_extractor.imaging.foreground import segment_foreground
from image_asset_extractor.imaging.grid import projection_cells
from image_asset_extractor.imaging.precheck import load_and_precheck
from image_asset_extractor.schemas.config import ExtractionConfig
from image_asset_extractor.services.extraction import ExtractionEngine


CASES = {
    "01_transparent_icons.png": 20,
    "17_products_collection.jpeg": 9,
    "07_regular_grid.png": 25,
    "09_mixed_sizes.png": 12,
    "10_close_same_color.png": 8,
    "12_slight_touching_pairs.png": 12,
    "14_outer_glow.png": 8,
    "19_low_resolution.png": 8,
    "21_white_on_white.png": 8,
    "23_translucent_glass.png": 6,
    "strong_gradient_background(2).png": 10,
    "test3.png": 80,
    "test5.png": 19,
}


def _source(name: str) -> Path:
    matches = list((Path.home() / "Desktop").rglob(name))
    if len(matches) != 1:
        pytest.skip(f"external known-issue fixture unavailable: {name}")
    return matches[0]


@pytest.fixture(scope="module")
def known_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("known-issue-regressions")
    collected = {}
    for name in CASES:
        path = _source(name)
        config = ExtractionConfig(include_zip=False)
        checked = load_and_precheck(path, config)
        background = classify_background(checked)
        structure, appearance = segment_foreground(checked, background, config)
        minimum = max(1, round(structure.size * config.min_area_ratio))
        regions, labels = analyze_components(structure, checked.rgba, minimum)
        clusters = cluster_regions(
            regions, (checked.height, checked.width), config.merge_distance_ratio,
            structure, projection_cells(structure),
        )
        output = root / path.stem
        result = ExtractionEngine().extract(path, output, config)
        collected[name] = {
            "path": path, "checked": checked, "background": background,
            "structure": structure, "appearance": appearance,
            "regions": regions, "labels": labels, "clusters": clusters,
            "result": result, "output": output,
        }
    return collected


@pytest.mark.parametrize("name,expected", CASES.items())
def test_known_issue_asset_counts(known_results, name, expected):
    assert len(known_results[name]["result"].assets) == expected


def test_multicolor_connected_icon_is_not_split(known_results):
    case = known_results["01_transparent_icons.png"]
    source = case["checked"].rgba[:, :, :3]
    colors = (np.array([150, 90, 220]), np.array([250, 170, 35]))
    per_asset = []
    for asset in case["result"].assets:
        box = asset.bbox
        local = source[box.y:box.y2, box.x:box.x2]
        per_asset.append([
            int(np.count_nonzero((asset.detection_mask > 0) & np.all(local == color, axis=2)))
            for color in colors
        ])
    assert any(first > 500 and second > 500 for first, second in per_asset)


def test_edge_clipped_asset_survives_below_global_area_threshold(known_results):
    case = known_results["07_regular_grid.png"]
    checked = case["checked"]
    minimum = round(checked.width * checked.height * ExtractionConfig().min_area_ratio)
    clipped = [asset for asset in case["result"].assets if asset.bbox.x2 == checked.width]
    assert len(clipped) == 1
    assert np.count_nonzero(clipped[0].detection_mask) < minimum
    assert clipped[0].status == "needs-review"


def test_touching_pair_pixels_follow_color_continuity(known_results):
    case = known_results["12_slight_touching_pairs.png"]
    source = case["checked"].rgba[:, :, :3]
    purple = np.array([150, 90, 220])
    teal = np.array([30, 180, 180])
    assigned = {"purple": 0, "teal": 0}
    contamination = 0
    for asset in case["result"].assets:
        box = asset.bbox
        local = source[box.y:box.y2, box.x:box.x2]
        active = asset.detection_mask > 0
        purple_count = int(np.count_nonzero(active & np.all(local == purple, axis=2)))
        teal_count = int(np.count_nonzero(active & np.all(local == teal, axis=2)))
        if purple_count > teal_count:
            assigned["purple"] += purple_count
            contamination += teal_count
        elif teal_count:
            assigned["teal"] += teal_count
            contamination += purple_count
    assert assigned["purple"] == int(np.count_nonzero(np.all(source == purple, axis=2)))
    assert assigned["teal"] == int(np.count_nonzero(np.all(source == teal, axis=2)))
    assert contamination == 0


def test_flattened_glow_recomposes_without_gray_fringe(known_results):
    case = known_results["14_outer_glow.png"]
    source = case["checked"].rgba[:, :, :3].astype(np.float32)
    background = np.asarray(case["background"].color, np.float32)
    errors = []
    gray_fringe = []
    for asset in case["result"].assets:
        exported = np.asarray(Image.open(case["output"] / asset.file).convert("RGBA"), np.float32)
        alpha = exported[:, :, 3:4] / 255.0
        recomposed = exported[:, :, :3] * alpha + background * (1.0 - alpha)
        box = asset.bbox
        original = source[box.y:box.y2, box.x:box.x2]
        errors.append(float(np.mean(np.abs(recomposed - original))))
        fringe = (exported[:, :, 3] > 5) & (exported[:, :, 3] < 230)
        if np.any(fringe):
            gray_fringe.append(float(np.mean(np.ptp(exported[:, :, :3][fringe], axis=1) < 35)))
    assert float(np.mean(errors)) < 1.0
    assert float(np.mean(gray_fringe)) < 0.05


def test_strong_gradient_uses_ten_sane_structure_regions_and_preserves_source_alpha(known_results):
    case = known_results["strong_gradient_background(2).png"]
    assert not case["checked"].alpha_valid
    assert case["background"].type == "gradient"
    assert np.mean(case["structure"] > 0) < 0.12
    assert len(case["regions"]) == 10
    assert len(case["clusters"]) == 10
    assert all(asset.bbox.width * asset.bbox.height < case["structure"].size * 0.2 for asset in case["result"].assets)
    assert any(np.any(asset.alpha_mask == 100) for asset in case["result"].assets)
    source_alpha_assets = [asset for asset in case["result"].assets if np.any(asset.alpha_mask == 100)]
    assert all(
        np.count_nonzero(asset.alpha_mask) <= np.count_nonzero(asset.detection_mask) * 1.05
        for asset in source_alpha_assets
    )


def test_test3_uses_repeated_macro_grid_cells(known_results):
    case = known_results["test3.png"]
    cells = projection_cells(case["structure"])
    assert len(cells) == 8 * 10
    assert len(case["clusters"]) == 8 * 10
    assert len(case["result"].assets) == 8 * 10


def test_sparse_internal_translucency_is_not_a_transparent_canvas():
    rgba = np.full((80, 120, 4), 255, np.uint8)
    rgba[:, :, :3] = (40, 80, 120)
    rgba[30:45, 50:70, 3] = 100
    stream = __import__("io").BytesIO()
    Image.fromarray(rgba, "RGBA").save(stream, format="PNG")
    checked = load_and_precheck(stream.getvalue(), ExtractionConfig())
    assert checked.has_alpha
    assert not checked.alpha_valid
