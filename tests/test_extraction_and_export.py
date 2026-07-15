from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from image_asset_extractor.schemas.asset import Asset, BBox, BackgroundInfo, ExtractionResult
from image_asset_extractor.schemas.config import ExtractionConfig
from image_asset_extractor.imaging.export import render_asset
from image_asset_extractor.services.extraction import ExtractionEngine
from image_asset_extractor.models.base import ModelCapability

from conftest import image_bytes, solid_sheet, transparent_sheet


@pytest.mark.parametrize("kind", ["transparent", "white-black", "white-color", "solid-color"])
def test_common_sheets_extract_two_assets(tmp_path, kind):
    if kind == "transparent":
        image = transparent_sheet()
    elif kind == "white-black":
        image = solid_sheet()
    elif kind == "white-color":
        image = solid_sheet(colored=True)
    else:
        image = solid_sheet((80, 110, 150), colored=True)
    result = ExtractionEngine().extract(image_bytes(image), tmp_path / kind, source_name=f"{kind}.png")
    assert len(result.assets) == 2
    assert [a.index for a in result.assets] == [1, 2]


def test_irregular_layout_sorting(tmp_path):
    image = Image.new("RGBA", (180, 140), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 10, 120, 30), fill="red")
    draw.rectangle((20, 15, 40, 35), fill="green")
    draw.rectangle((60, 90, 80, 110), fill="blue")
    result = ExtractionEngine().extract(image_bytes(image), tmp_path / "irregular")
    centers = [(a.bbox.y, a.bbox.x) for a in result.assets]
    assert len(centers) == 3
    assert result.assets[0].bbox.x < result.assets[1].bbox.x
    assert result.assets[2].bbox.y > result.assets[1].bbox.y


def test_regular_grid_and_close_assets(tmp_path):
    image = Image.new("RGBA", (120, 90), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for x, y in [(5, 5), (30, 5), (5, 40), (30, 40)]:
        draw.rectangle((x, y, x + 18, y + 18), fill=(255, 100, 20, 255))
    result = ExtractionEngine().extract(image_bytes(image), tmp_path / "grid")
    assert len(result.assets) == 4


def test_multi_component_asset_can_cluster(tmp_path):
    image = Image.new("RGBA", (100, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 15, 30, 25), fill="black")
    draw.rectangle((20, 28, 30, 38), fill="black")
    draw.rectangle((20, 41, 30, 51), fill="black")
    config = ExtractionConfig(merge_distance_ratio=0.05)
    result = ExtractionEngine().extract(image_bytes(image), tmp_path / "multi", config)
    assert len(result.assets) == 1
    assert len(result.assets[0].region_ids) == 3


def test_satellites_attach_to_one_body_without_merging_neighbor_icon(tmp_path):
    image = Image.new("RGBA", (260, 150), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((55, 50, 94, 89), fill=(240, 150, 20, 255))
    for box in ((68, 15, 80, 27), (68, 112, 80, 124), (18, 64, 30, 76), (118, 64, 130, 76)):
        draw.ellipse(box, fill=(240, 150, 20, 255))
    draw.rectangle((164, 48, 203, 91), fill=(240, 150, 20, 255))
    result = ExtractionEngine().extract(image_bytes(image), tmp_path / "satellites")
    assert len(result.assets) == 2
    assert sorted(len(asset.region_ids) for asset in result.assets) == [1, 5]


def test_noise_filtered_and_padding_safe(tmp_path):
    image = Image.new("RGBA", (100, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((1, 1, 25, 25), fill="red")
    draw.point((90, 70), fill="white")
    config = ExtractionConfig(min_area_ratio=0.001, padding_ratio=0.2)
    result = ExtractionEngine().extract(image_bytes(image), tmp_path / "padding", config)
    assert len(result.assets) == 1
    box = result.assets[0].bbox
    assert box.x == 0 and box.y == 0 and box.x2 <= 100 and box.y2 <= 80


def test_export_contract_and_alpha(tmp_path):
    root = tmp_path / "output"
    result = ExtractionEngine().extract(image_bytes(transparent_sheet()), root, source_name="sheet.png")
    metadata = json.loads((root / "result" / "metadata.json").read_text("utf-8"))
    assert metadata["algorithmVersion"]
    assert metadata["packageVersion"] == "0.2.0"
    assert metadata["processingConfig"] == {
        "backgroundType": "auto", "assetMode": "icon", "minAreaRatio": 0.0001,
        "mergeDistanceRatio": 0.01, "paddingRatio": 0.05, "backgroundTolerance": 15.0,
        "edgeFeather": 0.0, "splitStrength": 0.5, "useRmbg": False, "useSam31": False,
    }
    with (root / "result" / "metadata.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    pngs = sorted((root / "icons").glob("*.png"))
    assert len(pngs) == metadata["assetCount"] == len(rows) == len(result.assets)
    for index, path in enumerate(pngs, 1):
        rgba = np.asarray(Image.open(path))
        assert rgba.shape[2] == 4
        assert np.any(rgba[:, :, 3] > 0) and np.any(rgba[:, :, 3] < 255)
        assert np.all(rgba[rgba[:, :, 3] == 0, :3] == 0)
        assert path.name == f"icon_{index:03d}.png"
        assert rows[index - 1]["index"] == str(index)
    Image.open(root / "result" / "annotated.png").verify()
    with zipfile.ZipFile(root / "result" / "assets.zip") as archive:
        names = set(archive.namelist())
        assert "result/metadata.json" in names
        assert "result/metadata.csv" in names
        assert "result/annotated.png" in names
        assert all(f"icons/icon_{i:03d}.png" in names for i in range(1, 3))
        assert "result/assets.zip" not in names


def test_low_confidence_background_does_not_amplify_color_fringe():
    source = np.zeros((12, 12, 4), np.uint8)
    source[:, :, :3] = (25, 35, 45)
    source[:, :, 3] = 255
    alpha = np.zeros((12, 12), np.uint8)
    alpha[2:10, 2:10] = 45
    detection = (alpha > 0).astype(np.uint8) * 255
    asset = Asset("asset-001", 1, "icons/icon_001.png", BBox(0, 0, 12, 12), 64, 0.5, [], ["r"], [], "needs-review", detection, alpha)
    result = ExtractionResult("source.jpg", 12, 12, BackgroundInfo("near-solid", 0.5, (10, 10, 10), []), [asset])
    rendered = np.asarray(render_asset(result, asset, source, ExtractionConfig()))
    assert np.array_equal(rendered[5, 5, :3], source[5, 5, :3])


def test_rmbg_refines_each_candidate_with_fallback_isolation(tmp_path, monkeypatch):
    calls = []

    def capability(self):
        return ModelCapability("rmbg", True, True, "service", "configured", ["roi-soft-alpha"])

    def refine(self, rgba, **hints):
        calls.append(rgba.shape)
        return np.full(rgba.shape[:2], 200, dtype=np.uint8)

    monkeypatch.setenv("RMBG_SERVICE_URL", "http://rmbg:8000")
    monkeypatch.setattr("image_asset_extractor.models.rmbg_adapter.RMBGAdapter.capability", capability)
    monkeypatch.setattr("image_asset_extractor.models.rmbg_adapter.RMBGAdapter.refine", refine)
    result = ExtractionEngine().extract(
        image_bytes(solid_sheet()),
        tmp_path / "rmbg",
        ExtractionConfig(use_rmbg=True),
    )
    assert len(calls) == len(result.assets) == 2
    assert all("alpha-refined-by-rmbg" in asset.confidence_reasons for asset in result.assets)


def test_sam31_refines_each_candidate_through_service_adapter(tmp_path, monkeypatch):
    calls = []

    def capability(self):
        return ModelCapability("sam31", True, True, "service", "configured", ["box-prompt"])

    def refine(self, rgba, **hints):
        calls.append((rgba.shape, hints["box"]))
        return np.full(rgba.shape[:2], 255, dtype=np.uint8)

    monkeypatch.setenv("SAM31_SERVICE_URL", "http://sam31:8000")
    monkeypatch.setattr("image_asset_extractor.models.sam31_adapter.SAM31Adapter.capability", capability)
    monkeypatch.setattr("image_asset_extractor.models.sam31_adapter.SAM31Adapter.refine", refine)
    result = ExtractionEngine().extract(
        image_bytes(solid_sheet()),
        tmp_path / "sam31",
        ExtractionConfig(use_sam31=True),
    )
    assert len(calls) == len(result.assets) == 2
    assert all(any(reason.startswith("sam31-") or reason == "mask-refined-by-sam31" for reason in asset.confidence_reasons) for asset in result.assets)


@pytest.mark.parametrize("model", ["rmbg", "sam31"])
def test_model_candidate_that_deletes_subject_falls_back_to_baseline(tmp_path, monkeypatch, model):
    def capability(self):
        return ModelCapability(model, True, True, "service", "configured", ["mask"])

    def refine(self, rgba, **hints):
        mask = np.zeros(rgba.shape[:2], np.uint8)
        h, w = mask.shape
        mask[h // 2:h // 2 + 2, w // 2:w // 2 + 2] = 255
        return mask

    adapter = "rmbg_adapter.RMBGAdapter" if model == "rmbg" else "sam31_adapter.SAM31Adapter"
    monkeypatch.setenv("RMBG_SERVICE_URL" if model == "rmbg" else "SAM31_SERVICE_URL", "http://model:8000")
    monkeypatch.setattr(f"image_asset_extractor.models.{adapter}.capability", capability)
    monkeypatch.setattr(f"image_asset_extractor.models.{adapter}.refine", refine)
    config = ExtractionConfig(use_rmbg=model == "rmbg", use_sam31=model == "sam31")
    result = ExtractionEngine().extract(image_bytes(solid_sheet()), tmp_path / model, config)
    assert all(asset.mask_area > 100 for asset in result.assets)
    assert all(any(f"{model}-fallback-baseline" in reason for reason in asset.confidence_reasons) for asset in result.assets)
    assert all(asset.status == "needs-review" for asset in result.assets)
