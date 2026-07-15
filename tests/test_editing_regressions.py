from __future__ import annotations

import csv
import io
import json
import time
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from image_asset_extractor.api.app import create_app
from image_asset_extractor.errors import ExtractorError
from image_asset_extractor.schemas.config import ExtractionConfig
from image_asset_extractor.services.extraction import ExtractionEngine
from image_asset_extractor.services.task_manager import TaskManager
from image_asset_extractor.schemas.asset import BBox

from conftest import image_bytes, transparent_sheet


def create_completed(client: TestClient, image: Image.Image, **fields) -> tuple[str, dict]:
    response = client.post(
        "/api/asset-extraction/tasks",
        files={"file": ("sheet.png", image_bytes(image), "image/png")},
        data=fields,
    )
    assert response.status_code == 202, response.text
    task_id = response.json()["task_id"]
    for _ in range(200):
        task = client.get(f"/api/asset-extraction/tasks/{task_id}").json()
        if task["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    assert task["status"] == "completed", task
    return task_id, task


def test_two_lightly_touching_assets_split_into_two(tmp_path):
    image = Image.new("RGBA", (140, 70), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 15, 51, 54), fill=(220, 40, 40, 255))
    draw.rectangle((72, 15, 111, 54), fill=(40, 90, 220, 255))
    draw.rectangle((52, 33, 71, 35), fill=(120, 70, 120, 255))
    result = ExtractionEngine().extract(image_bytes(image), tmp_path / "touching")
    assert len(result.assets) == 2
    assert all(any(reason.startswith("touching-split:") for reason in asset.confidence_reasons) for asset in result.assets)


def test_tall_icon_and_same_color_narrow_connection_are_not_auto_split(tmp_path):
    tower = Image.new("RGBA", (100, 170), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tower)
    draw.rectangle((30, 20, 69, 149), fill=(100, 80, 220, 255))
    draw.ellipse((42, 75, 57, 90), fill=(0, 0, 0, 0))
    tower_result = ExtractionEngine().extract(image_bytes(tower), tmp_path / "tower")
    assert len(tower_result.assets) == 1

    connected = Image.new("RGBA", (150, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(connected)
    draw.ellipse((10, 15, 59, 64), fill=(40, 130, 220, 255))
    draw.ellipse((90, 15, 139, 64), fill=(40, 130, 220, 255))
    draw.rectangle((59, 37, 90, 41), fill=(40, 130, 220, 255))
    connected_result = ExtractionEngine().extract(image_bytes(connected), tmp_path / "connected")
    assert len(connected_result.assets) == 1
    assert connected_result.assets[0].status == "needs-review"
    assert "possible-split" in connected_result.assets[0].flags

    gray = Image.new("RGBA", (150, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(gray)
    draw.rectangle((10, 15, 59, 64), fill=(70, 70, 70, 255))
    draw.rectangle((90, 15, 139, 64), fill=(135, 135, 135, 255))
    draw.rectangle((59, 38, 90, 41), fill=(100, 100, 100, 255))
    gray_result = ExtractionEngine().extract(image_bytes(gray), tmp_path / "gray-connected")
    assert len(gray_result.assets) == 1


def test_three_disconnected_dots_form_one_icon(tmp_path):
    image = Image.new("RGBA", (90, 70), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for x in (25, 39, 53):
        draw.ellipse((x, 30, x + 5, 35), fill=(20, 20, 20, 255))
    result = ExtractionEngine().extract(
        image_bytes(image), tmp_path / "ellipsis", ExtractionConfig(merge_distance_ratio=0.08, min_area_ratio=0.0001)
    )
    assert len(result.assets) == 1
    assert len(result.assets[0].region_ids) == 3


def test_two_close_same_color_icons_are_not_merged(tmp_path):
    image = Image.new("RGBA", (100, 60), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 15, 39, 39), fill=(30, 120, 220, 255))
    draw.rectangle((43, 15, 62, 39), fill=(30, 120, 220, 255))
    result = ExtractionEngine().extract(image_bytes(image), tmp_path / "close")
    assert len(result.assets) == 2


def test_gradient_near_solid_jpeg_noise_and_low_contrast_paths(tmp_path):
    width, height = 180, 100
    x = np.linspace(215, 245, width, dtype=np.uint8)
    gradient = np.repeat(x[None, :, None], height, axis=0)
    gradient = np.repeat(gradient, 3, axis=2)
    gradient[25:70, 30:70] = (220, 40, 40)
    gradient[20:75, 115:155] = (20, 100, 220)
    result = ExtractionEngine().extract(image_bytes(Image.fromarray(gradient, "RGB")), tmp_path / "gradient")
    assert result.background.type == "gradient" and len(result.assets) == 2

    near = np.full((90, 140, 3), 232, dtype=np.uint8)
    near += np.random.default_rng(3).integers(0, 3, near.shape, dtype=np.uint8)
    near[20:65, 45:95] = (40, 60, 90)
    near_result = ExtractionEngine().extract(image_bytes(Image.fromarray(near, "RGB"), "JPEG"), tmp_path / "near")
    assert near_result.background.type in {"solid", "near-solid"} and near_result.assets

    low = Image.new("RGB", (120, 80), (255, 255, 255))
    ImageDraw.Draw(low).rectangle((30, 20, 85, 60), fill=(250, 250, 250))
    low_result = ExtractionEngine().extract(image_bytes(low), tmp_path / "low")
    assert low_result.background.confidence < 0.6
    assert low_result.assets[0].status == "needs-review"


def test_more_than_one_hundred_assets_and_square_canvas(tmp_path):
    image = Image.new("RGBA", (500, 460), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for row in range(10):
        for column in range(11):
            x, y = 8 + column * 44, 8 + row * 44
            draw.rectangle((x, y, x + 15, y + 15), fill=(30, 80 + row * 5, 180, 255))
    result = ExtractionEngine().extract(
        image_bytes(image), tmp_path / "many", ExtractionConfig(output_canvas="square", canvas_size=40, output_prefix="part", number_digits=4)
    )
    assert len(result.assets) == 110
    assert result.assets[0].file == "icons/part_0001.png"
    exported = Image.open(tmp_path / "many" / result.assets[0].file)
    assert exported.size == (40, 40)
    with pytest.raises(ExtractorError) as error:
        ExtractionEngine().extract(
            image_bytes(image), tmp_path / "over-limit", ExtractionConfig(max_assets=100, include_zip=False)
        )
    assert error.value.code == "TOO_MANY_ASSETS"


def test_delete_removes_png_metadata_and_zip_entry(tmp_path):
    with TestClient(create_app(str(tmp_path / "api"))) as client:
        task_id, task = create_completed(client, transparent_sheet())
        assets = client.get(f"/api/asset-extraction/tasks/{task_id}/assets").json()["assets"]
        deleted = assets[0]
        response = client.post(
            f"/api/asset-extraction/tasks/{task_id}/assets/{deleted['id']}/delete",
            json={"revision": task["revision"]},
        )
        assert response.status_code == 200, response.text
        metadata = client.get(f"/api/asset-extraction/tasks/{task_id}/files/metadata.json").json()
        assert metadata["assetCount"] == 1
        assert deleted["id"] not in {asset["id"] for asset in metadata["assets"]}
        assert not (tmp_path / "api" / task_id / deleted["file"]).exists()
        archive = client.get(f"/api/asset-extraction/tasks/{task_id}/export/download")
        with zipfile.ZipFile(io.BytesIO(archive.content)) as zf:
            assert deleted["file"] not in zf.namelist()


def test_complex_background_is_marked_needs_review(tmp_path):
    rng = np.random.default_rng(7)
    rgb = rng.integers(35, 205, size=(100, 140, 3), dtype=np.uint8)
    rgb[25:75, 45:95] = (245, 230, 30)
    image = Image.fromarray(rgb, "RGB")
    result = ExtractionEngine().extract(image_bytes(image), tmp_path / "complex")
    assert result.background.type in {"complex", "unknown"}
    assert result.background.confidence < 0.6
    assert result.assets and all(asset.status == "needs-review" for asset in result.assets)


def test_mask_edit_changes_alpha_and_all_exports_stay_consistent(tmp_path):
    root = tmp_path / "api"
    with TestClient(create_app(str(root))) as client:
        task_id, task = create_completed(client, transparent_sheet())
        asset = client.get(f"/api/asset-extraction/tasks/{task_id}/assets").json()["assets"][0]
        before = np.asarray(Image.open(io.BytesIO(client.get(
            f"/api/asset-extraction/tasks/{task_id}/assets/{asset['id']}/download"
        ).content)))[:, :, 3]
        center = {"x": asset["x"] + asset["width"] // 2, "y": asset["y"] + asset["height"] // 2}
        response = client.put(
            f"/api/asset-extraction/tasks/{task_id}/assets/{asset['id']}/mask",
            json={"revision": task["revision"], "mode": "erase", "strokes": [{"points": [center], "size": 8, "hardness": 1, "feather": 0}]},
        )
        assert response.status_code == 200, response.text
        after = np.asarray(Image.open(io.BytesIO(client.get(
            f"/api/asset-extraction/tasks/{task_id}/assets/{asset['id']}/download"
        ).content)))[:, :, 3]
        assert np.count_nonzero(after) < np.count_nonzero(before)

        metadata = client.get(f"/api/asset-extraction/tasks/{task_id}/files/metadata.json").json()
        csv_text = client.get(f"/api/asset-extraction/tasks/{task_id}/files/metadata.csv").content.decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        with zipfile.ZipFile(io.BytesIO(client.get(f"/api/asset-extraction/tasks/{task_id}/export/download").content)) as zf:
            names = set(zf.namelist())
        disk_pngs = list((root / task_id / "icons").glob("*.png"))
        assert metadata["assetCount"] == len(rows) == len(disk_pngs)
        assert {item["file"] for item in metadata["assets"]} == {row["file"] for row in rows}
        assert {item["file"] for item in metadata["assets"]} <= names
        Image.open(root / task_id / "result" / "annotated.png").verify()


def test_revision_conflict_undo_redo_merge_split_bbox_rename_and_sort(tmp_path):
    with TestClient(create_app(str(tmp_path / "api"))) as client:
        task_id, task = create_completed(client, transparent_sheet())
        assets = client.get(f"/api/asset-extraction/tasks/{task_id}/assets").json()["assets"]
        conflict = client.post(
            f"/api/asset-extraction/tasks/{task_id}/assets/{assets[0]['id']}/delete",
            json={"revision": task["revision"] + 1},
        )
        assert conflict.status_code == 409 and conflict.json()["code"] == "REVISION_CONFLICT"

        merged = client.post(
            f"/api/asset-extraction/tasks/{task_id}/merge",
            json={"revision": task["revision"], "assetIds": [asset["id"] for asset in assets]},
        ).json()
        assert merged["assetCount"] == 1 and merged["revision"] == task["revision"] + 1
        undone = client.post(f"/api/asset-extraction/tasks/{task_id}/undo", json={"revision": merged["revision"]}).json()
        assert undone["assetCount"] == 2
        redone = client.post(f"/api/asset-extraction/tasks/{task_id}/redo", json={"revision": undone["revision"]}).json()
        assert redone["assetCount"] == 1
        restored = client.post(f"/api/asset-extraction/tasks/{task_id}/undo", json={"revision": redone["revision"]}).json()
        assets = client.get(f"/api/asset-extraction/tasks/{task_id}/assets").json()["assets"]

        renamed = client.put(
            f"/api/asset-extraction/tasks/{task_id}/assets/{assets[0]['id']}/name",
            json={"revision": restored["revision"], "name": "custom.png"},
        ).json()
        ordered = client.post(
            f"/api/asset-extraction/tasks/{task_id}/sort",
            json={"revision": renamed["revision"], "assetIds": [assets[1]["id"], assets[0]["id"]]},
        ).json()
        current = client.get(f"/api/asset-extraction/tasks/{task_id}/assets").json()["assets"]
        assert current[0]["id"] == assets[1]["id"] and any(a["file"] == "icons/custom.png" for a in current)

        target = current[0]
        bbox = {"x": target["x"], "y": target["y"], "width": target["width"] + 1, "height": target["height"] + 1}
        updated = client.put(
            f"/api/asset-extraction/tasks/{task_id}/assets/{target['id']}/bbox",
            json={"revision": ordered["revision"], "bbox": bbox},
        )
        assert updated.status_code == 200, updated.text


def test_manual_split_line_changes_real_exports(tmp_path):
    image = Image.new("RGBA", (120, 70), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((20, 15, 95, 55), fill=(50, 120, 230, 255))
    with TestClient(create_app(str(tmp_path / "api"))) as client:
        task_id, task = create_completed(client, image)
        asset = client.get(f"/api/asset-extraction/tasks/{task_id}/assets").json()["assets"][0]
        before = np.asarray(Image.open(io.BytesIO(client.get(
            f"/api/asset-extraction/tasks/{task_id}/assets/{asset['id']}/download"
        ).content)))[:, :, 3]
        response = client.post(
            f"/api/asset-extraction/tasks/{task_id}/split",
            json={
                "revision": task["revision"], "assetId": asset["id"], "mode": "line",
                "line": {"x1": 58, "y1": 10, "x2": 58, "y2": 60, "width": 3},
            },
        )
        assert response.status_code == 200, response.text
        metadata = client.get(f"/api/asset-extraction/tasks/{task_id}/files/metadata.json").json()
        assert metadata["assetCount"] == 2
        assert all(item["status"] == "manually-edited" for item in metadata["assets"])
        after_area = 0
        for item in metadata["assets"]:
            alpha = np.asarray(Image.open(io.BytesIO(client.get(
                f"/api/asset-extraction/tasks/{task_id}/assets/{item['id']}/download"
            ).content)))[:, :, 3]
            after_area += np.count_nonzero(alpha)
        assert after_area == np.count_nonzero(before)


def test_bbox_restore_preserves_edits_and_restores_only_new_source_area(tmp_path):
    image = Image.new("RGBA", (120, 90), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((20, 20, 90, 65), fill=(40, 120, 220, 255))
    with TestClient(create_app(str(tmp_path / "api"))) as client:
        task_id, task = create_completed(client, image)
        original = client.get(f"/api/asset-extraction/tasks/{task_id}/assets").json()["assets"][0]
        shrunken_box = {
            "x": original["x"] + 10, "y": original["y"],
            "width": original["width"] - 20, "height": original["height"],
        }
        task = client.put(
            f"/api/asset-extraction/tasks/{task_id}/assets/{original['id']}/bbox",
            json={"revision": task["revision"], "bbox": shrunken_box},
        ).json()
        center = {"x": 55, "y": 42}
        task = client.put(
            f"/api/asset-extraction/tasks/{task_id}/assets/{original['id']}/mask",
            json={"revision": task["revision"], "mode": "erase", "strokes": [{"points": [center], "size": 10, "hardness": 1, "feather": 0}]},
        ).json()
        restored = client.put(
            f"/api/asset-extraction/tasks/{task_id}/assets/{original['id']}/bbox",
            json={
                "revision": task["revision"], "restoreFromSource": True,
                "bbox": {key: original[key] for key in ("x", "y", "width", "height")},
            },
        ).json()
        alpha = np.asarray(Image.open(io.BytesIO(client.get(
            f"/api/asset-extraction/tasks/{task_id}/assets/{original['id']}/download"
        ).content)))[:, :, 3]
        assert alpha[30 - original["y"], 22 - original["x"]] > 0
        assert alpha[center["y"] - original["y"], center["x"] - original["x"]] == 0
        undone = client.post(f"/api/asset-extraction/tasks/{task_id}/undo", json={"revision": restored["revision"]}).json()
        redone = client.post(f"/api/asset-extraction/tasks/{task_id}/redo", json={"revision": undone["revision"]}).json()
        assert redone["revision"] == restored["revision"] + 2
        metadata = client.get(f"/api/asset-extraction/tasks/{task_id}/files/metadata.json").json()
        assert metadata["assets"][0]["maskArea"] == int(np.count_nonzero(alpha))


def test_brush_hardness_has_hard_core_and_soft_edge():
    layer = TaskManager._brush_layer(
        (41, 41), [{"x": 20, "y": 20}], radius=10, hardness=0.5, feather=0, bbox=BBox(0, 0, 41, 41)
    )
    assert layer[20, 20] == 255
    assert layer[20, 24] == 255
    assert 0 < layer[20, 29] < 255
    assert layer[20, 31] == 0


def test_include_zip_false_contract_and_frontend(tmp_path):
    with TestClient(create_app(str(tmp_path / "api"))) as client:
        task_id, _ = create_completed(client, transparent_sheet(), includeZip="false")
        assert client.get(f"/api/asset-extraction/tasks/{task_id}/export/download").status_code == 400
        no_zip = client.post(f"/api/asset-extraction/tasks/{task_id}/export", json={"includeZip": False}).json()
        assert "downloadUrl" not in no_zip
        with_zip = client.post(f"/api/asset-extraction/tasks/{task_id}/export", json={"includeZip": True}).json()
        assert client.get(with_zip["downloadUrl"]).status_code == 200
        asset = client.get(f"/api/asset-extraction/tasks/{task_id}/assets").json()["assets"][0]
        for background in ("transparent", "checker", "white", "black"):
            preview = client.get(f"/api/asset-extraction/tasks/{task_id}/assets/{asset['id']}/preview?background={background}")
            assert preview.status_code == 200
            Image.open(io.BytesIO(preview.content)).verify()
        page = client.get("/")
        script = client.get("/static/app.js")
        styles = client.get("/static/app.css")
        assert page.status_code == script.status_code == styles.status_code == 200
        assert "uploadForm" in page.text and "/api/asset-extraction/tasks" in script.text
        assert 'class="asset-thumb"' in script.text
        assert ".asset-thumb img" in styles.text and "width:auto;height:auto" in styles.text
        assert "image-rendering:pixelated" not in styles.text
        for text in ("X（左上角横坐标，px）", "Y（左上角纵坐标，px）", "宽度（px）", "高度（px）", "坐标以原图左上角为 (0, 0)，单位为像素。"):
            assert text in page.text
        for element_id in ("bx", "by", "bw", "bh", "saveBbox"):
            assert f'id="{element_id}"' in page.text


def test_structured_parameter_validation(tmp_path):
    with TestClient(create_app(str(tmp_path / "api"))) as client:
        response = client.post(
            "/api/asset-extraction/tasks",
            files={"file": ("sheet.png", image_bytes(transparent_sheet()), "image/png")},
            data={"assetMode": "not-real"},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "INVALID_PARAMETER"

        disguised = client.post(
            "/api/asset-extraction/tasks",
            files={"file": ("fake.jpg", image_bytes(transparent_sheet()), "image/jpeg")},
        )
        task_id = disguised.json()["task_id"]
        for _ in range(100):
            task = client.get(f"/api/asset-extraction/tasks/{task_id}").json()
            if task["status"] == "failed":
                break
            time.sleep(0.01)
        assert task["error"]["code"] == "FORMAT_MISMATCH"


def test_cancel_reaches_engine_safe_checkpoint(tmp_path, monkeypatch):
    app = create_app(str(tmp_path / "api"))

    def slow_extract(*args, progress=None, cancel_check=None, **kwargs):
        for step in range(100):
            if progress:
                progress(step, "synthetic-slow-stage")
            if cancel_check and cancel_check():
                raise ExtractorError("TASK_CANCELLED", "cancelled at safe checkpoint")
            time.sleep(0.005)
        raise AssertionError("cancellation did not reach the engine")

    monkeypatch.setattr(app.state.task_manager.engine, "extract", slow_extract)
    with TestClient(app) as client:
        created = client.post(
            "/api/asset-extraction/tasks",
            files={"file": ("sheet.png", image_bytes(transparent_sheet()), "image/png")},
        ).json()
        task_id = created["task_id"]
        cancelled = client.delete(f"/api/asset-extraction/tasks/{task_id}")
        assert cancelled.status_code == 200
        for _ in range(100):
            task = client.get(f"/api/asset-extraction/tasks/{task_id}").json()
            if task["status"] == "cancelled":
                break
            time.sleep(0.01)
        assert task["status"] == "cancelled"
