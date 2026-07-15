from __future__ import annotations

import json
import subprocess
import sys
import time

from fastapi.testclient import TestClient

from image_asset_extractor.api.app import create_app
from image_asset_extractor.models import RMBGAdapter, SAM31Adapter

from conftest import image_bytes, transparent_sheet


def test_model_capabilities_are_clear(monkeypatch):
    monkeypatch.delenv("RMBG_MODEL_PATH", raising=False)
    assert not RMBGAdapter(True).capability().available
    sam = SAM31Adapter(False).capability()
    assert not sam.available and "disabled" in sam.reason


def test_cli_success_and_failure(tmp_path):
    source = tmp_path / "input.png"
    source.write_bytes(image_bytes(transparent_sheet()))
    output = tmp_path / "cli-output"
    success = subprocess.run(
        [sys.executable, "-m", "image_asset_extractor.cli", "extract", str(source), "--output", str(output)],
        capture_output=True, text=True, check=False,
    )
    assert success.returncode == 0
    assert json.loads(success.stdout)["assetCount"] == 2
    failure = subprocess.run(
        [sys.executable, "-m", "image_asset_extractor.cli", "extract", str(tmp_path / "missing.png"), "--output", str(tmp_path / "bad")],
        capture_output=True, text=True, check=False,
    )
    assert failure.returncode != 0
    assert json.loads(failure.stderr)["code"] == "IMAGE_UNREADABLE"


def test_health_capabilities_and_async_task(tmp_path):
    app = create_app(str(tmp_path / "api-output"))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        capabilities = client.get("/capabilities").json()
        assert capabilities["opencv"]["available"]
        assert set(capabilities) == {"opencv", "rmbg", "sam31"}
        response = client.post(
            "/api/asset-extraction/tasks",
            files={"file": ("sheet.png", image_bytes(transparent_sheet()), "image/png")},
        )
        assert response.status_code == 202
        task_id = response.json()["task_id"]
        for _ in range(100):
            task = client.get(f"/api/asset-extraction/tasks/{task_id}").json()
            if task["status"] in {"completed", "failed"}:
                break
            time.sleep(0.02)
        assert task["status"] == "completed", task
        assets = client.get(f"/api/asset-extraction/tasks/{task_id}/assets").json()
        assert assets["assetCount"] == 2
        export = client.post(f"/api/asset-extraction/tasks/{task_id}/export")
        assert export.status_code == 200
        download = client.get(export.json()["downloadUrl"])
        assert download.status_code == 200
        assert download.headers["content-type"] == "application/zip"


def test_api_structured_error(tmp_path):
    client = TestClient(create_app(str(tmp_path / "api-output")))
    response = client.get("/api/asset-extraction/tasks/not-found")
    assert response.status_code == 404
    assert set(response.json()) == {"code", "message", "details"}
