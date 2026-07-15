from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile, status
from fastapi.responses import FileResponse, Response

from ..schemas.config import ExtractionConfig

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/capabilities")
def capabilities(request: Request) -> dict:
    return request.app.state.task_manager.capabilities()


@router.post("/api/asset-extraction/tasks", status_code=status.HTTP_202_ACCEPTED)
async def create_task(
    request: Request,
    file: UploadFile = File(...),
    backgroundType: str = Form("auto"),
    assetMode: str = Form("icon"),
    shadowMode: str = Form("preserve"),
    textMode: str = Form("auto"),
    minAreaRatio: float = Form(0.0001),
    mergeDistanceRatio: float = Form(0.01),
    paddingRatio: float = Form(0.05),
    backgroundTolerance: float = Form(15),
    outputPrefix: str = Form("icon"),
    numberDigits: int = Form(3),
    outputCanvas: str = Form("tight"),
    outputFormat: str = Form("png"),
    canvasSize: int | None = Form(None),
    edgeFeather: float = Form(0),
    splitStrength: float = Form(0.5),
    includeZip: bool = Form(True),
    useRmbg: bool = Form(False),
    useSam31: bool = Form(False),
) -> dict:
    content = await file.read()
    config = ExtractionConfig(
        background_type=backgroundType, asset_mode=assetMode, shadow_mode=shadowMode, text_mode=textMode,
        min_area_ratio=minAreaRatio, merge_distance_ratio=mergeDistanceRatio,
        padding_ratio=paddingRatio, background_tolerance=backgroundTolerance,
        output_prefix=outputPrefix, number_digits=numberDigits, output_canvas=outputCanvas, output_format=outputFormat,
        canvas_size=canvasSize, edge_feather=edgeFeather, split_strength=splitStrength, include_zip=includeZip,
        use_rmbg=useRmbg, use_sam31=useSam31,
    )
    config.validate()
    return request.app.state.task_manager.create(content, file.filename or "upload.png", config).as_dict()


@router.get("/api/asset-extraction/tasks/{task_id}")
def get_task(task_id: str, request: Request) -> dict:
    return request.app.state.task_manager.get(task_id).as_dict()


@router.get("/api/asset-extraction/tasks/{task_id}/assets")
def get_assets(task_id: str, request: Request) -> dict:
    assets = request.app.state.task_manager.assets(task_id)
    return {"task_id": task_id, "assetCount": len(assets), "assets": assets}


@router.post("/api/asset-extraction/tasks/{task_id}/export")
def export_task(task_id: str, request: Request, payload: dict | None = None) -> dict:
    return request.app.state.task_manager.export(task_id, bool((payload or {}).get("includeZip", True)))


@router.get("/api/asset-extraction/tasks/{task_id}/export/download")
def download_export(task_id: str, request: Request) -> FileResponse:
    return FileResponse(request.app.state.task_manager.export_path(task_id), media_type="application/zip", filename="assets.zip")


@router.get("/api/asset-extraction/tasks/{task_id}/files/{name}")
def download_result_file(task_id: str, name: str, request: Request) -> FileResponse:
    return FileResponse(request.app.state.task_manager.result_file(task_id, name), filename=name)


@router.get("/api/asset-extraction/tasks/{task_id}/source")
def source_image(task_id: str, request: Request) -> Response:
    content, name = request.app.state.task_manager.source(task_id)
    suffix = name.lower().rsplit(".", 1)[-1]
    media = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(suffix, "application/octet-stream")
    return Response(content, media_type=media)


@router.get("/api/asset-extraction/tasks/{task_id}/assets/{asset_id}/download")
def download_asset(task_id: str, asset_id: str, request: Request) -> FileResponse:
    path = request.app.state.task_manager.asset_file(task_id, asset_id)
    return FileResponse(path, media_type="image/png", filename=path.name)


@router.get("/api/asset-extraction/tasks/{task_id}/assets/{asset_id}/preview")
def preview_asset(task_id: str, asset_id: str, request: Request, background: str = "checker") -> Response:
    return Response(request.app.state.task_manager.preview(task_id, asset_id, background), media_type="image/png")


@router.delete("/api/asset-extraction/tasks/{task_id}")
def cancel_task(task_id: str, request: Request) -> dict:
    return request.app.state.task_manager.cancel(task_id).as_dict()


@router.post("/api/asset-extraction/tasks/{task_id}/retry", status_code=status.HTTP_202_ACCEPTED)
def retry_task(task_id: str, request: Request) -> dict:
    return request.app.state.task_manager.retry(task_id).as_dict()


@router.post("/api/asset-extraction/tasks/{task_id}/assets/{asset_id}/delete")
def delete_asset(task_id: str, asset_id: str, request: Request, payload: dict) -> dict:
    return request.app.state.task_manager.delete_asset(task_id, asset_id, payload.get("revision")).as_dict()


@router.post("/api/asset-extraction/tasks/{task_id}/merge")
def merge_assets(task_id: str, request: Request, payload: dict) -> dict:
    return request.app.state.task_manager.merge_assets(task_id, payload.get("assetIds") or [], payload.get("revision")).as_dict()


@router.post("/api/asset-extraction/tasks/{task_id}/split")
def split_asset(task_id: str, request: Request, payload: dict) -> dict:
    return request.app.state.task_manager.split_asset(task_id, payload.get("assetId", ""), payload, payload.get("revision")).as_dict()


@router.put("/api/asset-extraction/tasks/{task_id}/assets/{asset_id}/bbox")
def update_bbox(task_id: str, asset_id: str, request: Request, payload: dict) -> dict:
    return request.app.state.task_manager.update_bbox(
        task_id, asset_id, payload.get("bbox") or payload, payload.get("revision"),
        bool(payload.get("restoreFromSource", False)),
    ).as_dict()


@router.put("/api/asset-extraction/tasks/{task_id}/assets/{asset_id}/mask")
def update_mask(task_id: str, asset_id: str, request: Request, payload: dict) -> dict:
    return request.app.state.task_manager.update_mask(task_id, asset_id, payload, payload.get("revision")).as_dict()


@router.post("/api/asset-extraction/tasks/{task_id}/sort")
def sort_assets(task_id: str, request: Request, payload: dict) -> dict:
    return request.app.state.task_manager.reorder(task_id, payload.get("assetIds") or [], payload.get("revision")).as_dict()


@router.put("/api/asset-extraction/tasks/{task_id}/assets/{asset_id}/name")
def rename_asset(task_id: str, asset_id: str, request: Request, payload: dict) -> dict:
    return request.app.state.task_manager.rename(task_id, asset_id, str(payload.get("name", "")), payload.get("revision")).as_dict()


@router.post("/api/asset-extraction/tasks/{task_id}/undo")
def undo(task_id: str, request: Request, payload: dict) -> dict:
    return request.app.state.task_manager.undo(task_id, payload.get("revision")).as_dict()


@router.post("/api/asset-extraction/tasks/{task_id}/redo")
def redo(task_id: str, request: Request, payload: dict) -> dict:
    return request.app.state.task_manager.redo(task_id, payload.get("revision")).as_dict()
