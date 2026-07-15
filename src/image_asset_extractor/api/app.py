from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..errors import ExtractorError
from ..services.task_manager import TaskManager
from .routes import router


def create_app(output_dir: str | None = None) -> FastAPI:
    application = FastAPI(title="Image Asset Extractor", version="0.2.0")
    application.state.task_manager = TaskManager(
        output_dir or os.getenv("IMAGE_ASSET_OUTPUT_DIR", "output"),
        max_workers=max(1, int(os.getenv("EXTRACTION_MAX_WORKERS", "2"))),
    )

    @application.exception_handler(ExtractorError)
    async def handle_extractor_error(request: Request, exc: ExtractorError) -> JSONResponse:
        status_code = 404 if exc.code in {"TASK_NOT_FOUND", "ASSET_NOT_FOUND", "FILE_NOT_FOUND"} else (409 if exc.code == "REVISION_CONFLICT" else 400)
        return JSONResponse(status_code=status_code, content=exc.as_dict())

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"code": "INVALID_REQUEST", "message": "Request parameters failed validation", "details": {"errors": exc.errors()}},
        )

    application.include_router(router)
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    application.mount("/static", StaticFiles(directory=frontend), name="static")

    @application.get("/", include_in_schema=False)
    def frontend_index() -> FileResponse:
        return FileResponse(frontend / "index.html")
    return application


app = create_app()
