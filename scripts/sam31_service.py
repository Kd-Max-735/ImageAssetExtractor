from __future__ import annotations

import io
import json
import os
import threading
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
from sam3.model_builder import build_sam3_predictor

CHECKPOINT = os.environ.get("SAM31_CHECKPOINT", r"D:\sam3.1\models\sam3.1_multiplex.pt")
WORK_ROOT = Path(os.environ.get("SAM31_WORK_ROOT", r"D:\ImageAssetExtractor\output\sam31-work"))
if not Path(CHECKPOINT).is_file():
    raise RuntimeError(f"SAM 3.1 checkpoint does not exist: {CHECKPOINT}")
WORK_ROOT.mkdir(parents=True, exist_ok=True)

PREDICTOR = build_sam3_predictor(
    checkpoint_path=CHECKPOINT,
    version="sam3.1",
    max_num_objects=1,
    multiplex_count=16,
    use_fa3=False,
    async_loading_frames=False,
)
LOCK = threading.Lock()
app = FastAPI(title="SAM 3.1 Isolated Service", version="1.0")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "SAM 3.1",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda": torch.cuda.is_available(),
        "checkpoint": CHECKPOINT,
    }


@app.post("/segment")
async def segment(
    file: UploadFile = File(...),
    text_prompt: str | None = Form(None),
    box: str | None = Form(None),
) -> StreamingResponse:
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"code": "INVALID_IMAGE", "message": str(exc)}) from exc
    original_size = image.size
    request_dir = WORK_ROOT / uuid.uuid4().hex
    request_dir.mkdir(parents=True, exist_ok=False)
    image.resize((1008, 1008), Image.Resampling.LANCZOS).save(request_dir / "0000.jpg", quality=95)
    kwargs: dict = {"text_str": text_prompt} if text_prompt else {}
    if box:
        try:
            parsed = json.loads(box)
            if len(parsed) != 4 or not all(0 <= float(value) <= 1 for value in parsed):
                raise ValueError("box must contain four normalized values")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail={"code": "INVALID_BOX", "message": str(exc)}) from exc
        kwargs.update(boxes_xywh=[parsed], box_labels=[1])
    if not kwargs:
        raise HTTPException(status_code=422, detail={"code": "PROMPT_REQUIRED", "message": "text_prompt or box is required"})
    with LOCK:
        state = PREDICTOR.model.init_state(resource_path=str(request_dir), async_loading_frames=False)
        _, result = PREDICTOR.model.add_prompt(state, frame_idx=0, **kwargs)
    masks = result["out_binary_masks"]
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks, dtype=bool)
    combined = np.any(masks, axis=0).astype(np.uint8) * 255
    output = Image.fromarray(combined, mode="L").resize(original_size, Image.Resampling.NEAREST)
    stream = io.BytesIO()
    output.save(stream, "PNG")
    stream.seek(0)
    return StreamingResponse(stream, media_type="image/png", headers={"X-Mask-Count": str(len(masks))})

