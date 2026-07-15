from __future__ import annotations

import io
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/RMBG20")
if not os.path.isdir(MODEL_PATH):
    raise RuntimeError(f"RMBG model directory does not exist: {MODEL_PATH}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = AutoModelForImageSegmentation.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)
if DEVICE == "cuda":
    torch.set_float32_matmul_precision("high")
MODEL.to(DEVICE).eval()

TRANSFORM = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = FastAPI(title="RMBG-2.0 Isolated Service", version="1.0")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "RMBG-2.0",
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "modelPath": MODEL_PATH,
    }


def remove_background(image: Image.Image) -> Image.Image:
    original_size = image.size
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        prediction = MODEL(tensor)[-1].sigmoid().cpu()[0].squeeze()
    mask = transforms.ToPILImage()(prediction).resize(original_size, Image.Resampling.LANCZOS)
    result = image.convert("RGBA")
    result.putalpha(mask)
    return result


async def process_upload(file: UploadFile) -> StreamingResponse:
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"code": "INVALID_IMAGE", "message": str(exc)}) from exc
    result = remove_background(image)
    stream = io.BytesIO()
    result.save(stream, "PNG", optimize=True)
    stream.seek(0)
    return StreamingResponse(stream, media_type="image/png")


@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)) -> StreamingResponse:
    return await process_upload(file)


@app.post("/remove-background", include_in_schema=False)
async def remove_background_alias(file: UploadFile = File(...)) -> StreamingResponse:
    return await process_upload(file)

