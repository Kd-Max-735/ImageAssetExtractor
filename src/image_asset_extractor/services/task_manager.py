from __future__ import annotations

import os
import threading
import uuid
import base64
import copy
from io import BytesIO
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..errors import ExtractorError
from ..models import RMBGAdapter, SAM31Adapter
from ..schemas.config import ExtractionConfig
from ..schemas.asset import Asset, BBox, ExtractionResult
from ..schemas.task import TaskRecord, utc_now
from ..imaging.export import export_result, render_asset
from ..imaging.background import classify_background
from ..imaging.foreground import segment_foreground
from ..imaging.precheck import load_and_precheck
from ..imaging.splitting import split_touching_mask
from .extraction import ExtractionEngine


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class TaskManager:
    def __init__(self, output_root: str | Path = "output", max_workers: int = 2):
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="asset-extraction")
        self._tasks: dict[str, TaskRecord] = {}
        self._sources: dict[str, tuple[bytes, str, ExtractionConfig]] = {}
        self._futures: dict[str, Future] = {}
        self._results: dict[str, ExtractionResult] = {}
        self._history: dict[str, list[list[Asset]]] = {}
        self._redo: dict[str, list[list[Asset]]] = {}
        self._edit_locks: dict[str, threading.RLock] = {}
        self._lock = threading.RLock()
        self.engine = ExtractionEngine()

    def create(self, content: bytes, filename: str, config: ExtractionConfig | None = None) -> TaskRecord:
        task_id = uuid.uuid4().hex
        record = TaskRecord(id=task_id, source_file=Path(filename).name, output_dir=str(self.output_root / task_id))
        cfg = config or ExtractionConfig()
        with self._lock:
            self._tasks[task_id] = record
            self._sources[task_id] = (content, filename, cfg)
            self._history[task_id], self._redo[task_id] = [], []
            self._edit_locks[task_id] = threading.RLock()
            self._futures[task_id] = self._executor.submit(self._run, task_id)
        return record

    def _run(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks[task_id]
            if task.cancelled:
                return
            task.status, task.progress, task.stage, task.updated_at = "processing", 1, "starting", utc_now()
            content, filename, config = self._sources[task_id]
        try:
            def progress(value: int, stage: str) -> None:
                with self._lock:
                    task.progress, task.stage, task.updated_at = value, stage, utc_now()

            result = self.engine.extract(
                content, task.output_dir, config, source_name=Path(filename).name,
                progress=progress, cancel_check=lambda: task.cancelled,
            )
            with self._lock:
                if task.cancelled:
                    task.status, task.stage = "cancelled", "cancelled"
                else:
                    self._results[task_id] = result
                    self._refresh_record(task_id)
                    task.status, task.progress, task.stage = "completed", 100, "completed"
                task.updated_at = utc_now()
        except ExtractorError as exc:
            with self._lock:
                if exc.code == "TASK_CANCELLED" or task.cancelled:
                    task.status, task.stage, task.error = "cancelled", "cancelled", None
                else:
                    task.status, task.stage, task.error = "failed", "failed", exc.as_dict()
                task.updated_at = utc_now()
        except Exception as exc:  # pragma: no cover - defensive API boundary
            with self._lock:
                task.status, task.stage = "failed", "failed"
                task.error = {"code": "INTERNAL_ERROR", "message": "Extraction failed", "details": {"reason": str(exc)}}
                task.updated_at = utc_now()

    def get(self, task_id: str) -> TaskRecord:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise ExtractorError("TASK_NOT_FOUND", "Task does not exist", {"task_id": task_id}) from exc

    def assets(self, task_id: str) -> list[dict[str, Any]]:
        task = self.get(task_id)
        if task.status not in {"completed", "editing"}:
            raise ExtractorError("TASK_NOT_READY", "Task assets are not ready", {"status": task.status})
        return list(task.assets)

    def _refresh_record(self, task_id: str) -> None:
        task, result = self._tasks[task_id], self._results[task_id]
        task.assets = [asset.metadata() for asset in result.assets]
        task.background_type = result.background.type
        task.background_confidence = round(result.background.confidence, 4)

    def _checked_result(self, task_id: str, revision: int | None = None) -> tuple[TaskRecord, ExtractionResult]:
        task = self.get(task_id)
        if task.status not in {"completed", "editing"} or task_id not in self._results:
            raise ExtractorError("TASK_NOT_READY", "Task is not ready for editing", {"status": task.status})
        if revision is None:
            raise ExtractorError("REVISION_REQUIRED", "The current task revision is required")
        if revision != task.revision:
            raise ExtractorError("REVISION_CONFLICT", "Task was modified by another editor", {"expected": task.revision, "received": revision})
        return task, self._results[task_id]

    def _edit_lock(self, task_id: str) -> threading.RLock:
        self.get(task_id)
        return self._edit_locks[task_id]

    @staticmethod
    def _find_asset(result: ExtractionResult, asset_id: str) -> Asset:
        for asset in result.assets:
            if asset.id == asset_id:
                return asset
        raise ExtractorError("ASSET_NOT_FOUND", "Asset does not exist", {"asset_id": asset_id})

    def _push_history(self, task_id: str, result: ExtractionResult) -> None:
        self._history[task_id].append(copy.deepcopy(result.assets))
        self._history[task_id] = self._history[task_id][-50:]
        self._redo[task_id].clear()

    def _source_rgba(self, task_id: str) -> np.ndarray:
        content, name, config = self._sources[task_id]
        return load_and_precheck(content, config, name).rgba

    def _source_segmentation(self, task_id: str) -> tuple[np.ndarray, np.ndarray]:
        content, name, config = self._sources[task_id]
        image = load_and_precheck(content, config, name)
        background = classify_background(image, config.background_type)
        return segment_foreground(image, background, config)

    def _regenerate(self, task_id: str, include_zip: bool | None = None) -> Path:
        task, result = self._tasks[task_id], self._results[task_id]
        config = copy.copy(self._sources[task_id][2])
        if include_zip is not None:
            config.include_zip = include_zip
        task.status, task.stage, task.progress = "editing", "exporting", 92
        export_result(result, self._source_rgba(task_id), task.output_dir, config.include_zip, config=config, overwrite=True)
        task.status, task.stage, task.progress, task.updated_at = "completed", "completed", 100, utc_now()
        self._refresh_record(task_id)
        return Path(task.output_dir)

    def _commit(self, task_id: str) -> TaskRecord:
        task = self._tasks[task_id]
        task.revision += 1
        task.updated_at = utc_now()
        self._regenerate(task_id)
        return task

    def export_path(self, task_id: str, *, ensure_zip: bool = False) -> Path:
        task = self.get(task_id)
        if task.status != "completed":
            raise ExtractorError("TASK_NOT_READY", "Task is not ready for export", {"status": task.status})
        path = Path(task.output_dir) / "result" / "assets.zip"
        if ensure_zip and not path.is_file():
            with self._edit_lock(task_id):
                self._regenerate(task_id, include_zip=True)
        if not path.is_file():
            raise ExtractorError("ZIP_NOT_REQUESTED", "ZIP was disabled; call the export endpoint with includeZip=true")
        return path

    def export(self, task_id: str, include_zip: bool = True) -> dict[str, Any]:
        self.get(task_id)
        with self._edit_lock(task_id):
            root = self._regenerate(task_id, include_zip=include_zip)
        response = {
            "task_id": task_id,
            "status": "completed",
            "revision": self._tasks[task_id].revision,
            "metadataJsonUrl": f"/api/asset-extraction/tasks/{task_id}/files/metadata.json",
            "metadataCsvUrl": f"/api/asset-extraction/tasks/{task_id}/files/metadata.csv",
            "annotatedUrl": f"/api/asset-extraction/tasks/{task_id}/files/annotated.png",
        }
        if include_zip:
            response["downloadUrl"] = f"/api/asset-extraction/tasks/{task_id}/export/download"
            response["file"] = str(root / "result" / "assets.zip")
        return response

    def result_file(self, task_id: str, name: str) -> Path:
        allowed = {"metadata.json", "metadata.csv", "annotated.png"}
        if name not in allowed:
            raise ExtractorError("FILE_NOT_FOUND", "Requested result file is not available")
        path = Path(self.get(task_id).output_dir) / "result" / name
        if not path.is_file():
            raise ExtractorError("FILE_NOT_FOUND", "Requested result file is not available", {"name": name})
        return path

    def asset_file(self, task_id: str, asset_id: str) -> Path:
        result = self._results.get(task_id)
        if result is None:
            raise ExtractorError("TASK_NOT_READY", "Task is not ready")
        asset = self._find_asset(result, asset_id)
        path = Path(self._tasks[task_id].output_dir) / asset.file
        if not path.is_file():
            raise ExtractorError("FILE_NOT_FOUND", "Asset PNG is unavailable")
        return path

    def source(self, task_id: str) -> tuple[bytes, str]:
        self.get(task_id)
        content, name, _ = self._sources[task_id]
        return content, name

    def preview(self, task_id: str, asset_id: str, background: str = "checker") -> bytes:
        if background not in {"transparent", "checker", "black", "white"}:
            raise ExtractorError("INVALID_PARAMETER", "Unsupported preview background")
        result = self._results.get(task_id)
        if result is None:
            raise ExtractorError("TASK_NOT_READY", "Task is not ready")
        asset = self._find_asset(result, asset_id)
        config = self._sources[task_id][2]
        image = render_asset(result, asset, self._source_rgba(task_id), config)
        if background != "transparent":
            if background == "checker":
                yy, xx = np.mgrid[:image.height, :image.width]
                shade = np.where(((xx // 12 + yy // 12) % 2)[..., None] == 0, 220, 175).astype(np.uint8)
                base = np.repeat(shade, 3, axis=2)
                canvas = Image.fromarray(base, "RGB").convert("RGBA")
            else:
                value = 0 if background == "black" else 255
                canvas = Image.new("RGBA", image.size, (value, value, value, 255))
            canvas.alpha_composite(image)
            image = canvas.convert("RGB")
        stream = BytesIO()
        image.save(stream, "PNG")
        return stream.getvalue()

    def cancel(self, task_id: str) -> TaskRecord:
        task = self.get(task_id)
        with self._lock:
            task.cancelled = True
            if task.status in {"pending", "processing"}:
                task.status = "cancelled"
                task.stage = "cancelling"
                future = self._futures.get(task_id)
                if future:
                    future.cancel()
            task.updated_at = utc_now()
        return task

    def retry(self, task_id: str) -> TaskRecord:
        old = self.get(task_id)
        if old.status not in {"failed", "cancelled"}:
            raise ExtractorError("TASK_NOT_RETRYABLE", "Only failed or cancelled tasks can be retried")
        content, filename, config = self._sources[task_id]
        return self.create(content, filename, config)

    def delete_asset(self, task_id: str, asset_id: str, revision: int | None) -> TaskRecord:
        with self._edit_lock(task_id):
            task, result = self._checked_result(task_id, revision)
            self._find_asset(result, asset_id)
            self._push_history(task_id, result)
            result.assets = [asset for asset in result.assets if asset.id != asset_id]
            return self._commit(task_id)

    def merge_assets(self, task_id: str, asset_ids: list[str], revision: int | None) -> TaskRecord:
        if len(set(asset_ids)) < 2:
            raise ExtractorError("INVALID_PARAMETER", "At least two distinct assets are required")
        with self._edit_lock(task_id):
            task, result = self._checked_result(task_id, revision)
            selected = [self._find_asset(result, asset_id) for asset_id in asset_ids]
            self._push_history(task_id, result)
            merged = copy.deepcopy(selected[0])
            x1 = min(asset.bbox.x for asset in selected)
            y1 = min(asset.bbox.y for asset in selected)
            x2 = max(asset.bbox.x2 for asset in selected)
            y2 = max(asset.bbox.y2 for asset in selected)
            merged.bbox = BBox(x1, y1, x2 - x1, y2 - y1)
            merged.detection_mask = np.zeros((merged.bbox.height, merged.bbox.width), np.uint8)
            merged.alpha_mask = np.zeros_like(merged.detection_mask)
            for item in selected:
                ox, oy = item.bbox.x - x1, item.bbox.y - y1
                area = np.s_[oy:oy + item.bbox.height, ox:ox + item.bbox.width]
                merged.detection_mask[area] = np.maximum(merged.detection_mask[area], item.detection_mask)
                merged.alpha_mask[area] = np.maximum(merged.alpha_mask[area], item.alpha_mask)
            merged.mask_area = int(np.count_nonzero(merged.alpha_mask))
            merged.region_ids = sorted({rid for asset in selected for rid in asset.region_ids})
            merged.status, merged.confidence = "manually-edited", min(asset.confidence for asset in selected)
            merged.confidence_reasons.append("manually-merged")
            first = min(result.assets.index(asset) for asset in selected)
            result.assets = [asset for asset in result.assets if asset.id not in set(asset_ids)]
            result.assets.insert(first, merged)
            self._renumber(result)
            return self._commit(task_id)

    @staticmethod
    def _renumber(result: ExtractionResult) -> None:
        for index, asset in enumerate(result.assets, 1):
            asset.index = index

    def split_asset(self, task_id: str, asset_id: str, payload: dict[str, Any], revision: int | None) -> TaskRecord:
        with self._edit_lock(task_id):
            task, result = self._checked_result(task_id, revision)
            asset = self._find_asset(result, asset_id)
            work = asset.detection_mask.copy()
            alpha = asset.alpha_mask.copy()
            mode = payload.get("mode", "auto")
            parts: list[np.ndarray]
            if mode == "line":
                line = payload.get("line") or {}
                try:
                    p1 = (int(line["x1"]) - asset.bbox.x, int(line["y1"]) - asset.bbox.y)
                    p2 = (int(line["x2"]) - asset.bbox.x, int(line["y2"]) - asset.bbox.y)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ExtractorError("INVALID_PARAMETER", "A valid split line is required") from exc
                original = work.copy()
                cv2.line(work, p1, p2, 0, max(1, int(line.get("width", 3))))
                seeds = self._component_masks(work)
                parts = self._assign_foreground_to_seeds(original, seeds)
            elif mode == "rectangles":
                parts = []
                for rectangle in payload.get("rectangles", []):
                    x, y, width, height = (int(rectangle[k]) for k in ("x", "y", "width", "height"))
                    x, y = x - asset.bbox.x, y - asset.bbox.y
                    piece = np.zeros_like(work)
                    piece[max(0, y):y + height, max(0, x):x + width] = work[max(0, y):y + height, max(0, x):x + width]
                    if np.any(piece):
                        parts.append(piece)
            elif mode == "auto":
                parts, _ = split_touching_mask(work, float(payload.get("strength", 0.7)))
            else:
                raise ExtractorError("INVALID_PARAMETER", "Unsupported split mode", {"mode": mode})
            if len(parts) < 2:
                raise ExtractorError("SPLIT_NOT_FOUND", "The requested split did not produce multiple valid assets")
            self._push_history(task_id, result)
            replacements: list[Asset] = []
            for number, part in enumerate(parts, 1):
                replacement = copy.deepcopy(asset)
                replacement.id = f"asset-{uuid.uuid4().hex[:10]}"
                replacement.file = f"icons/{Path(asset.file).stem}_{number}.png"
                replacement.detection_mask = part
                local_alpha = np.where(part > 0, alpha, 0).astype(np.uint8)
                ys, xs = np.nonzero(local_alpha)
                if not len(xs):
                    continue
                lx1, ly1, lx2, ly2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
                replacement.bbox = BBox(asset.bbox.x + lx1, asset.bbox.y + ly1, lx2 - lx1, ly2 - ly1)
                replacement.detection_mask = part[ly1:ly2, lx1:lx2].copy()
                replacement.alpha_mask = local_alpha[ly1:ly2, lx1:lx2].copy()
                replacement.mask_area = int(len(xs))
                replacement.status = "manually-edited"
                replacement.confidence_reasons.append(f"manually-split:{mode}")
                replacements.append(replacement)
            if len(replacements) < 2:
                raise ExtractorError("SPLIT_NOT_FOUND", "Split masks did not contain enough foreground")
            position = result.assets.index(asset)
            result.assets[position:position + 1] = replacements
            self._renumber(result)
            return self._commit(task_id)

    @staticmethod
    def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        areas = sorted((int(stats[label, cv2.CC_STAT_AREA]), label) for label in range(1, count))
        minimum = max(1, int(sum(area for area, _ in areas) * 0.02))
        return [(labels == label).astype(np.uint8) * 255 for area, label in areas if area >= minimum]

    @staticmethod
    def _assign_foreground_to_seeds(original: np.ndarray, seeds: list[np.ndarray]) -> list[np.ndarray]:
        if len(seeds) < 2:
            return seeds
        distances = [cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5) for seed in seeds]
        assignment = np.argmin(np.stack(distances), axis=0)
        return [((original > 0) & (assignment == index)).astype(np.uint8) * 255 for index in range(len(seeds))]

    def update_bbox(
        self,
        task_id: str,
        asset_id: str,
        bbox: dict[str, Any],
        revision: int | None,
        restore_from_source: bool = False,
    ) -> TaskRecord:
        with self._edit_lock(task_id):
            task, result = self._checked_result(task_id, revision)
            asset = self._find_asset(result, asset_id)
            try:
                new_box = BBox(*(int(bbox[key]) for key in ("x", "y", "width", "height")))
            except (KeyError, TypeError, ValueError) as exc:
                raise ExtractorError("INVALID_PARAMETER", "bbox requires integer x, y, width and height") from exc
            if new_box.x < 0 or new_box.y < 0 or new_box.width < 1 or new_box.height < 1 or new_box.x2 > result.source_width or new_box.y2 > result.source_height:
                raise ExtractorError("INVALID_PARAMETER", "bbox is outside the source image")
            ix1, iy1 = max(asset.bbox.x, new_box.x), max(asset.bbox.y, new_box.y)
            ix2, iy2 = min(asset.bbox.x2, new_box.x2), min(asset.bbox.y2, new_box.y2)
            if ix1 >= ix2 or iy1 >= iy2:
                raise ExtractorError("INVALID_PARAMETER", "bbox does not contain foreground")
            old_slice = np.s_[iy1 - asset.bbox.y:iy2 - asset.bbox.y, ix1 - asset.bbox.x:ix2 - asset.bbox.x]
            if not np.any(asset.alpha_mask[old_slice]):
                raise ExtractorError("INVALID_PARAMETER", "bbox does not contain foreground")
            self._push_history(task_id, result)
            if restore_from_source:
                source_detection, source_alpha = self._source_segmentation(task_id)
                new_alpha = source_alpha[new_box.y:new_box.y2, new_box.x:new_box.x2].copy()
                new_detection = source_detection[new_box.y:new_box.y2, new_box.x:new_box.x2].copy()
            else:
                new_alpha = np.zeros((new_box.height, new_box.width), np.uint8)
                new_detection = np.zeros_like(new_alpha)
            new_slice = np.s_[iy1 - new_box.y:iy2 - new_box.y, ix1 - new_box.x:ix2 - new_box.x]
            new_alpha[new_slice] = asset.alpha_mask[old_slice]
            new_detection[new_slice] = asset.detection_mask[old_slice]
            asset.alpha_mask, asset.detection_mask = new_alpha, new_detection
            asset.bbox, asset.status = new_box, "manually-edited"
            asset.mask_area = int(np.count_nonzero(new_alpha))
            asset.confidence_reasons.append("bbox-restored-from-source" if restore_from_source else "bbox-manually-updated")
            return self._commit(task_id)

    @staticmethod
    def _brush_layer(
        shape: tuple[int, int],
        points: list[dict[str, Any]],
        radius: int,
        hardness: float,
        feather: float,
        bbox: BBox,
    ) -> np.ndarray:
        outer = np.zeros(shape, np.uint8)
        local = [(int(point["x"]) - bbox.x, int(point["y"]) - bbox.y) for point in points]
        if len(local) == 1:
            cv2.circle(outer, local[0], radius, 255, -1, cv2.LINE_8)
        else:
            for p1, p2 in zip(local, local[1:]):
                cv2.line(outer, p1, p2, 255, radius * 2, cv2.LINE_8)
                cv2.circle(outer, p1, radius, 255, -1, cv2.LINE_8)
                cv2.circle(outer, p2, radius, 255, -1, cv2.LINE_8)
        soft_width = radius * max(0.0, 1.0 - min(1.0, max(0.0, hardness))) + feather
        if soft_width <= 0.1:
            return outer
        distance = cv2.distanceTransform((outer > 0).astype(np.uint8), cv2.DIST_L2, 5)
        return np.clip(distance * (255.0 / soft_width), 0, 255).astype(np.uint8)

    def update_mask(self, task_id: str, asset_id: str, payload: dict[str, Any], revision: int | None) -> TaskRecord:
        with self._edit_lock(task_id):
            task, result = self._checked_result(task_id, revision)
            asset = self._find_asset(result, asset_id)
            edited_alpha = asset.alpha_mask.copy()
            encoded = payload.get("maskPngBase64")
            if encoded:
                try:
                    raw = base64.b64decode(str(encoded).split(",")[-1], validate=True)
                    mask = np.asarray(Image.open(BytesIO(raw)).convert("L"))
                except Exception as exc:
                    raise ExtractorError("INVALID_MASK", "maskPngBase64 is not a valid grayscale PNG") from exc
                if mask.shape == asset.alpha_mask.shape:
                    edited_alpha = mask.astype(np.uint8)
                elif mask.shape == (result.source_height, result.source_width):
                    edited_alpha = mask[asset.bbox.y:asset.bbox.y2, asset.bbox.x:asset.bbox.x2].astype(np.uint8)
                else:
                    raise ExtractorError("INVALID_MASK", "Mask dimensions must match the source image or asset bbox")
            else:
                mode = payload.get("mode", "add")
                if mode not in {"add", "erase"}:
                    raise ExtractorError("INVALID_PARAMETER", "Mask mode must be add or erase")
                strokes = payload.get("strokes") or []
                if not strokes:
                    raise ExtractorError("INVALID_PARAMETER", "At least one mask stroke is required")
                for stroke in strokes:
                    radius = max(1, min(500, int(stroke.get("size", 20)) // 2))
                    hardness = float(stroke.get("hardness", 0.8))
                    feather = max(0.0, float(stroke.get("feather", 0)))
                    points = stroke.get("points") or []
                    layer = self._brush_layer(asset.alpha_mask.shape, points, radius, hardness, feather, asset.bbox)
                    if mode == "add":
                        edited_alpha = np.maximum(edited_alpha, layer)
                    else:
                        edited_alpha = np.minimum(edited_alpha, 255 - layer)
            mask_area = int(np.count_nonzero(edited_alpha))
            if not mask_area:
                raise ExtractorError("INVALID_MASK", "The edited mask cannot be empty")
            self._push_history(task_id, result)
            asset.alpha_mask = edited_alpha
            asset.detection_mask = (edited_alpha > 7).astype(np.uint8) * 255
            asset.mask_area = mask_area
            asset.status = "manually-edited"
            asset.confidence_reasons.append("mask-manually-updated")
            return self._commit(task_id)

    def reorder(self, task_id: str, asset_ids: list[str], revision: int | None) -> TaskRecord:
        with self._edit_lock(task_id):
            task, result = self._checked_result(task_id, revision)
            if set(asset_ids) != {asset.id for asset in result.assets} or len(asset_ids) != len(result.assets):
                raise ExtractorError("INVALID_PARAMETER", "assetIds must contain every active asset exactly once")
            self._push_history(task_id, result)
            by_id = {asset.id: asset for asset in result.assets}
            result.assets = [by_id[asset_id] for asset_id in asset_ids]
            self._renumber(result)
            for asset in result.assets:
                asset.status = "manually-edited"
            return self._commit(task_id)

    def rename(self, task_id: str, asset_id: str, name: str, revision: int | None) -> TaskRecord:
        safe = Path(name).name
        if safe != name or not safe.lower().endswith(".png") or len(safe) > 100:
            raise ExtractorError("INVALID_PARAMETER", "Asset name must be a safe .png filename")
        with self._edit_lock(task_id):
            task, result = self._checked_result(task_id, revision)
            asset = self._find_asset(result, asset_id)
            if any(other is not asset and Path(other.file).name.lower() == safe.lower() for other in result.assets):
                raise ExtractorError("INVALID_PARAMETER", "Asset filename already exists")
            self._push_history(task_id, result)
            asset.file, asset.status = f"icons/{safe}", "manually-edited"
            return self._commit(task_id)

    def undo(self, task_id: str, revision: int | None) -> TaskRecord:
        with self._edit_lock(task_id):
            task, result = self._checked_result(task_id, revision)
            if not self._history[task_id]:
                raise ExtractorError("UNDO_EMPTY", "There is no edit to undo")
            self._redo[task_id].append(copy.deepcopy(result.assets))
            result.assets = self._history[task_id].pop()
            return self._commit(task_id)

    def redo(self, task_id: str, revision: int | None) -> TaskRecord:
        with self._edit_lock(task_id):
            task, result = self._checked_result(task_id, revision)
            if not self._redo[task_id]:
                raise ExtractorError("REDO_EMPTY", "There is no edit to redo")
            self._history[task_id].append(copy.deepcopy(result.assets))
            result.assets = self._redo[task_id].pop()
            return self._commit(task_id)

    def capabilities(self) -> dict[str, Any]:
        rmbg = RMBGAdapter(_env_bool("ENABLE_RMBG", True), os.getenv("RMBG_MODEL_PATH", ""), os.getenv("RMBG_SERVICE_URL", ""))
        sam = SAM31Adapter(_env_bool("ENABLE_SAM31", False), os.getenv("SAM31_MODEL_PATH", ""), os.getenv("SAM31_SERVICE_URL", ""))
        return {
            "opencv": {"available": True, "capabilities": ["alpha", "solid-background", "components", "clustering", "png-export"]},
            "rmbg": rmbg.capability().as_dict(),
            "sam31": sam.capability().as_dict(),
        }
