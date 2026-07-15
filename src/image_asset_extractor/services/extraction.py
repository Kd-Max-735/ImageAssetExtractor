from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from ..errors import ExtractorError
from ..imaging.background import classify_background
from ..imaging.clustering import cluster_regions
from ..imaging.components import analyze_components
from ..imaging.export import export_result
from ..imaging.foreground import coarse_instance_envelope, segment_foreground
from ..imaging.grid import projection_cells
from ..imaging.masks import isolate_alpha
from ..imaging.precheck import PrecheckedImage, load_and_precheck
from ..imaging.splitting import split_touching_mask
from ..models import RMBGAdapter, SAM31Adapter
from ..schemas.asset import Asset, BBox, ExtractionResult
from ..schemas.config import ExtractionConfig


class ExtractionEngine:
    @staticmethod
    def _mask_quality(
        baseline_alpha: np.ndarray,
        baseline_detection: np.ndarray,
        candidate_alpha: np.ndarray,
    ) -> tuple[bool, str]:
        baseline = baseline_detection > 0
        candidate = candidate_alpha > 8
        intersection = int(np.count_nonzero(baseline & candidate))
        union = int(np.count_nonzero(baseline | candidate))
        baseline_area = int(np.count_nonzero(baseline))
        candidate_area = int(np.count_nonzero(candidate))
        iou = intersection / max(1, union)
        area_ratio = candidate_area / max(1, baseline_area)
        nonzero_alpha = baseline_alpha[baseline_alpha > 8]
        core_threshold = max(96, int(np.percentile(nonzero_alpha, 60))) if len(nonzero_alpha) else 255
        core = baseline_alpha >= core_threshold
        core_retention = np.count_nonzero(core & candidate) / max(1, np.count_nonzero(core))

        def structure(mask: np.ndarray) -> tuple[int, int, int, tuple[int, int]]:
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
            areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count)]
            contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            holes = int(np.count_nonzero(hierarchy[0, :, 3] >= 0)) if hierarchy is not None else 0
            ys, xs = np.nonzero(mask)
            span = (int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)) if len(xs) else (0, 0)
            return len(areas), max(areas, default=0), holes, span

        base_count, base_largest, base_holes, base_span = structure(baseline)
        cand_count, cand_largest, cand_holes, cand_span = structure(candidate)
        largest_retention = cand_largest / max(1, base_largest)
        span_retention = min(
            cand_span[0] / max(1, base_span[0]),
            cand_span[1] / max(1, base_span[1]),
        )
        accepted = (
            iou >= 0.68
            and core_retention >= 0.90
            and 0.72 <= area_ratio <= 1.50
            and cand_count <= base_count + 2
            and cand_holes <= base_holes + 2
            and largest_retention >= 0.72
            and span_retention >= 0.82
        )
        metrics = (
            f"iou={iou:.3f}, coreRetention={core_retention:.3f}, areaRatio={area_ratio:.3f}, "
            f"components={base_count}->{cand_count}, holes={base_holes}->{cand_holes}, "
            f"largestRetention={largest_retention:.3f}, spanRetention={span_retention:.3f}"
        )
        return accepted, metrics

    @staticmethod
    def _record_model_fallback(asset: Asset, model: str, reason: str, penalty: float = 0.05) -> None:
        asset.confidence_reasons.append(f"{model}-fallback-baseline: {reason}")
        previous = asset.confidence
        asset.confidence = max(0.0, asset.confidence - penalty)
        asset.confidence_reasons.append(f"confidence:{previous:.3f}->{asset.confidence:.3f}")
        if "low-confidence" not in asset.flags:
            asset.flags.append("low-confidence")
        asset.status = "needs-review"

    def extract(
        self,
        source: str | Path | bytes,
        output_dir: str | Path,
        config: ExtractionConfig | None = None,
        source_name: str | None = None,
        progress: Callable[[int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ExtractionResult:
        config = config or ExtractionConfig()
        config.validate()

        def checkpoint(value: int, stage: str) -> None:
            if cancel_check and cancel_check():
                raise ExtractorError("TASK_CANCELLED", "Extraction was cancelled", {"stage": stage})
            if progress:
                progress(value, stage)

        checkpoint(3, "precheck")
        image = load_and_precheck(source, config, source_name)
        checkpoint(12, "background-classification")
        background = classify_background(image, config.background_type)
        checkpoint(25, "foreground-segmentation")
        detection, soft_alpha = segment_foreground(image, background, config)
        coarse_mask, coarse_reason = coarse_instance_envelope(image, background, config)
        ownership_mask = coarse_mask if coarse_mask is not None else detection
        min_area = max(1, round(image.width * image.height * config.min_area_ratio))
        checkpoint(38, "connected-components")
        regions, labels = analyze_components(ownership_mask, image.rgba, min_area)
        if not regions:
            raise ExtractorError("NO_ASSETS", "No foreground assets were detected")
        checkpoint(48, "grid-projection")
        cells = projection_cells(ownership_mask)
        clusters = (
            [[region] for region in regions]
            if coarse_mask is not None
            else cluster_regions(regions, (image.height, image.width), config.merge_distance_ratio, ownership_mask, cells)
        )
        # Small components remain available for ownership, but noise-only and
        # compression-line clusters are not exportable. A clipped edge object
        # gets a lower, explicitly reviewable threshold.
        def exportable(cluster: list) -> bool:
            area = sum(region.area for region in cluster)
            has_body = any(not region.suspected_noise for region in cluster)
            edge_area = sum(region.area for region in cluster if region.touches_edge)
            edge_span = max(
                (max(region.bbox.width, region.bbox.height) for region in cluster if region.touches_edge),
                default=0,
            )
            clipped_body = edge_area >= min_area * 0.35 and edge_span >= np.sqrt(min_area) * 1.5
            return (area >= min_area and has_body) or clipped_body

        clusters = [cluster for cluster in clusters if exportable(cluster)]
        if not clusters:
            raise ExtractorError("NO_ASSETS", "No foreground assets remained after component ownership analysis")
        checkpoint(58, "touching-split")
        def candidate_masks() -> Iterable[tuple[np.ndarray, list, str | None, int, int]]:
            for cluster in clusters:
                if cancel_check and cancel_check():
                    raise ExtractorError("TASK_CANCELLED", "Extraction was cancelled", {"stage": "touching-split"})
                cx1 = min(region.bbox.x for region in cluster)
                cy1 = min(region.bbox.y for region in cluster)
                cx2 = max(region.bbox.x2 for region in cluster)
                cy2 = max(region.bbox.y2 for region in cluster)
                local_labels = labels[cy1:cy2, cx1:cx2]
                cluster_mask = np.isin(local_labels, [region.label for region in cluster]).astype(np.uint8) * 255
                local_rgb = image.rgba[cy1:cy2, cx1:cx2, :3]
                parts, split_reason = split_touching_mask(cluster_mask, config.split_strength, local_rgb) if len(cluster) == 1 else ([cluster_mask], None)
                for part in parts:
                    yield part, cluster, split_reason, cx1, cy1
        checkpoint(70, "fine-mask")
        assets = self._build_assets(
            image, candidate_masks(), ownership_mask, soft_alpha, config,
            background.type, background.confidence, coarse_reason,
        )
        if len(assets) > config.max_assets:
            raise ExtractorError("TOO_MANY_ASSETS", "Detected asset count exceeds configured limit", {"count": len(assets)})
        if config.use_rmbg:
            checkpoint(78, "rmbg-refinement")
            self._refine_with_rmbg(image, assets)
        if config.use_sam31:
            checkpoint(86, "sam31-refinement")
            self._refine_with_sam31(image, assets, background.type)
        name = source_name or (Path(source).name if not isinstance(source, bytes) else "upload.png")
        result = ExtractionResult(name, image.width, image.height, background, assets)
        checkpoint(93, "export")
        export_result(result, image.rgba, output_dir, config.include_zip, config=config)
        checkpoint(100, "completed")
        return result

    @staticmethod
    def _refine_with_rmbg(image: PrecheckedImage, assets: list[Asset]) -> None:
        adapter = RMBGAdapter(
            enabled=True,
            model_path=os.getenv("RMBG_MODEL_PATH", ""),
            service_url=os.getenv("RMBG_SERVICE_URL", ""),
        )
        capability = adapter.capability()
        if not capability.available:
            for asset in assets:
                asset.confidence_reasons.append(f"rmbg-unavailable: {capability.reason}")
            return
        for asset in assets:
            box = asset.bbox
            roi = image.rgba[box.y:box.y2, box.x:box.x2]
            try:
                refined = adapter.refine(roi)
                if not isinstance(refined, np.ndarray) or refined.ndim != 2 or refined.shape != (box.height, box.width):
                    raise ExtractorError("RMBG_INVALID_MASK", "RMBG returned a mask with an unexpected size")
                if not np.issubdtype(refined.dtype, np.number) or not np.all(np.isfinite(refined)):
                    raise ExtractorError("RMBG_INVALID_MASK", "RMBG returned a non-numeric mask")
                refined = np.clip(refined, 0, 255).astype(np.uint8)
                constraint = asset.detection_mask
                ys, xs = np.nonzero(constraint)
                subject_span = max(
                    int(xs.max() - xs.min() + 1) if len(xs) else box.width,
                    int(ys.max() - ys.min() + 1) if len(ys) else box.height,
                )
                radius = max(1, round(subject_span * 0.08))
                kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
                constraint = cv2.dilate(constraint, kernel)
                constrained = np.minimum(refined.astype(np.uint8), constraint)
                accepted, metrics = ExtractionEngine._mask_quality(asset.alpha_mask, asset.detection_mask, constrained)
                if not accepted:
                    ExtractionEngine._record_model_fallback(asset, "rmbg", f"candidate-rejected: {metrics}")
                    continue
                asset.alpha_mask = constrained
                asset.mask_area = int(np.count_nonzero(asset.alpha_mask))
                asset.confidence_reasons.extend(["alpha-refined-by-rmbg", f"rmbg-candidate-accepted: {metrics}"])
                previous = asset.confidence
                asset.confidence = min(1.0, asset.confidence + 0.01)
                asset.confidence_reasons.append(f"confidence:{previous:.3f}->{asset.confidence:.3f}")
            except ExtractorError as exc:
                ExtractionEngine._record_model_fallback(asset, "rmbg", exc.message, 0.08)

    @staticmethod
    def _refine_with_sam31(image: PrecheckedImage, assets: list[Asset], background_type: str) -> None:
        adapter = SAM31Adapter(
            enabled=True,
            model_path=os.getenv("SAM31_MODEL_PATH", ""),
            service_url=os.getenv("SAM31_SERVICE_URL", ""),
        )
        capability = adapter.capability()
        if not capability.available:
            for asset in assets:
                asset.confidence_reasons.append(f"sam31-unavailable: {capability.reason}")
            return
        for asset in assets:
            box = asset.bbox
            roi = image.rgba[box.y:box.y2, box.x:box.x2]
            try:
                segmented = adapter.refine(roi, box=[0.01, 0.01, 0.98, 0.98])
                if not isinstance(segmented, np.ndarray) or segmented.ndim != 2 or segmented.shape != (box.height, box.width):
                    raise ExtractorError("SAM31_INVALID_MASK", "SAM 3.1 returned a mask with an unexpected size")
                if not np.issubdtype(segmented.dtype, np.number) or not np.all(np.isfinite(segmented)):
                    raise ExtractorError("SAM31_INVALID_MASK", "SAM 3.1 returned a non-numeric mask")
                segmented = np.clip(segmented, 0, 255).astype(np.uint8)
                current = asset.alpha_mask.copy()
                candidate = np.where(segmented > 0, current, 0).astype(np.uint8)
                accepted, metrics = ExtractionEngine._mask_quality(current, asset.detection_mask, candidate)
                if accepted:
                    asset.alpha_mask = candidate
                    asset.mask_area = int(np.count_nonzero(asset.alpha_mask))
                    asset.confidence_reasons.extend(["mask-refined-by-sam31", f"sam31-candidate-accepted: {metrics}"])
                    previous = asset.confidence
                    asset.confidence = min(1.0, asset.confidence + 0.02)
                    asset.confidence_reasons.append(f"confidence:{previous:.3f}->{asset.confidence:.3f}")
                else:
                    ExtractionEngine._record_model_fallback(asset, "sam31", f"candidate-rejected: {metrics}")
            except ExtractorError as exc:
                ExtractionEngine._record_model_fallback(asset, "sam31", exc.message, 0.08)

    @staticmethod
    def _build_assets(
        image: PrecheckedImage,
        candidates: Iterable[tuple[np.ndarray, list, str | None, int, int]],
        structure_mask: np.ndarray,
        soft_alpha: np.ndarray,
        config: ExtractionConfig,
        background_type: str,
        background_confidence: float,
        coarse_reason: str | None,
    ) -> list[Asset]:
        built: list[tuple[BBox, np.ndarray, np.ndarray, list[str], list[str], float, list[str]]] = []
        for detection, cluster, split_reason, origin_x, origin_y in candidates:
            ys, xs = np.nonzero(detection)
            if not len(xs):
                continue
            tight_w, tight_h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
            candidate_scale = max(tight_w, tight_h)
            max_effect_radius = 0 if background_type in {"gradient", "complex", "unknown"} else max(1, int(round(candidate_scale * 0.15)))
            global_x1, global_y1 = origin_x + int(xs.min()), origin_y + int(ys.min())
            global_x2, global_y2 = origin_x + int(xs.max()) + 1, origin_y + int(ys.max()) + 1

            appearance_radius = 0
            if max_effect_radius:
                probe_x1, probe_y1 = max(0, global_x1 - max_effect_radius), max(0, global_y1 - max_effect_radius)
                probe_x2 = min(image.width, global_x2 + max_effect_radius)
                probe_y2 = min(image.height, global_y2 + max_effect_radius)
                probe_detection = np.zeros((probe_y2 - probe_y1, probe_x2 - probe_x1), np.uint8)
                source_x1, source_y1 = max(0, probe_x1 - origin_x), max(0, probe_y1 - origin_y)
                source_x2 = min(detection.shape[1], probe_x2 - origin_x)
                source_y2 = min(detection.shape[0], probe_y2 - origin_y)
                target_x, target_y = max(0, origin_x - probe_x1), max(0, origin_y - probe_y1)
                probe_detection[
                    target_y:target_y + source_y2 - source_y1,
                    target_x:target_x + source_x2 - source_x1,
                ] = detection[source_y1:source_y2, source_x1:source_x2]
                probe_distance = cv2.distanceTransform((probe_detection == 0).astype(np.uint8), cv2.DIST_L2, 5)
                appearance_evidence = (
                    (soft_alpha[probe_y1:probe_y2, probe_x1:probe_x2] > 1)
                    & (probe_detection == 0)
                    & (probe_distance <= max_effect_radius)
                )
                if np.any(appearance_evidence):
                    appearance_radius = min(
                        max_effect_radius,
                        max(1, int(np.ceil(np.percentile(probe_distance[appearance_evidence], 99)))),
                    )
            padding = max(int(round(max(tight_w, tight_h) * config.padding_ratio)), appearance_radius)
            x1 = max(0, global_x1 - padding)
            y1 = max(0, global_y1 - padding)
            x2 = min(image.width, global_x2 + padding)
            y2 = min(image.height, global_y2 + padding)
            box = BBox(x1, y1, x2 - x1, y2 - y1)
            local_detection = np.zeros((box.height, box.width), np.uint8)
            sx1, sy1 = max(0, box.x - origin_x), max(0, box.y - origin_y)
            sx2 = min(detection.shape[1], box.x2 - origin_x)
            sy2 = min(detection.shape[0], box.y2 - origin_y)
            tx, ty = max(0, origin_x - box.x), max(0, origin_y - box.y)
            local_detection[ty:ty + sy2 - sy1, tx:tx + sx2 - sx1] = detection[sy1:sy2, sx1:sx2]
            other_structure = (structure_mask[y1:y2, x1:x2] > 0) & (local_detection == 0)
            if background_type in {"gradient", "complex", "unknown"}:
                contours, _ = cv2.findContours(local_detection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                envelope = np.zeros(local_detection.shape, np.uint8)
                cv2.drawContours(envelope, contours, -1, 255, thickness=cv2.FILLED)
                allowed = envelope > 0
            else:
                distance_to_candidate = cv2.distanceTransform((local_detection == 0).astype(np.uint8), cv2.DIST_L2, 5)
                allowed = distance_to_candidate <= appearance_radius
                if np.any(other_structure):
                    distance_to_other = cv2.distanceTransform((~other_structure).astype(np.uint8), cv2.DIST_L2, 5)
                    allowed &= distance_to_candidate <= distance_to_other
            expanded = allowed.astype(np.uint8) * 255
            alpha = isolate_alpha(soft_alpha[y1:y2, x1:x2], expanded)
            if coarse_reason:
                core = cv2.erode((local_detection > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                source_alpha = image.rgba[y1:y2, x1:x2, 3]
                alpha[core] = source_alpha[core]
            if not image.alpha_valid:
                # Preserve the ownership support even where the inferred soft
                # alpha is zero; flattened-effect recovery uses this support.
                alpha[(expanded > 0) & (alpha == 0)] = 1
            flags: list[str] = []
            reasons: list[str] = []
            if coarse_reason:
                reasons.append(coarse_reason)
            confidence = 0.98
            if background_confidence < 0.6:
                flags.extend(["low-confidence", "background-needs-review"])
                reasons.append("low-background-confidence")
                confidence -= 0.3
            if split_reason and split_reason.startswith("possible-split:"):
                flags.extend(["possible-split", "low-confidence"])
                reasons.append(split_reason)
                confidence -= 0.25
            elif split_reason:
                reasons.append(f"touching-split:{split_reason}")
                confidence -= 0.08
            if any(r.touches_edge for r in cluster):
                flags.append("touches-edge")
                reasons.append("candidate-touches-image-edge")
                confidence -= 0.18
            if any(r.suspected_noise for r in cluster) and all(r.suspected_noise for r in cluster):
                flags.extend(["edge-small-candidate", "low-confidence"])
                reasons.append("edge-clipped-candidate-below-global-area-threshold")
                confidence = min(confidence, 0.70)
            if len(cluster) > 1:
                reasons.append("multiple-connected-regions-clustered")
                confidence -= min(0.12, 0.02 * (len(cluster) - 1))
            if image.width * image.height and np.count_nonzero(local_detection) / (box.width * box.height) < 0.08:
                flags.append("possible-merge")
                reasons.append("low-mask-fill-ratio")
                confidence -= 0.15
            image_area = image.width * image.height
            if image_area and (
                np.count_nonzero(local_detection) / image_area > 0.70
                or (box.width * box.height / image_area > 0.85 and any(r.touches_edge for r in cluster))
            ):
                flags.extend(["page-like-candidate", "low-confidence"])
                reasons.append("candidate-failed-page-coverage-sanity-check")
                confidence = min(confidence, 0.35)
            if confidence < 0.75 and "low-confidence" not in flags:
                flags.append("low-confidence")
            built.append((
                box,
                local_detection,
                alpha,
                [r.id for r in cluster], flags, max(0.0, confidence), reasons,
            ))

        if not built:
            raise ExtractorError("NO_ASSETS", "No exportable assets were detected")
        average_height = float(np.mean([item[0].height for item in built]))
        row_tolerance = max(1.0, average_height * 0.5)
        built.sort(key=lambda item: (round((item[0].y + item[0].height / 2) / row_tolerance), item[0].x))
        assets: list[Asset] = []
        for index, (box, detection, alpha, region_ids, flags, confidence, reasons) in enumerate(built, 1):
            status = "needs-review" if confidence < 0.75 else "auto-confirmed"
            filename = f"{config.output_prefix}_{index:0{config.number_digits}d}.png"
            assets.append(Asset(
                id=f"asset-{index:03d}", index=index, file=f"icons/{filename}", bbox=box,
                mask_area=int(np.count_nonzero(alpha)), confidence=confidence,
                confidence_reasons=reasons, region_ids=region_ids, flags=flags, status=status,
                detection_mask=detection, alpha_mask=alpha,
            ))
        return assets
