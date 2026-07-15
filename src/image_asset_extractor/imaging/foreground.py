from __future__ import annotations

import cv2
import numpy as np

from ..schemas.asset import BackgroundInfo
from ..schemas.config import ExtractionConfig
from .background import background_color_candidates, estimate_low_frequency_background
from .morphology import clean_detection_mask
from .precheck import PrecheckedImage


def _solid_delta(rgb: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(np.asarray(color, np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
    return np.linalg.norm(lab - bg_lab, axis=2)


def _segmentation_score(detection: np.ndarray) -> tuple[bool, float]:
    foreground = detection > 0
    ratio = float(np.mean(foreground))
    edge_ratio = float(np.mean(np.concatenate((foreground[0], foreground[-1], foreground[:, 0], foreground[:, -1]))))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
    largest = max((int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count)), default=0) / max(1, detection.size)
    abnormal = ratio > 0.92 or (largest > 0.85 and edge_ratio > 0.65)
    return abnormal, edge_ratio * 2.0 + ratio * 0.2


def _background_candidate_support(
    rgb: np.ndarray,
    color: tuple[int, int, int],
    tolerance: float,
) -> tuple[int, float, float]:
    similar = _solid_delta(rgb, color) <= max(5.0, tolerance * 0.65)
    height, width = similar.shape
    band = max(1, round(min(height, width) * 0.03))
    edges = [
        similar[:band], similar[-band:],
        similar[band:height - band or None, :band],
        similar[band:height - band or None, width - band:],
    ]
    edge_supports = [float(np.mean(edge)) for edge in edges]
    coverage = sum(value >= 0.08 for value in edge_supports)
    return coverage, float(np.mean(edge_supports)), float(np.mean(similar))


def _stable_component_mask(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    kept = np.zeros(mask.shape, np.uint8)
    boxes: list[tuple[int, int, int, int]] = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < min_area:
            continue
        kept[labels == label] = 255
        boxes.append((x, y, width, height))
    return kept, boxes


def _box_sets_stable(
    reference: list[tuple[int, int, int, int]],
    candidate: list[tuple[int, int, int, int]],
) -> bool:
    if not reference or not candidate:
        return False
    remaining = set(range(len(candidate)))
    matched = 0
    for x, y, width, height in reference:
        best_index, best_iou = None, 0.0
        for index in remaining:
            other_x, other_y, other_width, other_height = candidate[index]
            x1, y1 = max(x, other_x), max(y, other_y)
            x2, y2 = min(x + width, other_x + other_width), min(y + height, other_y + other_height)
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            union = width * height + other_width * other_height - intersection
            overlap = intersection / max(1, union)
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        if best_index is not None and best_iou >= 0.45:
            remaining.remove(best_index)
            matched += 1
    return matched / max(len(reference), len(candidate)) >= 0.85


def coarse_instance_envelope(
    image: PrecheckedImage,
    background: BackgroundInfo,
    config: ExtractionConfig,
) -> tuple[np.ndarray | None, str | None]:
    """Build a stable coarse ownership mask for low-confidence flat photos."""
    if (
        background.type != "near-solid"
        or background.confidence >= 0.75
        or background.color is None
        or image.alpha_valid
    ):
        return None, None
    delta = _solid_delta(image.rgba[:, :, :3], background.color)
    min_area = max(1, round(delta.size * config.min_area_ratio))
    kernel_size = max(3, round(min(delta.shape) * 0.015))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    trials: list[tuple[np.ndarray, list[tuple[int, int, int, int]]]] = []
    for scale in (1.20, 1.50, 1.75):
        candidate = (delta >= max(3.0, config.background_tolerance * scale)).astype(np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
        trials.append(_stable_component_mask(candidate, min_area))
    counts = [len(boxes) for _, boxes in trials]
    if not counts[0] or max(counts) - min(counts) > 1:
        return None, None
    selected, boxes = trials[1]
    if not all(_box_sets_stable(boxes, trial_boxes) for _, trial_boxes in trials):
        return None, None
    ratio = float(np.mean(selected > 0))
    largest = max((width * height for _, _, width, height in boxes), default=0) / max(1, selected.size)
    if ratio > 0.85 or largest > 0.35:
        return None, None
    envelope = np.zeros(selected.shape, np.uint8)
    contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(envelope, contours, -1, 255, thickness=cv2.FILLED)
    return envelope, f"stable-coarse-instance-envelope:{counts[0]}-{counts[1]}-{counts[2]}"


def _apply_content_modes(rgb: np.ndarray, detection: np.ndarray, alpha: np.ndarray, config: ExtractionConfig) -> tuple[np.ndarray, np.ndarray]:
    if config.text_mode == "remove":
        count, labels, stats, _ = cv2.connectedComponentsWithStats((detection > 0).astype(np.uint8), 8)
        image_area = detection.size
        for label in range(1, count):
            _, _, width, height, area = (int(v) for v in stats[label])
            if area < image_area * 0.0015 and height < max(10, detection.shape[0] * 0.08) and width < detection.shape[1] * 0.25:
                detection[labels == label] = 0
                alpha[labels == label] = 0
    return detection, alpha


def _apply_segmentation_sanity(background: BackgroundInfo, detection: np.ndarray) -> None:
    foreground = detection > 0
    ratio = float(np.mean(foreground))
    edge = np.concatenate((foreground[0], foreground[-1], foreground[:, 0], foreground[:, -1]))
    edge_ratio = float(np.mean(edge))
    count, _, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
    largest = max(
        (int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count)),
        default=0,
    ) / max(1, detection.size)
    abnormal = ratio > 0.80 or largest > 0.70 or (ratio > 0.55 and edge_ratio > 0.45)
    if abnormal:
        background.confidence = min(background.confidence, 0.35)
        for reason in (
            f"segmentation-foreground-ratio-{ratio:.3f}",
            f"segmentation-largest-component-ratio-{largest:.3f}",
            f"segmentation-edge-occupancy-{edge_ratio:.3f}",
            "segmentation-sanity-check-failed",
            "manual-review-required",
        ):
            if reason not in background.reasons:
                background.reasons.append(reason)


def _weak_candidate(delta: np.ndarray, tolerance: float, min_area: int) -> np.ndarray | None:
    low = delta >= max(2.5, tolerance * 0.20)
    high = delta >= max(3.5, tolerance * 0.23)
    union = int(np.count_nonzero(low))
    if union < min_area or union > delta.size * 0.35:
        return None
    stability = np.count_nonzero(low & high) / max(1, union)
    if stability < 0.55:
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(low.astype(np.uint8), 8)
    minimum = max(2, int(min_area * 0.15))
    keep = np.zeros(delta.shape, np.uint8)
    kept_areas = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= minimum:
            keep[labels == label] = 255
            kept_areas.append(area)
    if not kept_areas or len(kept_areas) > max(150, delta.size // 40):
        return None
    return keep


def segment_foreground(image: PrecheckedImage, background: BackgroundInfo, config: ExtractionConfig) -> tuple[np.ndarray, np.ndarray]:
    rgb = image.rgba[:, :, :3]
    if background.type == "transparent":
        alpha = image.rgba[:, :, 3].copy()
        nonzero = alpha[alpha > 0]
        structure_threshold = max(8, min(96, round(float(np.percentile(nonzero, 75)) * 0.50))) if len(nonzero) else 8
        detection = (alpha >= structure_threshold).astype(np.uint8) * 255
        detection, alpha = _apply_content_modes(rgb, detection, alpha, config)
        detection = clean_detection_mask(detection)
        _apply_segmentation_sanity(background, detection)
        return detection, alpha
    tolerance = max(2.0, float(config.background_tolerance))
    structure_tolerance = tolerance * 2.25
    if background.type in {"gradient", "complex", "unknown"}:
        local_background = estimate_low_frequency_background(rgb, tolerance).image
        delta = np.max(cv2.absdiff(rgb, local_background), axis=2).astype(np.float32)
        detection = (delta >= structure_tolerance).astype(np.uint8) * 255
    else:
        candidates = background_color_candidates(rgb)
        preferred = background.color or candidates[0]
        ordered = [preferred] + [color for color in candidates if color != preferred]
        trials = []
        for color in ordered:
            trial_delta = _solid_delta(rgb, color)
            trial = (trial_delta >= structure_tolerance).astype(np.uint8) * 255
            abnormal, score = _segmentation_score(trial)
            coverage, edge_support, global_support = _background_candidate_support(rgb, color, tolerance)
            trials.append((abnormal, score, color, trial_delta, trial, coverage, edge_support, global_support))
        preferred_trial = trials[0]
        eligible = [preferred_trial]
        if preferred_trial[0]:
            preferred_lab = cv2.cvtColor(np.asarray(preferred, np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
            for trial in trials[1:]:
                candidate_lab = cv2.cvtColor(np.asarray(trial[2], np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
                consistent = float(np.linalg.norm(candidate_lab - preferred_lab)) <= max(8.0, tolerance * 0.8)
                if trial[5] >= 3 and trial[6] >= 0.15 and trial[7] >= 0.15 and consistent:
                    eligible.append(trial)
        abnormal, _, selected, delta, detection, _, _, _ = min(eligible, key=lambda item: (item[0], item[1]))
        if selected != preferred:
            background.color = selected
            background.reasons.append("alternate-background-candidate-selected")
        if abnormal:
            background.confidence = min(background.confidence, 0.35)
            background.reasons.extend(["segmentation-sanity-check-failed", "manual-review-required"])
    if background.type == "complex":
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        saliency = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), sigmaX=max(3, min(rgb.shape[:2]) * 0.03)))
        detection = cv2.bitwise_or(detection, (saliency > max(8, tolerance * 0.7)).astype(np.uint8) * 255)
    min_area = max(1, round(detection.size * config.min_area_ratio))
    weak_used = False
    if np.count_nonzero(detection) < min_area:
        weak = _weak_candidate(delta, tolerance, min_area)
        if weak is not None:
            detection = weak
            weak_used = True
            background.confidence = min(background.confidence, 0.45)
            background.reasons.extend(["stable-low-contrast-candidate", "manual-review-required"])
    alpha = np.clip((delta - tolerance * 0.35) * (255.0 / max(1.0, tolerance * 0.9)), 0, 255).astype(np.uint8)
    if image.has_alpha:
        alpha = np.minimum(alpha, image.rgba[:, :, 3])
    if weak_used:
        alpha[detection > 0] = np.maximum(alpha[detection > 0], 128)
    alpha[detection == 0] = np.minimum(alpha[detection == 0], 96)
    detection, alpha = _apply_content_modes(rgb, detection, alpha, config)
    detection = clean_detection_mask(detection)
    _apply_segmentation_sanity(background, detection)
    return detection, alpha
