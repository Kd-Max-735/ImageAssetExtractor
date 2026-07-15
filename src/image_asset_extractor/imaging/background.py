from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from ..schemas.asset import BackgroundInfo
from .precheck import PrecheckedImage


@dataclass(slots=True)
class LowFrequencyBackground:
    image: np.ndarray
    degree: int
    retained_residual_p90: float
    foreground_ratio: float
    edge_occupancy: float
    largest_component_ratio: float


def _polynomial_design(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    return np.column_stack([
        (x ** x_power) * (y ** (total - x_power))
        for total in range(degree + 1)
        for x_power in range(total + 1)
    ]).astype(np.float32)


def _fit_robust_surface(samples: np.ndarray, x: np.ndarray, y: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = _polynomial_design(x, y, degree)
    keep = np.ones(len(samples), dtype=bool)
    coefficients = np.zeros((design.shape[1], 3), np.float32)
    residual = np.zeros(len(samples), np.float32)
    for _ in range(7):
        if np.count_nonzero(keep) <= design.shape[1] * 3:
            break
        coefficients, *_ = np.linalg.lstsq(design[keep], samples[keep], rcond=None)
        residual = np.linalg.norm(samples - design @ coefficients, axis=1)
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median))) * 1.4826
        updated = residual <= max(5.0, median + 2.5 * max(1.0, mad))
        if np.array_equal(updated, keep):
            break
        keep = updated
    return coefficients, residual, keep


def _surface_image(shape: tuple[int, int], coefficients: np.ndarray, degree: int) -> np.ndarray:
    height, width = shape
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x_grid, y_grid = x[None, :], y[:, None]
    output = np.zeros((height, width, 3), np.float32)
    term = 0
    for total in range(degree + 1):
        for x_power in range(total + 1):
            basis = (x_grid ** x_power) * (y_grid ** (total - x_power))
            output += basis[:, :, None] * coefficients[term]
            term += 1
    return np.clip(output, 0, 255).astype(np.uint8)


def _candidate_shape_metrics(detection: np.ndarray) -> tuple[float, float, float]:
    foreground = detection > 0
    ratio = float(np.mean(foreground))
    edge = np.concatenate((foreground[0], foreground[-1], foreground[:, 0], foreground[:, -1]))
    count, _, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
    largest = max(
        (int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count)),
        default=0,
    ) / max(1, detection.size)
    return ratio, float(np.mean(edge)), float(largest)


def estimate_low_frequency_background(rgb: np.ndarray, tolerance: float = 15.0) -> LowFrequencyBackground:
    """Fit a robust full-image low-frequency surface with controlled complexity."""
    height, width = rgb.shape[:2]
    sample_limit = 160
    scale = sample_limit / max(height, width)
    sample_width = max(24, round(width * min(1.0, scale)))
    sample_height = max(24, round(height * min(1.0, scale)))
    sampled = cv2.resize(rgb, (sample_width, sample_height), interpolation=cv2.INTER_AREA).astype(np.float32)
    y_grid, x_grid = np.mgrid[0:sample_height, 0:sample_width].astype(np.float32)
    x = (x_grid / max(1, sample_width - 1) * 2.0 - 1.0).ravel()
    y = (y_grid / max(1, sample_height - 1) * 2.0 - 1.0).ravel()
    pixels = sampled.reshape(-1, 3)
    maximum_degree = max(1, min(6, min(sample_width, sample_height) // 20))
    candidates = []
    for degree in range(1, maximum_degree + 1):
        coefficients, residual, keep = _fit_robust_surface(pixels, x, y, degree)
        predicted = (_polynomial_design(x, y, degree) @ coefficients).reshape(sample_height, sample_width, 3)
        delta = np.max(np.abs(sampled - predicted), axis=2)
        ratio, edge_occupancy, largest = _candidate_shape_metrics((delta >= tolerance).astype(np.uint8))
        retained_p90 = float(np.percentile(residual[keep], 90)) if np.any(keep) else float("inf")
        term_count = (degree + 1) * (degree + 2) / 2
        sanity_penalty = edge_occupancy * 10.0 + max(0.0, ratio - 0.35) * 20.0
        if largest > 0.30:
            sanity_penalty += (largest - 0.30) * 20.0
        score = retained_p90 + sanity_penalty + term_count * 0.01
        candidates.append((score, degree, coefficients, retained_p90, ratio, edge_occupancy, largest))
    _, degree, coefficients, retained_p90, ratio, edge_occupancy, largest = min(candidates, key=lambda item: item[0])
    return LowFrequencyBackground(
        _surface_image((height, width), coefficients, degree),
        degree,
        retained_p90,
        ratio,
        edge_occupancy,
        largest,
    )


def _edge_arrays(rgb: np.ndarray) -> list[np.ndarray]:
    h, w = rgb.shape[:2]
    band = max(1, round(min(h, w) * 0.03))
    return [rgb[:band], rgb[-band:], rgb[band:h - band or None, :band], rgb[band:h - band or None, -band:]]


def _filtered_edge_pixels(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    texture = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), 1.2)).astype(np.float32)
    rgb_edges, quality_edges, edge_ids = [], [], []
    h, w = rgb.shape[:2]
    band = max(1, round(min(h, w) * 0.03))
    slices = [
        np.s_[:band, :], np.s_[h - band:, :],
        np.s_[band:h - band or None, :band], np.s_[band:h - band or None, w - band:],
    ]
    for edge_id, area in enumerate(slices):
        pixels = rgb[area].reshape(-1, 3)
        score = (gradient[area] + texture[area] * 3.0).reshape(-1)
        limit = min(30.0, float(np.percentile(score, 70)))
        keep = score <= max(4.0, limit)
        if np.count_nonzero(keep) < max(8, len(pixels) // 20):
            keep = score <= float(np.percentile(score, 85))
        rgb_edges.append(pixels[keep])
        quality_edges.append(np.full(np.count_nonzero(keep), edge_id, np.int16))
        edge_ids.append(edge_id)
    return np.concatenate(rgb_edges), np.concatenate(quality_edges)


def background_color_candidates(rgb: np.ndarray, maximum: int = 5) -> list[tuple[int, int, int]]:
    """Return robust Lab modes that occur across multiple image edges."""
    pixels, edge_ids = _filtered_edge_pixels(rgb)
    if not len(pixels):
        return [tuple(int(v) for v in np.median(np.concatenate(_edge_arrays(rgb)).reshape(-1, 3), axis=0))]
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    bins = (lab // 6).astype(np.int16)
    grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(map(tuple, bins)):
        grouped[key].append(index)
    ranked = sorted(
        grouped.values(),
        key=lambda ids: (len(set(int(edge_ids[index]) for index in ids)), len(ids)),
        reverse=True,
    )
    candidates: list[tuple[int, int, int]] = []
    for ids in ranked:
        if len(ids) < max(4, len(pixels) * 0.005):
            continue
        color = tuple(int(v) for v in np.median(pixels[ids], axis=0))
        if all(np.linalg.norm(np.asarray(color, float) - np.asarray(old, float)) >= 10 for old in candidates):
            candidates.append(color)
        if len(candidates) >= maximum:
            break
    return candidates or [tuple(int(v) for v in np.median(pixels, axis=0))]


def _plane_residual(rgb: np.ndarray) -> float:
    h, w = rgb.shape[:2]
    edges = _edge_arrays(rgb)
    samples = np.concatenate([edge.reshape(-1, 3) for edge in edges])
    gray_edges = np.concatenate([cv2.cvtColor(edge, cv2.COLOR_RGB2GRAY).reshape(-1) for edge in edges])
    return float(np.percentile(np.abs(gray_edges.astype(np.float32) - np.median(gray_edges)), 90)) + float(np.mean(np.var(samples, axis=0))) ** 0.5 * 0.1


def classify_background(image: PrecheckedImage, requested: str = "auto") -> BackgroundInfo:
    if image.alpha_valid:
        return BackgroundInfo("transparent", 1.0, None, ["valid-alpha-channel"])
    rgb = image.rgba[:, :, :3]
    candidates = background_color_candidates(rgb)
    color = candidates[0]
    if requested != "auto":
        normalized = {"white": "solid", "pure": "solid"}.get(requested, requested)
        return BackgroundInfo(normalized, 1.0, color, ["user-selected"])
    pixels, edge_ids = _filtered_edge_pixels(rgb)
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    color_lab = cv2.cvtColor(np.asarray(color, np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
    distances = np.linalg.norm(lab - color_lab, axis=1)
    support = float(np.mean(distances <= 8.0))
    coverage = sum(float(np.mean(distances[edge_ids == edge] <= 8.0)) >= 0.08 for edge in range(4))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edge_texture = float(np.mean(cv2.Laplacian(gray, cv2.CV_32F) ** 2))
    if coverage >= 3 and support >= 0.65:
        confidence = min(0.99, 0.72 + support * 0.25 + coverage * 0.015)
        return BackgroundInfo("solid", confidence, color, ["robust-multi-edge-lab-mode"])
    if coverage >= 3 and support >= 0.35:
        if _plane_residual(rgb) >= 6.0:
            return BackgroundInfo("gradient", 0.62, color, ["smooth-multi-edge-gradient", "robust-plane-required"])
        confidence = min(0.88, 0.62 + support * 0.25)
        return BackgroundInfo("near-solid", confidence, color, ["robust-multi-edge-lab-mode"])
    if coverage == 4 and support >= 0.25:
        return BackgroundInfo(
            "near-solid", 0.55, color,
            ["edge-contaminated-multi-edge-background", "manual-review-required"],
        )
    if edge_texture < 100 and _plane_residual(rgb) < 22:
        return BackgroundInfo("gradient", 0.62, color, ["smooth-multi-edge-gradient", "robust-plane-required"])
    surface = estimate_low_frequency_background(rgb)
    if (
        surface.retained_residual_p90 <= 8.0
        and surface.foreground_ratio <= 0.35
        and surface.edge_occupancy <= 0.20
        and surface.largest_component_ratio <= 0.25
    ):
        confidence = min(0.86, 0.72 + max(0.0, 8.0 - surface.retained_residual_p90) * 0.02)
        return BackgroundInfo(
            "gradient", confidence, color,
            [f"robust-full-image-surface-degree-{surface.degree}", "iterative-outlier-rejection"],
        )
    return BackgroundInfo("complex", 0.35, color, ["no-stable-multi-edge-background", "manual-review-required"])
