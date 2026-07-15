from __future__ import annotations

import cv2
import numpy as np

from ..schemas.asset import BBox, ConnectedRegion


def analyze_components(mask: np.ndarray, rgba: np.ndarray, min_area: int) -> tuple[list[ConnectedRegion], np.ndarray]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    h, w = mask.shape
    regions: list[ConnectedRegion] = []
    for label in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label])
        local = (labels[y:y + height, x:x + width] == label).astype(np.uint8)
        contours, hierarchy = cv2.findContours(local, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
        pixels = rgba[labels == label]
        complexity = perimeter * perimeter / max(1.0, 4.0 * np.pi * area)
        touches = x == 0 or y == 0 or x + width >= w or y + height >= h
        has_holes = bool(hierarchy is not None and np.any(hierarchy[0, :, 3] >= 0))
        average_color = tuple(float(v) for v in pixels[:, :3].mean(axis=0))
        color_spread = max(average_color) - min(average_color)
        average_alpha = float(pixels[:, 3].mean())
        slender_noise = (
            min(width, height) <= max(2, round(np.sqrt(max(1, min_area)) * 0.4))
            and area < min_area * 2
        )
        regions.append(ConnectedRegion(
            id=f"region-{label}", area=area, bbox=BBox(x, y, width, height),
            centroid=(float(centroids[label][0]), float(centroids[label][1])),
            perimeter=perimeter, aspect_ratio=width / max(1, height),
            average_color=average_color,
            average_alpha=average_alpha, contour_complexity=float(complexity),
            min_distance=float("inf"), touches_edge=touches, has_holes=has_holes,
            suspected_noise=area < min_area or slender_noise,
            suspected_shadow=average_alpha < 120 and color_spread < 45,
            label=label,
        ))
    for i, region in enumerate(regions):
        distances = []
        a = region.bbox
        for j, other in enumerate(regions):
            if i == j:
                continue
            b = other.bbox
            dx = max(a.x - b.x2, b.x - a.x2, 0)
            dy = max(a.y - b.y2, b.y - a.y2, 0)
            distances.append(float(np.hypot(dx, dy)))
        region.min_distance = min(distances, default=float(max(h, w)))
    return regions, labels
