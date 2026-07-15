from __future__ import annotations

import cv2
import numpy as np


def projection_split(mask: np.ndarray, minimum_fraction: float = 0.15) -> list[tuple[int, int, int, int]]:
    """Return conservative zero-gap splits for a local candidate mask."""
    h, w = mask.shape
    for axis in (0, 1):
        projection = np.count_nonzero(mask, axis=axis)
        zeros = np.flatnonzero(projection == 0)
        for cut in zeros:
            length = w if axis == 0 else h
            if length * minimum_fraction <= cut <= length * (1 - minimum_fraction):
                if axis == 0:
                    return [(0, 0, int(cut), h), (int(cut + 1), 0, w - int(cut + 1), h)]
                return [(0, 0, w, int(cut)), (0, int(cut + 1), w, h - int(cut + 1))]
    return [(0, 0, w, h)]


def _split_colors(local_rgb: np.ndarray | None, parts: list[np.ndarray]) -> float | None:
    if local_rgb is None:
        return None
    colors = []
    for part in parts:
        pixels = local_rgb[part > 0]
        if not len(pixels):
            return None
        colors.append(np.median(pixels, axis=0).astype(np.float32))
    # Brightness changes across antialiased gray strokes or shadows are not
    # independent color evidence. Require chromatic information on at least
    # one side before color can authorize an automatic split.
    if all(float(np.ptp(color)) < 25.0 for color in colors):
        return 0.0
    return float(np.linalg.norm(colors[0] - colors[1]))


def _valid_parts(parts: list[np.ndarray], total: int) -> bool:
    areas = [int(np.count_nonzero(part)) for part in parts]
    if len(areas) != 2 or min(areas) < total * 0.25:
        return False
    for part, area in zip(parts, areas):
        ys, xs = np.nonzero(part)
        if not len(xs):
            return False
        fill = area / max(1, (int(xs.max() - xs.min() + 1) * int(ys.max() - ys.min() + 1)))
        if fill < 0.22:
            return False
    return True


def _restore_partition(local: np.ndarray, parts: list[np.ndarray], local_rgb: np.ndarray | None = None) -> list[np.ndarray]:
    """Assign all pixels by seed distance and, when available, color continuity."""
    distances = [cv2.distanceTransform((part == 0).astype(np.uint8), cv2.DIST_L2, 5) for part in parts]
    spatial = np.stack(distances) / max(2.0, min(local.shape) * 0.05)
    scores = spatial
    if local_rgb is not None:
        lab = cv2.cvtColor(local_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        colors = []
        for part in parts:
            pixels = lab[part > 0]
            colors.append(np.median(pixels, axis=0) if len(pixels) else np.zeros(3, np.float32))
        separation = float(np.linalg.norm(colors[0] - colors[1]))
        if separation >= 18.0:
            color_scores = np.stack([np.linalg.norm(lab - color, axis=2) for color in colors])
            scores = spatial + color_scores / max(12.0, separation * 0.30) * 2.0
    assignment = np.argmin(scores, axis=0)
    return [((local > 0) & (assignment == index)).astype(np.uint8) for index in range(len(parts))]


def split_touching_mask(
    mask: np.ndarray,
    strength: float = 0.5,
    rgb: np.ndarray | None = None,
) -> tuple[list[np.ndarray], str | None]:
    """Generate split candidates, then accept only independently supported ones."""
    binary = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(binary)
    if not len(xs):
        return [mask], None
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    local = binary[y1:y2, x1:x2]
    local_rgb = rgb[y1:y2, x1:x2] if rgb is not None else None
    h, w = local.shape
    aspect = max(w / max(1, h), h / max(1, w))
    if aspect < 1.35:
        return [mask], None

    possible_reason: str | None = None
    for axis in (0, 1):
        projection = np.count_nonzero(local, axis=axis).astype(np.float32)
        length = len(projection)
        margin = max(2, round(length * 0.2))
        if length <= margin * 2:
            continue
        smooth = cv2.GaussianBlur(projection.reshape(1, -1), (0, 0), sigmaX=max(1, length * 0.02)).ravel()
        cut = margin + int(np.argmin(smooth[margin:length - margin]))
        valley_ratio = float(smooth[cut] / max(1.0, float(smooth.max())))
        if valley_ratio > 0.48 - 0.18 * strength:
            continue
        selectors = (
            ((slice(None), slice(0, cut + 1)), (slice(None), slice(cut + 1, None)))
            if axis == 0
            else ((slice(0, cut + 1), slice(None)), (slice(cut + 1, None), slice(None)))
        )
        local_parts = []
        for selector in selectors:
            part = np.zeros_like(local)
            part[selector] = local[selector]
            local_parts.append(part)
        if not _valid_parts(local_parts, int(np.count_nonzero(local))):
            continue
        color_distance = _split_colors(local_rgb, local_parts)
        # Different body colors provide independent evidence that the valley
        # separates two assets. With no color evidence, only explicit/manual
        # callers may accept an exceptionally narrow bridge.
        accepted = color_distance is not None and color_distance >= 28.0
        accepted = accepted or (color_distance is None and valley_ratio <= 0.12)
        if accepted:
            local_parts = _restore_partition(local, local_parts, local_rgb)
            parts = []
            for local_part in local_parts:
                piece = np.zeros_like(mask, dtype=np.uint8)
                piece[y1:y2, x1:x2] = local_part * 255
                parts.append(piece)
            return parts, "projection-valley"
        possible_reason = "possible-split:projection-valley"

    distance = cv2.distanceTransform(local, cv2.DIST_L2, 5)
    if distance.max() <= 0:
        return [mask], possible_reason
    sure = (distance >= distance.max() * (0.58 + 0.12 * (1.0 - strength))).astype(np.uint8)
    count, markers = cv2.connectedComponents(sure)
    if count != 3:
        return [mask], possible_reason
    markers = markers + 1
    unknown = cv2.subtract(local, sure)
    markers[unknown > 0] = 0
    terrain = cv2.cvtColor((255 - np.clip(distance / distance.max() * 255, 0, 255)).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    watershed = cv2.watershed(terrain, markers.astype(np.int32))
    local_parts = [((watershed == label) & (local > 0)).astype(np.uint8) for label in (2, 3)]
    local_parts = _restore_partition(local, local_parts, local_rgb)
    total = int(np.count_nonzero(local))
    if not _valid_parts(local_parts, total):
        return [mask], possible_reason
    color_distance = _split_colors(local_rgb, local_parts)
    if color_distance is not None and color_distance >= 35.0:
        parts = []
        for local_part in local_parts:
            piece = np.zeros_like(mask, dtype=np.uint8)
            piece[y1:y2, x1:x2] = local_part * 255
            parts.append(piece)
        return parts, "distance-transform-watershed"
    return [mask], possible_reason or "possible-split:distance-transform-watershed"
