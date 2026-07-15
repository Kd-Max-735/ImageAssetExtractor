from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schemas.asset import BBox, ConnectedRegion


@dataclass(frozen=True)
class ClusteringThresholds:
    """Dimensionless ownership thresholds, relative to nearby components."""

    satellite_area_min: float = 0.015
    satellite_area_max: float = 0.35
    satellite_gap_scale: float = 1.45
    satellite_center_scale: float = 2.0
    satellite_color_distance: float = 58.0
    pattern_area_ratio: float = 0.45
    pattern_gap_scale: float = 1.8
    pattern_color_distance: float = 48.0
    pattern_alignment_scale: float = 0.6
    # Cross-cell ownership needs local geometric evidence. 0.20 preserves
    # nearby multi-stroke symbols while rejecting row/column-spanning jumps.
    cross_cell_gap_scale: float = 0.20
    max_cluster_span_scale: float = 6.0


THRESHOLDS = ClusteringThresholds()


def _bbox_gap(a: BBox, b: BBox) -> float:
    dx = max(a.x - b.x2, b.x - a.x2, 0)
    dy = max(a.y - b.y2, b.y - a.y2, 0)
    return float(np.hypot(dx, dy))


def _color_distance(a: ConnectedRegion, b: ConnectedRegion) -> float:
    return float(np.linalg.norm(np.asarray(a.average_color) - np.asarray(b.average_color)))


def _cell_index(region: ConnectedRegion, cells: list[tuple[int, int, int, int]] | None) -> int | None:
    if not cells:
        return None
    cx, cy = region.centroid
    return next((index for index, (x, y, w, h) in enumerate(cells) if x <= cx < x + w and y <= cy < y + h), None)


def _reliable_grid(cells: list[tuple[int, int, int, int]] | None) -> bool:
    if not cells or len(cells) < 2:
        return False
    centers = {(x * 2 + width, y * 2 + height) for x, y, width, height in cells}
    x_centers = {center[0] for center in centers}
    y_centers = {center[1] for center in centers}
    return len(centers) == len(cells) and len(x_centers) * len(y_centers) == len(cells)


def _cross_cell_gap_allowed(
    a: ConnectedRegion,
    b: ConnectedRegion,
    cells: list[tuple[int, int, int, int]] | None,
    global_distance: float,
    local_scale: float,
    enforce: bool,
) -> bool:
    if not enforce:
        return True
    a_cell, b_cell = _cell_index(a, cells), _cell_index(b, cells)
    if a_cell is None or b_cell is None or a_cell == b_cell:
        return True
    return _bbox_gap(a.bbox, b.bbox) <= max(
        global_distance,
        local_scale * THRESHOLDS.cross_cell_gap_scale,
    )


def _cluster_valid(indices: set[int], regions: list[ConnectedRegion], global_distance: float) -> bool:
    members = [regions[index] for index in indices]
    x1, y1 = min(r.bbox.x for r in members), min(r.bbox.y for r in members)
    x2, y2 = max(r.bbox.x2 for r in members), max(r.bbox.y2 for r in members)
    anchor = max(max(r.bbox.width, r.bbox.height) for r in members)
    limit = max(global_distance * 4.0, anchor * THRESHOLDS.max_cluster_span_scale)
    return x2 - x1 <= limit and y2 - y1 <= limit


def cluster_regions(
    regions: list[ConnectedRegion],
    image_shape: tuple[int, int],
    merge_ratio: float,
    mask: np.ndarray | None = None,
    grid_cells: list[tuple[int, int, int, int]] | None = None,
) -> list[list[ConnectedRegion]]:
    if not regions:
        return []
    global_distance = max(1.0, min(image_shape) * merge_ratio)
    enforce_cross_cell_guard = _reliable_grid(grid_cells)
    clusters: list[set[int]] = [{index} for index in range(len(regions))]

    def cluster_of(index: int) -> int:
        return next(position for position, members in enumerate(clusters) if index in members)

    def merge(indices: set[int]) -> bool:
        positions = sorted({cluster_of(index) for index in indices})
        combined = set().union(*(clusters[position] for position in positions))
        if not _cluster_valid(combined, regions, global_distance):
            return False
        clusters[positions[0]] = combined
        for position in reversed(positions[1:]):
            clusters.pop(position)
        return True

    # Locally adjacent strokes inside an icon are often similar in size. The
    # configured image-scale distance is retained as a strict cap here; wider
    # same-size pairs remain separate unless a larger repeat pattern supports
    # them.
    compact_pairs: list[tuple[float, int, int]] = []
    for i, a in enumerate(regions):
        for j in range(i + 1, len(regions)):
            b = regions[j]
            gap = _bbox_gap(a.bbox, b.bbox)
            overlaps_axis = not (
                a.bbox.x2 <= b.bbox.x or b.bbox.x2 <= a.bbox.x
            ) or not (
                a.bbox.y2 <= b.bbox.y or b.bbox.y2 <= a.bbox.y
            )
            different_cells = (
                _cell_index(a, grid_cells) is not None
                and _cell_index(b, grid_cells) is not None
                and _cell_index(a, grid_cells) != _cell_index(b, grid_cells)
            )
            same_cell = (
                _cell_index(a, grid_cells) is not None
                and _cell_index(a, grid_cells) == _cell_index(b, grid_cells)
            )
            contained = (
                a.bbox.x <= b.bbox.x and a.bbox.y <= b.bbox.y
                and a.bbox.x2 >= b.bbox.x2 and a.bbox.y2 >= b.bbox.y2
            ) or (
                b.bbox.x <= a.bbox.x and b.bbox.y <= a.bbox.y
                and b.bbox.x2 >= a.bbox.x2 and b.bbox.y2 >= a.bbox.y2
            )
            if (
                gap <= global_distance
                and overlaps_axis
                and not different_cells
                and not (a.suspected_noise and b.suspected_noise)
                and (
                    _color_distance(a, b) <= THRESHOLDS.satellite_color_distance
                    or (same_cell and contained)
                )
            ):
                compact_pairs.append((gap, i, j))
    for _, i, j in sorted(compact_pairs):
        merge({i, j})

    # Attach a smaller satellite only to its unique nearest compatible body.
    # This happens before any small singleton is treated as noise.
    for small_index, small in sorted(enumerate(regions), key=lambda item: item[1].area):
        compatible: list[tuple[float, int]] = []
        for large_index, large in enumerate(regions):
            if small_index == large_index or large.area <= small.area:
                continue
            ratio = small.area / max(1, large.area)
            if not THRESHOLDS.satellite_area_min <= ratio <= THRESHOLDS.satellite_area_max:
                continue
            scale = max(1.0, large.bbox.width, large.bbox.height)
            gap = _bbox_gap(small.bbox, large.bbox)
            center_distance = float(np.linalg.norm(np.asarray(small.centroid) - np.asarray(large.centroid)))
            same_cell = _cell_index(small, grid_cells) == _cell_index(large, grid_cells)
            allowance = 1.15 if same_cell else 1.0
            if (
                gap <= max(global_distance, scale * THRESHOLDS.satellite_gap_scale * allowance)
                and center_distance <= scale * THRESHOLDS.satellite_center_scale * allowance
                and _color_distance(small, large) <= THRESHOLDS.satellite_color_distance
                and _cross_cell_gap_allowed(
                    small, large, grid_cells, global_distance, scale, enforce_cross_cell_guard,
                )
            ):
                existing_satellites = len(clusters[cluster_of(large_index)]) - 1
                structure_bonus = min(0.24, existing_satellites * 0.08)
                compatible.append((gap / scale + center_distance / scale * 0.2 - structure_bonus, large_index))
        compatible.sort()
        if compatible and (len(compatible) == 1 or compatible[1][0] > compatible[0][0] * 1.25 + 0.12):
            merge({small_index, compatible[0][1]})

    # Three or more similarly sized, similarly colored, collinear pieces form a
    # repeat-pattern candidate (ellipsis, Wi-Fi arcs). A pair is intentionally
    # insufficient, which protects adjacent independent icons.
    graph: list[set[int]] = [set() for _ in regions]
    for i, a in enumerate(regions):
        for j in range(i + 1, len(regions)):
            b = regions[j]
            size_ratio = min(a.area, b.area) / max(a.area, b.area)
            scale = max(1.0, a.bbox.width, a.bbox.height, b.bbox.width, b.bbox.height)
            if (
                not (a.suspected_noise or b.suspected_noise)
                and
                size_ratio >= THRESHOLDS.pattern_area_ratio
                and _color_distance(a, b) <= THRESHOLDS.pattern_color_distance
                and _bbox_gap(a.bbox, b.bbox) <= max(global_distance, scale * THRESHOLDS.pattern_gap_scale)
            ):
                graph[i].add(j)
                graph[j].add(i)
    unseen = set(range(len(regions)))
    while unseen:
        seed = unseen.pop()
        group, pending = {seed}, [seed]
        while pending:
            current = pending.pop()
            additions = graph[current] & unseen
            unseen -= additions
            group |= additions
            pending.extend(additions)
        if len(group) < 3:
            continue
        centers = np.asarray([regions[index].centroid for index in group], dtype=np.float32)
        centered = centers - centers.mean(axis=0)
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        perpendicular = centered @ axes[-1]
        local_scale = float(np.median([max(r.bbox.width, r.bbox.height) for r in (regions[index] for index in group)]))
        cell_ids = [_cell_index(regions[index], grid_cells) for index in group]
        if all(cell is not None for cell in cell_ids) and len(set(cell_ids)) == len(cell_ids):
            nearest_gaps = []
            for index in group:
                nearest_gaps.append(min(
                    _bbox_gap(regions[index].bbox, regions[other].bbox)
                    for other in group if other != index
                ))
            if merge_ratio <= 0.02 and float(np.median(nearest_gaps)) <= local_scale * 0.25:
                continue
        if float(np.ptp(perpendicular)) <= max(2.0, local_scale * THRESHOLDS.pattern_alignment_scale):
            merge(group)

    # A repeat-pattern cluster can itself be the body of a multi-component
    # symbol (for example Wi-Fi arcs plus a dot). Resolve remaining singleton
    # satellites against the whole structure rather than one ambiguous arc.
    for singleton in list(range(len(regions))):
        if len(clusters[cluster_of(singleton)]) != 1:
            continue
        small = regions[singleton]
        candidates: list[tuple[float, int]] = []
        for position, members in enumerate(clusters):
            if len(members) < 2 or singleton in members:
                continue
            body = [regions[index] for index in members]
            body_area = sum(region.area for region in body)
            ratio = small.area / max(1, body_area)
            if not THRESHOLDS.satellite_area_min <= ratio <= THRESHOLDS.satellite_area_max:
                continue
            box = BBox(
                min(r.bbox.x for r in body), min(r.bbox.y for r in body),
                max(r.bbox.x2 for r in body) - min(r.bbox.x for r in body),
                max(r.bbox.y2 for r in body) - min(r.bbox.y for r in body),
            )
            scale = max(1.0, box.width, box.height)
            body_color = np.average(
                np.asarray([r.average_color for r in body]), axis=0,
                weights=np.asarray([r.area for r in body]),
            )
            color_distance = float(np.linalg.norm(np.asarray(small.average_color) - body_color))
            center = np.average(
                np.asarray([r.centroid for r in body]), axis=0,
                weights=np.asarray([r.area for r in body]),
            )
            center_distance = float(np.linalg.norm(np.asarray(small.centroid) - center))
            gap = _bbox_gap(small.bbox, box)
            body_cells = {_cell_index(region, grid_cells) for region in body}
            small_cell = _cell_index(small, grid_cells)
            cross_cell_allowed = (
                not enforce_cross_cell_guard
                or
                small_cell is None
                or None in body_cells
                or small_cell in body_cells
                or gap <= max(global_distance, scale * THRESHOLDS.cross_cell_gap_scale)
            )
            if (
                gap <= max(global_distance, scale * THRESHOLDS.satellite_gap_scale)
                and center_distance <= scale * THRESHOLDS.satellite_center_scale
                and color_distance <= THRESHOLDS.satellite_color_distance
                and cross_cell_allowed
            ):
                candidates.append((gap / scale + center_distance / scale * 0.2, next(iter(members))))
        candidates.sort()
        if candidates and (len(candidates) == 1 or candidates[1][0] > candidates[0][0] * 1.25 + 0.12):
            merge({singleton, candidates[0][1]})

    return [[regions[index] for index in sorted(cluster)] for cluster in clusters]
