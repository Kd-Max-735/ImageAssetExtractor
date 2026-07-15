from __future__ import annotations

import numpy as np


def occupied_intervals(projection: np.ndarray) -> list[tuple[int, int]]:
    active = projection > 0
    changes = np.diff(np.pad(active.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _regular_repeated_axis(intervals: list[tuple[int, int]]) -> bool:
    if len(intervals) < 3:
        return False
    centers = np.asarray([(start + end) * 0.5 for start, end in intervals], np.float32)
    gaps = np.diff(centers)
    median_gap = float(np.median(gaps))
    if median_gap <= 0:
        return False
    return float(np.max(np.abs(gaps - median_gap))) <= median_gap * 0.12


def projection_cells(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    y_intervals = occupied_intervals(np.count_nonzero(mask, axis=1))
    x_intervals = occupied_intervals(np.count_nonzero(mask, axis=0))
    if _regular_repeated_axis(x_intervals) and _regular_repeated_axis(y_intervals):
        return [
            (x1, y1, x2 - x1, y2 - y1)
            for y1, y2 in y_intervals
            for x1, x2 in x_intervals
        ]
    cells: list[tuple[int, int, int, int]] = []
    for y1, y2 in y_intervals:
        for x1, x2 in occupied_intervals(np.count_nonzero(mask[y1:y2], axis=0)):
            cells.append((x1, y1, x2 - x1, y2 - y1))
    return cells
