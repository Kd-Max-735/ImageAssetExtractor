from __future__ import annotations

import cv2
import numpy as np


def adaptive_kernel(shape: tuple[int, int], ratio: float, minimum: int = 1, maximum: int = 15) -> int:
    size = int(round(min(shape) * ratio))
    size = max(minimum, min(maximum, size))
    return size if size % 2 == 1 else size + 1


def clean_detection_mask(mask: np.ndarray) -> np.ndarray:
    size = adaptive_kernel(mask.shape, 0.0015)
    if size <= 1:
        return mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    original = mask.astype(np.uint8)
    # Closing repairs tiny gaps without erasing one-pixel strokes. Isolated
    # noise is classified after component ownership has been considered.
    closed = cv2.morphologyEx(original, cv2.MORPH_CLOSE, kernel)
    return np.maximum(original, closed)
