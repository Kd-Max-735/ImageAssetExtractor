from __future__ import annotations

import numpy as np


def isolate_alpha(alpha: np.ndarray, detection_mask: np.ndarray) -> np.ndarray:
    result = alpha.copy()
    result[detection_mask == 0] = 0
    return result.astype(np.uint8)
