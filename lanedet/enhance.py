"""
Frame enhancement utilities (contrast/brightness adjustments).
"""

from __future__ import annotations

import cv2
import numpy as np

from .constants import CLAHE


def apply_auto_enhance(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = CLAHE.apply(l_channel)
    enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def enhance_frame(frame: np.ndarray, contrast: float, brightness: float, auto: bool) -> np.ndarray:
    output = frame
    if auto:
        output = apply_auto_enhance(output)
    if contrast != 1.0 or brightness != 0.0:
        output = cv2.convertScaleAbs(output, alpha=contrast, beta=brightness)
    return output


__all__ = ["enhance_frame", "apply_auto_enhance"]
