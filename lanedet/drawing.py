"""
Drawing helpers for overlays and interactive UI elements.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .constants import (
    LANE_LINE_THICKNESS,
    LANE_TEXT_BG,
    LANE_TEXT_FG,
    LANE_TEXT_SCALE,
    LANE_TEXT_THICKNESS,
    ROI_FILL_ALPHA,
    ROI_LINE_COLOR,
)
from .types import LaneGroup, LaneSegment
from .utils import class_name_lookup


def draw_text_with_bg(frame: np.ndarray, text: str, position: Tuple[int, int]) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, LANE_TEXT_SCALE, LANE_TEXT_THICKNESS)
    x, y = position
    cv2.rectangle(frame, (x, y - th - baseline), (x + tw, y + baseline), LANE_TEXT_BG, thickness=cv2.FILLED)
    cv2.putText(
        frame,
        text,
        (x, y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        LANE_TEXT_SCALE,
        LANE_TEXT_FG,
        LANE_TEXT_THICKNESS,
        cv2.LINE_AA,
    )


def configure_window_to_frame(window_name: str, frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    cv2.resizeWindow(window_name, max(width, 320), max(height, 240))


def id_to_color(track_id: int) -> Tuple[int, int, int]:
    return (
        (37 * track_id) % 255,
        (17 * track_id) % 255,
        (29 * track_id) % 255,
    )


def draw_tracks(frame: np.ndarray, tracks: Optional[np.ndarray], names: Union[Sequence[str], dict]) -> np.ndarray:
    if tracks is None or len(tracks) == 0:
        return frame

    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for track in tracks:
        if len(track) < 7:
            continue
        x1, y1, x2, y2, track_id, score, cls_idx = track[:7]
        track_id = int(track_id)
        cls_idx = int(cls_idx)

        x1_int = max(0, min(int(round(x1)), width - 1))
        y1_int = max(0, min(int(round(y1)), height - 1))
        x2_int = max(0, min(int(round(x2)), width - 1))
        y2_int = max(0, min(int(round(y2)), height - 1))

        color = id_to_color(track_id)
        cv2.rectangle(annotated, (x1_int, y1_int), (x2_int, y2_int), color, 2)
        cv2.circle(
            annotated,
            (int((x1_int + x2_int) / 2), int((y1_int + y2_int) / 2)),
            2,
            color,
            -1,
        )
    return annotated


def draw_lane_overlay(
    frame: np.ndarray,
    lane_groups: Sequence[LaneGroup],
    lane_counts: dict,
) -> np.ndarray:
    annotated = frame.copy()
    for group in lane_groups:
        display_name = group.name.replace("_", " ").upper()
        total = sum(lane_counts.get(group.name, {}).values())
        label = display_name if total == 0 else f"{display_name} {total}"
        label_drawn = False
        for segment in group.segments:
            cv2.line(annotated, segment.start, segment.end, group.color, LANE_LINE_THICKNESS)
            if not label_drawn:
                mid_x, mid_y = segment.midpoint()
                draw_text_with_bg(annotated, label, (max(10, int(mid_x) - 60), max(25, int(mid_y) - 10)))
                label_drawn = True
    return annotated


def draw_roi_overlay(frame: np.ndarray, roi_polygon: Optional[Sequence[Tuple[int, int]]]) -> np.ndarray:
    if not roi_polygon or len(roi_polygon) < 3:
        return frame
    overlay = frame.copy()
    contour = np.array(roi_polygon, dtype=np.int32)
    cv2.fillPoly(overlay, [contour], ROI_LINE_COLOR)
    blended = cv2.addWeighted(overlay, ROI_FILL_ALPHA, frame, 1 - ROI_FILL_ALPHA, 0.0)
    cv2.polylines(blended, [contour], isClosed=True, color=ROI_LINE_COLOR, thickness=LANE_LINE_THICKNESS)
    return blended
