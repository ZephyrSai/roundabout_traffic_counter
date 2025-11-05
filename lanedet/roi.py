"""
ROI helpers including interactive drawing and polygon membership checks.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .constants import LANE_LINE_THICKNESS, ROI_LINE_COLOR
from .drawing import configure_window_to_frame, draw_roi_overlay, draw_text_with_bg
from .types import LaneSegment


def polygon_contains_point(point: Tuple[float, float], polygon: Sequence[Tuple[int, int]]) -> bool:
    if not polygon or len(polygon) < 3:
        return True
    contour = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def segment_within_polygon(segment: LaneSegment, polygon: Sequence[Tuple[int, int]]) -> bool:
    if not polygon or len(polygon) < 3:
        return True
    samples = [
        segment.start,
        segment.end,
        segment.midpoint(),
        (
            (segment.start[0] * 3 + segment.end[0]) / 4.0,
            (segment.start[1] * 3 + segment.end[1]) / 4.0,
        ),
        (
            (segment.start[0] + segment.end[0] * 3) / 4.0,
            (segment.start[1] + segment.end[1] * 3) / 4.0,
        ),
    ]
    return all(polygon_contains_point(sample, polygon) for sample in samples)


def segments_within_polygon(segments: Sequence[LaneSegment], polygon: Sequence[Tuple[int, int]]) -> bool:
    return all(segment_within_polygon(segment, polygon) for segment in segments)


def interactive_roi_polygon(
    frame: np.ndarray,
    existing_points: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    window_name = "Define ROI"
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    points: List[Tuple[int, int]] = []
    if existing_points:
        for px, py in existing_points:
            x = int(np.clip(px, 0, width - 1))
            y = int(np.clip(py, 0, height - 1))
            points.append((x, y))

    def mouse_handler(event, x, y, _flags, _param) -> None:
        nonlocal points
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    instructions = [
        "Define ROI polygon",
        "Left-click: add vertex",
        "Right-click: undo last vertex",
        "Enter/Space: confirm (>=3 points)  |  C: clear",
    ]

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_handler)
    configure_window_to_frame(window_name, frame)

    while True:
        preview = canvas.copy()
        if points:
            cv2.polylines(
                preview,
                [np.array(points, dtype=np.int32)],
                isClosed=False,
                color=ROI_LINE_COLOR,
                thickness=LANE_LINE_THICKNESS,
            )
            for pt in points:
                cv2.circle(preview, pt, 4, ROI_LINE_COLOR, -1)
        for idx, message in enumerate(instructions):
            draw_text_with_bg(preview, message, (10, 25 + idx * 22))
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10, 32):
            if len(points) >= 3:
                break
            cv2.setWindowTitle(window_name, "Need at least 3 points")
        if key in (27, ord("q"), ord("Q")):
            points = []
            break
        if key in (ord("c"), ord("C"), ord("r"), ord("R")):
            points.clear()

    cv2.destroyWindow(window_name)
    return points
