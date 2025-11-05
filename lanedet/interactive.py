"""
Interactive tools for drawing ROIs and lane groups.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .constants import DEFAULT_DIRECTION, DEFAULT_GROUP_COLORS
from .drawing import configure_window_to_frame, draw_roi_overlay, draw_text_with_bg
from .lanes import generate_default_lane_groups
from .roi import (
    interactive_roi_polygon as _interactive_roi_polygon,
    segment_within_polygon,
    segments_within_polygon,
)
from .types import LaneGroup, LaneSegment
from .utils import human_direction


def interactive_lane_segments(
    frame: np.ndarray,
    title: str,
    color: Tuple[int, int, int],
    initial_segments: Optional[Sequence[LaneSegment]] = None,
    roi_polygon: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[LaneSegment]:
    window_name = f"Define Lanes - {title}"
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    segments: List[LaneSegment] = []
    if initial_segments:
        for seg in initial_segments:
            sx = int(np.clip(seg.start[0], 0, width - 1))
            sy = int(np.clip(seg.start[1], 0, height - 1))
            ex = int(np.clip(seg.end[0], 0, width - 1))
            ey = int(np.clip(seg.end[1], 0, height - 1))
            segments.append(LaneSegment((sx, sy), (ex, ey)))

    pending_point: Optional[Tuple[int, int]] = None

    def mouse_handler(event, x, y, _flags, _param) -> None:
        nonlocal pending_point, segments
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        if event == cv2.EVENT_LBUTTONDOWN:
            if pending_point is None:
                pending_point = (x, y)
            else:
                segments.append(LaneSegment(pending_point, (x, y)))
                pending_point = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            if pending_point is not None:
                pending_point = None
            elif segments:
                segments.pop()

    instructions = [
        f"{title} - draw lane segments",
        "Left-click twice: add a segment",
        "Right-click: undo point/segment",
        "Enter/Space: confirm  |  C: clear",
    ]
    if roi_polygon and len(roi_polygon) >= 3:
        instructions.append("Segments must stay within the highlighted ROI")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_handler)
    configure_window_to_frame(window_name, frame)

    while True:
        preview = canvas.copy()
        if roi_polygon and len(roi_polygon) >= 3:
            preview = draw_roi_overlay(preview, roi_polygon)
        for seg in segments:
            cv2.line(preview, seg.start, seg.end, color, 2)
        if pending_point is not None:
            cv2.circle(preview, pending_point, 5, color, -1)
        for idx, message in enumerate(instructions):
            draw_text_with_bg(preview, message, (10, 25 + idx * 22))
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10, 32) and pending_point is None:
            break
        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("c"), ord("C"), ord("r"), ord("R")):
            segments.clear()
            pending_point = None
        if key in (8, ord("u"), ord("U")) and segments:
            segments.pop()

    cv2.destroyWindow(window_name)
    return segments


def interactive_lane_groups(
    frame: np.ndarray,
    group_names: Sequence[str],
    directions: Dict[str, str],
    existing_groups: Optional[Sequence[LaneGroup]] = None,
    roi_polygon: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[LaneGroup]:
    existing_map = {group.name: group for group in existing_groups or []}
    lane_groups: List[LaneGroup] = []
    for idx, name in enumerate(group_names):
        base = existing_map.get(name)
        color = base.color if base else DEFAULT_GROUP_COLORS[idx % len(DEFAULT_GROUP_COLORS)]
        direction = directions.get(name, DEFAULT_DIRECTION)
        while True:
            segments = interactive_lane_segments(
                frame.copy(),
                f"{name} ({human_direction(direction)})",
                color,
                base.segments if base else None,
                roi_polygon=roi_polygon,
            )
            if not segments:
                logging.warning("No segments drawn for group %s; skipping.", name)
                break
            if roi_polygon and not segments_within_polygon(segments, roi_polygon):
                logging.warning(
                    "Segments for group %s extend outside the ROI. Please redraw this stage.",
                    name,
                )
                base = None
                continue
            lane_groups.append(LaneGroup(name=name, color=color, direction=direction, segments=segments))
            break
    return lane_groups


def interactive_roi_polygon(frame: np.ndarray, existing_points=None):
    """Thin wrapper around ROI editor so callers import from one module."""
    return _interactive_roi_polygon(frame, existing_points)


__all__ = [
    "interactive_lane_groups",
    "interactive_lane_segments",
    "interactive_roi_polygon",
]
