"""
Geometry utilities for detecting when tracked centroids cross lane segments.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .constants import DEFAULT_DIRECTION, VALID_DIRECTIONS
from .roi import polygon_contains_point
from .types import LaneSegment


def point_within_segment_projection(point: Tuple[float, float], segment: LaneSegment, margin: float = 15.0) -> bool:
    x, y = point
    x_min = min(segment.start[0], segment.end[0]) - margin
    x_max = max(segment.start[0], segment.end[0]) + margin
    y_min = min(segment.start[1], segment.end[1]) - margin
    y_max = max(segment.start[1], segment.end[1]) + margin
    return x_min <= x <= x_max and y_min <= y <= y_max


def _orientation(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> float:
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    q1: Tuple[float, float],
    q2: Tuple[float, float],
    margin: float = 1.0,
) -> bool:
    p_min_x = min(p1[0], p2[0]) - margin
    p_max_x = max(p1[0], p2[0]) + margin
    p_min_y = min(p1[1], p2[1]) - margin
    p_max_y = max(p1[1], p2[1]) + margin

    q_min_x = min(q1[0], q2[0]) - margin
    q_max_x = max(q1[0], q2[0]) + margin
    q_min_y = min(q1[1], q2[1]) - margin
    q_max_y = max(q1[1], q2[1]) + margin

    if p_max_x < q_min_x or q_max_x < p_min_x or p_max_y < q_min_y or q_max_y < p_min_y:
        return False

    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    eps = 1e-6
    if (o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps):
        if (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps):
            return True

    def on_segment(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
        return (
            min(a[0], b[0]) - margin <= c[0] <= max(a[0], b[0]) + margin
            and min(a[1], b[1]) - margin <= c[1] <= max(a[1], b[1]) + margin
        )

    if abs(o1) <= eps and on_segment(p1, p2, q1):
        return True
    if abs(o2) <= eps and on_segment(p1, p2, q2):
        return True
    if abs(o3) <= eps and on_segment(q1, q2, p1):
        return True
    if abs(o4) <= eps and on_segment(q1, q2, p2):
        return True

    return False


def crosses_segment(previous: Tuple[float, float], current: Tuple[float, float], segment: LaneSegment, direction: str) -> bool:
    orientation, forward = VALID_DIRECTIONS.get(direction, VALID_DIRECTIONS[DEFAULT_DIRECTION])
    prev_x, prev_y = previous
    cur_x, cur_y = current

    if not segments_intersect(previous, current, segment.start, segment.end, margin=5.0):
        return False

    if orientation == "vertical":
        delta = cur_y - prev_y
        if abs(delta) < 1.0:
            return False
        return delta < 0 if forward >= 0 else delta > 0

    delta = cur_x - prev_x
    if abs(delta) < 1.0:
        return False
    return delta > 0 if forward >= 0 else delta < 0


__all__ = [
    "crosses_segment",
    "segments_intersect",
    "point_within_segment_projection",
]
