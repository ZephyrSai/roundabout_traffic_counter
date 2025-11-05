"""
Helpers to scale lane geometry and ROI polygons when frames are resized.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .types import LaneGroup, LaneSegment


def scale_point(point: Tuple[int, int], scale: float) -> Tuple[int, int]:
    return (int(round(point[0] * scale)), int(round(point[1] * scale)))


def scale_segment(segment: LaneSegment, scale: float) -> LaneSegment:
    if scale == 1.0:
        return LaneSegment(segment.start, segment.end)
    return LaneSegment(scale_point(segment.start, scale), scale_point(segment.end, scale))


def scale_lane_groups(lane_groups: Sequence[LaneGroup], scale: float) -> List[LaneGroup]:
    if scale == 1.0:
        return [LaneGroup(group.name, group.color, group.direction, list(group.segments)) for group in lane_groups]
    scaled: List[LaneGroup] = []
    for group in lane_groups:
        scaled_segments = [scale_segment(segment, scale) for segment in group.segments]
        scaled.append(LaneGroup(name=group.name, color=group.color, direction=group.direction, segments=scaled_segments))
    return scaled


def scale_polygon(polygon: Optional[Sequence[Tuple[int, int]]], scale: float) -> Optional[List[Tuple[int, int]]]:
    if not polygon or scale == 1.0:
        return list(polygon) if polygon else None
    return [scale_point(pt, scale) for pt in polygon]


__all__ = ["scale_lane_groups", "scale_polygon", "scale_point"]
