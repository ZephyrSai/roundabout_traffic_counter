"""
Lane configuration helpers: loading, saving, validation, and defaults.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from .constants import (
    BASE_STAGE_NAME,
    DEFAULT_DIRECTION,
    DEFAULT_DIRECTION_ALIASES,
    DEFAULT_GROUP_COLORS,
    INDEPENDENT_GROUPS,
    VALID_DIRECTIONS,
)
from .roi import segments_within_polygon
from .types import LaneGroup, LaneSegment


def generate_default_lane_groups(
    group_names: Sequence[str],
    directions: Dict[str, str],
    frame_width: int,
    frame_height: int,
) -> List[LaneGroup]:
    lane_groups: List[LaneGroup] = []
    total = max(1, len(group_names))
    for idx, name in enumerate(group_names):
        direction = directions.get(name, DEFAULT_DIRECTION)
        orientation, _ = VALID_DIRECTIONS.get(direction, VALID_DIRECTIONS[DEFAULT_DIRECTION])
        color = DEFAULT_GROUP_COLORS[idx % len(DEFAULT_GROUP_COLORS)]
        if orientation == "vertical":
            step = frame_height / (total + 1)
            y = int(round(step * (idx + 1)))
            segment = LaneSegment((0, y), (frame_width - 1, y))
        else:
            step = frame_width / (total + 1)
            x = int(round(step * (idx + 1)))
            segment = LaneSegment((x, 0), (x, frame_height - 1))
        lane_groups.append(LaneGroup(name=name, color=color, direction=direction, segments=[segment]))
    return lane_groups


def load_lane_groups(
    path: Path,
    fallback_names: Sequence[str],
    directions: Dict[str, str],
) -> Tuple[List[LaneGroup], Optional[List[Tuple[int, int]]]]:
    if not path.exists():
        raise FileNotFoundError(f"Lane config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return [], None

    roi_polygon: Optional[List[Tuple[int, int]]] = None
    if isinstance(data, dict) and "groups" in data:
        groups_payload = data.get("groups", [])
        roi_payload = data.get("roi")
        if isinstance(roi_payload, (list, tuple)) and len(roi_payload) >= 3:
            extracted: List[Tuple[int, int]] = []
            for point in roi_payload:
                if (
                    isinstance(point, (list, tuple))
                    and len(point) == 2
                    and all(isinstance(coord, (int, float)) for coord in point)
                ):
                    extracted.append((int(point[0]), int(point[1])))
                else:
                    logging.warning("Ignoring malformed ROI point: %s", point)
            if len(extracted) >= 3:
                roi_polygon = extracted
    elif isinstance(data, list):
        name = fallback_names[0] if fallback_names else "stage1"
        groups_payload = [{"name": name, "segments": data}]
    elif isinstance(data, dict) and "segments" in data:
        name = fallback_names[0] if fallback_names else "stage1"
        groups_payload = [{"name": name, "segments": data.get("segments", [])}]
    else:
        raise ValueError("Lane config must contain a 'groups' list or be a list of segments.")

    lane_groups: List[LaneGroup] = []
    for idx, entry in enumerate(groups_payload):
        if not isinstance(entry, dict):
            logging.warning("Ignoring malformed lane group entry: %s", entry)
            continue
        name = entry.get("name") or (fallback_names[idx] if idx < len(fallback_names) else f"stage{idx+1}")
        direction_raw = entry.get("direction", directions.get(name, DEFAULT_DIRECTION)).lower()
        if direction_raw not in VALID_DIRECTIONS:
            logging.warning(
                "Unknown direction '%s' in config for %s. Using default %s.",
                direction_raw,
                name,
                DEFAULT_DIRECTION,
            )
            direction_raw = directions.get(name, DEFAULT_DIRECTION)
        segments: List[LaneSegment] = []
        for seg in entry.get("segments", []):
            if (
                isinstance(seg, (list, tuple))
                and len(seg) == 2
                and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in seg)
            ):
                (x1, y1), (x2, y2) = seg
                segments.append(LaneSegment((int(x1), int(y1)), (int(x2), int(y2))))
            else:
                logging.warning("Ignoring malformed segment in group %s: %s", name, seg)
        if not segments:
            continue
        color_raw = entry.get("color")
        if (
            isinstance(color_raw, (list, tuple))
            and len(color_raw) == 3
            and all(isinstance(component, (int, float)) for component in color_raw)
        ):
            color = tuple(int(component) for component in color_raw)
        else:
            color = DEFAULT_GROUP_COLORS[idx % len(DEFAULT_GROUP_COLORS)]
        lane_groups.append(LaneGroup(name=name, color=color, direction=direction_raw, segments=segments))
    return lane_groups, roi_polygon


def save_lane_groups(
    path: Path,
    lane_groups: Sequence[LaneGroup],
    roi_polygon: Optional[Sequence[Tuple[int, int]]] = None,
) -> None:
    payload = {
        "groups": [
            {
                "name": group.name,
                "color": list(group.color),
                "direction": group.direction,
                "segments": [segment.as_list() for segment in group.segments],
            }
            for group in lane_groups
        ],
    }
    if roi_polygon and len(roi_polygon) >= 3:
        payload["roi"] = [[int(x), int(y)] for x, y in roi_polygon]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def ensure_lane_groups(
    group_names: Sequence[str],
    lane_groups: Sequence[LaneGroup],
    directions: Dict[str, str],
    frame_width: int,
    frame_height: int,
    *,
    auto_fill_missing: bool = True,
) -> List[LaneGroup]:
    existing_map = {group.name: group for group in lane_groups}
    final_groups: List[LaneGroup] = []
    for idx, name in enumerate(group_names):
        direction = directions.get(name, DEFAULT_DIRECTION)
        if name in existing_map:
            group = existing_map[name]
            if group.direction != direction:
                group = LaneGroup(name=name, color=group.color, direction=direction, segments=group.segments)
            final_groups.append(group)
        elif auto_fill_missing:
            group = generate_default_lane_groups([name], directions, frame_width, frame_height)[0]
            group.color = DEFAULT_GROUP_COLORS[idx % len(DEFAULT_GROUP_COLORS)]
            final_groups.append(group)
        else:
            logging.info("Skipping group %s: no segments provided and auto-fill disabled.", name)
    return final_groups


__all__ = [
    "generate_default_lane_groups",
    "load_lane_groups",
    "save_lane_groups",
    "ensure_lane_groups",
    "BASE_STAGE_NAME",
    "INDEPENDENT_GROUPS",
]
