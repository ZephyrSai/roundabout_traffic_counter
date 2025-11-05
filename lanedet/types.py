"""
Dataclasses and basic geometry helpers for lane analytics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class LaneSegment:
    """Line segment drawn across the frame for counting in/out traffic."""

    start: Tuple[int, int]
    end: Tuple[int, int]

    def midpoint(self) -> Tuple[float, float]:
        return ((self.start[0] + self.end[0]) / 2.0, (self.start[1] + self.end[1]) / 2.0)

    def as_list(self) -> List[List[int]]:
        return [
            [int(self.start[0]), int(self.start[1])],
            [int(self.end[0]), int(self.end[1])],
        ]


@dataclass
class LaneGroup:
    """Collection of lane segments with shared name, color, and direction."""

    name: str
    color: Tuple[int, int, int]
    direction: str
    segments: List[LaneSegment]
