"""
Application-wide constants related to vehicle detection and lane rendering.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2

# Vehicle labels supported by YOLO11 and filtered for analytics.
VEHICLE_LABELS = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "motorbike",
    "bicycle",
    "train",
    "van",
    "trailer",
}

CAR_LABELS = {"car"}
TRUCK_LIKE_LABELS = {"truck", "trailer"}

DEFAULT_GROUP_COLORS: List[Tuple[int, int, int]] = [
    (0, 255, 255),
    (0, 165, 255),
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 255),
]

VALID_DIRECTIONS: Dict[str, Tuple[str, int]] = {
    "south-north": ("vertical", 1),
    "north-south": ("vertical", -1),
    "west-east": ("horizontal", 1),
    "east-west": ("horizontal", -1),
}

DEFAULT_DIRECTION = "south-north"
BASE_STAGE_NAME = "base"

DEFAULT_DIRECTION_ALIASES: Dict[str, str] = {
    BASE_STAGE_NAME: "south-north",
    "straight": "south-north",
    "turn": "south-north",
    "far_turn": "south-north",
    "uturn": "north-south",
    "reverse": "north-south",
}

# Stages that can be counted without first crossing the base lane group.
INDEPENDENT_GROUPS = {BASE_STAGE_NAME, "reverse"}

# OpenCV drawing settings.
LANE_TEXT_BG = (0, 0, 0)
LANE_TEXT_FG = (255, 255, 255)
LANE_TEXT_SCALE = 0.5
LANE_TEXT_THICKNESS = 1
LANE_LINE_THICKNESS = 2

# CLAHE instance reused for auto-enhance to avoid recreating for every frame.
CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

ROI_LINE_COLOR = (255, 255, 0)
ROI_FILL_ALPHA = 0.18
