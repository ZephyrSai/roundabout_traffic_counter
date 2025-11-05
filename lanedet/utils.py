"""
General utility helpers used across the lane analytics modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import torch
import ultralytics
import yaml
from ultralytics.trackers.byte_tracker import BYTETracker

from .constants import DEFAULT_DIRECTION, DEFAULT_DIRECTION_ALIASES, VALID_DIRECTIONS


def normalize_class_name(name: str) -> str:
    return name.strip().lower()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_group_names(raw: str, fallback_count: int = 6) -> List[str]:
    if not raw:
        return [f"stage{i + 1}" for i in range(fallback_count)]
    return [name.strip() for name in raw.split(",") if name.strip()]


def parse_direction_map(raw: Optional[str], group_names: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if raw:
        for token in raw.split(","):
            token = token.strip()
            if not token or ":" not in token:
                continue
            name, direction = token.split(":", 1)
            direction = direction.strip().lower()
            if direction not in VALID_DIRECTIONS:
                logging.warning("Ignoring unknown direction '%s' for group '%s'.", direction, name)
                continue
            mapping[name.strip()] = direction
    for name in group_names:
        default_direction = DEFAULT_DIRECTION_ALIASES.get(name.lower(), DEFAULT_DIRECTION)
        mapping.setdefault(name, default_direction)
    return mapping


def human_direction(direction: str) -> str:
    labels = {
        "south-north": "south -> north",
        "north-south": "north -> south",
        "west-east": "west -> east",
        "east-west": "east -> west",
    }
    return labels.get(direction, direction)


def resolve_source(raw: str) -> Union[str, int]:
    lowered = raw.strip().lower()
    if lowered.startswith(("rtsp://", "http://", "https://")):
        return raw
    if raw.isdigit():
        return int(raw)
    path = Path(raw)
    if path.exists():
        return str(path.resolve())
    logging.warning("Source %s does not exist locally. Passing through to OpenCV unchanged.", raw)
    return raw


def ensure_torch_device() -> str:
    if torch is None:
        logging.error("PyTorch is missing. Install a CUDA-enabled build that matches your GPU/driver.")
        raise SystemExit(1)
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(torch.cuda.current_device())
        logging.info("Using CUDA device: %s", name)
        return "cuda"
    logging.warning("CUDA not available; defaulting to CPU. Performance may be limited.")
    return "cpu"


def build_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    fourcc = cv2.VideoWriter_fourcc(*("XVID" if suffix in (".avi", ".mov") else "mp4v"))
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def derive_default_save_path(source: Union[str, int]) -> Path:
    if isinstance(source, int):
        return Path(f"camera_{source}_detections.mp4")
    parsed = Path(str(source))
    if parsed.suffix:
        return parsed.with_name(f"{parsed.stem}_detections{parsed.suffix}")
    return Path(f"{parsed.name or 'stream'}_detections.mp4")


def derive_counts_path(source: Union[str, int]) -> Path:
    if isinstance(source, int):
        return Path(f"camera_{source}_counts.txt")
    parsed = Path(str(source))
    if parsed.suffix:
        return parsed.with_name(f"{parsed.stem}_counts.txt")
    return Path(f"{parsed.name or 'stream'}_counts.txt")


def load_tracker(config: str, frame_rate: float) -> BYTETracker:
    if config.lower() == "none":
        raise ValueError("Tracker disabled via configuration.")

    cfg_path = Path(config)
    if not cfg_path.exists():
        script_candidate = Path(__file__).resolve().parent / config
        if script_candidate.exists():
            cfg_path = script_candidate
    if not cfg_path.exists():
        default_root = Path(ultralytics.__file__).resolve().parent / "cfg" / "trackers"
        candidate = default_root / config
        if candidate.exists():
            cfg_path = candidate
    if not cfg_path.exists():
        raise FileNotFoundError(f"Tracker config not found: {config}")

    with cfg_path.open("r", encoding="utf-8") as stream:
        cfg_data = yaml.safe_load(stream)
    if not isinstance(cfg_data, dict):
        raise ValueError(f"Tracker config {cfg_path} is not a mapping.")
    return BYTETracker(SimpleNamespace(**cfg_data), frame_rate=max(1, int(round(frame_rate))))


def class_name_lookup(names: Union[List[str], dict], class_idx: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_idx, class_idx))
    if 0 <= class_idx < len(names):
        return str(names[class_idx])
    return str(class_idx)
