"""
Command-line interface (argument parsing) for the lane analytics app.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO11 staged lane analytics")
    parser.add_argument("--source", type=str, default="2025_0917_085148_010A.MP4")
    parser.add_argument("--model", type=str, default="yolo11l.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--view", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stream-buffer", type=int, default=2)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--tracker-config", type=str, default="bytetrack.yaml")
    parser.add_argument("--lane-groups", type=str, default="base,straight,turn,far_turn,uturn,reverse")
    parser.add_argument("--lane-directions", type=str, default=None)
    parser.add_argument("--lane-config", type=str, default=None)
    parser.add_argument("--save-lane-config", type=str, default=None)
    parser.add_argument("--interactive-lanes", action="store_true")
    parser.add_argument("--interactive-roi", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-process-dim", type=int, default=640)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--auto-enhance", action="store_true")
    parser.add_argument("--inference-backend", choices=["auto", "torch", "tflite"], default="auto")
    parser.add_argument("--debug-crossings", action="store_true")
    parser.add_argument("--counts-output", type=str, default=None)
    parser.add_argument("--lanes", type=int, default=6)
    parser.add_argument("--verbose", action="store_true")
    return parser


__all__ = ["build_parser"]
