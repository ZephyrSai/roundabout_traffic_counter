"""
Executable entry point for the lane analytics application.
"""

from __future__ import annotations

from lanedet.cli import build_parser
from lanedet.pipeline import LaneAnalyticsPipeline


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pipeline = LaneAnalyticsPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
