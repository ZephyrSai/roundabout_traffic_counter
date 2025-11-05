"""
Core modules for the staged lane analytics application.

This package exposes utility helpers, data models, interactive tools, and the
main processing pipeline used by :mod:`app`.
"""

from .pipeline import LaneAnalyticsPipeline

__all__ = ["LaneAnalyticsPipeline"]
