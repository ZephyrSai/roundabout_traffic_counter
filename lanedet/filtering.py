"""
Detection filtering logic to keep only relevant vehicle classes and handle overlaps.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from .constants import CAR_LABELS, TRUCK_LIKE_LABELS, VEHICLE_LABELS
from .utils import class_name_lookup, normalize_class_name


def filter_vehicle_boxes(boxes, names) -> Optional[object]:
    """
    Filter raw YOLO detections to keep relevant vehicle classes. Drops cars inside trucks/trailers.
    """
    if boxes is None:
        return None
    if len(boxes) == 0:
        return boxes

    cls_indices = boxes.cls.int().tolist()
    keep = [False] * len(cls_indices)
    for idx, cls_idx in enumerate(cls_indices):
        label = normalize_class_name(class_name_lookup(names, cls_idx))
        if label in VEHICLE_LABELS:
            keep[idx] = True

    if not any(keep):
        empty = boxes.data.new_zeros((0, boxes.data.shape[1]))
        return boxes.__class__(empty, boxes.orig_shape)

    xyxy = boxes.xyxy.cpu().numpy()
    for i, should_keep in enumerate(list(keep)):
        if not should_keep:
            continue
        label_i = normalize_class_name(class_name_lookup(names, cls_indices[i]))
        if label_i not in CAR_LABELS:
            continue
        for j, other_keep in enumerate(keep):
            if i == j or not other_keep:
                continue
            label_j = normalize_class_name(class_name_lookup(names, cls_indices[j]))
            if label_j not in TRUCK_LIKE_LABELS:
                continue
            car_box = xyxy[i]
            other_box = xyxy[j]
            inter_x1 = max(car_box[0], other_box[0])
            inter_y1 = max(car_box[1], other_box[1])
            inter_x2 = min(car_box[2], other_box[2])
            inter_y2 = min(car_box[3], other_box[3])
            inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
            car_area = max(0.0, car_box[2] - car_box[0]) * max(0.0, car_box[3] - car_box[1])
            truck_area = max(0.0, other_box[2] - other_box[0]) * max(0.0, other_box[3] - other_box[1])
            union = car_area + truck_area - inter_area
            iou = inter_area / union if union > 0 else 0.0
            contain_ratio = inter_area / car_area if car_area > 0 else 0.0
            if iou >= 0.5 or contain_ratio >= 0.8:
                keep[i] = False
                break

    if not any(keep):
        empty = boxes.data.new_zeros((0, boxes.data.shape[1]))
        return boxes.__class__(empty, boxes.orig_shape)

    if torch is None:
        raise RuntimeError("torch is required for filtering detections.")
    mask = torch.tensor(keep, dtype=torch.bool, device=boxes.data.device)
    return boxes[mask]


__all__ = ["filter_vehicle_boxes"]
