"""
High-level pipeline that orchestrates YOLO11 detection, tracking, and lane analytics.
"""

from __future__ import annotations

import logging
import signal
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import math
import numpy as np
from ultralytics import YOLO

from .constants import BASE_STAGE_NAME, INDEPENDENT_GROUPS, VEHICLE_LABELS
from .drawing import (
    configure_window_to_frame,
    draw_lane_overlay,
    draw_roi_overlay,
    draw_tracks,
)
from .enhance import enhance_frame
from .filtering import filter_vehicle_boxes
from .geometry import crosses_segment
from .interactive import interactive_lane_groups, interactive_roi_polygon
from .lanes import (
    ensure_lane_groups,
    load_lane_groups,
    save_lane_groups,
)
from .roi import polygon_contains_point, segment_within_polygon
from .scaling import scale_lane_groups, scale_polygon
from .types import LaneGroup, LaneSegment
from .utils import (
    build_writer,
    class_name_lookup,
    configure_logging,
    derive_counts_path,
    derive_default_save_path,
    ensure_torch_device,
    human_direction,
    load_tracker,
    parse_direction_map,
    parse_group_names,
    resolve_source,
)


class GracefulShutdown:
    """Capture SIGINT/SIGTERM and finish the current frame before exiting."""

    def __init__(self) -> None:
        self.stop_requested = False
        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            signal.signal(sig, self._handler)

    def _handler(self, signum, _frame) -> None:
        logging.info("Received signal %s; finishing current frame before exiting.", signum)
        self.stop_requested = True


class LaneAnalyticsPipeline:
    """Executable pipeline wrapping YOLO detection, ByteTrack tracking, and staged lane counting."""

    def __init__(self, args) -> None:
        self.args = args
        configure_logging(args.verbose)
        self.source = resolve_source(args.source)
        self.killer = GracefulShutdown()

        self.device = ensure_torch_device()
        self.backend = args.inference_backend
        self.model = self._load_model(args.model)
        self.names = self.model.model.names if hasattr(self.model.model, "names") else self.model.names
        self.use_half = self.device.startswith("cuda") and self.backend != "tflite"

    # --------------------------------------------------------------------- setup helpers
    def _load_model(self, weight_path: str) -> YOLO:
        try:
            if self.backend == "tflite":
                if not weight_path.lower().endswith(".tflite"):
                    logging.error("TFLite backend requires a .tflite model path.")
                    raise SystemExit(1)
                logging.info("Loading TFLite model %s. Ensure the tflite-runtime package is installed.", weight_path)
                return YOLO(weight_path)
            model = YOLO(weight_path)
            model.to(self.device)
            return model
        except Exception as exc:
            logging.error("Unable to load model %s: %s", weight_path, exc)
            raise SystemExit(1) from exc

    def _prepare_capture(self) -> Tuple[cv2.VideoCapture, float, int, int, Optional[np.ndarray]]:
        capture = cv2.VideoCapture(self.source)
        if self.args.stream_buffer >= 0:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, self.args.stream_buffer)
        if not capture.isOpened():
            logging.error("Failed to open video source %s", self.args.source)
            raise SystemExit(1)

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 1e-2:
            fps = 30.0
            logging.info("FPS metadata unavailable; defaulting to %.1f FPS for output.", fps)

        frame_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))) or 0
        frame_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) or 0

        preview_frame: Optional[np.ndarray] = None
        need_preview = self.args.interactive_lanes or (self.args.interactive_roi if self.args.interactive_roi is not None else self.args.interactive_lanes)
        if need_preview or frame_width <= 0 or frame_height <= 0:
            ret, frame = capture.read()
            if not ret:
                logging.error("Unable to read a frame for configuration.")
                raise SystemExit(1)
            preview_frame = enhance_frame(frame.copy(), self.args.contrast, self.args.brightness, self.args.auto_enhance)
            frame_height, frame_width = frame.shape[:2]
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return capture, fps, frame_width, frame_height, preview_frame

    def _configure_lanes(
        self,
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[List[LaneGroup], Optional[List[Tuple[int, int]]]]:
        group_names = parse_group_names(self.args.lane_groups, self.args.lanes)
        direction_map = parse_direction_map(self.args.lane_directions, group_names)

        lane_groups: List[LaneGroup] = []
        roi_polygon: Optional[List[Tuple[int, int]]] = None
        if self.args.lane_config:
            try:
                lane_groups, roi_polygon = load_lane_groups(Path(self.args.lane_config), group_names, direction_map)
                logging.info(
                    "Loaded %s lane groups from %s.",
                    len(lane_groups),
                    Path(self.args.lane_config).resolve(),
                )
            except Exception as exc:
                logging.error("Failed to load lane config: %s", exc)
                raise SystemExit(1) from exc

        interactive_roi = self.args.interactive_roi if self.args.interactive_roi is not None else self.args.interactive_lanes
        if interactive_roi:
            frame_for_roi = frame.copy()
            roi_candidate = interactive_roi_polygon(frame_for_roi, roi_polygon)
            if len(roi_candidate) >= 3:
                roi_polygon = roi_candidate
            else:
                logging.warning("ROI not defined; processing the full frame.")

        if self.args.interactive_lanes:
            lane_groups = interactive_lane_groups(
                frame.copy(),
                group_names,
                direction_map,
                lane_groups,
                roi_polygon=roi_polygon,
            )
            if self.args.save_lane_config and lane_groups:
                try:
                    save_lane_groups(Path(self.args.save_lane_config), lane_groups, roi_polygon=roi_polygon)
                    logging.info(
                        "Saved lane configuration (%s groups) to %s.",
                        len(lane_groups),
                        Path(self.args.save_lane_config).resolve(),
                    )
                except Exception as exc:
                    logging.warning("Failed to save lane config: %s", exc)

        lane_groups = ensure_lane_groups(
            group_names,
            lane_groups,
            direction_map,
            frame_width,
            frame_height,
            auto_fill_missing=not self.args.interactive_lanes,
        )
        if not lane_groups:
            logging.error("Lane configuration is invalid. Need at least one lane group.")
            raise SystemExit(1)

        if roi_polygon and len(roi_polygon) >= 3:
            sanitized_groups: List[LaneGroup] = []
            for group in lane_groups:
                valid_segments = [segment for segment in group.segments if segment_within_polygon(segment, roi_polygon)]
                if not valid_segments:
                    logging.warning("Skipping group %s; its segments fall outside the ROI.", group.name)
                    continue
                sanitized_groups.append(LaneGroup(name=group.name, color=group.color, direction=group.direction, segments=valid_segments))
            lane_groups = sanitized_groups
            if not lane_groups:
                logging.error("Lane configuration invalid: all segments are outside the ROI.")
                raise SystemExit(1)
            logging.info("ROI defined with %s vertices.", len(roi_polygon))

        for group in lane_groups:
            logging.info(
                "Lane group %s (%s) configured with %s segment(s).",
                group.name,
                human_direction(group.direction),
                len(group.segments),
            )
        return lane_groups, roi_polygon

    # --------------------------------------------------------------------- main processing
    def run(self) -> None:
        capture, fps, frame_width, frame_height, preview_frame = self._prepare_capture()
        lane_groups, roi_polygon = self._configure_lanes(preview_frame if preview_frame is not None else np.zeros((frame_height, frame_width, 3), dtype=np.uint8), frame_width, frame_height)

        processing_scale = 1.0
        max_dim = max(frame_width, frame_height)
        if self.args.max_process_dim and self.args.max_process_dim > 0 and max_dim > self.args.max_process_dim:
            processing_scale = self.args.max_process_dim / float(max_dim)
            logging.info(
                "Downscaling frames for processing: original %sx%s -> %sx%s (scale %.3f).",
                frame_width,
                frame_height,
                int(round(frame_width * processing_scale)),
                int(round(frame_height * processing_scale)),
                processing_scale,
            )

        processing_width = max(1, int(round(frame_width * processing_scale)))
        processing_height = max(1, int(round(frame_height * processing_scale)))
        scale_back = 1.0 / processing_scale if processing_scale != 0 else 1.0

        lane_groups_output = lane_groups
        lane_groups_runtime = scale_lane_groups(lane_groups_output, processing_scale)
        roi_polygon_runtime = scale_polygon(roi_polygon, processing_scale)
        roi_mask: Optional[np.ndarray]
        if roi_polygon_runtime and len(roi_polygon_runtime) >= 3:
            roi_mask = np.zeros((processing_height, processing_width), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [np.array(roi_polygon_runtime, dtype=np.int32)], 255)
        else:
            roi_mask = None

        if self.args.imgsz and self.args.imgsz > 0:
            inference_imgsz = max(32, min(self.args.imgsz, 1280))
        else:
            inference_imgsz = max(processing_width, processing_height)
        inference_imgsz = int(math.ceil(inference_imgsz / 32) * 32)

        if self.args.view:
            display_window = "YOLO11 detections"
            cv2.namedWindow(display_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(display_window, max(processing_width, 320), max(processing_height, 240))
        else:
            display_window = None

        counts_output_path = Path(self.args.counts_output) if self.args.counts_output else derive_counts_path(self.source)

        tracker = None
        if self.args.tracker_config.lower() != "none":
            tracker_fps = fps if fps and fps > 0 else (self.args.target_fps or 30.0)
            try:
                tracker = load_tracker(self.args.tracker_config, tracker_fps)
                logging.info("Initialized ByteTrack with nominal frame rate %.2f FPS.", tracker_fps)
            except Exception as exc:
                logging.error("Tracker initialization failed: %s", exc)
                raise SystemExit(1) from exc

        frame_skip = 1
        interval_seconds: Optional[float] = None
        last_process_time = time.perf_counter()
        target_fps = self.args.target_fps if self.args.target_fps and self.args.target_fps > 0 else None
        if target_fps:
            if fps > 0:
                frame_skip = max(1, int(round(fps / target_fps)))
                effective = fps / frame_skip
                logging.info("Frame skipping enabled: processing roughly %.2f FPS (1 of %s frames).", effective, frame_skip)
            else:
                interval_seconds = 1.0 / target_fps
                last_process_time -= interval_seconds
                logging.info("Live stream throttling enabled: processing at most %.2f FPS.", target_fps)

        output_fps = target_fps if target_fps else (fps / frame_skip if frame_skip > 1 else fps)

        writer: Optional[cv2.VideoWriter] = None
        save_path: Optional[Path] = None

        effective_fps = target_fps if target_fps else (fps / frame_skip if frame_skip > 1 else fps)
        if not effective_fps or effective_fps <= 0:
            effective_fps = 30.0
        alias_max_gap_frames = max(3, int(round(effective_fps * 1.5)))
        alias_distance_thresh = max(20.0, max(frame_width, frame_height) * 0.08)

        total_frames_read = 0
        processed_frames = 0
        detections_counter: Counter[str] = Counter()
        observed_track_ids: set[int] = set()
        track_states: Dict[int, Dict[str, Union[Tuple[float, float], set]]] = {}
        lane_counts: Dict[str, Counter[str]] = {group.name: Counter() for group in lane_groups_output}
        vehicle_group_history: Dict[int, set[str]] = {}
        track_aliases: Dict[int, int] = {}
        next_stable_id = 1

        start_time = time.time()
        logging.info("Starting inference on %s", self.args.source)

        base_group = next((group for group in lane_groups_output if group.name.lower() == BASE_STAGE_NAME), None)
        first_stage_name = base_group.name if base_group and base_group.segments else None
        first_stage_lower = first_stage_name.lower() if first_stage_name else None
        independent_groups = {name.lower() for name in INDEPENDENT_GROUPS}

        try:
            while not self.killer.stop_requested:
                ret, frame = capture.read()
                if not ret:
                    logging.info("No more frames available or stream ended.")
                    break

                total_frames_read += 1
                if frame_skip > 1 and ((total_frames_read - 1) % frame_skip != 0):
                    continue
                if interval_seconds is not None:
                    now = time.perf_counter()
                    if now - last_process_time < interval_seconds:
                        continue
                    last_process_time = now

                processed_frames += 1

                frame = enhance_frame(frame, self.args.contrast, self.args.brightness, self.args.auto_enhance)
                processing_frame = frame if processing_scale == 1.0 else cv2.resize(frame, (processing_width, processing_height))

                inference_frame = (
                    processing_frame
                    if roi_mask is None
                    else cv2.bitwise_and(processing_frame, processing_frame, mask=roi_mask)
                )

                results = self.model.predict(
                    inference_frame,
                    conf=self.args.conf,
                    iou=self.args.iou,
                    device=self.device,
                    half=self.use_half,
                    imgsz=inference_imgsz,
                    verbose=False,
                )
                result = results[0]
                result.orig_img = processing_frame.copy()

                boxes = result.boxes
                boxes_cpu = boxes.cpu() if boxes is not None else None
                filtered_boxes = filter_vehicle_boxes(boxes_cpu, self.names) if boxes_cpu is not None else None
                if filtered_boxes is not None:
                    if roi_polygon_runtime and len(roi_polygon_runtime) >= 3 and len(filtered_boxes):
                        xyxy = filtered_boxes.xyxy.cpu().numpy()
                        keep_mask = []
                        for bbox in xyxy:
                            cx = (bbox[0] + bbox[2]) / 2.0
                            cy = (bbox[1] + bbox[3]) / 2.0
                            keep_mask.append(polygon_contains_point((cx, cy), roi_polygon_runtime))
                        filtered_boxes = filtered_boxes[np.array(keep_mask, dtype=bool)]
                    result.boxes = filtered_boxes

                tracks_array = None
                if tracker is not None and filtered_boxes is not None and len(filtered_boxes):
                    tracks_array = tracker.update(filtered_boxes, processing_frame)

                annotated = result.plot()
                annotated = draw_tracks(annotated, tracks_array, self.names)
                if roi_polygon_runtime and len(roi_polygon_runtime) >= 3:
                    annotated = draw_roi_overlay(annotated, roi_polygon_runtime)

                if filtered_boxes is not None and len(filtered_boxes):
                    cls_ids = filtered_boxes.cls.int().tolist()
                    for cls_id in cls_ids:
                        class_idx = int(cls_id)
                        class_name = class_name_lookup(self.names, class_idx)
                        detections_counter[class_name] += 1

                active_tracker_ids: set[int] = set()
                active_stable_ids: set[int] = set()
                if tracks_array is not None and len(tracks_array):
                    for track in tracks_array:
                        if len(track) < 7:
                            continue
                        track_id = int(track[4])
                        active_tracker_ids.add(track_id)
                        cls_idx = int(track[6])
                        class_name = class_name_lookup(self.names, cls_idx)
                        normalized_label = class_name.lower()
                        if normalized_label not in VEHICLE_LABELS:
                            continue

                        x1, y1, x2, y2 = track[:4]
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        current_centroid_proc = (cx, cy)
                        current_centroid = (cx * scale_back, cy * scale_back)

                        stable_id = track_aliases.get(track_id)
                        if stable_id is None:
                            best_sid = None
                            best_dist = float("inf")
                            for sid, state in track_states.items():
                                if sid in active_stable_ids:
                                    continue
                                last_frame_seen = state.get("last_frame", -1)
                                if last_frame_seen >= 0 and processed_frames - last_frame_seen > alias_max_gap_frames:
                                    continue
                                stored_class = state.get("class")
                                if stored_class and stored_class != normalized_label:
                                    continue
                                last_centroid = state.get("last")
                                if last_centroid is None:
                                    continue
                                dist = math.hypot(current_centroid[0] - last_centroid[0], current_centroid[1] - last_centroid[1])
                                if dist < best_dist and dist <= alias_distance_thresh:
                                    best_dist = dist
                                    best_sid = sid
                            if best_sid is not None:
                                stable_id = best_sid
                            else:
                                stable_id = next_stable_id
                                next_stable_id += 1
                            track_aliases[track_id] = stable_id

                        active_stable_ids.add(stable_id)
                        observed_track_ids.add(stable_id)

                        state = track_states.setdefault(
                            stable_id,
                            {"last": None, "counted": set(), "class": normalized_label, "last_frame": processed_frames},
                        )
                        previous_centroid = state["last"]
                        state["last"] = current_centroid
                        counted_segments: set = state.setdefault("counted", set())
                        state["class"] = normalized_label
                        state["last_frame"] = processed_frames
                        crossed_groups = vehicle_group_history.setdefault(stable_id, set())
                        crossed_groups_normalized = {name.lower() for name in crossed_groups}

                        if previous_centroid is None:
                            continue
                        if roi_polygon and len(roi_polygon) >= 3:
                            if not (
                                polygon_contains_point(previous_centroid, roi_polygon)
                                or polygon_contains_point(current_centroid, roi_polygon)
                            ):
                                continue

                        for group in lane_groups_output:
                            group_name_lower = group.name.lower()
                            if (
                                first_stage_lower
                                and group_name_lower not in independent_groups
                                and first_stage_lower not in crossed_groups_normalized
                            ):
                                continue
                            group_counter = lane_counts[group.name]
                            for seg_idx, segment in enumerate(group.segments):
                                segment_key = (group.name, seg_idx)
                                if segment_key in counted_segments:
                                    continue
                                if crosses_segment(previous_centroid, current_centroid, segment, group.direction):
                                    group_counter[class_name] += 1
                                    counted_segments.add(segment_key)
                                    crossed_groups.add(group.name)
                                    crossed_groups_normalized.add(group_name_lower)
                                    if self.args.debug_crossings:
                                        logging.debug(
                                            "stable=%s(track=%s) class=%s crossed group=%s seg=%s",
                                            stable_id,
                                            track_id,
                                            class_name_lookup(self.names, cls_idx),
                                            group.name,
                                            seg_idx,
                                        )
                                    break

                    stale_tracker_ids = set(track_aliases.keys()) - active_tracker_ids
                    for tid in stale_tracker_ids:
                        track_aliases.pop(tid, None)

                    for sid, state in list(track_states.items()):
                        last_frame_seen = state.get("last_frame", -1)
                        if sid in active_stable_ids:
                            continue
                        if last_frame_seen >= 0 and processed_frames - last_frame_seen <= alias_max_gap_frames:
                            continue
                        track_states.pop(sid, None)
                        vehicle_group_history.pop(sid, None)
                else:
                    for sid, state in list(track_states.items()):
                        last_frame_seen = state.get("last_frame", -1)
                        if last_frame_seen >= 0 and processed_frames - last_frame_seen > alias_max_gap_frames:
                            track_states.pop(sid, None)
                            vehicle_group_history.pop(sid, None)
                    track_aliases.clear()

                annotated = draw_lane_overlay(annotated, lane_groups_runtime, lane_counts)

                if self.args.save:
                    if writer is None:
                        candidate = Path(self.args.save_path) if self.args.save_path else derive_default_save_path(self.source)
                        if not candidate.suffix:
                            candidate = candidate.with_suffix(".mp4")
                        save_path = candidate
                        writer = build_writer(save_path, output_fps, annotated.shape[1], annotated.shape[0])
                        logging.info("Saving annotated output to %s", save_path)
                    writer.write(annotated)

                if self.args.view and display_window:
                    cv2.imshow(display_window, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        logging.info("Exit requested from preview window.")
                        break

                if self.args.max_frames and processed_frames >= self.args.max_frames:
                    logging.info("Reached max frame limit (%s).", self.args.max_frames)
                    break
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received; stopping inference.")
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if self.args.view and display_window:
                cv2.destroyWindow(display_window)

        elapsed = time.time() - start_time
        fps_processed = processed_frames / elapsed if elapsed > 0 else 0.0
        logging.info(
            "Processed %s frames (read %s) in %.2f seconds (%.2f FPS).",
            processed_frames,
            total_frames_read,
            elapsed,
            fps_processed,
        )

        if detections_counter:
            summary = ", ".join(f"{cls}: {count}" for cls, count in detections_counter.most_common(10))
            logging.info("Detection counts (top 10): %s", summary)

        for group in lane_groups_output:
            counter = lane_counts.get(group.name, Counter())
            if counter:
                summary = ", ".join(f"{cls}: {count}" for cls, count in counter.most_common())
            else:
                summary = "no vehicles"
            logging.info("%s (%s) counts: %s", group.name, human_direction(group.direction), summary)

        total_counter = Counter()
        for counter in lane_counts.values():
            total_counter.update(counter)

        all_group_names = [group.name for group in lane_groups_output]
        vehicles_all_groups = (
            sum(1 for groups in vehicle_group_history.values() if set(all_group_names).issubset(groups))
            if all_group_names
            else 0
        )

        counts_output_path.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        lines.append("Lane Groups:")
        for group in lane_groups_output:
            lines.append(f"- {group.name}: color={group.color}, direction={group.direction}")
            for idx, segment in enumerate(group.segments, start=1):
                lines.append(f"    segment {idx}: {segment.as_list()}")
        lines.append("")
        for group in lane_groups_output:
            counter = lane_counts.get(group.name, Counter())
            lines.append(f"{group.name} ({human_direction(group.direction)}):")
            if counter:
                for cls, count in counter.most_common():
                    lines.append(f"  {cls}: {count}")
            else:
                lines.append("  (no vehicles detected)")
            lines.append("")
        lines.append("Vehicle crossing summary:")
        if all_group_names:
            lines.append(f"  Vehicles crossing all groups ({', '.join(all_group_names)}): {vehicles_all_groups}")
        lines.append(f"  Total unique tracked vehicles: {len(vehicle_group_history)}")
        if total_counter:
            lines.append("")
            lines.append("Total vehicles by class:")
            for cls, count in total_counter.most_common():
                lines.append(f"  {cls}: {count}")

        counts_output_path.write_text("\n".join(lines), encoding="utf-8")
        logging.info("Lane count report saved to: %s", counts_output_path.resolve())

        if observed_track_ids:
            logging.info("Unique ByteTrack IDs observed: %s", len(observed_track_ids))


__all__ = ["LaneAnalyticsPipeline"]
