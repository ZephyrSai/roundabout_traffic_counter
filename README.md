# Lane Analytics Pipeline

This project provides a modular Python application for running YOLO11-based vehicle detection, ByteTrack tracking, and staged lane analytics on video files or live streams.

## Features

- YOLO11 (PyTorch or TFLite) inference with optional CUDA half precision
- ByteTrack multi-object tracking to maintain vehicle identities
- Interactive lane and ROI definition with persistent YAML configs
- Multi-stage lane counting (straight, turn, far_turn, uturn, reverse, etc.)
- Optional auto-enhancement and resolution scaling to balance quality vs. speed
- Rich logging and optional debug tracing of lane crossings

## Project Layout

``
app.py                 # Entry point
lanedet/               # Core application package
    cli.py             # Argument parsing helpers
    constants.py       # Shared constants (classes, colors, fonts, CLAHE, etc.)
    drawing.py         # Overlay rendering utilities
    enhance.py         # Contrast / brightness adjustments
    filtering.py       # YOLO detection post-processing
    geometry.py        # Crossing and intersection math
    interactive.py     # ROI and lane drawing UIs
    lanes.py           # Load/save lane configs and defaults
    pipeline.py        # LaneAnalyticsPipeline orchestration
    roi.py             # ROI helpers and polygon utilities
    scaling.py         # Geometry scaling routines
    types.py           # Dataclasses for LaneGroup and LaneSegment
    utils.py           # General utilities (logging, tracker loading, etc.)
working.py             # Legacy wrapper (imports the new pipeline)
``

## Setup

1. Create or activate a Python 3.10+ virtual environment.
2. Install dependencies:
   `ash
   pip install -r requirements.txt
   `
3. (Optional) For TFLite inference, install the runtime:
   `ash
   pip install tflite-runtime==2.13.0
   `
   Adjust the version according to your platform if needed.

## Running the App

### Torch / CUDA backend

`powershell
python app.py --source TMC32SA.mp4 --lane-config lanes_TMC32SA.yaml \
  --max-process-dim 640 --imgsz 640 --target-fps 10 --view
`

- --lane-config reuses saved lanes/ROI.
- --interactive-roi and --interactive-lanes can be added to redraw.
- --save writes an annotated video; --counts-output stores counts.

### TFLite backend

1. Download a YOLO11 TFLite model (example for the large variant):
   `powershell
   Invoke-WebRequest 
     -Uri https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.tflite 
     -OutFile yolo11l.tflite
   `
2. Run the pipeline with TFLite:
   `powershell
   python app.py --source TMC32SA.mp4 --model yolo11l.tflite --inference-backend tflite \
     --lane-config lanes_TMC32SA.yaml --max-process-dim 512 --imgsz 512 --target-fps 10 --view
   `

### Debugging lane crossings

Add --debug-crossings to log every stable track/lane crossing for verification.

## Working Script

working.py remains as a compatibility wrapper; it simply imports the new pipeline and runs it with the same CLI.

## Notes

- Lane counts are computed in the original frame coordinate space to avoid scaling drift.
- ROI masks and lane overlays are scaled when processing at reduced resolutions.
- ByteTrack IDs are stabilised across gaps to prevent double counting at low frame rates.

## License

This project builds on the Ultralytics YOLO framework; review their license for model usage. Any bespoke code in this repository can be adapted to your project needs.

