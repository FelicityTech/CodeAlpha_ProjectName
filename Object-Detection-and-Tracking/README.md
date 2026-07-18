# Real-Time Object Detection and Tracking

Detects objects in a video stream (webcam or file) with YOLOv4-tiny and
tracks them across frames with a from-scratch SORT implementation
(Kalman filter + Hungarian algorithm), drawing bounding boxes, class
labels, and persistent tracking IDs in real time.

## Why YOLOv4-tiny via OpenCV DNN, not YOLOv8/PyTorch

This uses OpenCV's built-in `cv2.dnn` module to run YOLOv4-tiny directly
from its original Darknet `.cfg`/`.weights` files, instead of
`ultralytics`/PyTorch. Same detection approach (a real pretrained YOLO
model, COCO's 80 classes), but a much lighter dependency footprint —
just `opencv-python`, no multi-gigabyte PyTorch/CUDA install. Good
enough for CPU real-time use; see [Possible Improvements](#possible-improvements)
if you want the accuracy bump of a full YOLOv8 model later.

## Project Structure

```
object_tracker/
├── models/
│   ├── yolov4-tiny.cfg        # network architecture
│   ├── yolov4-tiny.weights    # pretrained weights (24MB)
│   └── coco.names             # 80 COCO class names
├── sample_data/
│   └── vtest.avi               # sample pedestrian video (for testing/demo)
├── detector.py                    # YOLOv4-tiny wrapper (cv2.dnn)
├── sort_tracker.py                 # SORT: Kalman filter + Hungarian matching
├── main.py                          # CLI entry point (webcam or video file)
├── download_models.py                # re-fetches model files if missing
├── test_pipeline.py                   # automated verification tests
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Model files are already included in `models/`. If they're missing (e.g.
git ignored the large `.weights` file), re-fetch them with:
```bash
python download_models.py
```

## Usage

**Webcam, all object classes:**
```bash
python main.py --source 0
```

**Video file, only track people and cars:**
```bash
python main.py --source path/to/video.mp4 --classes person car
```

**Headless run, save annotated output instead of displaying it:**
```bash
python main.py --source video.mp4 --save output.mp4 --no-display
```

Press `q` in the preview window to quit at any time.

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Webcam index or path to a video file |
| `--classes` | all | Space-separated COCO class names to track (e.g. `person car dog`) |
| `--confidence` | `0.5` | Minimum detection confidence (0–1) |
| `--save` | none | Path to write the annotated output video |
| `--no-display` | off | Skip the live preview window |
| `--max-frames` | none | Stop after N frames (handy for quick tests) |

## How It Works

```
video frame  →  YOLOv4-tiny detection  →  SORT tracking  →  draw boxes/IDs
              (detector.py)              (sort_tracker.py)   (main.py)
```

### Detection (`detector.py`)

Each frame is resized into a 416×416 blob and passed through YOLOv4-tiny.
Raw outputs are filtered by confidence, then collapsed with non-max
suppression (`cv2.dnn.NMSBoxes`) to remove duplicate boxes on the same
object. Returns `(x, y, w, h, confidence, class_name)` per detection.

### Tracking (`sort_tracker.py`)

A from-scratch SORT implementation, one `KalmanBoxTracker` per tracked
object:

1. **Predict** — each track's Kalman filter predicts where its box should
   be this frame, using a constant-velocity motion model over
   `[center_x, center_y, area, aspect_ratio]`.
2. **Associate** — predicted boxes are matched against this frame's real
   detections by IOU (intersection-over-union), solved optimally with the
   Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) rather than
   greedy nearest-match.
3. **Update** — matched tracks correct their Kalman state with the real
   detection; unmatched detections spawn new tracks; tracks unmatched for
   more than `max_age` frames are dropped.

A track is only reported once it's matched `min_hits` consecutive frames
(or the video itself is only `min_hits` frames old), which filters out
one-off false-positive detections from ever getting a flickering ID.

### Why this needed real debugging, not just "it ran"

Initial testing on the bundled sample video surfaced a genuine bug: single
frame "flicker" tracks (an ID appearing for exactly one frame, then gone).
The cause was using a track's own age as its confirmation grace period
instead of the *tracker's* global frame count — meaning a spurious
detection with no real object behind it could still get instantly reported
just because it happened to appear early. Switching to a proper
`hit_streak` (consecutive matches, reset on any miss) plus a global
frame-count grace period fixed it. `test_pipeline.py` asserts flicker
tracks stay under 30% of all tracks so this doesn't silently regress.

## Testing

```bash
python test_pipeline.py
```

Four tests, all running against the real bundled video (not synthetic
data, since a blank/synthetic video wouldn't meaningfully exercise a real
detector):

1. **IOU correctness** — identical, disjoint, and partially-overlapping
   boxes score 1.0, 0.0, and strictly-between as expected.
2. **Detector finds real objects** — asserts multiple people are detected
   in frame 0 of the sample video, with valid confidence scores and box
   dimensions.
3. **Tracks persist across frames** — asserts multiple tracks survive 10+
   consecutive frames (proving the Kalman prediction + Hungarian matching
   is actually working, not just assigning a new ID every frame) and that
   flicker tracks stay rare.
4. **Bounding box sanity** — tracked boxes never explode in size or go
   negative, which would indicate the Kalman filter diverging.

## Known Limitations

- **CPU-only, real-time-ish but not fast** — YOLOv4-tiny on CPU runs
  roughly 8–15 FPS depending on hardware and frame size (measured ~8.7 FPS
  in this environment); a GPU or a lighter model would be needed for
  higher frame rates.
- **SORT has no appearance model** — it tracks purely on motion + box
  overlap. Two people crossing paths and briefly occluding each other can
  cause an ID swap, which is exactly the scenario Deep SORT (which adds a
  learned appearance embedding to re-identify objects after occlusion)
  was built to fix.
- **YOLOv4-tiny trades accuracy for speed** — it's the "tiny" variant, so
  it'll miss small or partially-occluded objects that a full YOLOv4/YOLOv8
  model would catch.

## Possible Improvements

- **Deep SORT** — replace pure IOU matching with a learned appearance
  embedding (e.g. a small ReID CNN) so tracks survive occlusion without
  swapping IDs. This is the most impactful upgrade if occlusion-heavy
  scenes (crowds, crossing paths) are the main use case.
- **Full YOLOv8 via ultralytics** — meaningfully better accuracy at the
  cost of a much heavier install (PyTorch); worth it if accuracy matters
  more than setup simplicity, and GPU is available.
- **Multi-camera / multi-stream** — `main.py` currently handles one
  source; the detector and tracker are stateless enough per-frame that
  running multiple `SortTracker` instances in parallel (one per camera)
  would be a straightforward extension.

## Tech Stack

| Component | Library |
|---|---|
| Video I/O, drawing | OpenCV (`cv2`) |
| Object detection | YOLOv4-tiny (Darknet weights, run via `cv2.dnn`) |
| Kalman filtering, math | NumPy |
| Optimal detection-to-track assignment | SciPy (`linear_sum_assignment`) |
