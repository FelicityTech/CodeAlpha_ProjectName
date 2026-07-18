"""
Object Detection and Tracking — main pipeline.

Reads video from a webcam or file, runs YOLOv4-tiny detection on each frame,
tracks detected objects across frames with SORT, and displays the result
with bounding boxes, class labels, and persistent tracking IDs.

Usage:
    python main.py                          # webcam (device 0)
    python main.py --source path/to/video.mp4
    python main.py --source 0 --classes person car
    python main.py --source video.mp4 --save output.mp4 --no-display
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from detector import YOLODetector
from sort_tracker import SortTracker

# Deterministic-but-varied color per track ID, so a given ID keeps the same
# color for its whole lifetime (helps visually follow an object on screen).
def color_for_id(track_id: int):
    rng = np.random.default_rng(track_id * 7919)  # arbitrary prime for spread
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def draw_tracks(frame, tracks):
    for (x, y, w, h, track_id, class_name, confidence) in tracks:
        x, y, w, h = int(x), int(y), int(w), int(h)
        color = color_for_id(track_id)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label = f"ID {track_id} {class_name} {confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - text_h - baseline - 4), (x + text_w + 4, y), color, -1)
        cv2.putText(
            frame, label, (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return frame


def run(source, classes_of_interest, confidence_threshold, save_path, display, max_frames=None):
    detector = YOLODetector(confidence_threshold=confidence_threshold)
    tracker = SortTracker(max_age=15, min_hits=3, iou_threshold=0.3)

    # source is either a webcam index (int-like string) or a file path
    cap_source = int(source) if str(source).isdigit() else str(source)
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, fps_in, (width, height))

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame, classes_of_interest=classes_of_interest)
            tracks = tracker.update(detections)
            draw_tracks(frame, tracks)

            elapsed = time.time() - t_start
            live_fps = frame_count / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                frame, f"FPS: {live_fps:.1f}  Objects: {len(tracks)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )

            if writer:
                writer.write(frame)

            if display:
                cv2.imshow("Object Detection and Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    total_time = time.time() - t_start
    print(f"Processed {frame_count} frames in {total_time:.1f}s ({frame_count / total_time:.1f} FPS avg)")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time object detection and tracking")
    parser.add_argument(
        "--source", default="0",
        help="Webcam index (e.g. 0) or path to a video file. Default: 0 (webcam)",
    )
    parser.add_argument(
        "--classes", nargs="*", default=None,
        help="Only track these COCO classes (e.g. --classes person car). Default: all classes.",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Minimum detection confidence (0-1). Default: 0.5",
    )
    parser.add_argument(
        "--save", default=None,
        help="Path to save the annotated output video (e.g. output.mp4)",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Don't open a live preview window (useful for headless/server runs)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Stop after N frames (useful for testing)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    classes = set(args.classes) if args.classes else None
    run(
        source=args.source,
        classes_of_interest=classes,
        confidence_threshold=args.confidence,
        save_path=args.save,
        display=not args.no_display,
        max_frames=args.max_frames,
    )
