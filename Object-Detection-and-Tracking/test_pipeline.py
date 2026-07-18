"""
Verification tests for the detection + tracking pipeline.

Run with: python test_pipeline.py

Uses the bundled sample video (people walking, from OpenCV's own sample
data) as a real test case, since a synthetic/blank video wouldn't
meaningfully exercise a real object detector.
"""
import cv2

from detector import YOLODetector
from sort_tracker import SortTracker, iou

SAMPLE_VIDEO = "sample_data/vtest.avi"


def test_detector_finds_people():
    detector = YOLODetector(confidence_threshold=0.4)
    cap = cv2.VideoCapture(SAMPLE_VIDEO)
    ret, frame = cap.read()
    cap.release()
    assert ret, "Could not read sample video"

    detections = detector.detect(frame, classes_of_interest={"person"})
    assert len(detections) >= 2, f"Expected multiple people in frame 0, got {len(detections)}"

    for (x, y, w, h, conf, cls) in detections:
        assert cls == "person"
        assert 0.4 <= conf <= 1.0
        assert w > 0 and h > 0

    print(f"PASS: detector found {len(detections)} people in frame 0")


def test_iou():
    # Identical boxes -> IOU 1.0
    box = (10, 10, 50, 50)
    assert abs(iou(box, box) - 1.0) < 1e-6

    # Non-overlapping boxes -> IOU 0.0
    box_a = (0, 0, 10, 10)
    box_b = (100, 100, 110, 110)
    assert iou(box_a, box_b) == 0.0

    # Partial overlap -> IOU strictly between 0 and 1
    box_c = (0, 0, 10, 10)
    box_d = (5, 5, 15, 15)
    val = iou(box_c, box_d)
    assert 0.0 < val < 1.0

    print("PASS: IOU calculation correct for identical/disjoint/overlapping boxes")


def test_tracks_persist_across_frames():
    detector = YOLODetector(confidence_threshold=0.4)
    tracker = SortTracker(max_age=15, min_hits=2, iou_threshold=0.3)

    cap = cv2.VideoCapture(SAMPLE_VIDEO)
    id_frame_counts = {}

    for frame_idx in range(60):
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect(frame, classes_of_interest={"person"})
        tracks = tracker.update(detections)
        for (x, y, w, h, track_id, cls, conf) in tracks:
            id_frame_counts[track_id] = id_frame_counts.get(track_id, 0) + 1

    cap.release()

    assert len(id_frame_counts) > 0, "No tracks were produced at all"

    long_lived = [tid for tid, count in id_frame_counts.items() if count >= 10]
    assert len(long_lived) >= 2, (
        f"Expected at least 2 tracks to persist 10+ frames, got {len(long_lived)}. "
        f"Track lifetimes: {id_frame_counts}"
    )

    flicker_tracks = [tid for tid, count in id_frame_counts.items() if count == 1]
    flicker_ratio = len(flicker_tracks) / len(id_frame_counts)
    assert flicker_ratio < 0.3, (
        f"Too many single-frame flicker tracks ({len(flicker_tracks)}/{len(id_frame_counts)}) "
        f"— tracking is not stable"
    )

    print(
        f"PASS: {len(id_frame_counts)} total tracks, {len(long_lived)} persisted 10+ frames, "
        f"{len(flicker_tracks)} single-frame flickers"
    )


def test_bbox_stays_in_reasonable_bounds():
    """Sanity check: tracked boxes shouldn't explode in size or go wildly negative."""
    detector = YOLODetector(confidence_threshold=0.4)
    tracker = SortTracker(max_age=15, min_hits=2, iou_threshold=0.3)

    cap = cv2.VideoCapture(SAMPLE_VIDEO)
    frame_w, frame_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for frame_idx in range(40):
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect(frame, classes_of_interest={"person"})
        tracks = tracker.update(detections)
        for (x, y, w, h, track_id, cls, conf) in tracks:
            assert w < frame_w * 1.5, f"Track {track_id} box width exploded: {w}"
            assert h < frame_h * 1.5, f"Track {track_id} box height exploded: {h}"
            assert w > 0 and h > 0, f"Track {track_id} has non-positive size: {w}x{h}"

    cap.release()
    print("PASS: all tracked bounding boxes stayed within reasonable size bounds")


if __name__ == "__main__":
    tests = [test_iou, test_detector_finds_people, test_tracks_persist_across_frames, test_bbox_stays_in_reasonable_bounds]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {test.__name__}: {e}")

    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
    exit(0 if failed == 0 else 1)
