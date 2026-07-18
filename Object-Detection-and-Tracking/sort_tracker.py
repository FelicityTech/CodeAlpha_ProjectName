"""
A compact SORT (Simple Online and Realtime Tracking) implementation.

SORT tracks objects frame-to-frame by:
1. Predicting each existing track's next position with a Kalman filter
   (constant-velocity motion model over [x, y, aspect_ratio, height]).
2. Matching predictions to new detections by IOU (intersection-over-union),
   solved optimally with the Hungarian algorithm (scipy linear_sum_assignment).
3. Updating matched tracks with the real detection, aging out tracks that
   go unmatched for too long, and spawning new tracks for detections that
   matched nothing.

This is a from-scratch implementation of the well-known SORT algorithm
(Bewley et al., 2016) — it follows the published approach but is written
independently, not copied from any reference codebase.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(box_a, box_b):
    """IOU between two boxes in (x1, y1, x2, y2) format."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def xywh_to_xyxy(box):
    x, y, w, h = box
    return (x, y, x + w, y + h)


class KalmanBoxTracker:
    """
    Tracks a single object's bounding box over time with a constant-velocity
    Kalman filter. State vector: [cx, cy, s, r, vcx, vcy, vs]
    where s = area, r = aspect ratio (w/h, held constant).
    """

    _next_id = 1

    def __init__(self, bbox_xywh, class_name, confidence):
        x, y, w, h = bbox_xywh
        cx, cy = x + w / 2, y + h / 2
        s = w * h
        r = w / h if h != 0 else 1.0

        # State: [cx, cy, s, r, vcx, vcy, vs]  (r held constant, no vr term)
        self.state = np.array([cx, cy, s, r, 0, 0, 0], dtype=np.float64)

        # Constant-velocity transition matrix
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0

        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0  # high uncertainty on initial velocity

        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01
        self.Q[2, 2] = 0.1  # area changes faster than position

        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0  # less trust in area/ratio measurement noise

        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1

        self.class_name = class_name
        self.confidence = confidence
        self.hit_streak = 1     # consecutive frames matched (resets to 0 on a miss)
        self.age = 0
        self.time_since_update = 0

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        if self.state[2] < 0:
            self.state[2] = 0
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_bbox_xywh()

    def update(self, bbox_xywh, class_name, confidence):
        x, y, w, h = bbox_xywh
        cx, cy = x + w / 2, y + h / 2
        s = w * h
        r = w / h if h != 0 else 1.0
        z = np.array([cx, cy, s, r])

        y_residual = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y_residual
        self.P = (np.eye(7) - K @ self.H) @ self.P

        self.class_name = class_name
        self.confidence = confidence
        self.hit_streak += 1
        self.time_since_update = 0

    def get_bbox_xywh(self):
        cx, cy, s, r = self.state[:4]
        s = max(s, 0)
        w = np.sqrt(s * r) if r > 0 else 0
        h = s / w if w > 0 else 0
        x = cx - w / 2
        y = cy - h / 2
        return (float(x), float(y), float(w), float(h))


class SortTracker:
    def __init__(self, max_age: int = 15, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        max_age: frames a track can go unmatched before being deleted
        min_hits: consecutive matches needed before a track is reported
                  (filters out one-frame false-positive detections)
        iou_threshold: minimum IOU for a detection-to-track match
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections):
        """
        detections: list of (x, y, w, h, confidence, class_name)
        Returns list of (x, y, w, h, track_id, class_name, confidence)
        for tracks confirmed this frame.
        """
        self.frame_count += 1

        # Predict new locations for all existing tracks
        predicted_boxes = [t.predict() for t in self.tracks]

        matches, unmatched_dets, unmatched_tracks = self._associate(detections, predicted_boxes)

        for det_idx, track_idx in matches:
            x, y, w, h, conf, cls = detections[det_idx]
            self.tracks[track_idx].update((x, y, w, h), cls, conf)

        for det_idx in unmatched_dets:
            x, y, w, h, conf, cls = detections[det_idx]
            self.tracks.append(KalmanBoxTracker((x, y, w, h), cls, conf))

        # Drop tracks that have gone too long without a match
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        results = []
        for t in self.tracks:
            if t.time_since_update == 0 and (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                x, y, w, h = t.get_bbox_xywh()
                results.append((x, y, w, h, t.id, t.class_name, t.confidence))
        return results

    def _associate(self, detections, predicted_boxes):
        if not detections or not predicted_boxes:
            return [], list(range(len(detections))), list(range(len(predicted_boxes)))

        iou_matrix = np.zeros((len(detections), len(predicted_boxes)), dtype=np.float64)
        for d, det in enumerate(detections):
            det_box = xywh_to_xyxy(det[:4])
            for t, trk_box in enumerate(predicted_boxes):
                iou_matrix[d, t] = iou(det_box, xywh_to_xyxy(trk_box))

        # Hungarian algorithm minimizes cost, so we negate IOU (maximize overlap)
        row_idx, col_idx = linear_sum_assignment(-iou_matrix)

        matches, unmatched_dets, unmatched_tracks = [], [], []
        matched_det_set, matched_trk_set = set(), set()

        for d, t in zip(row_idx, col_idx):
            if iou_matrix[d, t] >= self.iou_threshold:
                matches.append((d, t))
                matched_det_set.add(d)
                matched_trk_set.add(t)

        unmatched_dets = [d for d in range(len(detections)) if d not in matched_det_set]
        unmatched_tracks = [t for t in range(len(predicted_boxes)) if t not in matched_trk_set]

        return matches, unmatched_dets, unmatched_tracks
