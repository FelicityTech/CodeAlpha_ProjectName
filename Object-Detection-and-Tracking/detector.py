"""
YOLOv4-tiny object detector using OpenCV's DNN module.

Deliberately avoids PyTorch/ultralytics — OpenCV's cv2.dnn backend runs
YOLOv4-tiny directly from the original Darknet .cfg/.weights files, which
keeps the dependency footprint small (just opencv-python) while still
giving real-time performance on CPU.
"""
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).parent / "models"


class YOLODetector:
    def __init__(
        self,
        cfg_path: Path = MODEL_DIR / "yolov4-tiny.cfg",
        weights_path: Path = MODEL_DIR / "yolov4-tiny.weights",
        names_path: Path = MODEL_DIR / "coco.names",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: int = 416,
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        with open(names_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = self.net.getLayerNames()
        unconnected = self.net.getUnconnectedOutLayers()
        # OpenCV versions differ on whether this returns a 1D or 2D array
        self.output_layers = [layer_names[i - 1] for i in unconnected.flatten()]

    def detect(self, frame: np.ndarray, classes_of_interest: set[str] | None = None):
        """
        Run detection on a single BGR frame.

        Returns a list of (x, y, w, h, confidence, class_name) tuples in
        pixel coordinates, already filtered by confidence + NMS.
        classes_of_interest: if given, only return detections of these
        COCO class names (e.g. {"person", "car"}). None = all classes.
        """
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence < self.confidence_threshold:
                    continue

                class_name = self.class_names[class_id]
                if classes_of_interest and class_name not in classes_of_interest:
                    continue

                cx, cy, bw, bh = detection[0:4] * np.array([w, h, w, h])
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)

                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(confidence)
                class_ids.append(class_id)

        # Non-max suppression to collapse overlapping boxes for the same object
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

        detections = []
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                x, y, bw, bh = boxes[i]
                detections.append((x, y, bw, bh, confidences[i], self.class_names[class_ids[i]]))

        return detections
