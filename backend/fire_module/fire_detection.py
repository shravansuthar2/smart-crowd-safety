"""
Fire Detection Module — Standalone, Plug-and-Play
===================================================
A completely independent fire detection component.
Does NOT depend on any existing project code.

Usage:
    from fire_module.fire_detection import FireDetector

    detector = FireDetector()                      # uses config_fire.yaml
    detector = FireDetector(config_path="custom.yaml")  # custom config

    results = detector.detect(frame)               # run detection
    annotated = detector.annotate(frame, results)  # draw boxes
    # OR in one step:
    frame, results = detector.detect_and_annotate(frame)
"""

import cv2
import numpy as np
import os
import yaml

from .model_loader import get_model


class FireDetector:
    """
    Standalone fire detection using YOLO.
    Safe to use — never crashes, returns empty results on failure.
    """

    def __init__(self, config_path=None):
        """
        Initialize the fire detector.

        Args:
            config_path: Path to config_fire.yaml. If None, uses default.
        """
        self.config = self._load_config(config_path)
        self.enabled = self.config.get("enabled", True)
        self.model = None
        self._persist_counter = 0
        self._last_boxes = []

        if self.enabled:
            self.model = get_model(
                model_path=self.config.get("model_path", "fire_model.pt"),
                device=self.config.get("device", "auto"),
            )
            if self.model is None:
                print("[FireDetector] Model unavailable — detection disabled")
                self.enabled = False

    def _load_config(self, config_path):
        """Load YAML config file with safe defaults."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config_fire.yaml")

        defaults = {
            "model_path": "fire_model.pt",
            "confidence_threshold": 0.5,
            "alert_threshold": 0.7,
            "device": "auto",
            "inference_size": 640,
            "persist_frames": 15,
            "enabled": True,
            "box_color": [0, 0, 255],
            "box_thickness": 3,
            "label_font_scale": 0.7,
            "banner_enabled": True,
        }

        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    loaded = yaml.safe_load(f) or {}
                defaults.update(loaded)
        except Exception as e:
            print(f"[FireDetector] Config error: {e}, using defaults")

        return defaults

    def detect(self, frame):
        """
        Run fire detection on a single frame.

        Args:
            frame: numpy array (BGR, from OpenCV)

        Returns:
            dict with:
                - fire_detected (bool)
                - smoke_detected (bool)
                - detections (list of dicts with x1,y1,x2,y2,confidence,label)
                - alert (bool) — True if any detection exceeds alert_threshold
        """
        result = {
            "fire_detected": False,
            "smoke_detected": False,
            "detections": [],
            "alert": False,
        }

        if not self.enabled or self.model is None or frame is None:
            return self._apply_persistence(result)

        try:
            conf = self.config["confidence_threshold"]
            imgsz = self.config["inference_size"]

            predictions = self.model(frame, conf=conf, imgsz=imgsz, verbose=False)

            for pred in predictions:
                for box in pred.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = pred.names[cls].lower()

                    detection = {
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "confidence": confidence,
                        "label": label,
                    }

                    if "fire" in label or "flame" in label:
                        result["fire_detected"] = True
                        result["detections"].append(detection)

                        if confidence >= self.config["alert_threshold"]:
                            result["alert"] = True

                    elif "smoke" in label:
                        result["smoke_detected"] = True
                        result["detections"].append(detection)

        except Exception as e:
            print(f"[FireDetector] Detection error: {e}")

        return self._apply_persistence(result)

    def _apply_persistence(self, result):
        """
        Temporal smoothing: keep fire status active for N frames
        after last detection to prevent blinking in video.
        """
        persist = self.config.get("persist_frames", 0)

        if result["fire_detected"]:
            self._persist_counter = persist
            self._last_boxes = result["detections"]
        elif self._persist_counter > 0:
            self._persist_counter -= 1
            result["fire_detected"] = True
            result["detections"] = self._last_boxes
            result["persisted"] = True

        return result

    def annotate(self, frame, result):
        """
        Draw fire detection results on a frame.

        Args:
            frame: numpy array (BGR)
            result: dict from detect()

        Returns:
            annotated frame (new copy, original untouched)
        """
        if not result["fire_detected"]:
            return frame

        annotated = frame.copy()
        color = tuple(self.config.get("box_color", [0, 0, 255]))
        thickness = self.config.get("box_thickness", 3)
        font_scale = self.config.get("label_font_scale", 0.7)

        # Draw bounding boxes
        for det in result.get("detections", []):
            x1, y1 = det["x1"], det["y1"]
            x2, y2 = det["x2"], det["y2"]
            conf = det["confidence"]
            label = det["label"]

            # Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Label
            text = f"FIRE {conf:.0%}" if "fire" in label else f"SMOKE {conf:.0%}"
            cv2.putText(annotated, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

        # Top banner
        if self.config.get("banner_enabled", True):
            h, w = annotated.shape[:2]
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 200), -1)
            annotated = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)
            cv2.putText(annotated, "FIRE DETECTED!", (w // 2 - 130, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Console alert
        if result.get("alert", False):
            print("[FireDetector] FIRE ALERT — high confidence detection")

        return annotated

    def detect_and_annotate(self, frame):
        """
        Convenience method: detect + annotate in one call.

        Args:
            frame: numpy array (BGR)

        Returns:
            (annotated_frame, result_dict)
        """
        result = self.detect(frame)
        annotated = self.annotate(frame, result)
        return annotated, result

    def reset_persistence(self):
        """Reset temporal smoothing (call when starting a new video)."""
        self._persist_counter = 0
        self._last_boxes = []
