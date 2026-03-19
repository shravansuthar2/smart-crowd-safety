"""
Emergency Detection Module — v6 (TRAINED FIRE MODEL)
======================================================
Fire detection using YOLOv8 trained on Kaggle fire dataset.
Precision: 100%, Recall: 90.8%, mAP@50: 97.8%
"""

import cv2
import numpy as np
import os

# Load trained fire model
FIRE_MODEL = None
try:
    from ultralytics import YOLO
    fire_model_path = os.path.join(os.path.dirname(__file__), "..", "fire_model.pt")
    if os.path.exists(fire_model_path):
        FIRE_MODEL = YOLO(fire_model_path)
        print("[Emergency] Fire model loaded (Kaggle trained, 97.8% mAP)")
    else:
        print("[Emergency] No fire_model.pt found — fire detection disabled")
except Exception as e:
    print(f"[Emergency] Fire model error: {e}")

print("[Emergency] v6 loaded")


def detect_emergency(frame):
    """Detect fire using trained YOLOv8 model only."""
    emergency = {
        "fire_detected": False,
        "smoke_detected": False,
        "fall_detected": False,
        "fight_detected": False,
        "frame": frame
    }

    if FIRE_MODEL is None:
        return emergency

    # Run fire detection with HIGH confidence (avoid false positives)
    results = FIRE_MODEL(frame, conf=0.6, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = result.names[cls].lower()

            if "fire" in label or "flame" in label:
                emergency["fire_detected"] = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"FIRE {conf:.0%}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif "smoke" in label:
                emergency["smoke_detected"] = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                cv2.putText(frame, f"SMOKE {conf:.0%}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

    if emergency["fire_detected"]:
        oh, ow = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (ow, 45), (0, 0, 200), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        cv2.putText(frame, "FIRE DETECTED!", (ow // 2 - 130, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    emergency["frame"] = frame
    return emergency
