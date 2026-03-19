"""
Crowd Density Detection Module — v3 (FIXED)
=============================================
Uses YOLOv8s with proper detection pipeline.

Key fix: Run YOLO directly on original image (YOLO handles resizing internally).
Only enhance for dark/blurry images, and run YOLO on both to merge results.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from config import (
    CROWD_DENSITY_THRESHOLD, CONFIDENCE_THRESHOLD,
    YOLO_MODEL, YOLO_IMG_SIZE, CROWD_ENHANCE
)

model = YOLO(YOLO_MODEL)
print(f"[Crowd Density] {YOLO_MODEL} model loaded (upgraded)")


def enhance_for_detection(frame):
    """Enhance dark/low-contrast images only"""
    enhanced = frame.copy()

    brightness = np.mean(enhanced)

    # Only enhance if actually dark or washed out
    if brightness < 80:
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.8, beta=40)
    elif brightness > 220:
        enhanced = cv2.convertScaleAbs(enhanced, alpha=0.7, beta=-20)
    else:
        return None  # no enhancement needed

    # CLAHE contrast
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return enhanced


def remove_duplicate_boxes(boxes, iou_threshold=0.5):
    """
    Remove duplicate detections using IoU ONLY.

    Previous bug: center distance check was removing valid people
    standing next to each other (their centers are close but they
    are different people).

    Now: only remove if boxes overlap > 50% (true duplicates).
    """
    if not boxes:
        return []

    # Filter tiny boxes (body parts like hands)
    min_area = 1000
    boxes = [b for b in boxes if
             (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]) > min_area]

    # Sort by confidence (keep highest)
    boxes = sorted(boxes, key=lambda x: x["confidence"], reverse=True)

    keep = []
    used = set()

    for i, box_a in enumerate(boxes):
        if i in used:
            continue
        keep.append(box_a)

        for j, box_b in enumerate(boxes[i + 1:], start=i + 1):
            if j in used:
                continue

            # IoU check only — no center distance (was causing false dedup)
            b1, b2 = box_a["bbox"], box_b["bbox"]
            xi1 = max(b1[0], b2[0])
            yi1 = max(b1[1], b2[1])
            xi2 = min(b1[2], b2[2])
            yi2 = min(b1[3], b2[3])
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
            a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            iou = inter / (a1 + a2 - inter + 1e-6)

            if iou > iou_threshold:
                used.add(j)

    return keep


def detect_crowd(frame):
    """
    Detect people in frame — v3 FIXED.

    Simple and reliable:
    1. Run YOLO directly on original frame (no resize bugs)
    2. If dark/blurry → also run on enhanced version
    3. Merge + dedup
    4. Draw on original frame
    """
    original = frame.copy()
    oh, ow = original.shape[:2]

    # Step 1: Run YOLO on original image directly
    results = model(original, classes=[0], conf=CONFIDENCE_THRESHOLD,
                    imgsz=YOLO_IMG_SIZE, verbose=False)

    all_boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            all_boxes.append({"bbox": [x1, y1, x2, y2], "confidence": conf})

    # Step 2: If image is dark/bright, also try enhanced
    if CROWD_ENHANCE:
        enhanced = enhance_for_detection(original)
        if enhanced is not None:
            results_enh = model(enhanced, classes=[0], conf=CONFIDENCE_THRESHOLD,
                                imgsz=YOLO_IMG_SIZE, verbose=False)
            for result in results_enh:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    all_boxes.append({"bbox": [x1, y1, x2, y2], "confidence": conf})

    # Step 3: Remove duplicates
    unique_boxes = remove_duplicate_boxes(all_boxes)
    person_count = len(unique_boxes)

    # Step 4: Draw on original frame
    for box_data in unique_boxes:
        x1, y1, x2, y2 = box_data["bbox"]
        conf = box_data["confidence"]

        # Clamp to bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(ow-1, x2), min(oh-1, y2)
        box_data["bbox"] = [x1, y1, x2, y2]

        # Color by confidence
        if conf > 0.6:
            color = (0, 255, 0)
        elif conf > 0.4:
            color = (0, 255, 255)
        else:
            color = (0, 165, 255)

        cv2.rectangle(original, (x1, y1), (x2, y2), color, 2)

        label = f"{conf:.0%}"
        lbl_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        cv2.rectangle(original, (x1, y1 - lbl_sz[1] - 4), (x1 + lbl_sz[0] + 4, y1), color, -1)
        cv2.putText(original, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    # Status overlay
    is_overcrowded = person_count > CROWD_DENSITY_THRESHOLD
    overlay = original.copy()
    cv2.rectangle(overlay, (0, 0), (350, 80), (0, 0, 0), -1)
    original = cv2.addWeighted(overlay, 0.7, original, 0.3, 0)

    count_color = (0, 0, 255) if is_overcrowded else (0, 255, 0)
    cv2.putText(original, f"People: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, count_color, 2)

    density_pct = min(100, int((person_count / max(CROWD_DENSITY_THRESHOLD, 1)) * 100))
    cv2.putText(original, f"Threshold: {CROWD_DENSITY_THRESHOLD} | Model: {YOLO_MODEL}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(original, f"Density: {density_pct}%", (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, count_color, 1)

    if is_overcrowded:
        cv2.putText(original, "! OVERCROWDED !", (ow - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return {
        "count": person_count,
        "is_overcrowded": is_overcrowded,
        "threshold": CROWD_DENSITY_THRESHOLD,
        "boxes": unique_boxes,
        "frame": original
    }


def get_density_heatmap(frame):
    """Generate crowd density heatmap"""
    results = model(frame, classes=[0], conf=CONFIDENCE_THRESHOLD,
                    imgsz=YOLO_IMG_SIZE, verbose=False)

    heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
    person_count = 0

    for result in results:
        for box in result.boxes:
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            radius = max(30, min(y2 - y1, 100))
            cv2.circle(heatmap, (cx, cy), radius, 1, -1)

    if person_count == 0:
        cv2.putText(frame, "Heatmap | No people detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        return frame

    heatmap = cv2.GaussianBlur(heatmap, (99, 99), 0)
    heatmap = np.uint8(255 * heatmap / (heatmap.max() + 1e-5))
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.5, heatmap_colored, 0.5, 0)
    cv2.putText(overlay, f"Heatmap | {person_count} people", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return overlay
