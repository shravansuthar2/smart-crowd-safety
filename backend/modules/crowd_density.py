"""
Crowd Density Detection + Heatmap Module — v4 (ADVANCED)
=========================================================
Features:
1. YOLOv8 person detection with dedup
2. Advanced temporal heatmap that accumulates over time
3. Decay effect — old data fades out
4. Grid-based density calculation
5. Region alerts when grid cell exceeds threshold
6. Works for both live camera and pre-recorded video
"""

from ultralytics import YOLO
import cv2
import numpy as np
from config import (
    CROWD_DENSITY_THRESHOLD, CONFIDENCE_THRESHOLD,
    YOLO_MODEL, YOLO_IMG_SIZE, CROWD_ENHANCE
)

model = YOLO(YOLO_MODEL)
print(f"[Crowd Density] {YOLO_MODEL} model loaded")


# ========== HEATMAP ENGINE (class-based, persistent) ==========

class CrowdHeatmap:
    """
    Advanced crowd density heatmap with temporal accumulation and decay.

    How it works:
    1. Each frame: detect people → add heat at their positions
    2. Heat ACCUMULATES over time (shows where people have been)
    3. Old heat DECAYS slowly (fades out over ~30 frames)
    4. Grid divides frame into cells for density alerts
    5. Colors: Blue(safe) → Green(low) → Yellow(medium) → Red(critical)
    """

    def __init__(self, width=1280, height=720, grid_cols=8, grid_rows=6,
                 decay_rate=0.92, cell_alert_threshold=5):
        """
        Args:
            width, height: frame dimensions
            grid_cols, grid_rows: divide frame into grid for density alerts
            decay_rate: 0.0-1.0, how fast old data fades (0.92 = slow fade)
            cell_alert_threshold: people per grid cell to trigger alert
        """
        self.width = width
        self.height = height
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.decay_rate = decay_rate
        self.cell_alert_threshold = cell_alert_threshold

        # Accumulated heatmap (float, persists between frames)
        self.heat_accumulator = np.zeros((height, width), dtype=np.float64)

        # Grid density tracking
        self.cell_w = width // grid_cols
        self.cell_h = height // grid_rows
        self.grid_density = np.zeros((grid_rows, grid_cols), dtype=np.int32)

        # Stats
        self.total_frames = 0
        self.peak_density = 0
        self.alert_cells = []

        self.initialized = False

    def _ensure_size(self, frame):
        """Resize accumulator if frame size changed"""
        h, w = frame.shape[:2]
        if h != self.height or w != self.width:
            self.width = w
            self.height = h
            self.heat_accumulator = np.zeros((h, w), dtype=np.float64)
            self.cell_w = w // self.grid_cols
            self.cell_h = h // self.grid_rows
            self.grid_density = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)

    def update(self, frame, person_boxes):
        """
        Update heatmap with new detections.

        Args:
            frame: current video frame (BGR)
            person_boxes: list of {"bbox": [x1,y1,x2,y2]} from YOLO

        Returns:
            heatmap overlay image (BGR)
        """
        self._ensure_size(frame)
        self.total_frames += 1

        # Step 1: Decay old heat (fade out)
        self.heat_accumulator *= self.decay_rate

        # Step 2: Reset grid density for this frame
        self.grid_density.fill(0)

        # Step 3: Add heat for each detected person
        for box_data in person_boxes:
            x1, y1, x2, y2 = box_data["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            person_h = y2 - y1

            # Heat radius based on person size (larger person = larger heat area)
            radius = max(40, int(person_h * 0.6))

            # Add gaussian-like heat at person's position
            # Feet position (bottom center) is more accurate for ground density
            foot_y = min(y2, self.height - 1)
            cv2.circle(self.heat_accumulator, (cx, foot_y), radius, 0.3, -1)

            # Also add heat at center
            cv2.circle(self.heat_accumulator, (cx, cy), radius // 2, 0.15, -1)

            # Update grid density
            grid_col = min(cx // max(self.cell_w, 1), self.grid_cols - 1)
            grid_row = min(cy // max(self.cell_h, 1), self.grid_rows - 1)
            self.grid_density[grid_row][grid_col] += 1

        # Step 4: Smooth the accumulated heatmap
        smoothed = cv2.GaussianBlur(self.heat_accumulator, (51, 51), 0)

        # Step 5: Normalize for visualization
        # Use fixed scale so colors are consistent
        # Values: 0-0.3 = blue(safe), 0.3-0.6 = green/yellow(medium), 0.6+ = red(critical)
        normalized = np.clip(smoothed / 1.5, 0, 1)
        heat_uint8 = np.uint8(255 * normalized)

        # Step 6: Apply colormap
        heatmap_colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

        # Step 7: Only overlay where there's meaningful heat
        mask = heat_uint8 > 15
        overlay = frame.copy()
        overlay[mask] = cv2.addWeighted(frame, 0.35, heatmap_colored, 0.65, 0)[mask]

        # Step 8: Draw grid lines and alert cells
        self.alert_cells = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                gx1 = col * self.cell_w
                gy1 = row * self.cell_h
                gx2 = gx1 + self.cell_w
                gy2 = gy1 + self.cell_h

                density = self.grid_density[row][col]

                # Draw grid lines (subtle)
                cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), (50, 50, 50), 1)

                # Alert cell if density exceeds threshold
                if density >= self.cell_alert_threshold:
                    self.alert_cells.append({
                        "row": row, "col": col,
                        "density": int(density),
                        "x": gx1, "y": gy1
                    })
                    # Red border on alert cells
                    cv2.rectangle(overlay, (gx1 + 1, gy1 + 1), (gx2 - 1, gy2 - 1), (0, 0, 255), 3)
                    cv2.putText(overlay, f"{density}", (gx1 + 5, gy1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                elif density > 0:
                    # Show count in cells with people
                    cv2.putText(overlay, f"{density}", (gx1 + 5, gy1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Step 9: Stats overlay
        self.peak_density = max(self.peak_density, len(person_boxes))
        people = len(person_boxes)

        cv2.putText(overlay, f"Heatmap | {people} people | Peak: {self.peak_density}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if self.alert_cells:
            cv2.putText(overlay, f"HIGH DENSITY in {len(self.alert_cells)} zone(s)!", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return overlay

    def get_grid_data(self):
        """Get grid density data for API response"""
        return {
            "grid": self.grid_density.tolist(),
            "alert_cells": self.alert_cells,
            "peak_density": self.peak_density,
            "total_frames": self.total_frames
        }

    def reset(self):
        """Reset heatmap for new video"""
        self.heat_accumulator.fill(0)
        self.grid_density.fill(0)
        self.total_frames = 0
        self.peak_density = 0
        self.alert_cells = []


# Global heatmap instance (persists across frames)
heatmap_engine = CrowdHeatmap()


# ========== IMAGE PREPROCESSING ==========

def enhance_for_detection(frame):
    """Enhance dark/low-contrast images only"""
    enhanced = frame.copy()
    brightness = np.mean(enhanced)

    if brightness < 80:
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.8, beta=40)
    elif brightness > 220:
        enhanced = cv2.convertScaleAbs(enhanced, alpha=0.7, beta=-20)
    else:
        return None

    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return enhanced


# ========== DEDUP ==========

def remove_duplicate_boxes(boxes, iou_threshold=0.5):
    """Remove duplicate detections using IoU"""
    if not boxes:
        return []

    min_area = 1000
    boxes = [b for b in boxes if
             (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]) > min_area]

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

            b1, b2 = box_a["bbox"], box_b["bbox"]
            xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
            a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            iou = inter / (a1 + a2 - inter + 1e-6)

            if iou > iou_threshold:
                used.add(j)

    return keep


# ========== MAIN DETECTION ==========

def detect_crowd(frame):
    """Detect people in frame"""
    original = frame.copy()
    oh, ow = original.shape[:2]

    results = model(original, classes=[0], conf=CONFIDENCE_THRESHOLD,
                    imgsz=YOLO_IMG_SIZE, verbose=False)

    all_boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            all_boxes.append({"bbox": [x1, y1, x2, y2], "confidence": conf})

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

    unique_boxes = remove_duplicate_boxes(all_boxes)
    person_count = len(unique_boxes)

    # Draw boxes
    for box_data in unique_boxes:
        x1, y1, x2, y2 = box_data["bbox"]
        conf = box_data["confidence"]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(ow-1, x2), min(oh-1, y2)
        box_data["bbox"] = [x1, y1, x2, y2]

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


# ========== HEATMAP FUNCTIONS ==========

def get_density_heatmap(frame):
    """
    Generate heatmap for a single frame (used in image upload).
    For video, use get_video_heatmap() which accumulates over time.
    """
    results = model(frame, classes=[0], conf=CONFIDENCE_THRESHOLD,
                    imgsz=YOLO_IMG_SIZE, verbose=False)

    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append({"bbox": [x1, y1, x2, y2]})

    if not boxes:
        cv2.putText(frame, "Heatmap | No people detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        return frame

    # Use heatmap engine for single frame (reset first)
    temp_heatmap = CrowdHeatmap(frame.shape[1], frame.shape[0])
    return temp_heatmap.update(frame, boxes)


def get_video_heatmap(frame, person_boxes):
    """
    Generate TEMPORAL heatmap for video — accumulates over time.
    Called once per processed frame.
    """
    global heatmap_engine
    return heatmap_engine.update(frame, person_boxes)


def reset_heatmap():
    """Reset heatmap for new video"""
    global heatmap_engine
    heatmap_engine.reset()


def get_heatmap_grid_data():
    """Get grid density data for API"""
    return heatmap_engine.get_grid_data()
