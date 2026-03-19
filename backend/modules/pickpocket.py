"""
Pickpocket Detection Module
Uses MediaPipe Pose to detect body keypoints.
Checks if someone's hand is suspiciously close to another person's pocket area.

How it works:
1. YOLO detects all persons → gives bounding boxes
2. MediaPipe Pose runs on each person → gives 33 body keypoints
3. For each pair of persons (A, B):
   - Check if A's WRIST is near B's HIP (pocket area)
   - If distance < threshold → SUSPICIOUS
4. Alert triggered

Body Keypoints Used:
    LEFT_WRIST (15), RIGHT_WRIST (16) → hand position
    LEFT_HIP (23), RIGHT_HIP (24)     → pocket area
"""

import mediapipe as mp
import cv2
import numpy as np
from config import SUSPICIOUS_PROXIMITY, POSE_CONFIDENCE

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    min_detection_confidence=POSE_CONFIDENCE,
    min_tracking_confidence=POSE_CONFIDENCE,
    model_complexity=1
)

print("[Pickpocket] MediaPipe Pose loaded")


def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_keypoints(landmarks, frame_shape):
    """
    Extract wrist and hip positions from pose landmarks.

    Args:
        landmarks: MediaPipe pose landmarks (33 points)
        frame_shape: (height, width, channels) of the frame

    Returns:
        dict with wrists and hips pixel coordinates
    """
    h, w = frame_shape[:2]

    points = {}
    # Wrists (hands)
    rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    points["right_wrist"] = (int(rw.x * w), int(rw.y * h), rw.visibility)
    points["left_wrist"] = (int(lw.x * w), int(lw.y * h), lw.visibility)

    # Hips (pocket area)
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    points["right_hip"] = (int(rh.x * w), int(rh.y * h))
    points["left_hip"] = (int(lh.x * w), int(lh.y * h))

    return points


def detect_pickpocket(frame):
    """
    Single-frame pickpocket detection.
    Detects pose and checks for suspicious hand-near-pocket behavior.

    Args:
        frame: numpy array (BGR image)

    Returns:
        dict with suspicious (bool), alerts (list), frame (annotated)
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    alerts = []

    if not results.pose_landmarks:
        return {"suspicious": False, "alerts": [], "frame": frame}

    landmarks = results.pose_landmarks.landmark
    keypoints = get_keypoints(landmarks, frame.shape)

    # Draw skeleton on frame
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

    # Check each wrist against each hip
    wrist_keys = ["right_wrist", "left_wrist"]
    hip_keys = ["right_hip", "left_hip"]

    for wrist_key in wrist_keys:
        wx, wy, visibility = keypoints[wrist_key]
        if visibility < 0.5:
            continue

        for hip_key in hip_keys:
            hx, hy = keypoints[hip_key]
            dist = distance((wx, wy), (hx, hy))

            if dist < SUSPICIOUS_PROXIMITY:
                alerts.append({
                    "type": "hand_near_pocket",
                    "wrist": wrist_key,
                    "hip": hip_key,
                    "distance_px": round(dist, 1),
                    "position": {"wrist": [wx, wy], "hip": [hx, hy]}
                })

                # Draw alert visuals
                cv2.circle(frame, (wx, wy), 12, (0, 0, 255), -1)      # Red dot on hand
                cv2.circle(frame, (hx, hy), 12, (255, 165, 0), -1)    # Orange dot on hip
                cv2.line(frame, (wx, wy), (hx, hy), (0, 0, 255), 2)   # Red line connecting
                cv2.putText(frame, f"SUSPICIOUS ({dist:.0f}px)",
                            (wx, wy - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

    # Status overlay
    status = "SUSPICIOUS ACTIVITY DETECTED" if alerts else "No suspicious activity"
    color = (0, 0, 255) if alerts else (0, 255, 0)
    cv2.putText(frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return {
        "suspicious": len(alerts) > 0,
        "alerts": alerts,
        "frame": frame
    }


def detect_pickpocket_multi(frame, person_boxes):
    """
    Multi-person pickpocket detection.
    Checks hand of Person A near hip of Person B.

    Args:
        frame: numpy array (BGR image)
        person_boxes: list of [x1, y1, x2, y2] from YOLO detection

    Returns:
        dict with suspicious (bool), alerts (list), frame (annotated)
    """
    all_persons = []

    # Run pose detection on each person crop
    for i, box in enumerate(person_boxes):
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_crop)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = get_keypoints(landmarks, crop.shape)

            # Offset keypoints to original frame coordinates
            for key in keypoints:
                if len(keypoints[key]) == 3:  # wrist (x, y, visibility)
                    kx, ky, vis = keypoints[key]
                    keypoints[key] = (kx + x1, ky + y1, vis)
                else:  # hip (x, y)
                    kx, ky = keypoints[key]
                    keypoints[key] = (kx + x1, ky + y1)

            all_persons.append({"id": i, "keypoints": keypoints, "box": box})

    # Cross-person check: A's hand near B's pocket
    alerts = []
    for person_a in all_persons:
        for person_b in all_persons:
            if person_a["id"] == person_b["id"]:
                continue

            for wrist_key in ["right_wrist", "left_wrist"]:
                wx, wy, vis = person_a["keypoints"][wrist_key]
                if vis < 0.5:
                    continue

                for hip_key in ["right_hip", "left_hip"]:
                    hx, hy = person_b["keypoints"][hip_key]
                    dist = distance((wx, wy), (hx, hy))

                    if dist < SUSPICIOUS_PROXIMITY:
                        alerts.append({
                            "type": "pickpocket_suspected",
                            "suspect": person_a["id"],
                            "victim": person_b["id"],
                            "distance_px": round(dist, 1)
                        })

                        # Draw on frame
                        cv2.line(frame, (wx, wy), (hx, hy), (0, 0, 255), 3)
                        cv2.putText(frame, "PICKPOCKET!",
                                    (wx, wy - 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 0, 255), 2)

    return {"suspicious": len(alerts) > 0, "alerts": alerts, "frame": frame}
