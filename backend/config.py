import os

# ========== CROWD DENSITY ==========
CROWD_DENSITY_THRESHOLD = 50    # max people before alert
CONFIDENCE_THRESHOLD = 0.2      # Lower to catch all people, dedup handles false positives
YOLO_MODEL = "yolov8s.pt"       # Pre-trained (best until we have 200+ labeled images)
YOLO_IMG_SIZE = 1280            # Higher resolution = detects smaller people
CROWD_ENHANCE = True            # Pre-process dark/blurry images before detection
CROWD_MULTI_SCALE = True        # Scan image at multiple zoom levels for small people

# ========== FACE RECOGNITION ==========
FACE_MATCH_THRESHOLD = 0.40     # Cosine distance threshold for embeddings (lower = stricter, 0.40 is good)
FACE_MODEL = "ArcFace"          # ArcFace = 99.41% accuracy on LFW dataset
FACE_DETECTOR = "retinaface"    # RetinaFace = best for small/blurry/angled faces
FACE_ENHANCE = True             # Pre-process blurry images before detection
FACE_MIN_CONFIDENCE = 0.3       # Lower = detect even low-quality faces

# ========== PICKPOCKET ==========
SUSPICIOUS_PROXIMITY = 50       # pixels - hand near pocket distance
POSE_CONFIDENCE = 0.5           # MediaPipe pose detection confidence

# ========== EMERGENCY ==========
FALL_BODY_RATIO = 1.5           # width/height ratio to detect fall (>1.5 = lying down)
FALL_HEAD_HIP_THRESHOLD = 30    # pixels - if head is this close to hip level = fallen
FIGHT_ELBOW_ANGLE = 120         # degrees - bent arm threshold for fighting stance

# ========== FIREBASE ==========
FIREBASE_CREDENTIALS = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")
FIREBASE_STORAGE_BUCKET = "your-project-id.appspot.com"  # Replace with your bucket

# ========== PATHS ==========
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
MISSING_PERSONS_DIR = os.path.join(UPLOAD_DIR, "missing_persons")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MISSING_PERSONS_DIR, exist_ok=True)

# ========== CAMERA FEEDS ==========
# Replace with your CCTV RTSP URLs
CAMERA_FEEDS = {
    "webcam": 0,                                    # 0 = default webcam
    "cam1": "rtsp://camera1-ip:554/stream",         # Camera 1
    "cam2": "rtsp://camera2-ip:554/stream",         # Camera 2
}
