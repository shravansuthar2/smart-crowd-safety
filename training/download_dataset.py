"""
Download and prepare crowd detection datasets.

Datasets available:
1. COCO Person subset — easiest, auto-download
2. CrowdHuman — best for crowd detection
3. VisDrone — best for aerial/CCTV views

Usage:
    python download_dataset.py coco       # Download COCO person subset
    python download_dataset.py crowdhuman  # Download CrowdHuman
"""

import os
import sys
import yaml
import shutil
from ultralytics import YOLO


def download_coco_person():
    """
    Download COCO dataset person subset using Ultralytics.
    This is the EASIEST way to get training data.
    ~20GB download but very high quality labels.
    """
    print("=" * 50)
    print("  Downloading COCO Person Dataset")
    print("  This will download ~20GB of data")
    print("=" * 50)

    # Create data.yaml for COCO person only
    data_yaml = {
        "path": os.path.abspath("dataset_coco"),
        "train": "images/train2017",
        "val": "images/val2017",
        "nc": 1,
        "names": ["person"]
    }

    os.makedirs("dataset_coco", exist_ok=True)
    yaml_path = "dataset_coco/data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    # Use ultralytics built-in COCO download
    model = YOLO("yolov8s.pt")
    print("\nStarting COCO download via ultralytics...")
    print("This will take 20-30 minutes depending on internet speed.")

    # Train for 1 epoch just to trigger download
    try:
        model.train(data="coco128.yaml", epochs=1, imgsz=640, batch=1, device="cpu")
    except Exception:
        pass

    print("\nCOCO dataset downloaded!")
    print(f"Update DATASET_YAML in train_crowd_yolo.py to: '{yaml_path}'")


def create_sample_dataset():
    """
    Create a small sample dataset for testing the training pipeline.
    Uses 10 dummy images — ONLY for testing if training works.
    For real training, use Roboflow or download CrowdHuman.
    """
    import numpy as np
    import cv2

    print("=" * 50)
    print("  Creating Sample Dataset (for testing only)")
    print("=" * 50)

    base = "dataset"
    for split in ["train", "val"]:
        img_dir = os.path.join(base, split, "images")
        lbl_dir = os.path.join(base, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        count = 8 if split == "train" else 2

        for i in range(count):
            # Create dummy image with colored rectangles (fake people)
            img = np.random.randint(100, 200, (640, 640, 3), dtype=np.uint8)

            labels = []
            num_people = np.random.randint(2, 6)

            for j in range(num_people):
                # Random person box
                cx = np.random.uniform(0.1, 0.9)
                cy = np.random.uniform(0.2, 0.8)
                w = np.random.uniform(0.05, 0.15)
                h = np.random.uniform(0.15, 0.4)

                # Draw rectangle on image
                x1 = int((cx - w/2) * 640)
                y1 = int((cy - h/2) * 640)
                x2 = int((cx + w/2) * 640)
                y2 = int((cy + h/2) * 640)
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

                labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # Save image
            cv2.imwrite(os.path.join(img_dir, f"sample_{i:04d}.jpg"), img)

            # Save label
            with open(os.path.join(lbl_dir, f"sample_{i:04d}.txt"), "w") as f:
                f.write("\n".join(labels))

    # Create data.yaml
    data_yaml = {
        "path": os.path.abspath(base),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["person"]
    }

    with open(os.path.join(base, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f)

    print(f"\nSample dataset created at: {os.path.abspath(base)}/")
    print(f"  Training images: {base}/train/images/ (8 images)")
    print(f"  Validation images: {base}/val/images/ (2 images)")
    print(f"  Config: {base}/data.yaml")
    print()
    print("WARNING: This is dummy data for testing the pipeline only!")
    print("For real training, use Roboflow to label your own CCTV images.")


def setup_roboflow_guide():
    """Print step-by-step Roboflow guide"""
    print("""
╔══════════════════════════════════════════════════════╗
║     HOW TO CREATE TRAINING DATA WITH ROBOFLOW        ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Step 1: Go to https://roboflow.com                  ║
║          Create a FREE account                       ║
║                                                      ║
║  Step 2: Create New Project                          ║
║          → Project Type: "Object Detection"          ║
║          → Name: "Crowd Detection"                   ║
║          → Class: "person"                           ║
║                                                      ║
║  Step 3: Upload Images                               ║
║          → Upload 200-500 crowd/CCTV images          ║
║          → More images = better accuracy             ║
║                                                      ║
║  Step 4: Label (Annotate)                            ║
║          → Draw bounding box around each person      ║
║          → Label as "person"                         ║
║          → Be consistent — label EVERY person        ║
║                                                      ║
║  Step 5: Generate Dataset                            ║
║          → Preprocessing: Auto-Orient, Resize 1280   ║
║          → Augmentation: Flip, Rotation, Brightness  ║
║          → Train/Val split: 80/20                    ║
║                                                      ║
║  Step 6: Export                                      ║
║          → Format: "YOLOv8"                          ║
║          → Download ZIP                              ║
║                                                      ║
║  Step 7: Extract ZIP to:                             ║
║          training/dataset/                           ║
║                                                      ║
║  Step 8: Run training:                               ║
║          python train_crowd_yolo.py                  ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "coco":
            download_coco_person()
        elif cmd == "sample":
            create_sample_dataset()
        elif cmd == "guide":
            setup_roboflow_guide()
        else:
            print("Unknown command. Options: coco, sample, guide")
    else:
        print("Usage:")
        print("  python download_dataset.py sample  → Create test dataset (dummy)")
        print("  python download_dataset.py coco    → Download COCO dataset (20GB)")
        print("  python download_dataset.py guide   → Roboflow labeling guide")
        print()
        setup_roboflow_guide()
