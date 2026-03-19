"""
Train YOLOv8 Fire & Smoke Detection Model
==========================================
Downloads a public fire detection dataset and trains YOLOv8.

Usage:
    python train_fire.py download   # Download fire dataset
    python train_fire.py train      # Train model
    python train_fire.py test img   # Test on image

After training, copy best.pt to backend/fire_model.pt
"""

from ultralytics import YOLO
import os
import sys
import yaml
import cv2
import numpy as np

DATASET_DIR = "fire_dataset"
DATASET_YAML = f"{DATASET_DIR}/data.yaml"
BASE_MODEL = "yolov8s.pt"
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 4
DEVICE = "cpu"


def create_fire_dataset():
    """
    Create a synthetic fire dataset for testing.
    For real training, use Roboflow fire datasets.
    """
    print("=" * 50)
    print("  Creating Fire Detection Dataset")
    print("=" * 50)

    os.makedirs(f"{DATASET_DIR}/train/images", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/train/labels", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/val/images", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/val/labels", exist_ok=True)

    classes = ["fire", "smoke"]

    def create_fire_image(idx, split):
        """Create synthetic image with fire-colored regions"""
        img = np.random.randint(30, 80, (640, 640, 3), dtype=np.uint8)

        labels = []

        if idx % 3 != 0:  # 2/3 images have fire
            # Draw fire-like orange/red region
            num_fires = np.random.randint(1, 3)
            for _ in range(num_fires):
                cx = np.random.randint(100, 540)
                cy = np.random.randint(100, 540)
                rx = np.random.randint(30, 100)
                ry = np.random.randint(40, 120)

                # Orange/red fire colors
                for _ in range(50):
                    px = cx + np.random.randint(-rx, rx)
                    py = cy + np.random.randint(-ry, ry)
                    if 0 <= px < 640 and 0 <= py < 640:
                        r = np.random.randint(200, 255)
                        g = np.random.randint(50, 180)
                        b = np.random.randint(0, 50)
                        cv2.circle(img, (px, py), np.random.randint(3, 15),
                                  (b, g, r), -1)

                # YOLO label
                ncx, ncy = cx / 640, cy / 640
                nw, nh = (rx * 2) / 640, (ry * 2) / 640
                labels.append(f"0 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")

        if idx % 4 == 0:  # some images have smoke
            sx = np.random.randint(100, 540)
            sy = np.random.randint(50, 300)
            sw = np.random.randint(80, 200)
            sh = np.random.randint(60, 150)

            # Gray smoke
            overlay = img.copy()
            cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh),
                         (180, 180, 180), -1)
            img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

            ncx, ncy = (sx + sw / 2) / 640, (sy + sh / 2) / 640
            nw, nh = sw / 640, sh / 640
            labels.append(f"1 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")

        cv2.imwrite(f"{DATASET_DIR}/{split}/images/fire_{idx:04d}.jpg", img)
        with open(f"{DATASET_DIR}/{split}/labels/fire_{idx:04d}.txt", "w") as f:
            f.write("\n".join(labels))

    # Create images
    for i in range(40):
        create_fire_image(i, "train")
    for i in range(10):
        create_fire_image(i + 100, "val")

    # data.yaml
    data = {
        "path": os.path.abspath(DATASET_DIR),
        "train": "train/images",
        "val": "val/images",
        "nc": 2,
        "names": classes
    }
    with open(DATASET_YAML, "w") as f:
        yaml.dump(data, f)

    print(f"Dataset created: 40 train + 10 val images")
    print(f"Classes: {classes}")
    print()
    print("For REAL accuracy, use a proper fire dataset from Roboflow:")
    print("  https://universe.roboflow.com/search?q=fire+detection")
    print("  Download as YOLOv8 format → extract to fire_dataset/")


def train():
    """Train YOLOv8 on fire detection dataset"""
    print("=" * 50)
    print("  Training Fire Detection Model")
    print("=" * 50)

    if not os.path.exists(DATASET_YAML):
        print("Dataset not found. Run: python train_fire.py download")
        return

    model = YOLO(BASE_MODEL)

    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project="runs",
        name="fire_detector",
        exist_ok=True,
        patience=15,
        pretrained=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        fliplr=0.5,
        mosaic=1.0,
    )

    print(f"\nTraining complete!")
    print(f"Best model: runs/fire_detector/weights/best.pt")
    print(f"\nTo use:")
    print(f"  cp runs/fire_detector/weights/best.pt ../backend/fire_model.pt")
    print(f"  Restart server — fire model auto-loads")


def test_image(path):
    """Test fire model on an image"""
    model_path = "runs/fire_detector/weights/best.pt"
    if not os.path.exists(model_path):
        print("No trained model. Run training first.")
        return

    model = YOLO(model_path)
    results = model(path, conf=0.3)
    for r in results:
        print(f"Detections: {len(r.boxes)}")
        r.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "download":
            create_fire_dataset()
        elif cmd == "train":
            train()
        elif cmd == "test" and len(sys.argv) > 2:
            test_image(sys.argv[2])
        else:
            print("Usage: python train_fire.py [download|train|test img.jpg]")
    else:
        print("Fire Detection Training")
        print("  python train_fire.py download  → Create dataset")
        print("  python train_fire.py train     → Train model")
        print("  python train_fire.py test img  → Test on image")
