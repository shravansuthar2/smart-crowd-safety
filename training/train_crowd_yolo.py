"""
Train YOLOv8 for Crowd Detection
=================================
This script fine-tunes YOLOv8 on your custom crowd images.

BEFORE RUNNING:
1. Prepare your dataset in the structure below
2. Label images using Roboflow (roboflow.com) or LabelImg
3. Export as YOLOv8 format
4. Place in training/dataset/ folder

Dataset structure:
    training/dataset/
    ├── train/
    │   ├── images/    (your training images)
    │   └── labels/    (YOLO format .txt files)
    ├── val/
    │   ├── images/    (validation images, ~20% of total)
    │   └── labels/
    └── data.yaml

RUN:
    cd smart-crowd-safety/training
    python train_crowd_yolo.py
"""

from ultralytics import YOLO
import os
import yaml

# ========== CONFIGURATION ==========
BASE_MODEL = "yolov8s.pt"       # Start from pre-trained YOLOv8s
DATASET_YAML = "dataset/data.yaml"
EPOCHS = 50                      # Reduced for testing (increase for real training)
IMG_SIZE = 640                   # Lower to save memory (increase to 1280 if you have GPU)
BATCH_SIZE = 4                   # Lower = less memory needed
DEVICE = "cpu"                   # "cpu" = works everywhere, "mps" for Mac M1/M2, "cuda" for NVIDIA
PROJECT = "runs"
NAME = "crowd_detector"

# ========== CHECK DATASET ==========
def check_dataset():
    """Verify dataset exists and is properly structured"""
    if not os.path.exists(DATASET_YAML):
        print("=" * 50)
        print("ERROR: Dataset not found!")
        print("=" * 50)
        print()
        print("Please prepare your dataset first:")
        print()
        print("Option 1: Use Roboflow (EASIEST)")
        print("  1. Go to https://roboflow.com")
        print("  2. Create free account")
        print("  3. Upload your CCTV/crowd images")
        print("  4. Draw boxes around each person")
        print("  5. Export as 'YOLOv8' format")
        print("  6. Download and extract to training/dataset/")
        print()
        print("Option 2: Download CrowdHuman dataset")
        print("  Run: python download_dataset.py")
        print()
        print("Option 3: Use LabelImg")
        print("  pip install labelimg")
        print("  labelimg  # opens GUI, draw boxes, save as YOLO")
        print()
        print(f"Expected path: {os.path.abspath(DATASET_YAML)}")

        # Create sample data.yaml as template
        os.makedirs("dataset/train/images", exist_ok=True)
        os.makedirs("dataset/train/labels", exist_ok=True)
        os.makedirs("dataset/val/images", exist_ok=True)
        os.makedirs("dataset/val/labels", exist_ok=True)

        sample_yaml = {
            "train": "./train/images",
            "val": "./val/images",
            "nc": 1,
            "names": ["person"]
        }
        with open(DATASET_YAML, "w") as f:
            yaml.dump(sample_yaml, f)

        print(f"\nCreated template: {DATASET_YAML}")
        print("Add your images to dataset/train/images/ and labels to dataset/train/labels/")
        return False

    # Check if images exist
    with open(DATASET_YAML) as f:
        data = yaml.safe_load(f)

    train_path = os.path.join("dataset", data.get("train", "train/images"))
    if os.path.exists(train_path):
        image_count = len([f for f in os.listdir(train_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"[Dataset] Found {image_count} training images")
        if image_count < 10:
            print("WARNING: Very few images. Aim for 500+ for good results.")
        return image_count > 0
    else:
        print(f"ERROR: Training images directory not found: {train_path}")
        return False


# ========== TRAIN ==========
def train():
    """Fine-tune YOLOv8 on your crowd dataset"""

    print("=" * 50)
    print("  YOLOv8 Crowd Detection Training")
    print("=" * 50)

    if not check_dataset():
        return

    # Load pre-trained model
    print(f"\n[1/3] Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    # Start training
    print(f"[2/3] Training for {EPOCHS} epochs...")
    print(f"       Image size: {IMG_SIZE}")
    print(f"       Batch size: {BATCH_SIZE}")
    print(f"       Device: {DEVICE}")
    print()

    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        exist_ok=True,

        # Training settings
        patience=20,           # Stop early if no improvement for 20 epochs
        save=True,             # Save checkpoints
        save_period=10,        # Save every 10 epochs
        workers=4,             # Data loading workers
        pretrained=True,       # Use pre-trained weights (transfer learning)

        # Augmentation (makes model more robust)
        augment=True,
        hsv_h=0.015,          # Hue variation
        hsv_s=0.7,            # Saturation variation
        hsv_v=0.4,            # Brightness variation
        degrees=10,           # Rotation
        translate=0.1,        # Translation
        scale=0.5,            # Scale variation
        fliplr=0.5,           # Horizontal flip
        mosaic=1.0,           # Mosaic augmentation (great for crowds)
        mixup=0.1,            # Mixup augmentation
    )

    # Results
    print("\n[3/3] Training complete!")
    print(f"       Best model saved at: {PROJECT}/{NAME}/weights/best.pt")
    print(f"       Results at: {PROJECT}/{NAME}/")
    print()
    print("To use your trained model:")
    print(f'  1. Copy {PROJECT}/{NAME}/weights/best.pt to backend/')
    print(f'  2. In config.py, change: YOLO_MODEL = "best.pt"')
    print(f'  3. Restart the server')

    return results


# ========== EVALUATE ==========
def evaluate():
    """Test the trained model on validation set"""
    best_model = f"{PROJECT}/{NAME}/weights/best.pt"
    if not os.path.exists(best_model):
        print(f"No trained model found at {best_model}")
        print("Run training first: python train_crowd_yolo.py")
        return

    model = YOLO(best_model)
    results = model.val(data=DATASET_YAML, imgsz=IMG_SIZE)

    print("\n========== EVALUATION RESULTS ==========")
    print(f"mAP@50:     {results.box.map50:.4f}")
    print(f"mAP@50-95:  {results.box.map:.4f}")
    print(f"Precision:  {results.box.mp:.4f}")
    print(f"Recall:     {results.box.mr:.4f}")


# ========== TEST ON SINGLE IMAGE ==========
def test_image(image_path):
    """Test trained model on a single image"""
    best_model = f"{PROJECT}/{NAME}/weights/best.pt"
    if not os.path.exists(best_model):
        print("No trained model found. Run training first.")
        return

    model = YOLO(best_model)
    results = model(image_path, conf=0.35, imgsz=IMG_SIZE)

    for result in results:
        count = len(result.boxes)
        print(f"Detected {count} people in {image_path}")
        result.show()  # Display image with boxes


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "eval":
            evaluate()
        elif sys.argv[1] == "test" and len(sys.argv) > 2:
            test_image(sys.argv[2])
        else:
            print("Usage:")
            print("  python train_crowd_yolo.py          # Train")
            print("  python train_crowd_yolo.py eval      # Evaluate")
            print("  python train_crowd_yolo.py test img  # Test on image")
    else:
        train()
