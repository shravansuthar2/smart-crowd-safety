# Smart Crowd Safety System

AI-powered crowd safety monitoring system for railway stations, stadiums, and festivals.

## Features

- **Crowd Density Detection** — YOLOv8 counts people in video feeds, alerts when overcrowded
- **Missing Person Finder** — InsightFace detects and matches missing persons in video with magenta highlight box
- **Fire Detection** — YOLOv8 trained on Kaggle fire dataset (97.8% mAP)
- **Real-time Video Processing** — Upload video, all detections run automatically frame by frame
- **Alert System** — Real-time alerts with acknowledge/resolve actions
- **Crowd Heatmap** — Visual density map showing where crowds concentrate
- **Dashboard** — Dark-themed responsive web dashboard

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Frontend | HTML, CSS, JavaScript |
| Person Detection | YOLOv8s |
| Face Recognition | InsightFace (ONNX) |
| Fire Detection | YOLOv8 (custom trained) |
| Database | Firebase Firestore (optional, works locally) |
| Video Processing | OpenCV |

## Quick Start

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/smart-crowd-safety.git
cd smart-crowd-safety
```

### 2. Setup Python Environment
```bash
cd backend
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install insightface onnxruntime
```

### 4. Download AI Models
Models download automatically on first run:
- **YOLOv8s** (~21MB) — person detection
- **InsightFace buffalo_sc** (~15MB) — face recognition

### 5. Train Fire Detection (Optional)
```bash
cd training
# Place fire dataset in fire_kaggle/ folder
python train_fire.py download   # Create sample dataset
python train_fire.py train      # Train model
cp runs/fire_kaggle/weights/best.pt ../backend/fire_model.pt
```

### 6. Start Server
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

### 7. Open Dashboard
Open `dashboard/index.html` in your browser.

Or visit: http://localhost:8000/dashboard

## How to Use

### Crowd Detection
1. Upload a video → crowd count shown automatically
2. Alerts trigger if count exceeds threshold (default: 50)

### Missing Person Finder
1. Register missing person (name + photo)
2. Upload video → system auto-searches every frame
3. If found → magenta box highlights the person + alert fires

### Fire Detection
1. Train fire model using Kaggle dataset
2. Upload video → fire detection runs automatically
3. If fire detected → critical alert

## Project Structure
```
smart-crowd-safety/
├── backend/
│   ├── main.py                 # FastAPI server
│   ├── config.py               # Settings & thresholds
│   ├── firebase_config.py      # Database (Firebase/local)
│   ├── modules/
│   │   ├── crowd_density.py    # YOLOv8 person counting
│   │   ├── face_finder.py      # InsightFace recognition
│   │   ├── emergency.py        # Fire detection
│   │   └── alert_manager.py    # Alert system
│   ├── routers/
│   │   ├── detection.py        # Detection + video APIs
│   │   ├── persons.py          # Missing person APIs
│   │   └── alerts.py           # Alert APIs
│   └── requirements.txt
├── dashboard/
│   ├── index.html              # Main UI
│   ├── style.css               # Dark theme
│   ├── app.js                  # Frontend logic
│   └── firebase-config.js      # Firebase (optional)
├── training/
│   ├── train_crowd_yolo.py     # Train crowd detection
│   ├── train_fire.py           # Train fire detection
│   └── download_dataset.py     # Dataset helpers
├── Dockerfile
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/detect/crowd` | Detect crowd in image |
| POST | `/api/detect/emergency` | Detect fire in image |
| POST | `/api/detect/heatmap` | Generate crowd heatmap |
| POST | `/api/detect/video` | Process video (all detections) |
| GET | `/api/detect/video/status/{id}` | Video processing progress |
| GET | `/api/detect/video/download/{id}` | Download processed video |
| POST | `/api/persons/register` | Register missing person |
| GET | `/api/persons/list` | List registered persons |
| DELETE | `/api/persons/delete/{name}` | Delete person |
| GET | `/api/alerts/` | List all alerts |
| DELETE | `/api/alerts/clear` | Clear all alerts |

## Configuration

Edit `backend/config.py`:
```python
CROWD_DENSITY_THRESHOLD = 50   # Max people before alert
CONFIDENCE_THRESHOLD = 0.2     # YOLO detection confidence
YOLO_IMG_SIZE = 1280           # Detection resolution
SIMILARITY_THRESHOLD = 0.35    # Face match threshold (in face_finder.py)
```

## Deployment (Free)

### Google Cloud Run
```bash
gcloud run deploy crowd-safety-api \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 2Gi
```

### Firebase Hosting (Dashboard)
```bash
firebase deploy --only hosting
```

## License

MIT
