"""
Smart Crowd Safety System - Backend Server
===========================================
FastAPI application that connects all AI modules:
- Crowd Density Detection (YOLOv8)
- Lost Person Finder (DeepFace)
- Pickpocket Detection (MediaPipe Pose)
- Emergency Detection (Fall/Fight)
- Alert Management (Firebase/Local)

Run:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routers import detection, alerts, persons
import os

app = FastAPI(
    title="Smart Crowd Safety System",
    description="AI-powered crowd safety monitoring using YOLOv8, DeepFace, and MediaPipe",
    version="1.0.0"
)

# ========== CORS ==========
# Allow frontend (dashboard) to call backend APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== ROUTERS ==========
app.include_router(detection.router)    # /api/detect/*
app.include_router(alerts.router)       # /api/alerts/*
app.include_router(persons.router)      # /api/persons/*

# ========== STATIC FILES ==========
# Serve uploaded images
uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

# Serve dashboard (optional - can also use Firebase Hosting)
dashboard_dir = os.path.join(os.path.dirname(__file__), "..", "dashboard")
if os.path.exists(dashboard_dir):
    app.mount("/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")


# ========== ROOT ENDPOINTS ==========
@app.get("/")
def root():
    return {
        "message": "Smart Crowd Safety System API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "crowd_detection": "POST /api/detect/crowd",
            "pickpocket": "POST /api/detect/pickpocket",
            "emergency": "POST /api/detect/emergency",
            "heatmap": "POST /api/detect/heatmap",
            "live_feed": "WS /api/detect/live/{camera_id}",
            "register_person": "POST /api/persons/register",
            "search_person": "POST /api/persons/search",
            "list_persons": "GET /api/persons/list",
            "alerts": "GET /api/alerts/",
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# ========== STARTUP EVENT ==========
@app.on_event("startup")
async def startup():
    print("=" * 50)
    print("  Smart Crowd Safety System")
    print("  API running at http://localhost:8000")
    print("  Docs at http://localhost:8000/docs")
    print("  Dashboard at http://localhost:8000/dashboard")
    print("=" * 50)


# ========== RUN DIRECTLY ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
