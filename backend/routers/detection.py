"""
Detection Router
API endpoints for all detection modules:
- POST /api/detect/crowd        → Crowd density detection (image)
- POST /api/detect/pickpocket   → Pickpocket detection (image)
- POST /api/detect/emergency    → Emergency detection (image)
- POST /api/detect/heatmap      → Crowd density heatmap (image)
- POST /api/detect/video        → Process full video with all detections
- GET  /api/detect/video/status → Check video processing progress
- WS   /api/detect/live/{id}    → Live camera WebSocket feed
"""

from fastapi import APIRouter, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from modules.crowd_density import detect_crowd, get_density_heatmap, get_video_heatmap, reset_heatmap
from modules.pickpocket import detect_pickpocket
from modules.emergency import detect_emergency
from modules.face_finder import search_in_frame as face_search, embedding_cache
from modules.alert_manager import create_alert
import cv2
import numpy as np
import base64
import os
import json
import time
import tempfile
import threading
from config import UPLOAD_DIR

router = APIRouter(prefix="/api/detect", tags=["Detection"])

# Video processing state (shared between threads)
video_jobs = {}


def decode_upload(contents: bytes):
    """Convert uploaded file bytes to OpenCV frame"""
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame


def encode_frame(frame):
    """Convert OpenCV frame to base64 string for frontend"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


# ========== CROWD DENSITY ==========
@router.post("/crowd")
async def detect_crowd_endpoint(file: UploadFile = File(...)):
    """Detect crowd density from uploaded image."""
    contents = await file.read()
    frame = decode_upload(contents)

    if frame is None:
        return {"error": "Invalid image"}

    result = detect_crowd(frame)

    if result["is_overcrowded"]:
        create_alert(
            alert_type="crowd_density",
            severity="high",
            location="uploaded_frame",
            details={"count": result["count"], "threshold": result["threshold"]}
        )

    return {
        "count": result["count"],
        "is_overcrowded": result["is_overcrowded"],
        "threshold": result["threshold"],
        "annotated_image": encode_frame(result["frame"])
    }


# ========== HEATMAP ==========
@router.post("/heatmap")
async def heatmap_endpoint(file: UploadFile = File(...)):
    """Generate crowd density heatmap from uploaded image."""
    contents = await file.read()
    frame = decode_upload(contents)

    if frame is None:
        return {"error": "Invalid image"}

    heatmap = get_density_heatmap(frame)
    return {"heatmap": encode_frame(heatmap)}


# ========== PICKPOCKET ==========
@router.post("/pickpocket")
async def detect_pickpocket_endpoint(file: UploadFile = File(...)):
    """Detect suspicious pickpocket behavior."""
    contents = await file.read()
    frame = decode_upload(contents)

    if frame is None:
        return {"error": "Invalid image"}

    result = detect_pickpocket(frame)

    if result["suspicious"]:
        create_alert(
            alert_type="pickpocket",
            severity="medium",
            location="uploaded_frame",
            details={"alert_count": len(result["alerts"])}
        )

    return {
        "suspicious": result["suspicious"],
        "alerts": result["alerts"],
        "annotated_image": encode_frame(result["frame"])
    }


# ========== EMERGENCY ==========
@router.post("/emergency")
async def detect_emergency_endpoint(file: UploadFile = File(...)):
    """Detect emergency situations (fall, fight)."""
    contents = await file.read()
    frame = decode_upload(contents)

    if frame is None:
        return {"error": "Invalid image"}

    result = detect_emergency(frame)

    if result["fire_detected"]:
        create_alert(
            alert_type="emergency", severity="critical",
            location="uploaded_frame",
            details={"type": "fire"}
        )

    if result.get("smoke_detected", False):
        create_alert(
            alert_type="emergency", severity="high",
            location="uploaded_frame",
            details={"type": "smoke"}
        )

    if result["fall_detected"]:
        create_alert(
            alert_type="emergency", severity="critical",
            location="uploaded_frame",
            details={"type": "fall"}
        )

    if result["fight_detected"]:
        create_alert(
            alert_type="emergency", severity="critical",
            location="uploaded_frame",
            details={"type": "fight"}
        )

    return {
        "fire_detected": result["fire_detected"],
        "smoke_detected": result.get("smoke_detected", False),
        "fall_detected": result["fall_detected"],
        "fight_detected": result["fight_detected"],
        "annotated_image": encode_frame(result["frame"])
    }


# ========== VIDEO PROCESSING ==========

def process_video_worker(job_id: str, video_path: str, output_path: str, process_every_n: int):
    """
    Background worker that processes a video file frame by frame.
    Runs all detections on each frame and writes annotated output video.

    For each frame:
    1. Crowd detection (count people)
    2. Pickpocket detection (suspicious hands)
    3. Emergency detection (fall/fight)
    4. Draw all results on frame
    5. Write to output video
    """
    job = video_jobs[job_id]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        job["status"] = "error"
        job["error"] = "Cannot open video file"
        return

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    job["total_frames"] = total_frames
    job["fps"] = fps
    job["duration"] = round(duration, 1)
    job["resolution"] = f"{width}x{height}"
    job["status"] = "processing"

    # Reset heatmap for new video
    reset_heatmap()

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Stats tracking
    frame_num = 0
    total_people = 0
    max_people = 0
    alerts_generated = 0
    overcrowded_frames = 0
    fire_alert_sent = False  # ensures only one fire alert per video
    smoke_alert_sent = False  # ensures only one smoke alert per video
    emergency_frames = 0
    frame_results = []
    # Track per-person face matches (one alert per person, not per frame)
    person_match_tracker = {}  # name → {"first_frame", "last_frame", "first_time", "last_time", "best_conf", "count"}

    # Anti-flicker: cache last annotation state AND box coordinates for skipped frames
    last_fire_active = False
    last_smoke_active = False
    last_overcrowded = False
    last_people_count = 0
    last_fire_boxes = []  # list of (x1, y1, x2, y2, conf) for fire boxes
    last_smoke_boxes = []  # list of (x1, y1, x2, y2, conf) for smoke boxes

    # Direct access to fire model for getting box coordinates
    from modules.emergency import FIRE_MODEL

    def get_fire_smoke_boxes(img):
        """Get fire/smoke box coordinates without drawing."""
        fire_boxes = []
        smoke_boxes = []
        if FIRE_MODEL is None:
            return fire_boxes, smoke_boxes
        try:
            results = FIRE_MODEL(img, conf=0.5, verbose=False)
            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    cf = float(b.conf[0])
                    label = r.names[int(b.cls[0])].lower()
                    if "fire" in label or "flame" in label:
                        fire_boxes.append((x1, y1, x2, y2, cf))
                    elif "smoke" in label:
                        smoke_boxes.append((x1, y1, x2, y2, cf))
        except Exception:
            pass
        return fire_boxes, smoke_boxes

    def draw_cached_boxes(img, fire_boxes, smoke_boxes):
        """Draw cached boxes + banner on a frame."""
        for (x1, y1, x2, y2, cf) in fire_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img, f"FIRE {cf:.0%}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for (x1, y1, x2, y2, cf) in smoke_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (50, 50, 50), 3)
            cv2.putText(img, f"SMOKE {cf:.0%}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        if fire_boxes or smoke_boxes:
            oh, ow = img.shape[:2]
            overlay = img.copy()
            color = (0, 0, 200) if fire_boxes else (50, 50, 50)
            cv2.rectangle(overlay, (0, 0), (ow, 45), color, -1)
            img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
            text = "FIRE DETECTED!" if fire_boxes else "SMOKE DETECTED!"
            cv2.putText(img, text, (ow // 2 - 130, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return img

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            job["current_frame"] = frame_num
            job["progress"] = round((frame_num / total_frames) * 100, 1)

            # Process every Nth frame (skip frames for speed)
            if frame_num % process_every_n != 0:
                # For skipped frames: redraw cached boxes to prevent flickering
                if last_fire_boxes or last_smoke_boxes:
                    frame = draw_cached_boxes(frame, last_fire_boxes, last_smoke_boxes)
                writer.write(frame)
                continue

            # ===== Run ALL detections =====

            # 1. Crowd Detection
            crowd_frame = frame.copy()
            crowd_result = detect_crowd(crowd_frame)
            people_count = crowd_result["count"]
            total_people += people_count
            max_people = max(max_people, people_count)

            if crowd_result["is_overcrowded"]:
                overcrowded_frames += 1
                if overcrowded_frames == 1:  # only FIRST time
                    create_alert(
                        alert_type="crowd_density", severity="high",
                        location="video",
                        details={"count": people_count, "first_seen": f"frame {frame_num}"}
                    )
                    alerts_generated += 1

            # 2. Emergency Detection (fire + smoke + fall + fight)
            # Run on clean frame for accuracy
            emg_result = detect_emergency(frame.copy())
            smoke_detected = emg_result.get("smoke_detected", False)

            # IMMEDIATE separate alert for fire when first detected
            if emg_result["fire_detected"] and not fire_alert_sent:
                fire_alert_sent = True
                create_alert(
                    alert_type="emergency", severity="critical",
                    location="video",
                    details={"type": "fire", "first_seen": f"frame {frame_num}"}
                )
                alerts_generated += 1

            # IMMEDIATE separate alert for smoke when first detected
            if smoke_detected and not smoke_alert_sent:
                smoke_alert_sent = True
                create_alert(
                    alert_type="emergency", severity="high",
                    location="video",
                    details={"type": "smoke", "first_seen": f"frame {frame_num}"}
                )
                alerts_generated += 1

            has_emergency = (emg_result["fire_detected"] or smoke_detected
                            or emg_result["fall_detected"] or emg_result["fight_detected"])
            if has_emergency:
                emergency_frames += 1
                if emergency_frames == 1:  # only FIRST time
                    emergency_types = []
                    if emg_result["fire_detected"]:
                        emergency_types.append("fire")
                    if smoke_detected:
                        emergency_types.append("smoke")
                    if emg_result["fall_detected"]:
                        emergency_types.append("fall")
                    if emg_result["fight_detected"]:
                        emergency_types.append("fight")
                    create_alert(
                        alert_type="emergency", severity="critical",
                        location="video",
                        details={
                            "first_seen": f"frame {frame_num}",
                            "types": emergency_types
                        }
                    )
                    alerts_generated += 1

            # ===== Draw combined results on frame =====
            annotated = crowd_result["frame"]

            # Draw fire/smoke boxes on the final annotated frame (so they appear in output)
            if emg_result["fire_detected"] or emg_result.get("smoke_detected", False):
                detect_emergency(annotated)

            # Update cache for next skipped frames (prevents flickering)
            last_fire_active = emg_result["fire_detected"]
            last_smoke_active = emg_result.get("smoke_detected", False)
            last_overcrowded = crowd_result["is_overcrowded"]
            last_people_count = people_count

            # Save actual box coordinates for skipped-frame redraw
            if last_fire_active or last_smoke_active:
                last_fire_boxes, last_smoke_boxes = get_fire_smoke_boxes(frame)
            else:
                last_fire_boxes, last_smoke_boxes = [], []

            # 4. Face Search (auto-detect missing persons)
            # MUST be AFTER annotated is set, so boxes draw on the FINAL frame
            face_matches = []
            if embedding_cache:
                try:
                    face_matches = face_search(frame, draw_on=annotated)
                    if face_matches:
                        for match in face_matches:
                            name = match["person_name"]
                            conf = match["confidence"]
                            t = round(frame_num / fps, 1)

                            if name not in person_match_tracker:
                                # FIRST TIME this person is seen → send alert IMMEDIATELY
                                person_match_tracker[name] = {
                                    "first_frame": frame_num,
                                    "last_frame": frame_num,
                                    "first_time": t,
                                    "last_time": t,
                                    "best_conf": conf,
                                    "count": 1,
                                    "alert_id": None
                                }
                                # Create alert right now
                                alert = create_alert(
                                    alert_type="missing_person", severity="critical",
                                    location="video",
                                    details={
                                        "person_name": name,
                                        "confidence": conf,
                                        "first_seen": f"frame {frame_num} ({t}s)",
                                        "status": "tracking"
                                    }
                                )
                                alerts_generated += 1
                                print(f"[Video] ALERT: {name} first seen at frame {frame_num} ({t}s)")
                            else:
                                # Already seen → just update tracker (no new alert)
                                tracker = person_match_tracker[name]
                                tracker["last_frame"] = frame_num
                                tracker["last_time"] = t
                                tracker["best_conf"] = max(tracker["best_conf"], conf)
                                tracker["count"] += 1
                except Exception as e:
                    print(f"[Video] Face search error at frame {frame_num}: {e}")

            # Draw emergency warnings
            emg_y = 30
            if emg_result["fire_detected"]:
                cv2.putText(annotated, "FIRE DETECTED!", (width - 300, emg_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                emg_y += 30
            if emg_result["fall_detected"]:
                cv2.putText(annotated, "FALL DETECTED!", (width - 300, emg_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                emg_y += 30
            if emg_result["fight_detected"]:
                cv2.putText(annotated, "FIGHT DETECTED!", (width - 300, emg_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Face match results already shown in pink box by face_finder

            # Frame counter
            cv2.putText(annotated, f"Frame: {frame_num}/{total_frames}", (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            writer.write(annotated)

            # Save frame result for timeline
            matched_names = [m["person_name"] for m in face_matches]
            frame_results.append({
                "frame": frame_num,
                "time": round(frame_num / fps, 1),
                "people": people_count,
                "overcrowded": crowd_result["is_overcrowded"],
                "fire": emg_result["fire_detected"],
                "fall": emg_result["fall_detected"],
                "fight": emg_result["fight_detected"],
                "face_match": matched_names
            })

            # Generate TEMPORAL heatmap (accumulates over time with decay)
            heatmap_frame = get_video_heatmap(frame.copy(), crowd_result["boxes"])
            job["current_heatmap"] = encode_frame(heatmap_frame)

            # Store latest annotated frame + ORIGINAL clean frame + people count
            job["latest_frame"] = encode_frame(annotated)
            job["original_frame"] = encode_frame(frame)
            job["current_people"] = people_count
            job["face_matches"] = [m["person_name"] for m in face_matches]

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"[Video] Error at frame {frame_num}: {e}")
    finally:
        cap.release()
        writer.release()

    # Calculate final stats
    processed_frames = len(frame_results)
    avg_people = round(total_people / max(processed_frames, 1), 1)

    # Count face match frames
    face_match_frames = len([f for f in frame_results if f.get("face_match")])
    all_matched_persons = set()
    for f in frame_results:
        for name in f.get("face_match", []):
            all_matched_persons.add(name)

    # Update existing alerts with final frame range (no new alerts)
    for name, tracker in person_match_tracker.items():
        print(f"[Video] {name}: seen in frames {tracker['first_frame']}-{tracker['last_frame']} ({tracker['count']} frames)")

    job["status"] = "completed"
    job["progress"] = 100
    job["stats"] = {
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "duration_seconds": round(duration, 1),
        "avg_people_per_frame": avg_people,
        "max_people_in_frame": max_people,
        "overcrowded_frames": overcrowded_frames,
        "emergency_frames": emergency_frames,
        "face_match_frames": face_match_frames,
        "persons_found": list(all_matched_persons),
        "alerts_generated": alerts_generated
    }
    job["timeline"] = frame_results
    job["output_video"] = output_path

    print(f"[Video] Completed: {processed_frames} frames processed, {alerts_generated} alerts")


@router.post("/video")
async def process_video_endpoint(
    file: UploadFile = File(...),
    skip_frames: int = 3
):
    """
    Upload and process a video file with all detections.

    Args:
        file: Video file (mp4, avi, mov, mkv)
        skip_frames: Process every Nth frame (default 3 = skip 2, process 1)
                     Higher = faster but less accurate
                     1 = process every frame (slow but thorough)

    Returns:
        job_id to track progress via /api/detect/video/status/{job_id}
    """
    # Validate file type
    allowed = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return {"error": f"Unsupported format: {ext}. Use: {', '.join(allowed)}"}

    # Save uploaded video
    os.makedirs(os.path.join(UPLOAD_DIR, "videos"), exist_ok=True)
    video_path = os.path.join(UPLOAD_DIR, "videos", f"input_{int(time.time())}{ext}")
    output_path = os.path.join(UPLOAD_DIR, "videos", f"output_{int(time.time())}.mp4")

    contents = await file.read()
    with open(video_path, "wb") as f:
        f.write(contents)

    # Get video info before processing
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    cap.release()

    # Create job
    job_id = f"video_{int(time.time())}"
    video_jobs[job_id] = {
        "id": job_id,
        "status": "starting",
        "progress": 0,
        "current_frame": 0,
        "total_frames": total_frames,
        "fps": fps,
        "duration": round(duration, 1),
        "resolution": f"{width}x{height}",
        "filename": file.filename,
        "skip_frames": skip_frames,
        "latest_frame": None,
        "stats": None,
        "timeline": None,
        "output_video": None,
        "error": None
    }

    # Start processing in background thread
    thread = threading.Thread(
        target=process_video_worker,
        args=(job_id, video_path, output_path, skip_frames),
        daemon=True
    )
    thread.start()

    return {
        "job_id": job_id,
        "message": f"Video processing started ({total_frames} frames, {round(duration, 1)}s)",
        "total_frames": total_frames,
        "estimated_process_frames": total_frames // skip_frames,
        "duration": round(duration, 1),
        "resolution": f"{width}x{height}",
        "check_status": f"/api/detect/video/status/{job_id}"
    }


@router.get("/video/status/{job_id}")
async def video_status(job_id: str):
    """
    Check video processing progress.
    Poll this endpoint to get real-time progress updates.
    """
    if job_id not in video_jobs:
        return {"error": "Job not found"}

    job = video_jobs[job_id]

    response = {
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "current_frame": job["current_frame"],
        "total_frames": job["total_frames"],
        "duration": job["duration"],
        "resolution": job["resolution"],
        "filename": job["filename"],
        "current_people": job.get("current_people", 0),
        "face_matches": job.get("face_matches", []),
    }

    # Include latest processed frame for live preview
    if job["latest_frame"]:
        response["latest_frame"] = job["latest_frame"]

    # Include original clean frame for face search
    if job.get("original_frame"):
        response["original_frame"] = job["original_frame"]

    # Include live heatmap
    if job.get("current_heatmap"):
        response["current_heatmap"] = job["current_heatmap"]

    # Include stats when completed
    if job["status"] == "completed":
        response["stats"] = job["stats"]
        response["timeline"] = job["timeline"]

    if job["error"]:
        response["error"] = job["error"]

    return response


@router.get("/video/download/{job_id}")
async def download_processed_video(job_id: str):
    """Download the processed video with annotations."""
    if job_id not in video_jobs:
        return {"error": "Job not found"}

    job = video_jobs[job_id]

    if job["status"] != "completed":
        return {"error": "Video not ready yet", "status": job["status"]}

    output_path = job["output_video"]
    if not output_path or not os.path.exists(output_path):
        return {"error": "Output video not found"}

    def video_stream():
        with open(output_path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk

    return StreamingResponse(
        video_stream(),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename=processed_{job['filename']}"}
    )


@router.get("/video/jobs")
async def list_video_jobs():
    """List all video processing jobs."""
    jobs = []
    for job_id, job in video_jobs.items():
        jobs.append({
            "id": job["id"],
            "status": job["status"],
            "progress": job["progress"],
            "filename": job["filename"],
            "duration": job["duration"]
        })
    return {"jobs": jobs}


# ========== LIVE WEBSOCKET FEED ==========
@router.websocket("/live/{camera_id}")
async def live_detection(websocket: WebSocket, camera_id: str):
    """WebSocket endpoint for live camera feed processing."""
    await websocket.accept()
    print(f"[WebSocket] Client connected to camera: {camera_id}")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        await websocket.send_json({"error": "Cannot open camera"})
        await websocket.close()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            crowd_result = detect_crowd(frame.copy())
            emergency_result = detect_emergency(frame.copy())

            await websocket.send_json({
                "crowd_count": crowd_result["count"],
                "is_overcrowded": crowd_result["is_overcrowded"],
                "fall_detected": emergency_result["fall_detected"],
                "fight_detected": emergency_result["fight_detected"],
                "frame": encode_frame(crowd_result["frame"])
            })

    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected from camera: {camera_id}")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
    finally:
        cap.release()
