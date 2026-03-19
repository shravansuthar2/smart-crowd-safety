"""
Missing Persons Router
API endpoints for registering and searching missing persons:
- POST /api/persons/register  → Register a missing person with photo
- POST /api/persons/search    → Search for missing persons in an image
- GET  /api/persons/list      → List all registered missing persons
"""

from fastapi import APIRouter, UploadFile, File, Form
from modules.face_finder import register_missing_person, search_in_frame, embedding_cache
from modules.alert_manager import create_alert
from firebase_config import save_missing_person, get_missing_persons, upload_image, delete_missing_person
import cv2
import numpy as np
import base64
import os
from config import MISSING_PERSONS_DIR

router = APIRouter(prefix="/api/persons", tags=["Missing Persons"])


@router.post("/register")
async def register_person(
    name: str = Form(...),
    details: str = Form(""),
    photo: UploadFile = File(...)
):
    """
    Register a missing person.

    Flow:
    1. Receive name + details + photo from frontend
    2. Save photo to uploads/missing_persons/
    3. DeepFace verifies a face exists in the photo
    4. Save person info to Firebase/local storage
    5. Return success/failure
    """
    # Save uploaded photo temporarily
    os.makedirs(MISSING_PERSONS_DIR, exist_ok=True)
    file_path = os.path.join(MISSING_PERSONS_DIR, f"{name}.jpg")

    contents = await photo.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # Register in face finder module
    result = register_missing_person(file_path, name, details)

    if result["success"]:
        # Upload to Firebase Storage (or get local path)
        image_url = upload_image(file_path, f"missing_persons/{name}.jpg")

        # Save to Firestore/local
        save_missing_person({
            "name": name,
            "details": details,
            "image_url": image_url,
            "status": "missing",
        })

    return result


@router.post("/search")
async def search_person(file: UploadFile = File(...)):
    """
    Search for missing persons in uploaded image.

    Flow:
    1. Receive camera frame/image
    2. Compare with all registered missing person photos
    3. If match found → create critical alert
    4. Return list of matches with confidence
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image", "matches": [], "count": 0}

    matches = search_in_frame(frame)

    # Create alerts for each match
    if matches:
        for match in matches:
            create_alert(
                alert_type="missing_person",
                severity="critical",
                location="uploaded_frame",
                details={
                    "person_name": match["person_name"],
                    "confidence": match["confidence"]
                }
            )

    # Return annotated frame with face boxes drawn
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated_image = base64.b64encode(buffer).decode('utf-8')

    return {
        "matches": matches,
        "count": len(matches),
        "annotated_image": annotated_image
    }


@router.get("/list")
async def list_missing_persons():
    """
    List all registered missing persons.
    Returns name, details, image URL, and status for each person.
    """
    persons = get_missing_persons()
    return {"persons": persons}


@router.delete("/delete/{name}")
async def delete_person(name: str):
    """Delete a registered missing person and their files."""
    # Delete image files
    deleted_files = []
    for ext in [".jpg", ".jpeg", ".png", "_face.jpg"]:
        path = os.path.join(MISSING_PERSONS_DIR, f"{name}{ext}")
        if os.path.exists(path):
            os.remove(path)
            deleted_files.append(path)

    # Remove from embedding cache
    if name in embedding_cache:
        del embedding_cache[name]

    # Remove from Firebase/local storage
    delete_missing_person(name)

    if deleted_files:
        return {"message": f"Deleted {name}", "files_removed": len(deleted_files)}
    else:
        return {"message": f"{name} not found", "files_removed": 0}
