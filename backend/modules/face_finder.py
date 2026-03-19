"""
Lost Person Finder — v6 INSIGHTFACE
=====================================
Uses InsightFace (ONNX) — NO TensorFlow/Keras dependency.
Detects faces + generates embeddings in one pass.

Pipeline:
  InsightFace.get(image) → returns faces with bbox + embedding
  Compare embeddings → cosine similarity → match

Why InsightFace:
  - ONNX runtime (no Keras/TF crashes)
  - Detects face + embedding in single call
  - 4 faces detected vs 2 with OpenCV Haar
  - 90% similarity on correct match vs 4% on wrong person
"""

import insightface
import cv2
import os
import numpy as np
from config import MISSING_PERSONS_DIR

print("[Face Finder] v6 — InsightFace (ONNX)")

# ========== MODEL ==========

model = insightface.app.FaceAnalysis(
    name="buffalo_sc",
    providers=["CPUExecutionProvider"]
)
model.prepare(ctx_id=-1, det_size=(640, 640))
print("[Face Finder] InsightFace model loaded")

# Similarity threshold (0.0 to 1.0, higher = stricter)
SIMILARITY_THRESHOLD = 0.35

# ========== EMBEDDING CACHE ==========

embedding_cache = {}  # person_name → 512-dim embedding


def cosine_similarity(emb1, emb2):
    """1.0 = identical, 0.0 = different, -1.0 = opposite"""
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10))


# ========== REGISTER ==========

def register_missing_person(image_path: str, person_name: str, details: str):
    """
    Register a missing person:
    1. Detect face in photo
    2. Extract embedding
    3. Save face crop + cache embedding
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"[Register] ERROR: Cannot read {image_path}")
            return {"success": False, "message": "Cannot read image"}

        print(f"[Register] Image: {img.shape[1]}x{img.shape[0]}")

        # Detect faces
        faces = model.get(img)
        print(f"[Register] Faces detected: {len(faces)}")

        if not faces:
            print(f"[Register] ERROR: No face found")
            return {"success": False, "message": "No face detected. Use a clear front-facing photo."}

        # Use largest face (highest area)
        best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = best_face.bbox.astype(int)
        score = best_face.det_score
        embedding = best_face.embedding

        print(f"[Register] Best face: [{x1},{y1},{x2},{y2}], score={score:.2f}, emb_dim={len(embedding)}")

        # Save face crop
        pad = 20
        cy1 = max(0, y1 - pad)
        cx1 = max(0, x1 - pad)
        cy2 = min(img.shape[0], y2 + pad)
        cx2 = min(img.shape[1], x2 + pad)
        face_crop = img[cy1:cy2, cx1:cx2]

        face_path = os.path.join(MISSING_PERSONS_DIR, f"{person_name}_face.jpg")
        cv2.imwrite(face_path, face_crop)

        # Save full image
        full_path = os.path.join(MISSING_PERSONS_DIR, f"{person_name}.jpg")
        cv2.imwrite(full_path, img)

        # Cache embedding
        embedding_cache[person_name] = embedding
        print(f"[Register] SUCCESS: {person_name} registered (face: {x2-x1}x{y2-y1}px)")

        return {
            "success": True,
            "message": f"Registered {person_name} (face: {x2-x1}x{y2-y1}px, score: {score:.0%})",
            "face_count": len(faces),
            "face_size": f"{x2-x1}x{y2-y1}"
        }

    except Exception as e:
        print(f"[Register] ERROR: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}


# ========== SEARCH IN FRAME ==========

def search_in_frame(frame, draw_on=None):
    """
    Search for registered missing persons in a frame.

    Args:
        frame: image to detect faces in (should be clean for best accuracy)
        draw_on: if provided, draw boxes on THIS frame instead of 'frame'
                 (used in video pipeline: detect on clean frame, draw on annotated)
    """
    matches = []
    canvas = draw_on if draw_on is not None else frame
    oh, ow = canvas.shape[:2]

    # Load embeddings if cache is empty
    if not embedding_cache:
        _load_all_embeddings()

    if not embedding_cache:
        print("[Search] No registered persons")
        return matches

    print(f"[Search] Registered: {list(embedding_cache.keys())}")

    # Detect all faces + get embeddings in one call
    faces = model.get(frame)
    print(f"[Search] Faces detected: {len(faces)}")

    if not faces:
        print("[Search] No faces in frame")
        return matches

    # For each detected face
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox.astype(int)
        score = face.det_score
        face_emb = face.embedding
        fw, fh = x2 - x1, y2 - y1

        print(f"[Search] Face {i+1}: [{x1},{y1},{x2},{y2}] {fw}x{fh}px, score={score:.2f}")

        # Draw cyan box on detected face
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Compare against each registered person
        for person_name, reg_emb in embedding_cache.items():
            sim = cosine_similarity(face_emb, reg_emb)
            is_match = sim > SIMILARITY_THRESHOLD

            print(f"[Search]   vs {person_name}: similarity={sim:.4f}, match={is_match}")

            if is_match:
                already = any(m["person_name"] == person_name for m in matches)
                if not already:
                    matches.append({
                        "person_name": person_name,
                        "confidence": round(sim, 2),
                        "distance": round(1 - sim, 4),
                        "face_index": i + 1,
                        "face_size": f"{fw}x{fh}"
                    })

                    # MAGENTA/PINK box — completely different from green YOLO boxes
                    # So user can EASILY spot the missing person in a crowd
                    color_box = (255, 0, 255)    # bright magenta/pink
                    color_label = (255, 0, 255)

                    # Expand around face to cover full head area
                    pad = int(fw * 0.5)
                    bx1 = max(0, x1 - pad)
                    by1 = max(0, y1 - pad)
                    bx2 = min(ow, x2 + pad)
                    by2 = min(oh, y2 + int(fh * 0.8))

                    # Double border for visibility (outer white + inner magenta)
                    cv2.rectangle(canvas, (bx1 - 2, by1 - 2), (bx2 + 2, by2 + 2), (255, 255, 255), 5)
                    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color_box, 4)

                    # Corner marks (makes it look like a target/scan)
                    corner_len = min(20, bx2 - bx1)
                    # Top-left
                    cv2.line(canvas, (bx1, by1), (bx1 + corner_len, by1), color_box, 6)
                    cv2.line(canvas, (bx1, by1), (bx1, by1 + corner_len), color_box, 6)
                    # Top-right
                    cv2.line(canvas, (bx2, by1), (bx2 - corner_len, by1), color_box, 6)
                    cv2.line(canvas, (bx2, by1), (bx2, by1 + corner_len), color_box, 6)
                    # Bottom-left
                    cv2.line(canvas, (bx1, by2), (bx1 + corner_len, by2), color_box, 6)
                    cv2.line(canvas, (bx1, by2), (bx1, by2 - corner_len), color_box, 6)
                    # Bottom-right
                    cv2.line(canvas, (bx2, by2), (bx2 - corner_len, by2), color_box, 6)
                    cv2.line(canvas, (bx2, by2), (bx2, by2 - corner_len), color_box, 6)

                    # Large label with magenta background
                    label = f"MISSING: {person_name} ({sim:.0%})"
                    lbl_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(canvas, (bx1, by1 - lbl_sz[1] - 16), (bx1 + lbl_sz[0] + 12, by1 - 2), color_label, -1)
                    cv2.putText(canvas, label, (bx1 + 6, by1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Summary at bottom
    if matches:
        names = ", ".join(m["person_name"] for m in matches)
        cv2.putText(canvas, f"MATCHED: {names}", (10, oh - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print(f"[Search] TOTAL MATCHES: {len(matches)}")
    return matches


# ========== SEARCH IN YOLO PERSON BOXES (video pipeline) ==========

def search_in_person_boxes(frame, person_boxes):
    """
    Search inside YOLO-detected person crops.
    For video pipeline where YOLO already found people.
    """
    matches = []

    if not embedding_cache:
        _load_all_embeddings()
    if not embedding_cache:
        return matches

    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box)
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        faces = model.get(person_crop)
        if not faces:
            continue

        best_face = max(faces, key=lambda f: f.det_score)

        for person_name, reg_emb in embedding_cache.items():
            sim = cosine_similarity(best_face.embedding, reg_emb)
            if sim > SIMILARITY_THRESHOLD:
                already = any(m["person_name"] == person_name for m in matches)
                if not already:
                    matches.append({
                        "person_name": person_name,
                        "confidence": round(sim, 2),
                        "distance": round(1 - sim, 4)
                    })
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, f"FOUND: {person_name}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return matches


# ========== HELPERS ==========

def _load_all_embeddings():
    """Load embeddings for all registered persons from their images"""
    if not os.path.exists(MISSING_PERSONS_DIR):
        return

    for filename in os.listdir(MISSING_PERSONS_DIR):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        if '_face' in filename:
            continue  # skip face crops, use full image

        person_name = os.path.splitext(filename)[0]
        if person_name in embedding_cache:
            continue

        img_path = os.path.join(MISSING_PERSONS_DIR, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = model.get(img)
        if faces:
            best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            embedding_cache[person_name] = best.embedding
            print(f"[Cache] Loaded: {person_name}")
        else:
            print(f"[Cache] No face in: {person_name}")


def get_registered_persons():
    """Get list of all registered missing persons"""
    if not os.path.exists(MISSING_PERSONS_DIR):
        return []
    return [
        {"name": os.path.splitext(f)[0], "image_path": os.path.join(MISSING_PERSONS_DIR, f)}
        for f in os.listdir(MISSING_PERSONS_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '_face' not in f
    ]
