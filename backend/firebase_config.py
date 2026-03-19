import os
import firebase_admin
from firebase_admin import credentials, firestore, storage
from config import FIREBASE_CREDENTIALS, FIREBASE_STORAGE_BUCKET

# ========== FIREBASE INITIALIZATION ==========
# Check if serviceAccountKey.json exists
# If not, run in LOCAL MODE (no Firebase, store in memory)

USE_FIREBASE = os.path.exists(FIREBASE_CREDENTIALS)

if USE_FIREBASE:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    firebase_admin.initialize_app(cred, {
        "storageBucket": FIREBASE_STORAGE_BUCKET
    })
    db = firestore.client()
    bucket = storage.bucket()
    print("[Firebase] Connected successfully")
else:
    db = None
    bucket = None
    print("[Firebase] serviceAccountKey.json not found — running in LOCAL MODE")
    print("[Firebase] Alerts and data will be stored in memory (lost on restart)")

# ========== LOCAL STORAGE (fallback) ==========
local_alerts = []
local_missing_persons = []
alert_counter = 0


# ========== ALERT FUNCTIONS ==========
def save_alert(alert_data: dict):
    """Save alert to Firestore or local memory"""
    global alert_counter

    if USE_FIREBASE:
        db.collection("alerts").add(alert_data)
    else:
        alert_counter += 1
        alert_data["id"] = f"local-{alert_counter}"
        local_alerts.insert(0, alert_data)
        # Keep only last 100 alerts in memory
        if len(local_alerts) > 100:
            local_alerts.pop()


def get_alerts(limit=50):
    """Get recent alerts from Firestore or local memory"""
    if USE_FIREBASE:
        docs = db.collection("alerts").order_by(
            "timestamp", direction=firestore.Query.DESCENDING
        ).limit(limit).stream()
        return [{"id": doc.id, **doc.to_dict()} for doc in docs]
    else:
        return local_alerts[:limit]


def update_alert(alert_id: str, data: dict):
    """Update alert status"""
    if USE_FIREBASE:
        db.collection("alerts").document(alert_id).update(data)
    else:
        for alert in local_alerts:
            if alert.get("id") == alert_id:
                alert.update(data)
                break


def clear_alerts():
    """Clear all alerts"""
    global local_alerts, alert_counter
    if USE_FIREBASE:
        # Delete all docs in alerts collection
        docs = db.collection("alerts").stream()
        for doc in docs:
            doc.reference.delete()
    else:
        local_alerts.clear()
        alert_counter = 0


# ========== IMAGE UPLOAD ==========
def upload_image(file_path: str, destination: str):
    """Upload image to Firebase Storage or return local path"""
    if USE_FIREBASE:
        blob = bucket.blob(destination)
        blob.upload_from_filename(file_path)
        blob.make_public()
        return blob.public_url
    else:
        return f"/uploads/{os.path.basename(file_path)}"


# ========== MISSING PERSONS ==========
def save_missing_person(person_data: dict):
    """Save missing person to Firestore or local memory"""
    if USE_FIREBASE:
        db.collection("missing_persons").add(person_data)
    else:
        person_data["id"] = f"person-{len(local_missing_persons) + 1}"
        local_missing_persons.append(person_data)


def get_missing_persons():
    """Get all missing persons"""
    if USE_FIREBASE:
        docs = db.collection("missing_persons").stream()
        return [{"id": doc.id, **doc.to_dict()} for doc in docs]
    else:
        return local_missing_persons


def delete_missing_person(name: str):
    """Delete a missing person by name"""
    global local_missing_persons
    if USE_FIREBASE:
        docs = db.collection("missing_persons").where("name", "==", name).stream()
        for doc in docs:
            doc.reference.delete()
    else:
        local_missing_persons = [p for p in local_missing_persons if p.get("name") != name]
