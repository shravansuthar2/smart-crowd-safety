"""
Alert Manager Module
Handles creating, acknowledging, and resolving alerts.
Stores alerts in Firebase Firestore or local memory (fallback).
"""

from datetime import datetime, timezone
from firebase_config import save_alert, get_alerts, update_alert, clear_alerts as firebase_clear_alerts
from modules.gemini_advisor import get_alert_advice


def create_alert(alert_type: str, severity: str, location: str, details: dict = None):
    """
    Create and save a new alert.

    Args:
        alert_type: "crowd_density" | "missing_person" | "pickpocket" | "emergency"
        severity: "low" | "medium" | "high" | "critical"
        location: camera ID or description
        details: extra info about the alert

    Returns:
        dict - the created alert
    """
    ai_advice = get_alert_advice(alert_type, severity, location, details)

    alert = {
        "type": alert_type,
        "severity": severity,
        "location": location,
        "details": details or {},
        "ai_advice": ai_advice,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "active",
        "acknowledged": False
    }

    save_alert(alert)
    print(f"[Alert] {severity.upper()} - {alert_type} at {location}")
    if ai_advice:
        print(f"[Gemini] {ai_advice}")

    return alert


def acknowledge_alert(alert_id: str):
    """Mark alert as acknowledged (seen by operator)"""
    update_alert(alert_id, {
        "acknowledged": True,
        "status": "acknowledged",
        "acknowledged_at": datetime.now(timezone.utc).isoformat()
    })
    print(f"[Alert] Acknowledged: {alert_id}")


def resolve_alert(alert_id: str):
    """Mark alert as resolved (handled)"""
    update_alert(alert_id, {
        "status": "resolved",
        "resolved_at": datetime.now(timezone.utc).isoformat()
    })
    print(f"[Alert] Resolved: {alert_id}")


def get_recent_alerts(limit=50):
    """Get recent alerts sorted by newest first"""
    return get_alerts(limit)


def clear_all_alerts():
    """Clear all alerts"""
    firebase_clear_alerts()
    print("[Alert] All alerts cleared")
