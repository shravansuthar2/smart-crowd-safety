"""
Alerts Router
API endpoints for managing alerts:
- GET  /api/alerts/                    → List all recent alerts
- PUT  /api/alerts/{id}/acknowledge    → Mark alert as seen
- PUT  /api/alerts/{id}/resolve        → Mark alert as handled
"""

from fastapi import APIRouter
from modules.alert_manager import get_recent_alerts, acknowledge_alert, resolve_alert, clear_all_alerts

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


@router.get("/")
async def list_alerts(limit: int = 50):
    """
    Get all recent alerts, sorted newest first.

    Query params:
        limit: max number of alerts to return (default 50)
    """
    alerts = get_recent_alerts(limit)
    return {"alerts": alerts, "count": len(alerts)}


@router.delete("/clear")
async def clear_alerts():
    """Clear all alerts from memory"""
    clear_all_alerts()
    return {"message": "All alerts cleared"}


@router.put("/{alert_id}/acknowledge")
async def ack_alert(alert_id: str):
    """Mark alert as acknowledged (operator has seen it)"""
    acknowledge_alert(alert_id)
    return {"message": "Alert acknowledged", "id": alert_id}


@router.put("/{alert_id}/resolve")
async def res_alert(alert_id: str):
    """Mark alert as resolved (situation handled)"""
    resolve_alert(alert_id)
    return {"message": "Alert resolved", "id": alert_id}
