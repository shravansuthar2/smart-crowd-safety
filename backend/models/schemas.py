from pydantic import BaseModel
from typing import Optional


class AlertResponse(BaseModel):
    type: str
    severity: str
    location: str
    details: Optional[dict] = None
    timestamp: str
    status: str


class CrowdDetectionResponse(BaseModel):
    count: int
    is_overcrowded: bool
    threshold: int
    annotated_image: str  # base64


class PickpocketResponse(BaseModel):
    suspicious: bool
    alerts: list
    annotated_image: str


class EmergencyResponse(BaseModel):
    fall_detected: bool
    fight_detected: bool
    annotated_image: str


class MissingPersonRegister(BaseModel):
    name: str
    details: Optional[str] = ""


class SearchResult(BaseModel):
    person_name: str
    confidence: float
    distance: float
