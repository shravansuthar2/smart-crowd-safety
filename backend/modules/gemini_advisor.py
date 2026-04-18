"""
Gemini Advisor — Google AI-powered safety recommendations.
Generates real-time response advice for detected incidents using Gemini.
"""

import os

GEMINI_MODEL = None
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

try:
    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")
        print("[Gemini] Advisor loaded — Google AI active")
    else:
        print("[Gemini] No GEMINI_API_KEY set — advisor disabled")
except Exception as e:
    print(f"[Gemini] Init error: {e}")


def get_alert_advice(alert_type: str, severity: str, location: str, details: dict = None) -> str:
    """Ask Gemini for a 1-2 sentence response recommendation for the alert."""
    if GEMINI_MODEL is None:
        return ""

    details = details or {}
    prompt = f"""You are a public safety AI advisor for a crowd monitoring system.
A {severity} alert just fired:
- Type: {alert_type}
- Location: {location}
- Details: {details}

Respond with ONE short, actionable recommendation (under 25 words) for the security operator.
No preamble. No "I recommend". Just the action."""

    try:
        response = GEMINI_MODEL.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini] Advice error: {e}")
        return ""
