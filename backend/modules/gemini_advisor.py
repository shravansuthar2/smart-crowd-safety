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

        candidates = [
            "gemini-flash-latest",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash-latest",
            "gemini-pro",
        ]
        try:
            available = {m.name.split("/")[-1] for m in genai.list_models()
                         if "generateContent" in m.supported_generation_methods}
            print(f"[Gemini] Available models: {sorted(available)[:10]}")
        except Exception as e:
            available = set()
            print(f"[Gemini] list_models failed: {e}")

        chosen = next((m for m in candidates if m in available), None) or (sorted(available)[0] if available else candidates[0])
        GEMINI_MODEL = genai.GenerativeModel(chosen)
        print(f"[Gemini] Advisor loaded — using {chosen}")
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
