"""
Model Loader — Singleton pattern for YOLO fire model.
=====================================================
Loads the model ONCE at startup and reuses it for all detections.
Thread-safe, GPU-aware, with graceful fallback.
"""

import os
import threading

_model = None
_lock = threading.Lock()


def get_model(model_path="fire_model.pt", device="auto"):
    """
    Load and return the YOLO fire model (singleton).

    Args:
        model_path: Path to .pt weights file.
        device: "auto", "cpu", or "cuda:0".

    Returns:
        YOLO model instance, or None if loading fails.
    """
    global _model

    if _model is not None:
        return _model

    with _lock:
        # Double-check after acquiring lock
        if _model is not None:
            return _model

        try:
            from ultralytics import YOLO
            import torch

            # Resolve model path (check multiple locations)
            search_paths = [
                model_path,
                os.path.join(os.path.dirname(__file__), "..", model_path),
                os.path.join(os.path.dirname(__file__), model_path),
            ]

            resolved_path = None
            for p in search_paths:
                if os.path.exists(p):
                    resolved_path = os.path.abspath(p)
                    break

            if resolved_path is None:
                print(f"[FireModule] Model not found: {model_path}")
                print(f"[FireModule] Searched: {search_paths}")
                return None

            # Determine device
            if device == "auto":
                device = "cuda:0" if torch.cuda.is_available() else "cpu"

            _model = YOLO(resolved_path)
            print(f"[FireModule] Model loaded: {resolved_path}")
            print(f"[FireModule] Device: {device}")
            print(f"[FireModule] Classes: {_model.names}")
            return _model

        except Exception as e:
            print(f"[FireModule] Failed to load model: {e}")
            return None


def reset_model():
    """Reset the singleton (for testing or model swap)."""
    global _model
    with _lock:
        _model = None
