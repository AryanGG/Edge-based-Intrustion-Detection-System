# =============================================================================
# ids_utils.py — Shared Utility Functions
# =============================================================================

from __future__ import annotations

import csv
import math
import os
import struct
import wave
from datetime import datetime
from typing import Tuple

import cv2
import numpy as np

import config

# ── Type aliases ─────────────────────────────────────────────────────────────
BBox = Tuple[int, int, int, int]   # (x1, y1, x2, y2)
ROI  = Tuple[int, int, int, int]   # (x1, y1, x2, y2)


# ─────────────────────────────────────────────────────────────────────────────
# ROI / Detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def bbox_overlaps_roi(bbox: BBox, roi: ROI) -> bool:
    """Return True if *bbox* overlaps the *roi* rectangle (axis-aligned)."""
    bx1, by1, bx2, by2 = bbox
    rx1, ry1, rx2, ry2 = roi
    # Overlap exists when the rectangles are NOT separated on either axis
    return not (bx2 < rx1 or bx1 > rx2 or by2 < ry1 or by1 > ry2)


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_roi(frame: np.ndarray, roi: ROI, color: Tuple[int, int, int],
             thickness: int = 2) -> None:
    """Draw the ROI rectangle on *frame* in-place."""
    rx1, ry1, rx2, ry2 = roi
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, thickness)
    label = "ROI Zone"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (rx1, ry1 - lh - 6), (rx1 + lw + 4, ry1), color, -1)
    cv2.putText(frame, label, (rx1 + 2, ry1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_bbox(frame: np.ndarray, bbox: BBox, label: str,
              color: Tuple[int, int, int], thickness: int = 2) -> None:
    """Draw a labelled bounding box on *frame* in-place."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_status_banner(frame: np.ndarray, intrusion: bool) -> None:
    """Render a full-width status banner at the top of *frame*."""
    h, w = frame.shape[:2]
    if intrusion:
        color = (0, 0, 200)
        text  = "  !!!  INTRUSION DETECTED  !!!"
    else:
        color = (0, 150, 0)
        text  = "  MONITORING"

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, text, (8, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 1, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Render an FPS counter in the bottom-right corner."""
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (w - 110, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot / logging
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create snapshots directory and log file (with header) if they don't exist."""
    os.makedirs(config.SNAPSHOTS_DIR, exist_ok=True)
    if not os.path.isfile(config.LOG_FILE):
        with open(config.LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "confidence", "snapshot_path"])


def save_snapshot(frame: np.ndarray, timestamp: datetime) -> str:
    """Save *frame* as a PNG and return its file path."""
    filename = timestamp.strftime("%Y%m%d_%H%M%S_%f") + ".png"
    path = os.path.join(config.SNAPSHOTS_DIR, filename)
    cv2.imwrite(path, frame)
    return path


def log_event(timestamp: datetime, confidence: float, snapshot_path: str) -> None:
    """Append one intrusion event row to the CSV log."""
    with open(config.LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
            f"{confidence:.4f}",
            snapshot_path,
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Audio
# ─────────────────────────────────────────────────────────────────────────────

def generate_beep_wav(path: str = config.BEEP_FILE,
                      frequency: float = 880.0,
                      duration: float = 0.35,
                      sample_rate: int = 44100,
                      amplitude: float = 0.6) -> None:
    """
    Synthesise a sine-wave beep and write it to *path* as a 16-bit mono WAV.
    Uses only numpy + the stdlib `wave` module — no extra dependencies.
    """
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Apply a short fade-in/out to avoid clicking artefacts
    fade_len = int(sample_rate * 0.02)
    envelope = np.ones(n_samples, dtype=np.float64)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)

    sine = np.sin(2 * math.pi * frequency * t) * amplitude * envelope
    samples = (sine * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
