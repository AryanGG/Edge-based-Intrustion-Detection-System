# =============================================================================
# ids_main.py — Edge-Based Intrusion Detection System — Main Loop
# =============================================================================
# Prerequisite: esp32cam_sim.py must already be running in another terminal.
#
# Run:  python ids_main.py
# Exit: press 'q' in the live-feed window
#
# The detector reads from an MJPEG stream (config.ESP32CAM_STREAM_URL).
# By default this points to the local simulator.  To use a real ESP32-CAM,
# change ESP32CAM_STREAM_URL in config.py to the camera's IP address.
# =============================================================================

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

import config
import ids_utils as utils

# ── Playsound import (graceful fallback) ──────────────────────────────────────
try:
    from playsound import playsound as _playsound
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[WARN] playsound not found — audio alerts disabled. "
          "Install with: pip install playsound==1.2.2")


# ─────────────────────────────────────────────────────────────────────────────
# Audio helper
# ─────────────────────────────────────────────────────────────────────────────

def _play_beep_async(beep_path: str) -> None:
    """Fire-and-forget: play beep in a daemon thread so the main loop is never blocked."""
    if not AUDIO_AVAILABLE:
        return
    if not os.path.isfile(beep_path):
        return

    def _play() -> None:
        try:
            _playsound(beep_path)
        except Exception as exc:                # Silently swallow audio errors
            print(f"[WARN] Audio playback error: {exc}")

    t = threading.Thread(target=_play, daemon=True)
    t.start()


# ─────────────────────────────────────────────────────────────────────────────
# Main detection loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Setup ──────────────────────────────────────────────────────────────
    utils.ensure_dirs()

    if not os.path.isfile(config.BEEP_FILE):
        print("[INFO] Generating beep.wav …")
        utils.generate_beep_wav()

    print(f"[INFO] Loading model: {config.MODEL_NAME}")
    model = YOLO(config.MODEL_NAME)   # Downloads ~6 MB to ~/.ultralytics on first run

    roi = (config.ROI_X1, config.ROI_Y1, config.ROI_X2, config.ROI_Y2)

    stream_url = config.ESP32CAM_STREAM_URL
    print(f"[INFO] Connecting to ESP32-CAM stream: {stream_url}")
    print("[INFO] (Make sure esp32cam_sim.py is running in another terminal)")

    # Retry loop — the sim server may still be starting up
    cap: cv2.VideoCapture | None = None
    for attempt in range(1, 11):
        cap = cv2.VideoCapture(stream_url)
        if cap.isOpened():
            break
        print(f"[WARN] Stream not reachable (attempt {attempt}/10) — retrying in 2 s …")
        cap.release()
        time.sleep(2.0)
    else:
        sys.exit(
            f"[ERROR] Could not connect to stream at {stream_url}\n"
            "        Is esp32cam_sim.py running?  Check ESP32CAM_STREAM_URL in config.py."
        )

    print("[INFO] Stream connected. Press 'q' to quit.")

    # ── 2. Cooldown trackers ──────────────────────────────────────────────────
    last_beep_time: float     = 0.0
    last_snapshot_time: float = 0.0

    # ── 3. FPS tracking ───────────────────────────────────────────────────────
    prev_tick  = cv2.getTickCount()
    fps: float = 0.0

    # ── 4. Main loop ──────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame — retrying …")
            time.sleep(0.05)
            continue

        # ── FPS calc ──────────────────────────────────────────────────────────
        curr_tick = cv2.getTickCount()
        elapsed   = (curr_tick - prev_tick) / cv2.getTickFrequency()
        fps       = 0.9 * fps + 0.1 * (1.0 / elapsed if elapsed > 0 else fps)
        prev_tick = curr_tick

        # ── YOLO inference ────────────────────────────────────────────────────
        results = model(
            frame,
            verbose=False,
            classes=[0],                          # class 0 = person
            conf=config.CONFIDENCE_THRESHOLD,
        )

        intrusion_this_frame = False
        best_conf: float = 0.0

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_val        = float(box.conf[0])
                bbox            = (x1, y1, x2, y2)

                overlaps = utils.bbox_overlaps_roi(bbox, roi)

                # Colour indicates threat level
                box_color  = (0, 0, 220)  if overlaps else (0, 200, 0)
                label      = f"Person {conf_val:.2f}"
                utils.draw_bbox(frame, bbox, label, box_color)

                if overlaps:
                    intrusion_this_frame = True
                    if conf_val > best_conf:
                        best_conf = conf_val

        # ── ROI overlay ───────────────────────────────────────────────────────
        roi_color = (0, 0, 220) if intrusion_this_frame else (0, 220, 0)
        utils.draw_roi(frame, roi, roi_color)

        # ── Status banner ─────────────────────────────────────────────────────
        utils.draw_status_banner(frame, intrusion_this_frame)
        utils.draw_fps(frame, fps)

        # ── Intrusion actions ─────────────────────────────────────────────────
        if intrusion_this_frame:
            now    = datetime.now()
            t_now  = time.monotonic()

            # Snapshot (throttled)
            if t_now - last_snapshot_time >= config.SNAPSHOT_COOLDOWN_SECONDS:
                snap_path          = utils.save_snapshot(frame, now)
                last_snapshot_time = t_now
                utils.log_event(now, best_conf, snap_path)
                print(f"[ALERT] Intrusion! conf={best_conf:.2f}  snap={snap_path}")

            # Audio (throttled)
            if t_now - last_beep_time >= config.BEEP_COOLDOWN_SECONDS:
                _play_beep_async(config.BEEP_FILE)
                last_beep_time = t_now

        # ── Display ───────────────────────────────────────────────────────────
        cv2.imshow(config.WINDOW_TITLE, frame)

        # ── Exit on 'q' ───────────────────────────────────────────────────────
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit signal received. Shutting down …")
            break

    # ── 5. Cleanup ────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
