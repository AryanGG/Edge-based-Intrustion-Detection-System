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
#
# Face whitelisting:
#   Place reference photos in known_faces/ to register trusted people.
#   Persons whose faces match the whitelist are shown with a teal box and
#   their name — no alert fires.  Unknown or unreadable faces trigger the
#   full alert pipeline (red overlay, beep, snapshot, log entry).
# =============================================================================

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

import config
import face_whitelist
import ids_utils as utils

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ids_main")

# ── Display colour palette ────────────────────────────────────────────────────
COLOR_ALERT   = (0,   0, 220)   # Red   — unknown intruder in ROI
COLOR_SAFE    = (0, 200,   0)   # Green — person outside ROI
COLOR_KNOWN   = (180, 200,  0)  # Teal  — whitelisted person inside ROI

# ── Playsound import (graceful fallback) ──────────────────────────────────────
try:
    from playsound import playsound as _playsound
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    log.warning(
        "playsound not found — audio alerts disabled. "
        "Install with: pip install playsound==1.2.2"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Audio helper
# ─────────────────────────────────────────────────────────────────────────────

def _play_beep_async(beep_path: str) -> None:
    """Fire-and-forget: play beep in a daemon thread so the main loop is never blocked."""
    if not AUDIO_AVAILABLE or not os.path.isfile(beep_path):
        return

    def _play() -> None:
        try:
            _playsound(beep_path)
        except Exception as exc:
            log.warning("Audio playback error: %s", exc)

    threading.Thread(target=_play, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Main detection loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Setup ──────────────────────────────────────────────────────────────
    utils.ensure_dirs()

    if not os.path.isfile(config.BEEP_FILE):
        log.info("Generating beep.wav …")
        utils.generate_beep_wav()

    # ── 2. Face whitelist ─────────────────────────────────────────────────────
    whitelist: face_whitelist.Whitelist = {}
    if config.FACE_RECOGNITION_ENABLED:
        if face_whitelist.FACE_RECOGNITION_AVAILABLE:
            whitelist = face_whitelist.load_whitelist(config.KNOWN_FACES_DIR)
            if not whitelist:
                log.info(
                    "Whitelist is empty — all ROI persons will trigger alerts. "
                    "Add photos to known_faces/ to register trusted people."
                )
        else:
            log.warning(
                "FACE_RECOGNITION_ENABLED=True but the face-recognition "
                "library is not installed. Falling back to alert-on-all-ROI mode."
            )
    else:
        log.info("Face recognition disabled (FACE_RECOGNITION_ENABLED=False).")

    whitelist_active = bool(whitelist)

    # ── 3. YOLO model ─────────────────────────────────────────────────────────
    log.info("Loading model: %s", config.MODEL_NAME)
    model = YOLO(config.MODEL_NAME)

    roi = (config.ROI_X1, config.ROI_Y1, config.ROI_X2, config.ROI_Y2)

    # ── 4. Connect to stream ───────────────────────────────────────────────────
    stream_url = config.ESP32CAM_STREAM_URL
    log.info("Connecting to ESP32-CAM stream: %s", stream_url)
    log.info("(Make sure esp32cam_sim.py is running in another terminal)")

    cap: cv2.VideoCapture | None = None
    for attempt in range(1, 11):
        cap = cv2.VideoCapture(stream_url)
        if cap.isOpened():
            break
        log.warning("Stream not reachable (attempt %d/10) — retrying in 2 s …", attempt)
        cap.release()
        time.sleep(2.0)
    else:
        sys.exit(
            f"[ERROR] Could not connect to stream at {stream_url}\n"
            "        Is esp32cam_sim.py running?  Check ESP32CAM_STREAM_URL in config.py."
        )

    log.info("Stream connected. Press 'q' to quit.")

    # ── 5. Cooldown trackers ───────────────────────────────────────────────────
    last_beep_time: float     = 0.0
    last_snapshot_time: float = 0.0

    # ── 6. FPS tracking ───────────────────────────────────────────────────────
    prev_tick  = cv2.getTickCount()
    fps: float = 0.0

    # ── 7. Main loop ──────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Failed to read frame — retrying …")
            time.sleep(0.05)
            continue

        # FPS calc
        curr_tick = cv2.getTickCount()
        elapsed   = (curr_tick - prev_tick) / cv2.getTickFrequency()
        fps       = 0.9 * fps + 0.1 * (1.0 / elapsed if elapsed > 0 else fps)
        prev_tick = curr_tick

        # ── YOLO inference ────────────────────────────────────────────────────
        results = model(
            frame,
            verbose=False,
            classes=[0],                       # class 0 = person
            conf=config.CONFIDENCE_THRESHOLD,
        )

        intrusion_this_frame = False
        best_conf: float     = 0.0

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_val        = float(box.conf[0])
                bbox            = (x1, y1, x2, y2)

                in_roi = utils.bbox_overlaps_roi(bbox, roi)

                if not in_roi:
                    # ── Outside ROI: always safe, green box ──────────────────
                    utils.draw_bbox(frame, bbox, f"Person {conf_val:.2f}", COLOR_SAFE)
                    continue

                # ── Person is inside ROI — run face identification ────────────
                identity: str | None = None
                if whitelist_active:
                    identity = face_whitelist.identify_person(
                        frame,
                        bbox,
                        whitelist,
                        tolerance=config.FACE_MATCH_TOLERANCE,
                        model=config.FACE_DETECTION_MODEL,
                    )

                if identity is not None:
                    # ── Known / whitelisted person ────────────────────────────
                    utils.draw_bbox(
                        frame, bbox,
                        f"\u2713 {identity}  {conf_val:.2f}",
                        COLOR_KNOWN,
                    )
                    # Do NOT set intrusion_this_frame

                else:
                    # ── Unknown person (or face not legible) — INTRUSION ──────
                    utils.draw_bbox(
                        frame, bbox,
                        f"\u26a0 Unknown  {conf_val:.2f}",
                        COLOR_ALERT,
                    )
                    intrusion_this_frame = True
                    if conf_val > best_conf:
                        best_conf = conf_val

        # ── ROI overlay ───────────────────────────────────────────────────────
        roi_color = COLOR_ALERT if intrusion_this_frame else (0, 220, 0)
        utils.draw_roi(frame, roi, roi_color)

        # ── Status banner ─────────────────────────────────────────────────────
        utils.draw_status_banner(frame, intrusion_this_frame)
        utils.draw_fps(frame, fps)

        # ── Whitelist indicator (top-right corner) ────────────────────────────
        if config.FACE_RECOGNITION_ENABLED:
            wl_text  = (
                f"Whitelist: {len(whitelist)} person(s)"
                if whitelist_active else "Whitelist: empty"
            )
            wl_color = (180, 200, 0) if whitelist_active else (100, 100, 100)
            h_frame = frame.shape[0]
            cv2.putText(
                frame, wl_text,
                (8, h_frame - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, wl_color, 1, cv2.LINE_AA,
            )

        # ── Intrusion actions ─────────────────────────────────────────────────
        if intrusion_this_frame:
            now   = datetime.now()
            t_now = time.monotonic()

            # Snapshot (throttled)
            if t_now - last_snapshot_time >= config.SNAPSHOT_COOLDOWN_SECONDS:
                snap_path          = utils.save_snapshot(frame, now)
                last_snapshot_time = t_now
                utils.log_event(now, best_conf, snap_path)
                log.info(
                    "ALERT: Unknown intruder | conf=%.2f | snap=%s",
                    best_conf, snap_path,
                )

            # Audio (throttled)
            if t_now - last_beep_time >= config.BEEP_COOLDOWN_SECONDS:
                _play_beep_async(config.BEEP_FILE)
                last_beep_time = t_now

        # ── Display ───────────────────────────────────────────────────────────
        cv2.imshow(config.WINDOW_TITLE, frame)

        # ── Exit on 'q' ───────────────────────────────────────────────────────
        if cv2.waitKey(1) & 0xFF == ord("q"):
            log.info("Quit signal received. Shutting down …")
            break

    # ── 8. Cleanup ────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    log.info("Done.")


if __name__ == "__main__":
    main()
