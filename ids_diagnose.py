# =============================================================================
# ids_diagnose.py — Face Recognition Diagnostic Tool
# =============================================================================
# Run this INSTEAD of ids_main.py to debug face recognition issues.
# It connects to the same MJPEG stream and shows in real-time:
#   - Whether a face is found in each detected person's region
#   - The raw distance score to every whitelisted person
#   - The crop region being searched (yellow rectangle)
#   - Saves face crops to debug_crops/ so you can inspect them
#
# Run:  python ids_diagnose.py
# Exit: press 'q'
# =============================================================================

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

import config
import face_whitelist
import ids_utils as utils

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ids_diagnose")

DEBUG_CROPS_DIR = "debug_crops"


def main() -> None:
    os.makedirs(DEBUG_CROPS_DIR, exist_ok=True)

    # ── Load whitelist ─────────────────────────────────────────────────────────
    if not face_whitelist.FACE_RECOGNITION_AVAILABLE:
        sys.exit("[ERROR] face-recognition library is not installed.")

    log.info("Loading whitelist …")
    whitelist = face_whitelist.load_whitelist(config.KNOWN_FACES_DIR)

    if not whitelist:
        log.warning(
            "Whitelist is EMPTY.  Add photos to known_faces/ and restart.\n"
            "  Check: did the startup log say 'Registered ...'?\n"
            "  If not, dlib could not detect a face in your reference photo."
        )
    else:
        log.info(
            "Whitelist contains: %s",
            ", ".join(f"'{n}' ({len(v)} encoding(s))" for n, v in whitelist.items()),
        )

    # ── YOLO ──────────────────────────────────────────────────────────────────
    log.info("Loading YOLO model …")
    model = YOLO(config.MODEL_NAME)
    roi = (config.ROI_X1, config.ROI_Y1, config.ROI_X2, config.ROI_Y2)

    # ── Stream ────────────────────────────────────────────────────────────────
    stream_url = config.ESP32CAM_STREAM_URL
    log.info("Connecting to stream: %s", stream_url)
    cap: cv2.VideoCapture | None = None
    for attempt in range(1, 11):
        cap = cv2.VideoCapture(stream_url)
        if cap.isOpened():
            break
        log.warning("Attempt %d/10 failed — retrying …", attempt)
        cap.release()
        time.sleep(2.0)
    else:
        sys.exit(f"[ERROR] Cannot connect to {stream_url}")

    log.info("Stream connected.  Press 'q' to quit.")
    log.info("Yellow box = face search region.  Watch the terminal for distance scores.")

    # Flatten whitelist for distance computation
    all_encodings: list[np.ndarray] = []
    all_names: list[str] = []
    for name, enc_list in whitelist.items():
        for enc in enc_list:
            all_encodings.append(enc)
            all_names.append(name)

    import face_recognition as _fr

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_idx += 1
        display = frame.copy()

        # Draw ROI
        utils.draw_roi(display, roi, (0, 220, 0))

        # YOLO detect persons
        results = model(frame, verbose=False, classes=[0], conf=config.CONFIDENCE_THRESHOLD)

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_val = float(box.conf[0])
                bbox = (x1, y1, x2, y2)
                in_roi = utils.bbox_overlaps_roi(bbox, roi)

                # Draw person bbox (white in diagnostics)
                cv2.rectangle(display, (x1, y1), (x2, y2), (200, 200, 200), 1)
                cv2.putText(display, f"person {conf_val:.2f}", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

                if not in_roi:
                    continue

                # ── Build crop candidates ──────────────────────────────────────
                # Try three crops: upper-33%, upper-50%, full bbox
                h_frame, w_frame = frame.shape[:2]
                bbox_h = y2 - y1
                bbox_w = x2 - x1
                pad_x = max(10, bbox_w // 8)
                pad_top = max(15, bbox_h // 8)

                crops = {
                    "upper-33%": (
                        max(0, x1 - pad_x), max(0, y1 - pad_top),
                        min(w_frame, x2 + pad_x), min(h_frame, y1 + int(bbox_h * 0.33)),
                    ),
                    "upper-55%": (
                        max(0, x1 - pad_x), max(0, y1 - pad_top),
                        min(w_frame, x2 + pad_x), min(h_frame, y1 + int(bbox_h * 0.55)),
                    ),
                    "full-bbox": (
                        max(0, x1 - pad_x), max(0, y1 - pad_top),
                        min(w_frame, x2 + pad_x), min(h_frame, y2),
                    ),
                }

                face_found = False
                for crop_name, (cx1, cy1, cx2, cy2) in crops.items():
                    if cy2 <= cy1 or cx2 <= cx1:
                        continue

                    face_crop = frame[cy1:cy2, cx1:cx2]
                    if face_crop.size == 0:
                        continue

                    rgb_crop = face_crop[:, :, ::-1]

                    # Draw the crop region being tried (yellow)
                    cv2.rectangle(display, (cx1, cy1), (cx2, cy2), (0, 220, 220), 1)

                    face_locs = _fr.face_locations(rgb_crop, model=config.FACE_DETECTION_MODEL)

                    if not face_locs:
                        print(f"  [DIAG frame {frame_idx}] crop={crop_name:12s}  → NO FACE DETECTED in crop")
                        continue

                    encodings = _fr.face_encodings(rgb_crop, face_locs)
                    if not encodings:
                        print(f"  [DIAG frame {frame_idx}] crop={crop_name:12s}  → face located but encoding failed")
                        continue

                    # We have a face encoding — compare against whitelist
                    face_found = True
                    query_enc = encodings[0]

                    if all_encodings:
                        distances = _fr.face_distance(all_encodings, query_enc)
                        print(f"\n  [DIAG frame {frame_idx}] crop={crop_name:12s}  → FACE FOUND  ({len(face_locs)} location(s))")
                        for name, dist in zip(all_names, distances):
                            match_sym = "✓ MATCH" if dist <= config.FACE_MATCH_TOLERANCE else "✗ no match"
                            print(f"     distance to '{name}': {dist:.4f}  [{match_sym}]  (tolerance={config.FACE_MATCH_TOLERANCE})")

                        best_idx = int(np.argmin(distances))
                        best_name = all_names[best_idx]
                        best_dist = distances[best_idx]

                        if best_dist <= config.FACE_MATCH_TOLERANCE:
                            label = f"\u2713 {best_name} d={best_dist:.3f}"
                            color = (180, 200, 0)
                        else:
                            label = f"Unknown (closest: {best_name} d={best_dist:.3f})"
                            color = (0, 80, 220)

                        cv2.putText(display, label, (x1, y2 + 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                    else:
                        print(f"  [DIAG frame {frame_idx}] crop={crop_name:12s}  → FACE FOUND but whitelist is empty")

                    # Save the crop for inspection (once every 60 frames)
                    if frame_idx % 60 == 0:
                        crop_path = os.path.join(
                            DEBUG_CROPS_DIR,
                            f"frame{frame_idx}_{crop_name.replace('%','pct')}.jpg"
                        )
                        cv2.imwrite(crop_path, face_crop)
                        print(f"     Saved crop: {crop_path}")

                    break  # Stop trying crops once we find a face

                if not face_found and frame_idx % 10 == 0:
                    print(f"  [DIAG frame {frame_idx}] All crops exhausted — no face detected for person in ROI")

        cv2.imshow("IDS Diagnostics (yellow=crop region, q=quit)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    log.info("Done. Check debug_crops/ for saved face crop images.")


if __name__ == "__main__":
    main()
