# =============================================================================
# register_face.py — Capture your reference photo directly from webcam
# =============================================================================
# Run:  python register_face.py
#
# This opens your webcam, shows you a live preview, and when it detects your
# face clearly it saves the shot directly to known_faces/<your_name>.jpg.
# This guarantees dlib CAN encode the photo before saving it.
# =============================================================================

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

import config

# Load dlib
try:
    import dlib
    from face_recognition_models import (
        face_recognition_model_location,
        pose_predictor_model_location,
    )
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(pose_predictor_model_location())
    face_rec  = dlib.face_recognition_model_v1(face_recognition_model_location())
except Exception as e:
    sys.exit(f"[ERROR] dlib not available: {e}")


def encode_image(img_rgb: np.ndarray) -> list[np.ndarray]:
    """
    Return 128-d face encodings. dlib 20.x compatible:
    - Ensures C-contiguous array (channel-flip slices are not contiguous)
    - Manually crops face with padding and resizes to 150x150
    - Calls compute_face_descriptor(crop) — signature 2, no landmarks needed
    """
    img_rgb = np.ascontiguousarray(img_rgb)
    h, w = img_rgb.shape[:2]
    dets = detector(img_rgb, 1)
    encodings = []
    for det in dets:
        pad    = int((det.bottom() - det.top()) * 0.30)
        top    = max(0, det.top()    - pad)
        bottom = min(h, det.bottom() + pad)
        left   = max(0, det.left()   - pad)
        right  = min(w, det.right()  + pad)
        crop   = img_rgb[top:bottom, left:right]
        if crop.size == 0:
            continue
        crop_150 = np.ascontiguousarray(cv2.resize(crop, (150, 150)))
        desc = face_rec.compute_face_descriptor(crop_150)
        encodings.append(np.array(desc))
    return encodings


# ── Get name ──────────────────────────────────────────────────────────────────
name = input("Enter your name (used as filename, e.g. 'aryan'): ").strip()
if not name:
    sys.exit("Name cannot be empty.")

save_path = os.path.join(config.KNOWN_FACES_DIR, f"{name.lower().replace(' ','_')}.jpg")

# ── Open webcam ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
if not cap.isOpened():
    sys.exit(f"[ERROR] Cannot open webcam index {config.CAMERA_INDEX}.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

print()
print(f"  Registering: '{name}'")
print(f"  Save path  : {save_path}")
print()
print("  Look directly at the camera.")
print("  The system will capture automatically when it detects a good face,")
print("  OR press SPACE to capture manually, or 'q' to quit.")
print()

saved = False
countdown_start: float | None = None
COUNTDOWN_SECS   = 2.0   # auto-capture after face is stable for 2 s

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()
    rgb     = frame[:, :, ::-1]

    dets = detector(rgb, 1)
    face_detected = len(dets) > 0

    if face_detected:
        for det in dets:
            l, t, r, b = det.left(), det.top(), det.right(), det.bottom()
            cv2.rectangle(display, (l, t), (r, b), (0, 220, 0), 2)

        if countdown_start is None:
            countdown_start = time.monotonic()

        elapsed  = time.monotonic() - countdown_start
        remain   = max(0.0, COUNTDOWN_SECS - elapsed)
        bar_fill = int((elapsed / COUNTDOWN_SECS) * (config.FRAME_WIDTH - 20))
        bar_fill = min(bar_fill, config.FRAME_WIDTH - 20)

        # Progress bar
        cv2.rectangle(display, (10, config.FRAME_HEIGHT - 25),
                      (config.FRAME_WIDTH - 10, config.FRAME_HEIGHT - 10),
                      (60, 60, 60), -1)
        cv2.rectangle(display, (10, config.FRAME_HEIGHT - 25),
                      (10 + bar_fill, config.FRAME_HEIGHT - 10),
                      (0, 220, 0), -1)

        status = f"Face found! Capturing in {remain:.1f}s  (SPACE = now)"
        status_col = (0, 220, 0)

        if elapsed >= COUNTDOWN_SECS:
            # Auto-capture
            cv2.imshow("Register Face", display)
            cv2.waitKey(1)
            # Verify dlib can actually encode it before saving
            encs = []
            try:
                img_rgb_c = np.ascontiguousarray(rgb)
                encs = encode_image(img_rgb_c)
            except Exception as ex:
                print(f"  [WARN] Encoding failed: {ex} — retrying")
                countdown_start = None
                continue

            cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"\n  Saved: {save_path}")
            print(f"  Encoding verified — 128-d descriptor length: {len(encs[0])}")
            print(f"\n  Done!  '{name}' is now registered.")
            print(f"  Restart ids_main.py to activate the whitelist.")
            saved = True
            break
    else:
        countdown_start = None
        status     = "NO FACE DETECTED — look directly at the camera"
        status_col = (0, 80, 220)

    # Header
    cv2.rectangle(display, (0, 0), (display.shape[1], 44), (25, 25, 25), -1)
    cv2.putText(display, status, (8, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, status_col, 1, cv2.LINE_AA)
    hint = f"Registering: {name}  |  SPACE = capture now  |  q = quit"
    cv2.putText(display, hint, (8, display.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)

    cv2.imshow("Register Face", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Cancelled.")
        break
    elif key == ord(" ") and face_detected:
        # Manual capture
        try:
            img_rgb_c = np.ascontiguousarray(rgb)
            encs = encode_image(img_rgb_c)
        except Exception as ex:
            print(f"  [WARN] Encoding failed: {ex} — try again")
            countdown_start = None
            continue
        cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"\n  Saved: {save_path}")
        print(f"  Encoding verified. '{name}' is now registered.")
        print(f"  Restart ids_main.py to activate the whitelist.")
        saved = True
        break

cap.release()
cv2.destroyAllWindows()

if saved:
    # Quick double-check: reload from disk and encode
    print("\n  Verifying saved file can be loaded and encoded ...")
    from face_whitelist import load_whitelist
    wl = load_whitelist(config.KNOWN_FACES_DIR)
    if name.title() in wl or name.lower().replace("_"," ").title() in wl:
        print("  Whitelist check: PASSED. You will NOT trigger intrusion alerts.")
    else:
        print("  Whitelist is still empty — check the output above for errors.")
