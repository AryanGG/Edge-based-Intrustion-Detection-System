# =============================================================================
# test_face.py — Minimal Face Recognition Test (no YOLO, no stream)
# =============================================================================
# Run: python test_face.py
# (No simulator needed — opens webcam directly)
# =============================================================================

import os
import sys
import time

import cv2
import numpy as np

import config

# ── Load dlib directly (bypasses the broken face_recognition wrapper) ─────────
try:
    import dlib
    from face_recognition_models import (
        face_recognition_model_location,
        pose_predictor_model_location,
    )
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(pose_predictor_model_location())
    face_rec  = dlib.face_recognition_model_v1(face_recognition_model_location())
    print("[OK] dlib models loaded.")
except Exception as e:
    sys.exit(f"[ERROR] Could not load dlib: {e}\n"
             "  Run:  pip install dlib-bin face-recognition-models")


def encode_image(img_rgb: np.ndarray, upsample: int = 1) -> list[np.ndarray]:
    """Return 128-d encodings for all faces. dlib 20.x compatible."""
    img_rgb = np.ascontiguousarray(img_rgb)
    h, w = img_rgb.shape[:2]
    dets = detector(img_rgb, upsample)
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


def face_distance(known: list[np.ndarray], query: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.array(known) - query, axis=1)


# ── Step 1: Load whitelist ────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading reference photos from known_faces/")
print("=" * 60)

faces_dir = config.KNOWN_FACES_DIR
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not os.path.isdir(faces_dir) or not os.listdir(faces_dir):
    sys.exit(
        f"[ERROR] '{faces_dir}/' is empty or missing.\n"
        "  Add a photo:  known_faces/your_name.jpg"
    )

photos: list[tuple[str, str]] = []
for entry in sorted(os.listdir(faces_dir)):
    path = os.path.join(faces_dir, entry)
    if os.path.isfile(path) and os.path.splitext(entry)[1].lower() in EXTS:
        photos.append((entry, path))
    elif os.path.isdir(path):
        for f in sorted(os.listdir(path)):
            if os.path.splitext(f)[1].lower() in EXTS:
                photos.append((f"{entry}/{f}", os.path.join(path, f)))

if not photos:
    sys.exit(f"[ERROR] No supported images found in '{faces_dir}/'.")

print(f"  Found {len(photos)} photo(s):")
for name, _ in photos:
    print(f"    {name}")

known_encodings: list[np.ndarray] = []
known_names: list[str] = []

print()
for photo_name, photo_path in photos:
    print(f"  Encoding '{photo_name}' ... ", end="", flush=True)
    img_bgr = cv2.imread(photo_path)
    if img_bgr is None:
        print("FAILED — could not load image file.")
        continue
    img_rgb  = img_bgr[:, :, ::-1]
    encs     = encode_image(img_rgb, upsample=1)
    if not encs:
        print("FAILED — no face detected in photo.")
        print(
            f"    >> Use a clear, well-lit, front-facing photo "
            f"(no sunglasses, no heavy shadow)."
        )
        continue
    stem = os.path.splitext(os.path.basename(photo_name))[0]
    name = stem.replace("_", " ").replace("-", " ").strip().title()
    known_encodings.append(encs[0])
    known_names.append(name)
    print(f"OK  ({len(encs)} face(s) found → registered as '{name}')")

if not known_encodings:
    sys.exit("\n[ERROR] No faces could be encoded — fix your reference photo first.")

print(f"\n  Whitelist: {known_names}")

# ── Step 2: Live webcam test ───────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 2: Live webcam test  (no YOLO, no stream)")
print("=" * 60)

cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
if not cap.isOpened():
    sys.exit(f"[ERROR] Could not open webcam index {config.CAMERA_INDEX}.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
print("  Webcam opened.  Look at the camera — press 'q' to quit.\n")

last_print = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()
    rgb     = frame[:, :, ::-1]

    # Detect all faces in the full frame
    dets          = detector(rgb, 1)
    header        = ""
    header_color  = (80, 80, 80)

    if not dets:
        header       = "NO FACE DETECTED — move closer or improve lighting"
        header_color = (0, 80, 220)
        if time.monotonic() - last_print > 1.5:
            print("  [LIVE] No face detected in frame")
            last_print = time.monotonic()

    for det in dets:
        rgb_c    = np.ascontiguousarray(rgb)
        encs     = encode_image(rgb_c)
        if not encs:
            continue
        enc      = encs[0]
        dists    = face_distance(known_encodings, enc)
        best_idx = int(np.argmin(dists))
        best_d   = float(dists[best_idx])
        best_n   = known_names[best_idx]
        matched  = best_d <= config.FACE_MATCH_TOLERANCE

        # Draw box
        l, t, r, b = det.left(), det.top(), det.right(), det.bottom()
        color = (0, 200, 0) if matched else (0, 0, 220)
        cv2.rectangle(display, (l, t), (r, b), color, 2)

        id_lbl   = f"{'MATCH: ' + best_n if matched else 'UNKNOWN'}"
        dist_lbl = f"dist={best_d:.3f}  tol={config.FACE_MATCH_TOLERANCE}"
        cv2.putText(display, id_lbl,   (l, t - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,  color,       2, cv2.LINE_AA)
        cv2.putText(display, dist_lbl, (l, t - 5),  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 0), 1, cv2.LINE_AA)

        if matched:
            header       = f"RECOGNISED: {best_n}  (dist={best_d:.3f})"
            header_color = (0, 200, 0)
        else:
            header       = f"UNKNOWN  |  closest: {best_n}  dist={best_d:.3f}"
            header_color = (0, 0, 220)

        if time.monotonic() - last_print > 0.5:
            if matched:
                print(f"  [LIVE] MATCH: '{best_n}'  dist={best_d:.4f}")
            else:
                sug = best_d + 0.03
                print(f"  [LIVE] NO MATCH  |  dist to '{best_n}' = {best_d:.4f}  "
                      f"(tolerance={config.FACE_MATCH_TOLERANCE:.2f})")
                print(f"         >> Try setting FACE_MATCH_TOLERANCE = {sug:.2f} in config.py")
            last_print = time.monotonic()

    # Header banner
    cv2.rectangle(display, (0, 0), (display.shape[1], 44), (25, 25, 25), -1)
    cv2.putText(display, header, (8, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, header_color, 1, cv2.LINE_AA)
    hint = "Direct webcam  |  no YOLO  |  no stream  |  q = quit"
    cv2.putText(display, hint, (8, display.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1, cv2.LINE_AA)

    cv2.imshow("Face Recognition Test", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
