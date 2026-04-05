# =============================================================================
# face_whitelist.py — Face Recognition Whitelist Module
# =============================================================================
# Uses dlib directly (via dlib-bin) instead of the face_recognition wrapper,
# which is incompatible with dlib 20.x.  The same pre-trained model files
# from face_recognition_models are used — accuracy is identical.
# =============================================================================

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("face_whitelist")

# ── Load dlib + models (graceful fallback if not installed) ───────────────────
try:
    import dlib
    from face_recognition_models import (
        face_recognition_model_location,
        pose_predictor_model_location,
    )

    _detector  = dlib.get_frontal_face_detector()
    _predictor = dlib.shape_predictor(pose_predictor_model_location())
    _face_rec  = dlib.face_recognition_model_v1(face_recognition_model_location())

    FACE_RECOGNITION_AVAILABLE = True
    log.debug("dlib face recognition models loaded OK.")

except Exception as _e:
    FACE_RECOGNITION_AVAILABLE = False
    _detector = _predictor = _face_rec = None  # type: ignore
    log.warning(
        "dlib face recognition unavailable (%s). "
        "All ROI detections will be treated as intrusions. "
        "Install with: pip install dlib-bin face-recognition-models",
        _e,
    )

# ── Types ─────────────────────────────────────────────────────────────────────
Whitelist = Dict[str, List[np.ndarray]]   # name → list of 128-d encodings

_SUPPORTED_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal dlib helpers  (replaces the broken face_recognition wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_rgb(img_rgb: np.ndarray, upsample: int = 1) -> list[np.ndarray]:
    """
    Return 128-d face encodings for every face in img_rgb.

    dlib 20.x compatible strategy:
      1. HOG detector -> bounding box
      2. Manually crop face region with padding (avoids get_face_chip issues)
      3. np.ascontiguousarray -> ensures C-contiguous memory dlib requires
      4. Resize to 150x150
      5. compute_face_descriptor(crop) — signature 2, no landmarks needed
    """
    # img_rgb MUST be C-contiguous — channel-flip slices are not
    img_rgb = np.ascontiguousarray(img_rgb)
    h, w = img_rgb.shape[:2]
    dets = _detector(img_rgb, upsample)
    encodings: list[np.ndarray] = []
    for det in dets:
        # Expand bbox by 30% on each side for context
        pad    = int((det.bottom() - det.top()) * 0.30)
        top    = max(0, det.top()    - pad)
        bottom = min(h, det.bottom() + pad)
        left   = max(0, det.left()   - pad)
        right  = min(w, det.right()  + pad)
        crop   = img_rgb[top:bottom, left:right]
        if crop.size == 0:
            continue
        crop_150 = np.ascontiguousarray(
            __import__("cv2").resize(crop, (150, 150))
        )
        desc = _face_rec.compute_face_descriptor(crop_150)
        encodings.append(np.array(desc))
    return encodings


def _face_distance(known: list[np.ndarray], query: np.ndarray) -> np.ndarray:
    """Euclidean distance between *query* and every vector in *known*."""
    return np.linalg.norm(np.array(known) - query, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Whitelist loading
# ─────────────────────────────────────────────────────────────────────────────

def _encode_photo(photo_path: str, name: str, whitelist: Whitelist) -> None:
    """Load one image file, encode faces, add to whitelist."""
    try:
        img_bgr = __import__("cv2").imread(photo_path)
        if img_bgr is None:
            raise ValueError("cv2.imread returned None — check file path/format")
        img_rgb = img_bgr[:, :, ::-1]   # BGR → RGB

        encodings = _encode_rgb(img_rgb, upsample=1)
        if not encodings:
            log.warning(
                "No face detected in '%s' — skipped. "
                "Use a well-lit, front-facing photo.",
                os.path.basename(photo_path),
            )
            return
        if len(encodings) > 1:
            log.warning(
                "'%s' has %d faces; using only the first.",
                os.path.basename(photo_path), len(encodings),
            )
        whitelist.setdefault(name, []).append(encodings[0])
        log.info("  Registered '%s'  ←  %s", name, os.path.basename(photo_path))
    except Exception as exc:
        log.warning("Could not process '%s': %s", photo_path, exc)


def load_whitelist(faces_dir: str) -> Whitelist:
    """
    Scan *faces_dir* and return a dict mapping person name → list of encodings.

    Layout options
    --------------
    Flat files:   known_faces/john_doe.jpg        → "John Doe"
    Subdirs:      known_faces/John Doe/photo.jpg  → "John Doe"
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return {}

    os.makedirs(faces_dir, exist_ok=True)
    whitelist: Whitelist = {}

    entries = os.listdir(faces_dir)
    if not entries:
        log.info(
            "known_faces/ is empty — whitelist disabled. "
            "Add reference photos to enable face whitelisting."
        )
        return whitelist

    log.info("Loading face whitelist from '%s' …", faces_dir)

    for entry in sorted(entries):
        entry_path = os.path.join(faces_dir, entry)

        if os.path.isdir(entry_path):
            person_name = entry.strip()
            found = 0
            for photo in sorted(os.listdir(entry_path)):
                if os.path.splitext(photo)[1].lower() in _SUPPORTED_EXTS:
                    _encode_photo(os.path.join(entry_path, photo), person_name, whitelist)
                    found += 1
            if found == 0:
                log.warning("Subdirectory '%s' has no supported images.", entry)

        elif os.path.isfile(entry_path):
            stem, ext = os.path.splitext(entry)
            if ext.lower() not in _SUPPORTED_EXTS:
                continue
            person_name = stem.replace("_", " ").replace("-", " ").strip().title()
            _encode_photo(entry_path, person_name, whitelist)

    total = sum(len(v) for v in whitelist.values())
    log.info(
        "Whitelist loaded: %d person(s), %d encoding(s) — %s",
        len(whitelist), total,
        ", ".join(f"'{n}'" for n in whitelist) or "none",
    )
    return whitelist


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame identification
# ─────────────────────────────────────────────────────────────────────────────

def identify_person(
    frame: np.ndarray,
    person_bbox: Tuple[int, int, int, int],
    whitelist: Whitelist,
    tolerance: float = 0.6,
    model: str = "hog",   # kept for API compatibility; dlib always uses HOG here
) -> Optional[str]:
    """
    Try to recognise the face inside *person_bbox* against the whitelist.

    Returns the matched person's name, or None (= unknown / no face found).
    """
    if not FACE_RECOGNITION_AVAILABLE or not whitelist:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = person_bbox
    bbox_h = y2 - y1
    bbox_w = x2 - x1
    pad_x   = max(8, bbox_w // 8)
    pad_top = max(10, bbox_h // 10)

    # Two crop candidates: upper-50 % first, full bbox as fallback
    crops = [
        (max(0, x1 - pad_x), max(0, y1 - pad_top),
         min(w, x2 + pad_x), min(h, y1 + int(bbox_h * 0.50))),
        (max(0, x1 - pad_x), max(0, y1 - pad_top),
         min(w, x2 + pad_x), min(h, y2)),
    ]

    encodings: list[np.ndarray] = []
    for cx1, cy1, cx2, cy2 in crops:
        if cy2 <= cy1 or cx2 <= cx1:
            continue
        face_crop = frame[cy1:cy2, cx1:cx2]
        if face_crop.size == 0:
            continue
        rgb_crop = face_crop[:, :, ::-1]
        try:
            encodings = _encode_rgb(rgb_crop, upsample=1)
        except Exception:
            continue
        if encodings:
            break   # face found — stop trying larger crops

    if not encodings:
        return None   # No face visible — fail-safe → unknown

    # Compare first encoding against every whitelisted vector
    all_encs: list[np.ndarray] = []
    all_names: list[str] = []
    for name, enc_list in whitelist.items():
        for enc in enc_list:
            all_encs.append(enc)
            all_names.append(name)

    if not all_encs:
        return None

    distances = _face_distance(all_encs, encodings[0])
    best_idx  = int(np.argmin(distances))
    best_dist = float(distances[best_idx])
    best_name = all_names[best_idx]

    log.debug(
        "Face distance to '%s': %.4f  (tolerance=%.2f)  %s",
        best_name, best_dist, tolerance,
        "MATCH" if best_dist <= tolerance else "NO MATCH",
    )

    return best_name if best_dist <= tolerance else None
