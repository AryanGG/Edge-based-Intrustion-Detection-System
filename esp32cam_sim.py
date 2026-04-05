# =============================================================================
# esp32cam_sim.py — Simulated ESP32-CAM MJPEG Stream Server
# =============================================================================
# This script mimics the behaviour of a physical ESP32-CAM module broadcasting
# a live MJPEG stream over Wi-Fi.  It reads the laptop's webcam and re-serves
# it as a proper multipart/x-mixed-replace HTTP stream — the same format that
# every real ESP32-CAM uses out of the box.
#
# Run:  python esp32cam_sim.py
#       python esp32cam_sim.py --port 8080   (override port)
#       python esp32cam_sim.py --camera 1    (use a different camera index)
#
# Stop: Ctrl+C
#
# Stream endpoint:  http://localhost:<PORT>/stream
# Status page:      http://localhost:<PORT>/
# =============================================================================

from __future__ import annotations

import argparse
import io
import logging
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import cv2
import numpy as np

import config

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("esp32cam_sim")


# ─────────────────────────────────────────────────────────────────────────────
# Shared frame buffer  (written by capture thread, read by HTTP handlers)
# ─────────────────────────────────────────────────────────────────────────────

class FrameBuffer:
    """Thread-safe holder for the most recently captured JPEG frame."""

    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._frame: Optional[bytes] = None
        self._event = threading.Event()   # signals that a new frame is ready

    def put(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self._frame = jpeg_bytes
        self._event.set()
        self._event.clear()

    def get(self, timeout: float = 2.0) -> Optional[bytes]:
        """Block until a new frame is available (or *timeout* seconds pass)."""
        self._event.wait(timeout=timeout)
        with self._lock:
            return self._frame


# Singleton buffer shared across all request handler instances
_buffer = FrameBuffer()


# ─────────────────────────────────────────────────────────────────────────────
# Webcam capture thread
# ─────────────────────────────────────────────────────────────────────────────

def capture_loop(camera_index: int, width: int, height: int,
                 jpeg_quality: int = 85) -> None:
    """
    Continuously reads frames from the webcam, JPEG-encodes them, and pushes
    them into the shared FrameBuffer.  Runs in a daemon thread.
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)   # CAP_DSHOW = faster init on Windows
    if not cap.isOpened():
        # Retry without backend hint (Linux / macOS)
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log.error("Cannot open camera index %d. Try --camera 1 or 2.", camera_index)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Keep buffer minimal → lower latency

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Camera %d opened at %d×%d", camera_index, actual_w, actual_h)

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame grab failed — retrying in 50 ms …")
            time.sleep(0.05)
            continue

        # Stamp a small "ESP32-CAM SIM" watermark so you can tell it apart
        # from a direct-webcam feed during testing.
        _stamp_watermark(frame)

        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            _buffer.put(buf.tobytes())


def _stamp_watermark(frame: np.ndarray) -> None:
    """Draw a subtle watermark in the top-right corner."""
    h, w = frame.shape[:2]
    text  = "ESP32-CAM SIM"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = w - tw - 8
    y = th + 6
    # Dark shadow for readability on any background
    cv2.putText(frame, text, (x + 1, y + 1), font, scale, (0, 0, 0),   thick + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x,     y),     font, scale, (0, 220, 180), thick,    cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP request handler  —  mirrors the ESP32-CAM web server routes
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ESP32-CAM Simulator</title>
  <style>
    body {{ font-family: monospace; background: #111; color: #0f0; margin: 40px; }}
    h1   {{ color: #0cf; }}
    a    {{ color: #0f9; }}
    img  {{ border: 2px solid #0f0; margin-top: 16px; max-width: 640px; }}
  </style>
</head>
<body>
  <h1>ESP32-CAM Simulator</h1>
  <p>Stream endpoint: <a href="/stream">/stream</a></p>
  <p>Serving live MJPEG — connect your detector to:
     <code>http://localhost:{port}/stream</code></p>
  <img src="/stream" alt="Live feed">
</body>
</html>
"""

_BOUNDARY = b"frame"
_PART_HEADER_TEMPLATE = (
    b"--frame\r\n"
    b"Content-Type: image/jpeg\r\n"
    b"Content-Length: {length}\r\n"
    b"\r\n"
)


class ESP32CamHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves a real-ESP32-CAM-compatible MJPEG stream."""

    # Suppress per-request access log noise — we have our own logging.
    def log_message(self, fmt: str, *args) -> None:  # type: ignore[override]
        pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/stream", "/mjpeg/1"):
            self._serve_stream()
        elif self.path in ("/", "/index.html"):
            self._serve_status()
        else:
            self.send_error(404, "Not Found")

    # ── Status page ───────────────────────────────────────────────────────────

    def _serve_status(self) -> None:
        port = self.server.server_address[1]
        body = _STATUS_HTML.format(port=port).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ── MJPEG stream ──────────────────────────────────────────────────────────

    def _serve_stream(self) -> None:
        self.send_response(200)
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={_BOUNDARY.decode()}",
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        log.info("Client %s connected to stream", self.client_address[0])
        frames_sent = 0

        try:
            while True:
                jpeg = _buffer.get(timeout=3.0)
                if jpeg is None:
                    continue

                header = _PART_HEADER_TEMPLATE.replace(
                    b"{length}", str(len(jpeg)).encode()
                )
                self.wfile.write(header + jpeg + b"\r\n")
                self.wfile.flush()
                frames_sent += 1

        except (BrokenPipeError, ConnectionResetError):
            pass   # Client disconnected — normal
        except Exception as exc:
            log.debug("Stream error for %s: %s", self.client_address[0], exc)
        finally:
            log.info(
                "Client %s disconnected after %d frames",
                self.client_address[0], frames_sent,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Threaded HTTP server
# ─────────────────────────────────────────────────────────────────────────────

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Handle each client connection in its own thread."""
    daemon_threads = True
    allow_reuse_address = True


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulated ESP32-CAM MJPEG stream server",
    )
    p.add_argument(
        "--port", "-p",
        type=int,
        default=config.ESP32CAM_SIM_PORT,
        help=f"HTTP port to listen on (default: {config.ESP32CAM_SIM_PORT})",
    )
    p.add_argument(
        "--camera", "-c",
        type=int,
        default=config.CAMERA_INDEX,
        help=f"Webcam index to capture from (default: {config.CAMERA_INDEX})",
    )
    p.add_argument(
        "--quality", "-q",
        type=int,
        default=85,
        metavar="1-100",
        help="JPEG compression quality (default: 85)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Start webcam capture in background
    capture_thread = threading.Thread(
        target=capture_loop,
        args=(args.camera, config.FRAME_WIDTH, config.FRAME_HEIGHT, args.quality),
        daemon=True,
        name="webcam-capture",
    )
    capture_thread.start()

    # Brief wait to let the capture thread open the camera
    log.info("Waiting for camera to initialise …")
    time.sleep(1.5)

    # Start HTTP server
    server = ThreadedHTTPServer(("0.0.0.0", args.port), ESP32CamHandler)

    log.info("=" * 60)
    log.info("ESP32-CAM Simulator running")
    log.info("  Stream URL : http://localhost:%d/stream", args.port)
    log.info("  Status page: http://localhost:%d/", args.port)
    log.info("  Camera     : index %d", args.camera)
    log.info("  Press Ctrl+C to stop")
    log.info("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutdown requested — stopping server …")
    finally:
        server.shutdown()
        log.info("Server stopped.")


if __name__ == "__main__":
    main()
