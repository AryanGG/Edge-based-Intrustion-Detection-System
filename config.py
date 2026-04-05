# =============================================================================
# config.py — Edge-Based Intrusion Detection System Configuration
# =============================================================================
# Edit these values to customise the system behaviour without touching any
# other source file.
# =============================================================================

# ── ESP32-CAM Stream ─────────────────────────────────────────────────────────
# URL the detection script connects to.  When using the simulator this is
# localhost.  To deploy on real hardware, replace with the ESP32-CAM's IP:
#   ESP32CAM_STREAM_URL = "http://192.168.1.42/stream"
ESP32CAM_STREAM_URL: str = "http://localhost:8080/stream"

# Port the simulation server (esp32cam_sim.py) listens on.
ESP32CAM_SIM_PORT: int = 8080

# ── Simulation Camera ─────────────────────────────────────────────────────────
# Webcam index used by esp32cam_sim.py to capture frames for the fake stream.
# (Not used by ids_main.py — it always reads from ESP32CAM_STREAM_URL.)
CAMERA_INDEX: int = 0          # 0 = default webcam; try 1 or 2 for external cams

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME: str = "yolov8n.pt"          # YOLOv8-nano (auto-downloaded on first run)
CONFIDENCE_THRESHOLD: float = 0.40     # Minimum detection confidence [0.0 – 1.0]

# ── Region of Interest (ROI) ─────────────────────────────────────────────────
# Pixel coordinates of the virtual boundary rectangle drawn on the live feed.
# Format: (x1, y1) = top-left corner,  (x2, y2) = bottom-right corner.
# Defaults cover the centre-left quarter of a 640×480 frame.
ROI_X1: int = 100
ROI_Y1: int = 50
ROI_X2: int = 540
ROI_Y2: int = 430

# ── Output Paths ─────────────────────────────────────────────────────────────
SNAPSHOTS_DIR: str = "snapshots"        # Folder for saved intrusion images
LOG_FILE: str = "intrusion_log.csv"     # CSV log of all intrusion events
BEEP_FILE: str = "beep.wav"             # Alert sound (auto-generated if absent)

# ── Behaviour ─────────────────────────────────────────────────────────────────
BEEP_COOLDOWN_SECONDS: float = 3.0     # Min seconds between consecutive beeps
SNAPSHOT_COOLDOWN_SECONDS: float = 3.0  # Min seconds between consecutive snapshots

# ── Display ───────────────────────────────────────────────────────────────────
WINDOW_TITLE: str = "Edge IDS — Live Feed"
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
