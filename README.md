# Edge-Based Intrusion Detection System

An **entirely on-device**, real-time intrusion detection system that simulates the exact workflow of a physical **ESP32-CAM** deployment — no cloud, no external API, all AI inference runs locally.

---

## Architecture

```
┌─────────────────────────────────────┐      HTTP MJPEG      ┌──────────────────────────────┐
│      esp32cam_sim.py                │ ──────────────────►  │      ids_main.py             │
│  (Terminal 1)                       │  http://localhost:    │  (Terminal 2)                │
│                                     │  8080/stream          │                              │
│  Laptop webcam → JPEG-encode        │                       │  Read MJPEG stream           │
│  → serve as MJPEG over HTTP         │                       │  → YOLOv8-nano inference     │
│                                     │                       │  → ROI overlap check         │
│  Mimics ESP32-CAM Wi-Fi broadcast   │                       │  → Alerts / logs / snapshots │
└─────────────────────────────────────┘                       └──────────────────────────────┘
         ▲ In real deployment, replace this with a
           physical ESP32-CAM on your Wi-Fi network
```

The detector (`ids_main.py`) **never touches the webcam directly**.
It only speaks HTTP — exactly as it would to a real ESP32-CAM over Wi-Fi.
Swapping to real hardware requires changing **one line** in `config.py`.

---

## Features

| Capability | Detail |
|---|---|
| 📡 MJPEG stream source | Simulated ESP32-CAM or real hardware — identical code path |
| 🤖 On-device AI | YOLOv8-nano (ultralytics) — no cloud, no GPU required |
| 🔴 Intrusion alert | Red banner + bounding box when a person enters the ROI |
| 🔊 Audio beep | Non-blocking 880 Hz beep (auto-generated WAV, no file needed) |
| 📸 Snapshot | Timestamped PNG saved to `snapshots/` on each intrusion |
| 📋 CSV log | Every event logged with timestamp, confidence and snapshot path |
| 🟢 Monitoring status | Green "MONITORING" shown when the scene is clear |
| 📊 Summary viewer | `ids_summary.py` prints a formatted table of all past events |

---

## Project Structure

```
.
├── esp32cam_sim.py    # 📡 Simulated ESP32-CAM MJPEG server  ← Terminal 1
├── ids_main.py        # 🔍 Main detection loop               ← Terminal 2
├── ids_summary.py     # 📊 Log viewer
├── ids_utils.py       # 🛠  Shared utilities (drawing, logging, audio)
├── config.py          # ✏️  All user-configurable settings live here
├── requirements.txt   # Python dependencies
├── beep.wav           # Auto-generated alert sound
├── snapshots/         # Auto-created; intrusion screenshots saved here
└── intrusion_log.csv  # Auto-created; CSV event log
```

---

## Requirements

- Python 3.9 or higher
- A working webcam
- ~150 MB disk space (PyTorch + YOLOv8-nano model, downloaded on first run)

---

## Setup

### 1. Create and activate a virtual environment (recommended)

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note** — `playsound` is pinned to `1.2.2` because versions ≥ 1.3.0 have known playback issues on Windows.

---

## Running the System

The system uses a **two-terminal workflow** that mirrors real hardware deployment.

### Terminal 1 — Start the ESP32-CAM simulator

```bash
python esp32cam_sim.py
```

This opens the laptop webcam and serves it as a live MJPEG stream on:

```
http://localhost:8080/stream
```

You should see output like:
```
[12:00:00] INFO  ESP32-CAM Simulator running
[12:00:00] INFO    Stream URL : http://localhost:8080/stream
[12:00:00] INFO    Status page: http://localhost:8080/
[12:00:00] INFO    Press Ctrl+C to stop
```

You can also open `http://localhost:8080/` in a browser to preview the live feed.

### Terminal 2 — Start the intrusion detector

```bash
python ids_main.py
```

The detector connects to the simulator's stream URL and begins inference:

```
[INFO] Connecting to ESP32-CAM stream: http://localhost:8080/stream
[INFO] Stream connected. Press 'q' to quit.
```

A window titled **"Edge IDS — Live Feed"** will open with the annotated live view.

- Walk into the ROI zone (default: green rectangle) → red **INTRUSION DETECTED** banner fires
- Press **`q`** in the live-feed window to exit cleanly

### Optional flags for the simulator

```bash
python esp32cam_sim.py --port 9090     # Use a different port
python esp32cam_sim.py --camera 1      # Use a different webcam
python esp32cam_sim.py --quality 70    # Lower JPEG quality (higher FPS)
```

If you change the port, update `ESP32CAM_STREAM_URL` in `config.py` to match.

### View past intrusion events

```bash
python ids_summary.py
```

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `ESP32CAM_STREAM_URL` | `http://localhost:8080/stream` | URL the detector reads from |
| `ESP32CAM_SIM_PORT` | `8080` | Port the simulator listens on |
| `CAMERA_INDEX` | `0` | Webcam used by the simulator |
| `ROI_X1/Y1/X2/Y2` | `100,50,540,430` | Virtual boundary (pixel coords) |
| `CONFIDENCE_THRESHOLD` | `0.40` | Minimum detection confidence |
| `BEEP_COOLDOWN_SECONDS` | `3.0` | Min gap between audio alerts |
| `SNAPSHOT_COOLDOWN_SECONDS` | `3.0` | Min gap between saved snapshots |

---

## Deploying on Real ESP32-CAM Hardware

Switching from the simulator to a physical ESP32-CAM requires **a single line change**:

### 1. Flash your ESP32-CAM with the standard CameraWebServer sketch
The default Arduino sketch exposes a stream at `http://<device-ip>/stream`.

### 2. Find the ESP32-CAM's IP address
It is printed in the Arduino Serial Monitor after connecting to Wi-Fi.

### 3. Update `config.py`

```python
# Before (simulator)
ESP32CAM_STREAM_URL: str = "http://localhost:8080/stream"

# After (real hardware)
ESP32CAM_STREAM_URL: str = "http://192.168.1.42/stream"   # ← your device's IP
```

### 4. Run only the detector (no simulator needed)

```bash
python ids_main.py
```

The detector connects to the real camera over Wi-Fi — no code changes required.

---

## How Detection Works

```
MJPEG stream (HTTP)
     │
     ▼
cv2.VideoCapture(stream_url)   ← identical API for local sim or real ESP32-CAM
     │
     ▼
YOLOv8-nano — class 0 (person), conf ≥ threshold
     │
     ▼
bbox_overlaps_roi()
  ├── YES → INTRUSION
  │           ├── Red banner + red bounding box on frame
  │           ├── Save PNG snapshot to snapshots/  (throttled every 3 s)
  │           ├── Append row to intrusion_log.csv
  │           └── Play 880 Hz beep              (throttled every 3 s)
  └── NO  → Green "MONITORING" banner
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not connect to stream` | Start `esp32cam_sim.py` first; check port matches config |
| `Cannot open camera index 0` | Run `esp32cam_sim.py --camera 1` or `--camera 2` |
| Detector shows old/frozen frame | Restart the simulator; check USB / Wi-Fi connection |
| No audio | Ensure `playsound==1.2.2` is installed; check system audio |
| Low FPS | Lower `--quality` on the simulator; or reduce `FRAME_WIDTH`/`FRAME_HEIGHT` |
| False positives | Raise `CONFIDENCE_THRESHOLD` to e.g. `0.55` in `config.py` |
| Model download fails | Internet needed on first run (~6 MB); cached in `~/.ultralytics/` |

---

## License

MIT — free to use, modify, and distribute.
