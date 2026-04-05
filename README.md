# Edge-Based Intrusion Detection System

An **entirely on-device**, real-time intrusion detection system with **ESP32-CAM simulation** and **face recognition whitelisting**. No cloud, no external API — all AI inference runs locally.

---

## Architecture

```
┌─────────────────────────────┐   HTTP MJPEG    ┌────────────────────────────────────────────┐
│   esp32cam_sim.py           │ ──────────────► │   ids_main.py                              │
│   (Terminal 1)              │  localhost:8080  │   (Terminal 2)                             │
│                             │                  │                                            │
│  Webcam → JPEG → HTTP       │                  │  1. Read MJPEG stream                      │
│  Mimics ESP32-CAM Wi-Fi     │                  │  2. YOLOv8-nano: detect persons            │
│  broadcast                  │                  │  3. Check if person is inside ROI          │
└─────────────────────────────┘                  │  4. Face recognition whitelist check       │
         ▲ Replace stream URL with real           │     ├── Known person → teal box, no alert │
           ESP32-CAM IP to deploy on             │     └── Unknown/no face → INTRUSION       │
           actual hardware                        │  5. Alert: red overlay + beep + snapshot  │
                                                  └────────────────────────────────────────────┘
```

---

## Features

| Capability | Detail |
|---|---|
| 📡 MJPEG stream source | Simulated ESP32-CAM or real hardware — identical code path |
| 🤖 On-device AI | YOLOv8-nano person detection — no cloud, no GPU required |
| 🧑‍🤝‍🧑 Face whitelist | Known people pass through — only unknown faces trigger alerts |
| 🔴 Intrusion alert | Red banner + bounding box when an unknown person enters the ROI |
| 🔊 Audio beep | Non-blocking 880 Hz beep (auto-generated, no extra file needed) |
| 📸 Snapshot | Timestamped PNG saved to `snapshots/` on each intrusion |
| 📋 CSV log | Every intrusion event logged with timestamp and confidence |
| 🟢 Monitoring status | Green "MONITORING" shown when the scene is clear |
| 📊 Summary viewer | `ids_summary.py` prints a formatted table of all past events |

---

## Project Structure

```
.
├── esp32cam_sim.py    # 📡 Simulated ESP32-CAM MJPEG server      ← Terminal 1
├── ids_main.py        # 🔍 Main detection loop                    ← Terminal 2
├── ids_summary.py     # 📊 Past event log viewer
├── ids_utils.py       # 🛠  Shared utilities (drawing, logging, audio)
├── face_whitelist.py  # 🧑‍🤝‍🧑 Face recognition whitelist (uses dlib directly)
├── register_face.py   # 📷 Register yourself via webcam          ← use this first!
├── test_face.py       # 🔬 Standalone face recognition tester
├── ids_diagnose.py    # 🩺 Full pipeline diagnostic tool
├── config.py          # ✏️  All user-configurable settings
├── requirements.txt   # Python dependencies
├── beep.wav           # Auto-generated alert sound
├── known_faces/       # 📁 Reference photos of trusted people
├── snapshots/         # Auto-created; intrusion screenshots saved here
└── intrusion_log.csv  # Auto-created; CSV event log
```

---

## Requirements

- Python 3.9 or higher
- A working webcam
- ~500 MB disk space (PyTorch + YOLOv8 + dlib face models on first run)

---

## Setup

### 1. Create and activate a virtual environment (recommended)

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install core dependencies

```bash
pip install ultralytics opencv-python pandas "playsound==1.2.2"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install face recognition

> [!IMPORTANT]
> On Windows, install `dlib` as a **pre-built binary** using `dlib-bin` — in this exact order.
> Installing plain `dlib` will attempt a C++ source build and fail without Visual Studio.

```bash
pip install dlib-bin
pip install face-recognition-models
pip install face-recognition --no-deps
```

---

## Registering Known People (Most Important Step)

> [!CAUTION]
> Do **not** just drop a random photo into `known_faces/` — dlib's face detector is strict.
> If it cannot find a face in your photo, you will still trigger intrusion alerts.
> **Always use `register_face.py` to capture your reference photo.** It verifies the encoding works before saving.

### The right way — use `register_face.py`

```bash
python register_face.py
```

This opens your webcam live, waits until your face is clearly detected, then:
- **Auto-captures after 2 seconds** of stable face detection
- **OR press `SPACE`** to capture immediately
- Verifies dlib can encode the face before saving
- Saves a verified photo to `known_faces/your_name.jpg`
- Confirms the whitelist loaded successfully

**Tips for a good capture:**
- Sit in good lighting facing the camera directly
- Keep your face at a normal webcam distance (not too close, not too far)
- Don't wear sunglasses or anything obscuring the face

### Adding more reference photos (optional but improves accuracy)

The more angles and lighting conditions you register, the more robust recognition becomes. You can run `register_face.py` multiple times. Use subdirectories if you want to group multiple shots per person:

```
known_faces/
    aryan.jpg             ← registered via register_face.py  (required: 1 photo minimum)
    Aryan/
        front.jpg         ← optional additional shots
        glasses.jpg
        different_light.jpg
```

### How the whitelist works

When a person enters the ROI:

```
Person in ROI
  │
  ▼
Crop upper 50% of YOLO bounding box (face/head region)
  │  └─ Fallback: full bounding box if no face found in upper crop
  │
  ▼
dlib HOG detector: is a face visible?
  ├── No  →  Face not visible (turned away, dark, too far) → INTRUSION (fail-safe)
  └── Yes →  Compute 128-d face descriptor via dlib ResNet
               │
               ▼
               Compare to all whitelist encodings
               ├── Distance ≤ tolerance → Known person → SAFE (teal box + name)
               └── Distance >  tolerance → Unknown person → INTRUSION
```

**The fail-safe rule:** if a face cannot be clearly detected (turned away, poor light, too small), the system defaults to treating the person as an unknown intruder. This prevents whitelisted people from bypassing the alarm by hiding their face.

---

## Running the System

### Step 1 — Terminal 1: Start the ESP32-CAM simulator

```bash
python esp32cam_sim.py
```

The stream will be live at `http://localhost:8080/stream`. You can preview it in a browser.

### Step 2 — Terminal 2: Start the intrusion detector

```bash
python ids_main.py
```

Startup output will confirm the whitelist loaded:

```
[12:00:00] INFO  Loading face whitelist from 'known_faces/' ...
[12:00:00] INFO    Registered 'Aryan'  <-  aryan.jpg
[12:00:00] INFO  Whitelist loaded: 1 person(s), 1 encoding(s)
[12:00:00] INFO  Stream connected. Press 'q' to quit.
```

### What you will see on screen

| Situation | Bounding box | Banner | Alert fires? |
|---|---|---|---|
| Person outside ROI | 🟢 Green | ● MONITORING | No |
| Whitelisted person inside ROI | 🔵 Teal + name | ● MONITORING | No |
| Unknown person inside ROI | 🔴 Red + "Unknown" | ⚠ INTRUSION DETECTED | Yes |
| No people | — | ● MONITORING | No |

Press **`q`** in the live-feed window to exit, then `Ctrl+C` in Terminal 1.

### View past intrusion events

```bash
python ids_summary.py
```

---

## Troubleshooting & Diagnostics

### Still triggering alerts even after registering?

**Step 1 — Test face recognition in isolation (no YOLO, no stream)**

```bash
python test_face.py
```

This opens the webcam directly and shows:
- Whether dlib detects your face in the live frame
- The raw distance score to your registered encoding
- `MATCH` in green if recognition worked, or the actual distance if it didn't

Watch the terminal output. It will tell you exactly what value to set for `FACE_MATCH_TOLERANCE` in `config.py` if your distance is just slightly over the threshold.

**Step 2 — Full pipeline diagnostic (YOLO + stream + face)**

```bash
python ids_diagnose.py
```

Shows YOLO detections, the face crop region (yellow box), and live distance scores all in one view. Also saves face crops to `debug_crops/` for inspection.

### Common issues

| Problem | Cause | Fix |
|---|---|---|
| Whitelist empty at startup | dlib couldn't detect a face in your reference photo | Re-register using `register_face.py` |
| `MATCH` in test_face but still alerting in ids_main | MJPEG compression degrading image quality | Lower `FACE_MATCH_TOLERANCE` to ~0.65 in `config.py` |
| `NO FACE DETECTED` in test_face | Poor lighting or extreme angle | Improve lighting; face the camera directly |
| Distance shows e.g. `0.72` but tolerance is `0.6` | Photo or live lighting mismatch | Raise `FACE_MATCH_TOLERANCE` to `dist + 0.03` |
| `Could not connect to stream` | Simulator not running | Start `esp32cam_sim.py` first |
| `Cannot open camera index 0` | Wrong webcam index | Run `esp32cam_sim.py --camera 1` or `--camera 2` |
| `dlib build fails` | Tried to build from source | Use `pip install dlib-bin` (pre-built binary) |
| No audio | playsound version mismatch | Ensure `playsound==1.2.2` is installed |

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `ESP32CAM_STREAM_URL` | `http://localhost:8080/stream` | URL the detector reads frames from |
| `ESP32CAM_SIM_PORT` | `8080` | Port the simulator listens on |
| `CAMERA_INDEX` | `0` | Webcam used by the simulator |
| `CONFIDENCE_THRESHOLD` | `0.40` | YOLO minimum person detection confidence |
| `ROI_X1/Y1/X2/Y2` | `100, 50, 540, 430` | Virtual boundary in pixel coordinates |
| `FACE_RECOGNITION_ENABLED` | `True` | Set `False` to disable face ID (faster) |
| `KNOWN_FACES_DIR` | `known_faces` | Folder with reference photos |
| `FACE_MATCH_TOLERANCE` | `0.6` | Match strictness: lower = stricter (0.0–1.0) |
| `FACE_DETECTION_MODEL` | `"hog"` | `"hog"` (CPU) or `"cnn"` (GPU) |

---

## Deploying on Real ESP32-CAM Hardware

**One line change** in `config.py`:

```python
# Change this:
ESP32CAM_STREAM_URL: str = "http://localhost:8080/stream"
# To this (use your device's actual IP):
ESP32CAM_STREAM_URL: str = "http://192.168.1.42/stream"
```

Then run only `python ids_main.py` — the simulator is not needed for real hardware.

Flash your ESP32-CAM with the standard **CameraWebServer** Arduino sketch. The stream endpoint `/stream` is provided by that sketch out of the box.

---

## Notes on the Face Recognition Implementation

This project uses **dlib directly** (via `dlib-bin`) rather than the `face_recognition` wrapper library. The `face_recognition` package (v1.3.0) has not been updated to support dlib 20.x and throws a `TypeError` at runtime. By calling dlib's HOG detector, shape predictor, and ResNet face descriptor model directly, we get identical accuracy with full compatibility.

The pre-trained model weights are provided by the `face_recognition_models` package and are unchanged.

---

## License

MIT — free to use, modify, and distribute.
