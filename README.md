# 🎬 AI-Powered Sports Commentary System

Automatically analyze sports footage (optimized for soccer/football), track player movements, and generate an AI-driven text and audio play-by-play commentary. Finished with an animated tech-stack credits slide — all processed locally using **YOLOv8** and **ByteTrack**.

---

## How It Works

```text
Input video
    │
    ▼
[OpenCV] Read frames (Memory-safe stream-to-disk architecture)
    │
    ▼  
[YOLOv8 + ByteTrack] Detect players & ball, assign persistent IDs (Player A, B, C...)
    │   
    ▼
[Physics Engine] Analyze spatial possession and velocity vectors
    │   • Calculates distance from ball to player's feet
    │   • Detects passes, intercepts, and strikes based on speed and ID changes
    ▼
[Event Engine] Generate commentary strings & trigger TTS audio (gTTS)
    │
    ▼
[OpenCV] Overlay styled commentary banners + tracking trails
    │
    ▼
[Tech Slide] Append 5-second animated credits slide
    │
    ▼
[FFmpeg] Merge silent video stream with generated TTS audio track
    │
    ▼
Output video  (original_name_smart_commentary.mp4)

```

---

## Quick Start

### 1. Install Python Dependencies

```bash
pip install ultralytics opencv-python numpy gTTS

```

### 2. Install FFmpeg (System Requirement)

This project requires `ffmpeg` to merge the generated audio commentary with the video track.

* **macOS:** `brew install ffmpeg`
* **Windows:** Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add to PATH.
* **Linux:** `sudo apt install ffmpeg`

### 3. Run

```bash
# Auto-generate output filename  →  game_smart_commentary.mp4
python smart_commentary.py game.mp4

# Specify custom output filename
python smart_commentary.py game.mp4 custom_output.mp4

```

> **Hardware Acceleration:** The system automatically detects and utilizes Apple Silicon (MPS) or NVIDIA (CUDA) if available via PyTorch to drastically speed up rendering for longer (3+ minute) videos.

---

## Supported Inputs

Any video format that OpenCV can read:
`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, and more.

**Optimized Sport:** This system's physics engine is specifically hardcoded for **Soccer/Football**. It calculates possession based on the distance between the ball and the bottom bounding-box coordinates (the player's feet).

---

## Commentary Examples

| Importance | Example event |
| --- | --- |
| 🔥 HIGH | "Player B steals the ball from Player A!" <br>

<br> "Player C strikes it hard!" |
| ▶ MEDIUM | "Player A plays a forward pass." |
| · LOW | "Player D takes control of the ball." |

---

## Configuration (Top of `smart_commentary.py`)

| Constant | Default | Meaning |
| --- | --- | --- |
| `YOLO_MODEL` | `"yolov8n.pt"` | Weights file (nano for speed, switch to `yolov8m.pt` for accuracy) |
| `POSSESSION_THRESH` | `80` | Max pixel distance from feet to ball to claim possession |
| `PASS_SPEED_THRESH` | `15` | Minimum pixel velocity per frame to register a pass/shot |
| `COMMENTARY_DISPLAY_SECS` | `2.5` | How long each visual caption stays on screen |
| `TECH_SLIDE_DURATION_SECS` | `5` | Length of the animated credits slide at the end |

---

## Tech Stack

| Component | Library / Technology |
| --- | --- |
| AI Vision & Object Detection | **YOLOv8** (Ultralytics) |
| Persistent ID Tracking | **ByteTrack** |
| Video I/O & Image Processing | **OpenCV 4.x** |
| Speech Synthesis / TTS | **gTTS** (Google Text-to-Speech) |
| Audio/Video Multiplexing | **FFmpeg** |
| Language | **Python 3** |

---

## Output Features

The processed video contains:

* **Original footage** — Processed via memory-safe disk streaming to prevent RAM overflow.
* **Player & Ball Tracking** — Dynamic bounding circles, persistent ID tags (Player A, B, etc.), and movement trails.
* **Commentary Overlays** — Professional, color-coded text banners appearing at the bottom of the screen during key events.
* **Audio Broadcast** — Real-time synthesized voice commentary perfectly synced to the video events.
* **Tech Stack Slide** — Animated credits slide appended to the final output.