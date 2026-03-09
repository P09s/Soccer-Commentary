# 🎬 AI-Powered Sports Commentary System

Automatically analyse any sports video and overlay AI-generated play-by-play commentary,
then finish with a tech-stack credits slide — all powered by **Claude Vision**.

---

## How It Works

```
Input video
    │
    ▼
[OpenCV] Read all frames
    │
    ▼  every 2 s
[Frame sampler] Select key frames
    │
    ▼  4 frames per call
[Claude claude-sonnet-4 Vision] Identify sport & detect events
    │   • passes, shots, blocks, steals …
    │   • scored as low / medium / high importance
    ▼
[Event mapper] Assign commentary to frame ranges (3.5 s display window)
    │
    ▼
[OpenCV] Overlay styled commentary + timestamp + progress bar
    │
    ▼
[Tech slide] 6-second animated credits slide
    │
    ▼
Output video  (original_name_commentary.mp4)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows PowerShell
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Run

```bash
# Auto-generate output filename  →  game_commentary.mp4
python sport_commentary.py game.mp4

# Specify output filename
python sport_commentary.py game.mp4 game_with_commentary.mp4
```

---

## Supported Inputs

Any video format that OpenCV can read:
`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, and more.

Works with **any sport**: football, basketball, soccer, tennis, cricket, hockey, etc.
The AI identifies the sport automatically from the footage.

---

## Commentary Examples

| Importance | Example event |
|---|---|
| 🔥 HIGH | "Striker fires a shot into the top corner" |
| ▶ MEDIUM | "Midfielder threads a through-ball to the winger" |
| · LOW | "Defender repositions to cover the near post" |

---

## Configuration (top of `sport_commentary.py`)

| Constant | Default | Meaning |
|---|---|---|
| `SAMPLE_EVERY_N_SECONDS` | `2.0` | How often a frame is sent to Claude |
| `BATCH_SIZE` | `4` | Frames per API call |
| `COMMENTARY_DISPLAY_SECS` | `3.5` | How long each caption stays on screen |
| `TECH_SLIDE_DURATION_SECS` | `6` | Length of the credits slide |
| `JPEG_QUALITY` | `70` | JPEG quality for API frames (lower = faster/cheaper) |

---

## Tech Stack

| Component | Library / Service |
|---|---|
| Video I/O & rendering | **OpenCV 4.x** |
| AI vision & commentary | **Claude claude-sonnet-4** (Anthropic) |
| Language | **Python 3.10+** |
| Numerical computing | **NumPy** |
| API client | **anthropic-sdk-python** |

---

## Output

The processed video contains:
- **Original footage** — untouched quality
- **Commentary overlays** — styled captions at the bottom with colour-coded importance
- **Timestamp** — HH:MM top-right corner
- **Progress bar** — thin green bar at the bottom edge
- **Tech stack slide** — animated 6-second credits at the end
