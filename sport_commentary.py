"""
Smart Sports Commentary System
Powered by YOLOv8, ByteTrack, gTTS, and ffmpeg.
"""

import cv2
import numpy as np
import subprocess
import tempfile
import os
import sys
import math
from pathlib import Path
from gtts import gTTS
from collections import deque
from ultralytics import YOLO

# ─── Configuration ────────────────────────────────────────────────────────────
YOLO_MODEL = "yolov8n.pt"  
PERSON_CLASS = 0
BALL_CLASS = 32

POSSESSION_THRESH = 80     
PASS_SPEED_THRESH = 15     
TRAIL_LENGTH = 15
COMMENTARY_DISPLAY_SECS = 2.5
TECH_SLIDE_DURATION_SECS = 5   # <--- Added duration for the ending tech slide

# ─── Utilities ────────────────────────────────────────────────────────────────

class PlayerRegistry:
    def __init__(self):
        self._map  = {}   
        self._next = 0

    def name(self, tid):
        if tid not in self._map:
            self._map[tid] = chr(ord("A") + self._next % 26)
            self._next += 1
        return f"Player {self._map[tid]}"

# ─── AI Vision & Event Engine ─────────────────────────────────────────────────

class SportsAnalyticsEngine:
    def __init__(self, fps, registry):
        self.fps = fps
        self.registry = registry
        self.model = YOLO(YOLO_MODEL)
        
        self.ball_trail = deque(maxlen=TRAIL_LENGTH)
        self.player_trails = {} 
        
        self.last_possessor_id = None
        self.ball_velocity = 0.0
        self.frame_count = 0
        self.cooldown = int(fps * 2) 
        self.last_event_frame = -999

    def process_frame(self, frame):
        self.frame_count += 1
        
        results = self.model.track(frame, persist=True, classes=[PERSON_CLASS, BALL_CLASS], 
                                   tracker="bytetrack.yaml", verbose=False)
        
        players = {} 
        ball_pos = None
        event = None
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            
            for box, cls, track_id in zip(boxes, clss, ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                tid = int(track_id)
                
                if cls == PERSON_CLASS:
                    players[tid] = (cx, cy, y2)
                    if tid not in self.player_trails:
                        self.player_trails[tid] = deque(maxlen=TRAIL_LENGTH)
                    self.player_trails[tid].append((cx, cy))
                    
                elif cls == BALL_CLASS:
                    ball_pos = (cx, cy)
                    self.ball_trail.append(ball_pos)

        if len(self.ball_trail) >= 2:
            dx = self.ball_trail[-1][0] - self.ball_trail[-2][0]
            dy = self.ball_trail[-1][1] - self.ball_trail[-2][1]
            self.ball_velocity = math.hypot(dx, dy)
        else:
            self.ball_velocity = 0.0

        current_possessor = None
        min_dist = float('inf')
        
        if ball_pos:
            bx, by = ball_pos
            for pid, (px, py, feet_y) in players.items():
                dist = math.hypot(bx - px, by - feet_y)
                if dist < POSSESSION_THRESH and dist < min_dist:
                    min_dist = dist
                    current_possessor = pid

        if (self.frame_count - self.last_event_frame) > self.cooldown:
            ts = round(self.frame_count / self.fps, 2)
            
            if current_possessor is not None and self.last_possessor_id is not None and current_possessor != self.last_possessor_id:
                p_new = self.registry.name(current_possessor)
                p_old = self.registry.name(self.last_possessor_id)
                event = {"timestamp_seconds": ts, "event": f"{p_new} steals the ball from {p_old}!", "importance": "high"}
                self.last_possessor_id = current_possessor
                self.last_event_frame = self.frame_count
                
            elif current_possessor is not None and self.last_possessor_id is None:
                p_name = self.registry.name(current_possessor)
                event = {"timestamp_seconds": ts, "event": f"{p_name} takes control of the ball.", "importance": "low"}
                self.last_possessor_id = current_possessor
                self.last_event_frame = self.frame_count
                
            elif self.last_possessor_id is not None and self.ball_velocity > PASS_SPEED_THRESH and current_possessor is None:
                p_name = self.registry.name(self.last_possessor_id)
                if self.ball_velocity > PASS_SPEED_THRESH * 1.8:
                    event = {"timestamp_seconds": ts, "event": f"{p_name} strikes it hard!", "importance": "high"}
                else:
                    event = {"timestamp_seconds": ts, "event": f"{p_name} plays a forward pass.", "importance": "medium"}
                self.last_possessor_id = None 
                self.last_event_frame = self.frame_count

        if current_possessor is not None:
            self.last_possessor_id = current_possessor

        return players, ball_pos, event

# ─── TTS & Audio ──────────────────────────────────────────────────────────────

def tts_to_wav(text, out_wav):
    mp3 = out_wav.replace(".wav", ".mp3")
    gTTS(text=text, lang="en", slow=False).save(mp3)
    subprocess.run(["ffmpeg","-y","-i",mp3,out_wav], capture_output=True, check=True)
    os.remove(mp3)
    r = subprocess.run(
        ["ffprobe","-v","error","-show_entries","format=duration",
         "-of","default=noprint_wrappers=1:nokey=1", out_wav],
        capture_output=True, text=True)
    return float(r.stdout.strip() or "2.0")

def build_audio_track(events, total_duration, tmp_dir):
    print("\n🔊 Generating TTS Broadcast…")
    base = os.path.join(tmp_dir, "silence.wav")
    subprocess.run(["ffmpeg","-y","-f","lavfi","-i","anullsrc=r=44100:cl=stereo",
                    "-t", str(total_duration), base], capture_output=True, check=True)

    clips, last_end = [], 0.0
    for ev in events:
        ts, text = ev["timestamp_seconds"], ev["event"]
        if ts < last_end: continue
        wav = os.path.join(tmp_dir, f"clip_{len(clips)}.wav")
        try:
            dur = tts_to_wav(text, wav)
            ev["tts_duration"] = dur
            clips.append((ts, wav))
            last_end = ts + dur
            print(f"   🎙 [{ts:.1f}s] {text}")
        except Exception as e:
            print(f"   ⚠ TTS Error: {e}")

    if not clips: return base

    inputs = ["-i", base]
    for _, wav in clips: inputs += ["-i", wav]

    fs, prev = "", "[0:a]"
    for i, (ts, _) in enumerate(clips):
        dm = int(ts * 1000)
        fs += f"[{i+1}:a]adelay={dm}|{dm}[d{i}];"
        fs += f"{prev}[d{i}]amix=inputs=2:duration=first:dropout_transition=0[m{i}];"
        prev = f"[m{i}]"
    fs += f"{prev}atrim=0:{total_duration},asetpts=PTS-STARTPTS[final]"

    final = os.path.join(tmp_dir, "final.wav")
    subprocess.run(["ffmpeg","-y"] + inputs + ["-filter_complex", fs, "-map","[final]", final],
                   capture_output=True, check=True)
    return final

# ─── Drawing Overlays & Tech Slide ────────────────────────────────────────────

def draw_commentary(frame, event):
    h, w = frame.shape[:2]
    text = event.get("event", "")
    if not text: return frame
    
    colors = {"high":(50,50,220), "medium":(40,170,40), "low":(200,140,30)}
    accent = colors.get(event.get("importance", "medium"))
    
    font = cv2.FONT_HERSHEY_DUPLEX
    fs = max(0.6, w/1280.0)
    thick = 2
    
    (tw, th), _ = cv2.getTextSize(text, font, fs, thick)
    pad = 15
    bx, by = 20, h - th - 40
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by-pad), (bx+tw+pad*2, by+th+pad), (10,10,10), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    cv2.rectangle(frame, (bx, by-pad), (bx+8, by+th+pad), accent, -1)
    cv2.putText(frame, text, (bx+pad+8, by+th), font, fs, (255,255,255), thick, cv2.LINE_AA)
    return frame

# --- NEW: Tech Slide Rendering ---
TECH_ITEMS = [
    ("Vision & Tracking", "YOLOv8 + ByteTrack"),
    ("Image Processing",  "OpenCV"),
    ("Speech / TTS",      "gTTS"),
    ("Media Merge",       "FFmpeg"),
    ("Language Stack",    "Python 3")
]

def make_tech_slide_frame(w, h, progress):
    frame = np.zeros((h,w,3), dtype=np.uint8)
    for y in range(h):
        t = y/h
        frame[y] = [int(20+t*20),int(5+t*10),int(35+t*25)]
    fb, fr = cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_SIMPLEX
    sb = w/1280.0
    title = "Built With Following Tech Stack"
    st = sb*1.4
    (tw,_),_ = cv2.getTextSize(title,fb,st,2)
    at = min(1.0, progress*3)
    cv2.putText(frame,title,((w-tw)//2,int(h*0.15)), fb,st,(int(60*at),int(210*at),int(255*at)),2,cv2.LINE_AA)
    
    lp = min(1.0, progress*4)
    lx1 = int(w*0.06)
    cv2.line(frame,(lx1,int(h*0.22)),(int(lx1+w*0.88*lp),int(h*0.22)),(50,200,255),2)
    
    rh, sy = int(h*0.105), int(h*0.27)
    for i, (label, tech) in enumerate(TECH_ITEMS):
        ia = min(1.0, max(0.0, (progress-i*0.15)*4))
        if ia <= 0: continue
        ry = sy + i*rh
        bg = frame.copy()
        cv2.rectangle(bg, (int(w*.05), ry-5), (int(w*.95), ry+rh-12), (50,50,70), -1)
        cv2.addWeighted(bg, ia*.35, frame, 1-ia*.35, 0, frame)
        cv2.circle(frame, (int(w*.07), ry+8), 5, tuple(int(c*ia) for c in (50,200,255)), -1)
        cv2.putText(frame, label, (int(w*.09), ry+int(rh*.6)), fb, sb*.62, tuple(int(c*ia) for c in (140,220,255)), 1, cv2.LINE_AA)
        cv2.putText(frame, tech, (int(w*.35), ry+int(rh*.6)), fr, sb*.60, tuple(int(c*ia) for c in (220,220,220)), 1, cv2.LINE_AA)
        
    fa = min(1.0, max(0.0, (progress-.7)*5))
    if fa > 0:
        footer = "AI Sports Commentary System"
        (fw,_),_ = cv2.getTextSize(footer, fr, sb*.58, 1)
        cv2.putText(frame, footer, ((w-fw)//2, int(h*.95)), fr, sb*.58, tuple(int(c*fa) for c in (100,100,120)), 1, cv2.LINE_AA)
    return frame

def generate_tech_slide(w, h, fps):
    n = int(fps * TECH_SLIDE_DURATION_SECS)
    return [make_tech_slide_frame(w, h, min(1.0, i/(fps*2.5))) for i in range(n)]

# ─── Main Pipeline (Memory Optimized for 3+ Minute Videos) ───────────────

def process_video(input_path, output_path=None):
    if not os.path.exists(input_path):
        sys.exit(f"❌ File not found: {input_path}")
        
    p = Path(input_path)
    if output_path is None:
        output_path = str(p.parent / f"{p.stem}_smart_commentary.mp4")
    silent_tmp = str(p.parent / f"{p.stem}_silent_tmp.mp4")

    print(f"\n🎬 Smart Sports Commentary — {p.name}")
    print("=" * 60)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    duration_video_only = n_frames / fps
    total_duration = duration_video_only + TECH_SLIDE_DURATION_SECS
    
    registry = PlayerRegistry()
    engine = SportsAnalyticsEngine(fps, registry)
    
    # OPTIMIZATION: Check for Apple Silicon (MPS) to speed up YOLO on MacBooks
    import torch
    if torch.backends.mps.is_available():
        print("⚡ Apple Silicon detected. Hardware acceleration enabled (MPS).")
        engine.model.to('mps')
    
    all_events = []
    active_commentary = {} 
    idx = 0

    # OPTIMIZATION: Open the VideoWriter immediately to stream to disk
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(silent_tmp, fourcc, fps, (width, height))

    print("\n⏳ Processing vision and tracking (Stream-to-Disk active)…\n")
    while True:
        ok, frame = cap.read()
        if not ok: break

        players, ball_pos, event = engine.process_frame(frame)
        
        if event:
            all_events.append(event)
            # Display commentary banner for N seconds
            end_f = min(idx + int(fps * COMMENTARY_DISPLAY_SECS), n_frames)
            for f in range(idx, end_f):
                active_commentary[f] = event

        # Draw overlays
        for pid, (cx, cy, feet_y) in players.items():
            name = registry.name(pid)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 100), -1)
            cv2.putText(frame, name, (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 2)
            
            trail = list(engine.player_trails[pid])
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i-1], trail[i], (0, 150, 50), 2)

        if ball_pos:
            cv2.circle(frame, ball_pos, 6, (0, 165, 255), -1)
            trail = list(engine.ball_trail)
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i-1], trail[i], (0, 165, 255), 2)

        if idx in active_commentary:
            frame = draw_commentary(frame, active_commentary[idx])

        # Write frame instantly, discarding it from RAM
        out.write(frame)
        idx += 1
        
        if idx % int(fps * 5) == 0:  # Update console every 5 seconds of video
            print(f"   {idx/n_frames*100:5.1f}% complete... ({idx}/{n_frames} frames)")

    cap.release()
    
    print(f"\n🖥  Appending Tech Stack Slide...")
    slide_frames = generate_tech_slide(width, height, fps)
    for f in slide_frames: 
        out.write(f)
    
    out.release() # Safely close the silent video file

    # -- Audio & Merge --
    with tempfile.TemporaryDirectory() as tmp_dir:
        final_audio = build_audio_track(all_events, total_duration, tmp_dir)
        
        print(f"\n🎧 Merging Audio & Video via ffmpeg…")
        r = subprocess.run([
            "ffmpeg", "-y",
            "-i", silent_tmp, "-i", final_audio,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
            "-shortest", output_path
        ], capture_output=True, text=True)
        
        if r.returncode != 0:
            print(f"⚠ FFmpeg merge failed: {r.stderr}")

    if os.path.exists(silent_tmp): os.remove(silent_tmp)
    print(f"\n✅ Done! Assignment ready. Saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python smart_commentary.py <video.mp4>")
        sys.exit(0)
    process_video(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)