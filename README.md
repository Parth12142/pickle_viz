# 🏓 Pickleball Vision AI - Player & Ball Tracking Pipeline

An AI-powered computer vision system designed to analyze pickleball match footage. It accurately detects, tracks, and visualizes **players** and the **ball** in real time, delivering broadcast-quality analytics and trajectory visualization.

---

## 🌟 Key Features

### 1. 🏃 Elite Player Tracking (`tracking/player_tracker.py`)
- **Hybrid Cost Association**: Combines Intersection-over-Union (IoU) with normalized center distance to handle fast player crossovers without identity swaps.
- **Persistent Canonical Slots**: Keeps player IDs locked to canonical tags (`P1`, `P2`, `P3`, `P4`) instead of fragile transient IDs.
- **Court Perspective Locking**: Distinguishes near-court and far-court players to prevent misidentifications across the net.
- **Occlusion Resilience**: Holds active player memory for up to 2.5 seconds (75 frames) across serves, swings, and net occlusions.
- **Footstep Movement Trails**: Renders smooth fading paths under each player's feet illustrating court coverage.
- **Anti-Aliased Badging**: Modern broadcast-style player pill badges and ground halo indicators.

### 2. ⚡ Precision Ball Trajectory Tracking (`detection/ball_detector.py`)
- **Custom Trained YOLO Model**: Fine-tuned on Roboflow Universe pickleball datasets (`weights/ball_best.pt`) running at high-resolution input (`imgsz: 1312`).
- **Kinematic Velocity Gating**: Filters rogue candidates using physical momentum vectors $\vec{v} = (v_x, v_y)$ to eliminate false positives (shoes, court lines, reflections).
- **Parabolic Arc Interpolation**: Fills detection gaps (1–5 frames) along the true gravitational ballistic arc ($sag = 4\alpha(1-\alpha)g$).
- **Dynamic Speed-Responsive Laser Trail**:
  - 🔥 **Fast Smash (> 22 px/f)**: Vibrant Electric Orange glow.
  - ⚡ **Rally Play (10–22 px/f)**: Neon Chartreuse Laser Yellow glow.
  - 💧 **Drop / Dink (< 10 px/f)**: Ice Cyan laser glow.
- **High-Visibility Reticle**: Dual-layer circular tracking reticle with center pip.

### 3. 📺 Broadcast Analytics HUD
- Live translucent telemetry bar displaying:
  - Match timecode and frame counter.
  - Active players tracked count.
  - Ball tracking status indicator.

---

## 📁 Repository Structure

```
├── data/
│   └── clips/
│       ├── match.mp4                 # Input match footage
│       └── rally_sample.mp4          # 10s rally test clip
├── outputs/
│   ├── rally_sample_players-ball.mp4 # Tracked rally output with HUD & laser trail
│   └── match_players-ball.mp4        # Full annotated match video
├── weights/
│   └── ball_best.pt                  # Fine-tuned YOLO ball detection weights
├── tracking/
│   └── player_tracker.py             # Advanced player tracking engine
├── detection/
│   └── ball_detector.py              # Ball detection & parabolic trajectory engine
├── render/
│   └── overlay.py                    # Visual elements & overlay renderers
├── results/                          # Performance benchmarks, confusion matrices, and run logs
├── config.yaml                       # Central pipeline configuration
├── requirements.txt                  # Python dependencies
├── run.py                            # Main execution CLI
└── video_io.py                       # FFmpeg H.264 CRF 18 video reader/writer
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Parth12142/pickleball-vision-ai.git
cd pickleball-vision-ai

# Create virtual environment
python -m venv .venv

# Activate on Linux/macOS:
source .venv/bin/activate
# Or on Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Pipeline
Run with default configuration:
```bash
python run.py
```

Run on a specific clip:
```bash
python run.py --clip data/clips/rally_sample.mp4
```

Processed videos will be saved directly into `outputs/` with full analytics logs in `results/runs/`.
