# crybaby — Baby Monitor & Dashboard

Detects baby cry, caregiver yelling, and loud noise events in Xiaomi (or any)
camera recordings, and presents them in a **web dashboard** with a day-by-day
timeline — similar to Xiaomi Home — where events are shown inline so you can
jump directly to any moment in the video.

No training data required — powered by **YAMNet** (Google, pre-trained on AudioSet).

---

## How it works

```
Audio file
   │
   ▼
librosa  ──►  16 kHz mono waveform
   │
   ▼
YAMNet   ──►  521-class softmax scores  (one row per ~0.96 s frame, hop 0.48 s)
   │
   ▼
Filter   ──►  keep frames where "Baby cry, infant cry" score ≥ threshold
   │
   ▼
Merge    ──►  join nearby frames into continuous episodes
   │
   ▼
Report   ──►  timestamped list of cry episodes
```

### Why YAMNet?

- Pre-trained on **AudioSet** (2 million YouTube clips, 521 classes).
- One of those classes is **"Baby cry, infant cry"** — no fine-tuning needed.
- Robust to real-world background noise because it was trained on diverse,
  noisy internet video.
- Sliding-window design automatically handles files of any length.

### Noise handling

YAMNet assigns a separate probability to every sound class in every frame.
Baby cry competes with traffic, TV, speech, etc. in a softmax layer, so the
model naturally suppresses frames that are dominated by other sounds.  
You can tune `--threshold` and `--min-duration` to match a noisier environment.

---

## Installation

```bash
# 1. Clone / open the project
cd crybaby

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3)?**  
> Replace `tensorflow` in `requirements.txt` with `tensorflow-macos` and
> `tensorflow-metal` for GPU acceleration.

---

## Usage

```bash
# Analyse a recording (default settings)
python main.py baby_monitor.wav

# Lower the threshold to catch quiet or muffled cries
python main.py recording.mp3 --threshold 0.15

# Stricter detection in a noisier environment
python main.py noisy_room.wav --threshold 0.40

# Save the report to a text file
python main.py recording.wav --output report.txt

# Show per-frame confidence scores
python main.py recording.wav --verbose
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold FLOAT` | `0.25` | Per-frame confidence cut-off (0–1). Lower = more sensitive. |
| `--merge-gap SECONDS` | `1.5` | Gap between flagged frames that are still joined into one episode. |
| `--min-duration SECONDS` | `0.5` | Discard episodes shorter than this (removes false-positive blips). |
| `--output FILE` | — | Also write the report to a text file. |
| `--verbose` / `-v` | — | Show per-frame scores and debug logs. |

### Example output

```
────────────────────────────────────────────────────────────
  Baby Cry Detection Report
────────────────────────────────────────────────────────────
  File      : baby_monitor.wav
  Threshold : 0.25
────────────────────────────────────────────────────────────
  Result    : 3 cry episode(s) found.

  Episode  1 : [00:12.48 → 00:26.40]  duration=13.9s  peak=0.82  mean=0.61
  Episode  2 : [01:44.64 → 01:51.36]  duration=6.7s   peak=0.71  mean=0.54
  Episode  3 : [03:02.40 → 03:09.12]  duration=6.7s   peak=0.68  mean=0.51
────────────────────────────────────────────────────────────
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | No cry detected (quiet recording). |
| `1` | One or more cry episodes detected. |

This makes it easy to integrate with shell scripts or cron jobs.

---

## Using the detector as a Python library

```python
from detector import BabyCryDetector

det = BabyCryDetector(threshold=0.25, merge_gap=1.5, min_duration=0.5)
episodes = det.detect("recording.wav")

for ep in episodes:
    print(ep)              # human-readable line
    print(ep.start)        # float seconds
    print(ep.end)          # float seconds
    print(ep.peak_confidence)
```

---

## Tuning guide

| Situation | Recommended change |
|-----------|--------------------|
| Missing quiet cries | Lower `--threshold` (e.g. 0.15) |
| Too many false alarms in noisy room | Raise `--threshold` (e.g. 0.40) |
| Short hiccup-like false positives | Raise `--min-duration` (e.g. 1.0) |
| Brief cries split into two episodes | Raise `--merge-gap` (e.g. 3.0) |

---

## Supported audio formats

Any format supported by **librosa** / **soundfile**: `.wav`, `.mp3`, `.flac`,
`.ogg`, `.m4a`, `.aac`, and more.  For `.mp3` and `.m4a` you may need
`ffmpeg` installed on your system (`brew install ffmpeg` on macOS).

---

## Dashboard

### Quick start

```bash
# Install dependencies (includes FastAPI + uvicorn)
pip install -r requirements.txt
brew install ffmpeg          # needed to extract audio from .mp4

# Launch the dashboard (opens browser automatically)
python run_dashboard.py
```

Open [http://localhost:7860](http://localhost:7860).

### Xiaomi camera filename format

Files named like `video_0282_0_10_20260224194418_20260224200356.mp4` are
parsed automatically — device ID and start/end timestamps are extracted from
the filename.  Any other format with the timestamp pattern
`…_YYYYMMDDHHmmss_YYYYMMDDHHmmss.ext` will also work.

### Workflow

1. **⚙️ Settings** (gear icon) — configure working days (default Sun–Thu),
   working hours, and detection thresholds.
2. **Enter a path** — paste the absolute path to a directory of recordings
   (or a single file) into the input box.
3. **▶ Analyse** — the pipeline processes each file:
   - Extracts audio via ffmpeg
   - Runs silence detection (silent regions are shown as grey stripes)
   - Runs YAMNet for baby cry, yelling, loud noise, and abuse co-occurrence
   - Saves results to `data/crybaby.db` (SQLite cache — files are not
     re-processed on the next run unless you tick *re-process*)
4. **Select a date** in the sidebar to open the timeline for that day.

### Timeline UI

```
00:00    02:00    04:00    …    19:00    20:00    21:00
         ┌── [19:44–20:03] ─────────────────────────┐
         │  ░░░░░░░░  🟡 🔴  ░░░░  🟡              │
         └──────────────────────────────────────────┘
             grey = silence
             🟡 = baby cry
             🔴 = yell / abuse alert
             💜 = loud noise
```

- Each recording is a row positioned at its real wall-clock time.
- Hover over an event marker to see its type, offset, and severity.
- **Click an event** (on the timeline or in the list below) to open the
  video player seeked to that exact offset.

### Project structure

```
crybaby/
├── pipeline/
│   ├── file_parser.py       parse filenames, filter weekday/hours
│   ├── audio_extractor.py   ffmpeg mp4 → wav
│   ├── silence_detector.py  RMS-based silence map
│   ├── event_detector.py    unified YAMNet inference (all event types)
│   └── processor.py         orchestrator
├── dashboard/
│   ├── app.py               FastAPI backend + SSE progress stream
│   ├── database.py          SQLite cache (recordings + events)
│   └── static/
│       ├── index.html
│       ├── app.js           timeline UI
│       └── style.css
├── detector.py              standalone BabyCryDetector class
├── abuse_detector.py        standalone AbuseDetector class
├── main.py                  CLI — baby cry
├── main_abuse.py            CLI — abuse detection
├── run_dashboard.py         one-command dashboard launcher
├── requirements.txt
└── data/
    └── crybaby.db           auto-created SQLite database
```
# lullaby-monitor
