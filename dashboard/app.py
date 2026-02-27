"""
FastAPI dashboard application.

Start with:
    uvicorn dashboard.app:app --reload --port 7860

Or via the convenience script:
    python run_dashboard.py

API routes
----------
  GET  /                           → serve index.html
  GET  /api/dates                  → list of dates that have recordings
  GET  /api/date/{date}            → recordings + events for one date
  GET  /api/stats                  → global summary stats
  POST /api/process                → kick off processing (body: ProcessRequest)
  GET  /api/process/status         → SSE stream of processing progress
  GET  /video                      → stream a video file  (?path=…)
  GET  /api/settings               → current filter settings
  POST /api/settings               → save filter settings (weekdays, hours)
  POST /api/export                 → clip and export an event as video or audio
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from datetime import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import subprocess

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dashboard.database import Database
from pipeline.file_parser import FileScanner, VideoFile
from pipeline.processor import Processor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

BASE_DIR     = Path(__file__).parent
STATIC_DIR   = BASE_DIR / "static"
DATA_DIR     = BASE_DIR.parent / "data"
DB_PATH      = DATA_DIR / "crybaby.db"
SETTINGS_PATH = DATA_DIR / "settings.json"
EXPORT_DIR   = DATA_DIR / "exports"

app = FastAPI(title="Crybaby Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

db = Database(DB_PATH)

# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------

_DEFAULT_SETTINGS = {
    "work_days":    [6, 0, 1, 2, 3],   # Sun-Thu (Python weekday)
    "hours_start":  "00:00",
    "hours_end":    "23:59",
    "cry_threshold":   0.25,
    "yell_threshold":  0.20,
    "noise_threshold": 0.30,
    "talk_threshold":  0.40,
    "co_window":       2.0,
    "merge_gap":       1.5,
    "min_event_dur":   0.5,
    "silence_db":     -45.0,
    "min_silence_dur": 1.0,
}


def load_settings() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH) as f:
            s = json.load(f)
        # Merge with defaults so new keys always exist
        merged = {**_DEFAULT_SETTINGS, **s}
        return merged
    return dict(_DEFAULT_SETTINGS)


def save_settings(settings: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


# ---------------------------------------------------------------------------
# Processing state (one job at a time)
# ---------------------------------------------------------------------------

class ProcessingState:
    def __init__(self) -> None:
        self.running     = False
        self.progress_q: queue.Queue = queue.Queue()
        self.lock        = threading.Lock()

    def push(self, msg: str, current: int, total: int) -> None:
        self.progress_q.put({"msg": msg, "current": current, "total": total})

    def done(self, summary: dict) -> None:
        self.progress_q.put({"done": True, "summary": summary})
        with self.lock:
            self.running = False

    def error(self, err: str) -> None:
        self.progress_q.put({"error": err})
        with self.lock:
            self.running = False


_state = ProcessingState()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ProcessRequest(BaseModel):
    source:    str                = Field(..., description="Absolute path to a file or directory.")
    force_reprocess: bool         = Field(False, description="Re-process files already in cache.")


class ExportRequest(BaseModel):
    file_path:    str   = Field(..., description="Absolute path to source video.")
    offset_start: float = Field(..., description="Event start offset in seconds within the file.")
    offset_end:   float = Field(..., description="Event end offset in seconds within the file.")
    abs_start:    str   = Field(..., description="Absolute wall-clock start (ISO-8601) for filename.")
    event_type:   str   = Field(..., description="Event type label used in filename.")
    pre_seconds:  float = Field(5.0,  description="Seconds to include before event start.")
    post_seconds: float = Field(20.0, description="Seconds to include after event end.")
    mode:         str   = Field("video", description="'video' or 'audio'")


class SettingsModel(BaseModel):
    work_days:       List[int]    = Field(..., description="Python weekday ints: 0=Mon … 6=Sun")
    hours_start:     str          = Field("00:00", description="HH:MM")
    hours_end:       str          = Field("23:59", description="HH:MM")
    cry_threshold:   float        = 0.25
    yell_threshold:  float        = 0.20
    noise_threshold: float        = 0.30
    talk_threshold:  float        = 0.40
    co_window:       float        = 2.0
    merge_gap:       float        = 1.5
    min_event_dur:   float        = 0.5
    silence_db:      float        = -45.0
    min_silence_dur: float        = 1.0


# ---------------------------------------------------------------------------
# Static / HTML
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Video streaming
# ---------------------------------------------------------------------------

@app.get("/video")
async def stream_video(path: str = Query(..., description="Absolute path to the video file.")):
    """Stream a local video file so the browser <video> element can play it."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Video file not found.")
    return FileResponse(p, media_type="video/mp4")


# ---------------------------------------------------------------------------
# Data API
# ---------------------------------------------------------------------------

@app.get("/api/dates")
async def get_dates() -> List[str]:
    return db.get_dates()


@app.get("/api/date/{date}")
async def get_date(date: str) -> List[dict]:
    """
    Return all recordings (with embedded events) for *date* (YYYY-MM-DD).
    """
    return db.get_recordings_for_date(date)


@app.get("/api/stats")
async def get_stats() -> dict:
    return db.get_stats()


# ---------------------------------------------------------------------------
# Settings API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Export API
# ---------------------------------------------------------------------------

@app.post("/api/export")
async def export_event(req: ExportRequest) -> dict:
    """
    Clip a segment from *req.file_path* using ffmpeg and save it to
    data/exports/.  Returns the path of the exported file.
    """
    src = Path(req.file_path)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source file not found.")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Clip boundaries (clamped to 0)
    clip_start = max(0.0, req.offset_start - req.pre_seconds)
    clip_end   = req.offset_end + req.post_seconds
    duration   = clip_end - clip_start

    # Filename:  20260224_194418_baby_cry.mp4 / .mp3
    ts = req.abs_start.replace("-", "").replace("T", "_").replace(":", "")[:15]
    ext  = "mp4" if req.mode == "video" else "mp3"
    stem = f"{ts}_{req.event_type}"
    # Avoid overwriting if multiple exports of the same event
    out_path = EXPORT_DIR / f"{stem}.{ext}"
    counter  = 1
    while out_path.exists():
        out_path = EXPORT_DIR / f"{stem}_{counter}.{ext}"
        counter += 1

    if req.mode == "video":
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(clip_start),
            "-i", str(src),
            "-t",  str(duration),
            "-c",  "copy",           # fast stream copy — no re-encode
            str(out_path),
        ]
    else:  # audio only
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(clip_start),
            "-i", str(src),
            "-t",  str(duration),
            "-vn",                   # drop video
            "-ac",  "1",
            "-ar",  "44100",
            "-q:a", "2",             # VBR quality
            str(out_path),
        ]

    logger.info("Exporting: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg error: %s", result.stderr)
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg failed: {result.stderr[-300:]}",
        )

    return {
        "ok":       True,
        "path":     str(out_path),
        "filename": out_path.name,
        "size_kb":  round(out_path.stat().st_size / 1024, 1),
    }


@app.get("/api/export/list")
async def list_exports() -> List[dict]:
    """Return all files in data/exports/ sorted newest-first."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(EXPORT_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {
            "filename": p.name,
            "path":     str(p),
            "size_kb":  round(p.stat().st_size / 1024, 1),
            "mtime":    p.stat().st_mtime,
        }
        for p in files if p.is_file()
    ]


@app.get("/api/export/download")
async def download_export(path: str = Query(...)) -> FileResponse:
    """Download / stream an exported file."""
    p = Path(path)
    if not p.exists() or not str(p).startswith(str(EXPORT_DIR)):
        raise HTTPException(status_code=404, detail="File not found.")
    media = "video/mp4" if p.suffix == ".mp4" else "audio/mpeg"
    return FileResponse(p, media_type=media, filename=p.name)


@app.get("/api/settings")
async def get_settings() -> dict:
    return load_settings()


@app.post("/api/settings")
async def post_settings(body: SettingsModel) -> dict:
    d = body.dict()
    save_settings(d)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Processing API
# ---------------------------------------------------------------------------

@app.post("/api/process")
async def start_process(req: ProcessRequest) -> dict:
    with _state.lock:
        if _state.running:
            raise HTTPException(status_code=409, detail="A processing job is already running.")
        _state.running = True
        # Drain any stale messages
        while not _state.progress_q.empty():
            _state.progress_q.get_nowait()

    settings = load_settings()

    def _run() -> None:
        try:
            hs = time(*[int(x) for x in settings["hours_start"].split(":")])
            he = time(*[int(x) for x in settings["hours_end"].split(":")])

            scanner = FileScanner(
                work_days=set(settings["work_days"]),
                work_hours_start=hs,
                work_hours_end=he,
            )
            _state.push("Scanning files…", 0, 0)
            videos = scanner.scan(req.source)

            if not videos:
                _state.push("No matching video files found.", 0, 0)
                _state.done({"processed": 0, "skipped": 0, "errors": 0})
                return

            _state.push(f"Found {len(videos)} file(s).  Starting analysis…", 0, len(videos))

            skip = set() if req.force_reprocess else db.get_cached_paths()

            processor = Processor(
                cry_threshold=settings["cry_threshold"],
                yell_threshold=settings["yell_threshold"],
                noise_threshold=settings["noise_threshold"],
                talk_threshold=settings["talk_threshold"],
                co_window=settings["co_window"],
                merge_gap=settings["merge_gap"],
                min_event_dur=settings["min_event_dur"],
                silence_db_thresh=settings["silence_db"],
                min_silence_dur=settings["min_silence_dur"],
            )

            processed = skipped = errors = 0
            total = len(videos)

            for idx, vf in enumerate(videos, 1):
                if str(vf.path) in skip:
                    skipped += 1
                    _state.push(f"Skipping (cached): {vf.path.name}", idx, total)
                    continue

                _state.push(f"Analysing: {vf.path.name}", idx, total)
                result = processor.process_one(vf)
                db.save_result(result)

                if result.status == "ok":
                    processed += 1
                else:
                    errors += 1

            _state.done({
                "processed": processed,
                "skipped":   skipped,
                "errors":    errors,
                "total":     total,
            })

        except Exception as exc:  # noqa: BLE001
            logger.exception("Processing failed")
            _state.error(str(exc))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"ok": True, "message": "Processing started."}


@app.get("/api/process/status")
async def process_status() -> StreamingResponse:
    """
    Server-Sent Events stream.  Each event is a JSON object:
      { "msg": "…", "current": N, "total": M }       — progress
      { "done": true, "summary": { … } }              — completion
      { "error": "…" }                                — failure
    """
    async def _generate():
        loop = asyncio.get_event_loop()
        while True:
            try:
                item = await loop.run_in_executor(
                    None, lambda: _state.progress_q.get(timeout=0.5)
                )
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("done") or item.get("error"):
                    break
            except queue.Empty:
                # Keep-alive heartbeat
                yield ": heartbeat\n\n"
                await asyncio.sleep(0.25)

    return StreamingResponse(_generate(), media_type="text/event-stream")
