"""
Orchestrator — ties together file parsing, audio extraction,
silence detection, and event detection into a single processing pipeline.

Each video is processed as follows:
  1. Extract audio to a temp WAV via ffmpeg.
  2. Load the waveform with librosa.
  3. Run silence detection to get the active regions.
  4. Run unified YAMNet event detection, skipping silent frames.
  5. Return a :class:`ProcessingResult` (stored in the DB by the caller).

Progress is reported through a simple callback so the dashboard can
show a live processing bar.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from pipeline.file_parser import VideoFile
from pipeline.audio_extractor import temporary_audio
from pipeline.silence_detector import SilenceMap, detect_silence
from pipeline.event_detector import Event, detect_events

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]   # (message, current, total)


@dataclass
class ProcessingResult:
    video:          VideoFile
    status:         str                        # "ok" | "error" | "skipped"
    duration:       float                      # audio duration in seconds
    silence_map:    Optional[SilenceMap]       # None on error
    events:         List[Event]                # empty on error / skipped
    error_message:  str = ""

    def to_dict(self) -> dict:
        sm = self.silence_map
        return {
            "file":           str(self.video.path),
            "device_id":      self.video.device_id,
            "start":          self.video.start.isoformat(),
            "end":            self.video.end.isoformat(),
            "status":         self.status,
            "duration":       round(self.duration, 2),
            "silent_fraction": round(sm.silent_fraction, 3) if sm else 0.0,
            "silent_regions": [[round(s, 3), round(e, 3)] for s, e in (sm.silent  if sm else [])],
            "active_regions": [[round(s, 3), round(e, 3)] for s, e in (sm.active  if sm else [])],
            "events":         [ev.to_dict() for ev in self.events],
            "error_message":  self.error_message,
        }


class Processor:
    """
    Process a list of :class:`VideoFile` objects through the full pipeline.

    Parameters
    ----------
    cry_threshold    : float  YAMNet threshold for baby-cry frames.
    yell_threshold   : float  YAMNet threshold for yelling frames.
    noise_threshold  : float  YAMNet threshold for loud-noise frames.
    co_window        : float  Seconds within which cry+yell counts as ABUSE.
    merge_gap        : float  Seconds — gaps within which events are merged.
    min_event_dur    : float  Seconds — events shorter than this are discarded.
    silence_db_thresh: float  dBFS below which a frame is silent.
    min_silence_dur  : float  Seconds — minimum silence segment duration.
    """

    def __init__(
        self,
        cry_threshold:     float = 0.25,
        yell_threshold:    float = 0.20,
        noise_threshold:   float = 0.30,
        talk_threshold:    float = 0.40,
        co_window:         float = 2.0,
        merge_gap:         float = 1.5,
        min_event_dur:     float = 0.5,
        silence_db_thresh: float = -45.0,
        min_silence_dur:   float = 1.0,
    ) -> None:
        self.cry_threshold     = cry_threshold
        self.yell_threshold    = yell_threshold
        self.noise_threshold   = noise_threshold
        self.talk_threshold    = talk_threshold
        self.co_window         = co_window
        self.merge_gap         = merge_gap
        self.min_event_dur     = min_event_dur
        self.silence_db_thresh = silence_db_thresh
        self.min_silence_dur   = min_silence_dur

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_all(
        self,
        videos: List[VideoFile],
        progress: Optional[ProgressCallback] = None,
        skip_paths: Optional[set] = None,
    ) -> List[ProcessingResult]:
        """
        Process *videos* in order.

        Parameters
        ----------
        progress    : optional callback(message, current, total)
        skip_paths  : set of Path strings already cached in the DB — skip them.
        """
        skip_paths = skip_paths or set()
        results: List[ProcessingResult] = []
        total = len(videos)

        for idx, vf in enumerate(videos, start=1):
            if str(vf.path) in skip_paths:
                logger.info("[%d/%d] Skipping (cached): %s", idx, total, vf.path.name)
                if progress:
                    progress(f"Skipping (cached): {vf.path.name}", idx, total)
                continue

            logger.info("[%d/%d] Processing: %s", idx, total, vf.path.name)
            if progress:
                progress(f"Processing: {vf.path.name}", idx, total)

            result = self.process_one(vf)
            results.append(result)

        return results

    def process_one(self, vf: VideoFile) -> ProcessingResult:
        """Process a single :class:`VideoFile`."""
        try:
            return self._run(vf)
        except Exception as exc:   # noqa: BLE001
            msg = traceback.format_exc()
            logger.error("Error processing %s:\n%s", vf.path.name, msg)
            return ProcessingResult(
                video=vf,
                status="error",
                duration=0.0,
                silence_map=None,
                events=[],
                error_message=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, vf: VideoFile) -> ProcessingResult:
        # SAFETY: temporary_audio() always writes the WAV to a system temp
        # directory (tempfile.TemporaryDirectory).  The source file at
        # vf.path is passed to ffmpeg as a read-only input and is never
        # modified.  extract_audio() also contains an explicit guard that
        # raises ValueError if the output would land in the source directory.
        with temporary_audio(vf.path) as wav_path:
            waveform = self._load(wav_path)

        sr       = 16_000
        duration = len(waveform) / sr

        # Silence detection
        silence_map = detect_silence(
            waveform,
            sample_rate=sr,
            db_thresh=self.silence_db_thresh,
            min_silence=self.min_silence_dur,
        )
        logger.info(
            "  Silence: %.0f%%  Active regions: %d",
            100 * silence_map.silent_fraction,
            len(silence_map.active),
        )

        # Event detection (pass active regions so silent frames are skipped)
        events = detect_events(
            waveform,
            cry_threshold=self.cry_threshold,
            yell_threshold=self.yell_threshold,
            noise_threshold=self.noise_threshold,
            talk_threshold=self.talk_threshold,
            co_window=self.co_window,
            merge_gap=self.merge_gap,
            min_duration=self.min_event_dur,
            silence_mask=silence_map.active,
        )
        logger.info("  Events detected: %d", len(events))

        return ProcessingResult(
            video=vf,
            status="ok",
            duration=duration,
            silence_map=silence_map,
            events=events,
        )

    @staticmethod
    def _load(wav_path: Path) -> np.ndarray:
        import librosa  # noqa: PLC0415
        waveform, _ = librosa.load(str(wav_path), sr=16_000, mono=True)
        return waveform.astype(np.float32)
