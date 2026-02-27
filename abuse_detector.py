"""
Caregiver-yelling + baby-cry co-occurrence detector.

IMPORTANT DISCLAIMER
--------------------
This tool is an *assistive* signal — it uses audio pattern recognition
and will produce both false positives and false negatives.  It is NOT
a substitute for direct observation, social services assessments, or
professional judgment.  Use it as one data point among many.

How it works
------------
YAMNet is run once over the recording.  For every 0.96-second frame we
extract two class groups:

  • Baby cry   — AudioSet "Baby cry, infant cry"
  • Adult yell — AudioSet "Shout", "Yell", "Screaming", "Bellow",
                 "Whoop"; also includes generic "Crying, sobbing"
                 (can indicate a distressed adult voice)

A frame is flagged as a **concern frame** when *both* groups exceed
their respective confidence thresholds at the same time, or when a yell
frame occurs within `co_window` seconds of a cry frame (babies often
pause between sobs).

Consecutive concern frames are merged into **ConcernEpisode** objects
with a severity level:

  HIGH   — peak yell confidence ≥ 0.55 and peak cry confidence ≥ 0.40
  MEDIUM — either of the above
  LOW    — both below those sub-thresholds but both above the detection
            thresholds

All timestamps are in MM:SS.ss format.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
_YAMNET_HOP = 0.48   # seconds between frame centres
_YAMNET_FRAME_DURATION = 0.96  # seconds, one analysis window

# AudioSet class substrings to search for (all matched case-insensitively)
_CRY_SUBSTRINGS   = ["baby cry", "infant cry"]
_YELL_SUBSTRINGS  = ["shout", "yell", "scream", "bellow", "whoop",
                     "crying, sobbing"]

# Sub-thresholds for severity classification
_HIGH_YELL_THRESH = 0.55
_HIGH_CRY_THRESH  = 0.40


class Severity(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


@dataclass
class ConcernEpisode:
    """
    A time window where adult yelling and baby crying overlap.
    """
    start: float
    end: float
    severity: Severity
    peak_cry_confidence: float
    peak_yell_confidence: float
    mean_cry_confidence: float
    mean_yell_confidence: float
    cry_frame_confidences: List[float]  = field(repr=False)
    yell_frame_confidences: List[float] = field(repr=False)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __str__(self) -> str:
        def _fmt(t: float) -> str:
            m, s = divmod(t, 60)
            return f"{int(m):02d}:{s:05.2f}"

        return (
            f"[{_fmt(self.start)} → {_fmt(self.end)}]  "
            f"duration={self.duration:.1f}s  "
            f"severity={self.severity.value:<6}  "
            f"cry={self.peak_cry_confidence:.2f}  "
            f"yell={self.peak_yell_confidence:.2f}"
        )


class AbuseDetector:
    """
    Detects potential adult-yelling-at-baby events in an audio file.

    Parameters
    ----------
    cry_threshold : float
        Minimum YAMNet confidence to flag a frame as containing a baby cry.
        Default: 0.20 (slightly lower than standalone detection because we
        require yelling to co-occur, so more cry sensitivity is acceptable).
    yell_threshold : float
        Minimum YAMNet confidence to flag a frame as containing adult yelling.
        Default: 0.20.
    co_window : float
        Seconds.  A yell frame and a cry frame are considered co-occurring
        if they are within this many seconds of each other.  Default: 2.0.
    merge_gap : float
        Gap in seconds between concern frames still merged into one episode.
        Default: 2.0.
    min_duration : float
        Episodes shorter than this are discarded.  Default: 0.5 s.
    """

    def __init__(
        self,
        cry_threshold: float = 0.20,
        yell_threshold: float = 0.20,
        co_window: float = 2.0,
        merge_gap: float = 2.0,
        min_duration: float = 0.5,
    ) -> None:
        self.cry_threshold  = cry_threshold
        self.yell_threshold = yell_threshold
        self.co_window      = co_window
        self.merge_gap      = merge_gap
        self.min_duration   = min_duration

        self._model = None
        self._cry_indices:  List[int] = []
        self._yell_indices: List[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, audio_path: str | Path) -> List[ConcernEpisode]:
        """
        Analyse *audio_path* and return a list of :class:`ConcernEpisode`.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform = self._load_audio(audio_path)
        scores   = self._run_yamnet(waveform)         # (N, 521)

        cry_scores  = self._pool_scores(scores, self._cry_indices)   # (N,)
        yell_scores = self._pool_scores(scores, self._yell_indices)  # (N,)

        episodes = self._build_episodes(cry_scores, yell_scores)
        logger.debug(
            "'%s': %d frames → %d concern episode(s).",
            audio_path.name, len(cry_scores), len(episodes),
        )
        return episodes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            import tensorflow_hub as hub   # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "tensorflow-hub is required — run: pip install -r requirements.txt"
            ) from exc

        logger.info("Loading YAMNet from TF Hub (first run downloads ~30 MB)…")
        self._model = hub.load(_YAMNET_URL)

        import tensorflow as tf  # noqa: PLC0415

        class_map_path = self._model.class_map_path().numpy().decode()
        with tf.io.gfile.GFile(class_map_path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                name  = row["display_name"].lower()
                index = int(row["index"])
                if any(s in name for s in _CRY_SUBSTRINGS):
                    self._cry_indices.append(index)
                    logger.debug("  Cry class:  [%d] %s", index, row["display_name"])
                if any(s in name for s in _YELL_SUBSTRINGS):
                    self._yell_indices.append(index)
                    logger.debug("  Yell class: [%d] %s", index, row["display_name"])

        if not self._cry_indices:
            raise RuntimeError("Could not find any baby-cry class in YAMNet class map.")
        if not self._yell_indices:
            raise RuntimeError("Could not find any yelling class in YAMNet class map.")

        logger.info(
            "  Cry classes: %d  |  Yell classes: %d",
            len(self._cry_indices), len(self._yell_indices),
        )

    def _load_audio(self, path: Path) -> np.ndarray:
        try:
            import librosa  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "librosa is required — run: pip install -r requirements.txt"
            ) from exc

        logger.info("Loading '%s'…", path.name)
        waveform, _ = librosa.load(str(path), sr=16_000, mono=True)
        logger.info(
            "  Duration: %.1f s  |  Samples: %d",
            len(waveform) / 16_000, len(waveform),
        )
        return waveform.astype(np.float32)

    def _run_yamnet(self, waveform: np.ndarray) -> np.ndarray:
        self._ensure_model_loaded()

        import tensorflow as tf  # noqa: PLC0415

        logger.info("Running YAMNet inference…")
        scores, _embeddings, _spec = self._model(
            tf.constant(waveform, dtype=tf.float32)
        )
        return scores.numpy()   # (N, 521)

    @staticmethod
    def _pool_scores(scores: np.ndarray, indices: List[int]) -> np.ndarray:
        """Max-pool across a group of class indices → shape (N,)."""
        return scores[:, indices].max(axis=1)

    def _build_episodes(
        self,
        cry_scores:  np.ndarray,
        yell_scores: np.ndarray,
    ) -> List[ConcernEpisode]:
        """
        Find frames where baby cry AND adult yelling co-occur, then merge them
        into :class:`ConcernEpisode` objects.
        """
        n = len(cry_scores)
        co_frame_gap = int(np.ceil(self.co_window / _YAMNET_HOP))

        # Build sets of flagged frame indices
        cry_flagged  = {i for i in range(n) if cry_scores[i]  >= self.cry_threshold}
        yell_flagged = {i for i in range(n) if yell_scores[i] >= self.yell_threshold}

        if not cry_flagged or not yell_flagged:
            return []

        # For each yell frame, check if a cry frame is within co_frame_gap
        concern_indices: List[int] = []
        for yi in sorted(yell_flagged):
            for ci in cry_flagged:
                if abs(yi - ci) <= co_frame_gap:
                    # Collect the range of frames spanning both
                    lo = min(yi, ci)
                    hi = max(yi, ci)
                    concern_indices.extend(range(lo, hi + 1))
                    break   # found a co-occurring cry, no need to keep searching

        if not concern_indices:
            return []

        concern_indices = sorted(set(concern_indices))

        # Merge consecutive / near-consecutive concern frames
        merge_frame_gap = int(np.ceil(self.merge_gap / _YAMNET_HOP))
        groups: List[List[int]] = [[concern_indices[0]]]
        for idx in concern_indices[1:]:
            if idx - groups[-1][-1] <= merge_frame_gap + 1:
                groups[-1].append(idx)
            else:
                groups.append([idx])

        episodes: List[ConcernEpisode] = []
        for group in groups:
            start = group[0]  * _YAMNET_HOP
            end   = group[-1] * _YAMNET_HOP + _YAMNET_FRAME_DURATION

            if (end - start) < self.min_duration:
                continue

            cry_conf  = [float(cry_scores[i])  for i in group]
            yell_conf = [float(yell_scores[i]) for i in group]

            peak_cry  = max(cry_conf)
            peak_yell = max(yell_conf)

            # Severity classification
            if peak_yell >= _HIGH_YELL_THRESH and peak_cry >= _HIGH_CRY_THRESH:
                severity = Severity.HIGH
            elif peak_yell >= _HIGH_YELL_THRESH or peak_cry >= _HIGH_CRY_THRESH:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW

            episodes.append(
                ConcernEpisode(
                    start=start,
                    end=end,
                    severity=severity,
                    peak_cry_confidence=peak_cry,
                    peak_yell_confidence=peak_yell,
                    mean_cry_confidence=float(np.mean(cry_conf)),
                    mean_yell_confidence=float(np.mean(yell_conf)),
                    cry_frame_confidences=cry_conf,
                    yell_frame_confidences=yell_conf,
                )
            )

        return episodes
