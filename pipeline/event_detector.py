"""
Unified event detector — runs YAMNet once and extracts multiple event types.

Event types
-----------
  BABY_CRY   — "Baby cry, infant cry"
  YELL       — "Shout / Yell / Screaming / Bellow / Whoop"
  LOUD_NOISE — "Explosion", "Bang", "Thud", "Slam", "Crash"
  ABUSE      — co-occurrence of BABY_CRY + YELL (uses AbuseDetector logic)

Running YAMNet once and dispatching to multiple class groups is much more
efficient than running it separately for each event type.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
_HOP        = 0.48    # seconds between YAMNet frame centres
_FRAME_DUR  = 0.96    # seconds, one analysis window

# --- class substring maps ---------------------------------------------------
_CLASS_SUBSTRINGS: Dict[str, List[str]] = {
    "baby_cry":   ["baby cry", "infant cry"],
    "yell":       ["shout", "yell", "scream", "bellow", "whoop",
                   "crying, sobbing"],
    "loud_noise": ["explosion", "bang", "thud", "slam", "crash",
                   "gunshot", "glass"],
    "talk":       ["speech", "conversation", "narration", "monologue",
                   "male speech", "female speech", "child speech",
                   "talking", "singing"],
}

# HIGH-severity sub-thresholds (used to classify ABUSE severity)
_HIGH_YELL = 0.55
_HIGH_CRY  = 0.40


class EventType(str, Enum):
    BABY_CRY   = "baby_cry"
    YELL       = "yell"
    LOUD_NOISE = "loud_noise"
    ABUSE      = "abuse"       # cry + yell co-occurrence
    TALK       = "talk"        # any speech / conversation


class Severity(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


@dataclass
class Event:
    """A detected audio event with a start/end time within the recording."""
    type:             EventType
    start:            float   # seconds from start of audio file
    end:              float   # seconds
    severity:         Severity
    peak_confidence:  float
    mean_confidence:  float
    # For ABUSE events, both cry and yell scores are stored
    peak_secondary:   Optional[float] = None   # e.g. peak cry score in ABUSE
    extra:            Dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "type":            self.type.value,
            "start":           round(self.start, 3),
            "end":             round(self.end,   3),
            "severity":        self.severity.value,
            "peak_confidence": round(self.peak_confidence, 3),
            "mean_confidence": round(self.mean_confidence, 3),
            "peak_secondary":  round(self.peak_secondary, 3) if self.peak_secondary else None,
        }


# ---------------------------------------------------------------------------
# YAMNet singleton (shared across processing runs)
# ---------------------------------------------------------------------------

_model      = None
_class_idx: Dict[str, List[int]] = {}


def _ensure_loaded() -> None:
    global _model, _class_idx  # noqa: PLW0603

    if _model is not None:
        return

    try:
        import tensorflow_hub as hub   # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "tensorflow-hub is required — run: pip install -r requirements.txt"
        ) from exc

    logger.info("Loading YAMNet from TF Hub …")
    _model = hub.load(_YAMNET_URL)

    import tensorflow as tf  # noqa: PLC0415

    class_map_path = _model.class_map_path().numpy().decode()
    idx_map: Dict[str, List[int]] = {k: [] for k in _CLASS_SUBSTRINGS}

    with tf.io.gfile.GFile(class_map_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name  = row["display_name"].lower()
            index = int(row["index"])
            for group, substrings in _CLASS_SUBSTRINGS.items():
                if any(s in name for s in substrings):
                    idx_map[group].append(index)

    _class_idx = idx_map
    for group, indices in idx_map.items():
        logger.debug("  %-12s → %d classes", group, len(indices))


def _pool(scores: np.ndarray, indices: List[int]) -> np.ndarray:
    if not indices:
        return np.zeros(scores.shape[0])
    return scores[:, indices].max(axis=1)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def detect_events(
    waveform: np.ndarray,
    *,
    cry_threshold:    float = 0.25,
    yell_threshold:   float = 0.20,
    noise_threshold:  float = 0.30,
    talk_threshold:   float = 0.40,
    co_window:        float = 2.0,
    merge_gap:        float = 1.5,
    min_duration:     float = 0.5,
    silence_mask:     Optional[List[Tuple[float, float]]] = None,
) -> List[Event]:
    """
    Run unified YAMNet inference on *waveform* (16 kHz mono float32).

    Parameters
    ----------
    silence_mask : optional list of (start, end) *active* (non-silent) regions.
        Frames that fall entirely within silent regions are not scored.
    
    Returns
    -------
    List of :class:`Event` objects sorted by start time.
    """
    _ensure_loaded()

    import tensorflow as tf  # noqa: PLC0415

    logger.info("Running YAMNet inference …")
    scores_tf, _, _ = _model(tf.constant(waveform, dtype=tf.float32))
    scores: np.ndarray = scores_tf.numpy()   # (N, 521)

    # Build active-frame boolean mask from silence_mask
    n = scores.shape[0]
    active_frame = np.ones(n, dtype=bool)
    if silence_mask is not None:
        # Mark frames that overlap with silent regions as inactive
        for i in range(n):
            t = i * _HOP
            t_end = t + _FRAME_DUR
            # frame is inactive if its midpoint is in a silent zone
            mid = (t + t_end) / 2
            in_active = any(s <= mid <= e for s, e in silence_mask)
            active_frame[i] = in_active

    # Per-group per-frame scores
    cry_s   = _pool(scores, _class_idx["baby_cry"])
    yell_s  = _pool(scores, _class_idx["yell"])
    noise_s = _pool(scores, _class_idx["loud_noise"])
    talk_s  = _pool(scores, _class_idx["talk"])

    events: List[Event] = []

    # Single-class event types
    _signal_events(cry_s,   EventType.BABY_CRY,   cry_threshold,   merge_gap, min_duration, active_frame, events)
    _signal_events(yell_s,  EventType.YELL,        yell_threshold,  merge_gap, min_duration, active_frame, events)
    _signal_events(noise_s, EventType.LOUD_NOISE,  noise_threshold, merge_gap, min_duration, active_frame, events)
    _signal_events(talk_s,  EventType.TALK,         talk_threshold,  merge_gap, min_duration, active_frame, events)

    # Co-occurrence (ABUSE)
    co_gap_frames = int(np.ceil(co_window / _HOP))
    cry_flag  = np.where((cry_s  >= cry_threshold)  & active_frame)[0]
    yell_flag = np.where((yell_s >= yell_threshold) & active_frame)[0]

    if len(cry_flag) and len(yell_flag):
        concern_idx: List[int] = []
        for yi in yell_flag:
            dists = np.abs(cry_flag - yi)
            if dists.min() <= co_gap_frames:
                ci = cry_flag[dists.argmin()]
                concern_idx.extend(range(min(yi, ci), max(yi, ci) + 1))

        if concern_idx:
            concern_idx = sorted(set(concern_idx))
            abuse_events = _merge_into_events(
                concern_idx,
                primary_scores=yell_s,
                secondary_scores=cry_s,
                event_type=EventType.ABUSE,
                merge_gap=merge_gap,
                min_duration=min_duration,
                high_primary=_HIGH_YELL,
                high_secondary=_HIGH_CRY,
            )
            events.extend(abuse_events)

    events.sort(key=lambda e: e.start)
    return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signal_events(
    scores:       np.ndarray,
    event_type:   EventType,
    threshold:    float,
    merge_gap:    float,
    min_duration: float,
    active_mask:  np.ndarray,
    results:      List[Event],
) -> None:
    flagged = list(np.where((scores >= threshold) & active_mask)[0])
    if not flagged:
        return
    results.extend(
        _merge_into_events(
            flagged,
            primary_scores=scores,
            secondary_scores=None,
            event_type=event_type,
            merge_gap=merge_gap,
            min_duration=min_duration,
        )
    )


def _merge_into_events(
    indices:          List[int],
    primary_scores:   np.ndarray,
    secondary_scores: Optional[np.ndarray],
    event_type:       EventType,
    merge_gap:        float,
    min_duration:     float,
    high_primary:     float = 0.55,
    high_secondary:   float = 0.40,
) -> List[Event]:
    merge_frames = int(np.ceil(merge_gap / _HOP))
    groups: List[List[int]] = [[indices[0]]]
    for idx in indices[1:]:
        if idx - groups[-1][-1] <= merge_frames + 1:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    events: List[Event] = []
    for group in groups:
        start = group[0]  * _HOP
        end   = group[-1] * _HOP + _FRAME_DUR
        if (end - start) < min_duration:
            continue

        pconf = [float(primary_scores[i]) for i in group]
        peak  = max(pconf)
        mean  = float(np.mean(pconf))

        peak_sec = None
        if secondary_scores is not None:
            sconf    = [float(secondary_scores[i]) for i in group]
            peak_sec = max(sconf)
            # Severity for co-occurrence events
            if peak >= high_primary and peak_sec >= high_secondary:
                severity = Severity.HIGH
            elif peak >= high_primary or peak_sec >= high_secondary:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW
        else:
            # Severity for single-class events
            if peak >= 0.70:
                severity = Severity.HIGH
            elif peak >= 0.45:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW

        events.append(Event(
            type=event_type,
            start=start,
            end=end,
            severity=severity,
            peak_confidence=peak,
            mean_confidence=mean,
            peak_secondary=peak_sec,
        ))
    return events
