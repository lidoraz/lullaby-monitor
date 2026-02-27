"""
Silence / low-energy region detector.

Uses short-time RMS energy to identify segments of audio where no
meaningful sound is present.  The result is used by the processor to:

  1. Skip inference on silent frames (saves compute).
  2. Display silent regions in the dashboard as grey "dead zones".

A ``SilenceMap`` exposes two methods:
  • ``is_silent(t)``            → bool — is timestamp t (seconds) silent?
  • ``active_regions()``        → list of (start, end) tuples of *non-silent* audio
  • ``silent_regions()``        → list of (start, end) tuples of silent audio
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_FRAME_LEN   = 0.05    # seconds per RMS frame
_DEFAULT_HOP_LEN     = 0.025   # seconds per hop
_DEFAULT_DB_THRESH   = -45.0   # dBFS below which a frame is "silent"
_DEFAULT_MIN_SILENCE = 1.0     # seconds — shorter silent stretches are ignored
_DEFAULT_MIN_ACTIVE  = 0.5     # seconds — shorter active stretches are absorbed


@dataclass
class SilenceMap:
    """Result of silence detection over a waveform."""
    duration: float           # total audio duration in seconds
    silent:   List[Tuple[float, float]]   # (start, end) pairs
    active:   List[Tuple[float, float]]   # (start, end) pairs

    def is_silent(self, t: float) -> bool:
        for s, e in self.silent:
            if s <= t <= e:
                return True
        return False

    @property
    def silent_fraction(self) -> float:
        total = sum(e - s for s, e in self.silent)
        return total / self.duration if self.duration > 0 else 0.0


def detect_silence(
    waveform: np.ndarray,
    sample_rate: int = 16_000,
    frame_len:   float = _DEFAULT_FRAME_LEN,
    hop_len:     float = _DEFAULT_HOP_LEN,
    db_thresh:   float = _DEFAULT_DB_THRESH,
    min_silence: float = _DEFAULT_MIN_SILENCE,
    min_active:  float = _DEFAULT_MIN_ACTIVE,
) -> SilenceMap:
    """
    Analyse *waveform* (1-D float32 array at *sample_rate* Hz) and
    return a :class:`SilenceMap`.

    Parameters
    ----------
    frame_len   : RMS analysis window in seconds.
    hop_len     : Hop between windows in seconds.
    db_thresh   : Frames below this dBFS level are considered silent.
    min_silence : Minimum duration of a silent segment to keep as silent.
    min_active  : Minimum active segment; shorter ones are merged with silence.
    """
    frame_samples = int(frame_len * sample_rate)
    hop_samples   = int(hop_len   * sample_rate)
    n_frames      = 1 + (len(waveform) - frame_samples) // hop_samples

    if n_frames <= 0:
        # Very short clip — treat as active
        duration = len(waveform) / sample_rate
        return SilenceMap(
            duration=duration,
            silent=[],
            active=[(0.0, duration)],
        )

    # Compute RMS per frame
    rms = np.array([
        np.sqrt(np.mean(
            waveform[i * hop_samples: i * hop_samples + frame_samples] ** 2
        ))
        for i in range(n_frames)
    ])

    # Convert to dBFS (add tiny epsilon to avoid log(0))
    db = 20.0 * np.log10(rms + 1e-10)

    is_silent_frame = db < db_thresh

    # Convert frame flags to time ranges
    hop_s = hop_len
    raw_silence: List[Tuple[float, float]] = []
    in_silence = False
    seg_start  = 0.0

    for i, silent in enumerate(is_silent_frame):
        t = i * hop_s
        if silent and not in_silence:
            in_silence = True
            seg_start  = t
        elif not silent and in_silence:
            in_silence = False
            raw_silence.append((seg_start, t))

    if in_silence:
        raw_silence.append((seg_start, n_frames * hop_s))

    duration = len(waveform) / sample_rate

    # Apply min_silence filter
    silence = [(s, e) for s, e in raw_silence if (e - s) >= min_silence]

    # Invert to get active regions
    active = _invert_regions(silence, duration)

    # Apply min_active filter (short active blips inside silence are swallowed)
    active  = [(s, e) for s, e in active  if (e - s) >= min_active]
    silence = _invert_regions(active, duration)

    logger.debug(
        "Silence map: %.1f s total | %d silent regions (%.0f%%) | %d active",
        duration, len(silence), 100 * sum(e - s for s, e in silence) / max(duration, 1),
        len(active),
    )

    return SilenceMap(duration=duration, silent=silence, active=active)


def _invert_regions(
    regions: List[Tuple[float, float]],
    duration: float,
) -> List[Tuple[float, float]]:
    """Return the complement of *regions* within [0, duration]."""
    inverted: List[Tuple[float, float]] = []
    prev = 0.0
    for s, e in sorted(regions):
        if prev < s:
            inverted.append((prev, s))
        prev = e
    if prev < duration:
        inverted.append((prev, duration))
    return inverted
