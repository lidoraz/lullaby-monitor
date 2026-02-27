"""
Baby cry detector using YAMNet (Google, pre-trained on AudioSet).

YAMNet processes audio in overlapping ~0.96 s frames and predicts
probabilities for 521 AudioSet sound classes, one of which is
"Baby cry, infant cry".  We scan every frame, collect windows that
exceed the confidence threshold, then merge adjacent windows into
continuous cry *episodes*.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# YAMNet model hosted on TensorFlow Hub
_YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

# AudioSet display name to look for (case-insensitive substring match)
_CRY_CLASS_SUBSTRING = "baby cry"

# YAMNet frame hop in seconds (fixed by the model architecture)
_YAMNET_HOP_SECONDS = 0.48


@dataclass
class CryEpisode:
    """A continuous segment of audio that contains a baby cry."""
    start: float          # seconds
    end: float            # seconds
    peak_confidence: float
    mean_confidence: float
    frame_confidences: List[float] = field(repr=False)

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
            f"peak={self.peak_confidence:.2f}  "
            f"mean={self.mean_confidence:.2f}"
        )


class BabyCryDetector:
    """
    Detects baby cry events in an audio file.

    Parameters
    ----------
    threshold : float
        Per-frame confidence threshold (0–1).  A frame is flagged as a
        potential cry when YAMNet's "Baby cry, infant cry" score ≥ threshold.
        Lower → more sensitive (more false positives).
        Higher → more strict (may miss quiet cries).  Default: 0.25.
    merge_gap : float
        Maximum gap in seconds between flagged frames that will still be
        merged into the same episode.  Default: 1.5 s.
    min_duration : float
        Episodes shorter than this (seconds) are discarded as noise
        artefacts.  Default: 0.5 s.
    """

    def __init__(
        self,
        threshold: float = 0.25,
        merge_gap: float = 1.5,
        min_duration: float = 0.5,
    ) -> None:
        self.threshold = threshold
        self.merge_gap = merge_gap
        self.min_duration = min_duration

        self._model = None          # lazy-load on first use
        self._cry_class_index: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, audio_path: str | Path) -> List[CryEpisode]:
        """
        Analyse *audio_path* and return a (possibly empty) list of
        :class:`CryEpisode` objects, ordered by start time.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform = self._load_audio(audio_path)
        scores = self._run_yamnet(waveform)          # shape (N, 521)
        cry_scores = scores[:, self._cry_class_index]  # shape (N,)

        episodes = self._build_episodes(cry_scores)
        logger.debug(
            "File '%s': %d frame(s) scored, %d episode(s) found.",
            audio_path.name,
            len(cry_scores),
            len(episodes),
        )
        return episodes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Import heavy dependencies and download the model on first use."""
        if self._model is not None:
            return

        try:
            import tensorflow_hub as hub  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "tensorflow-hub is required.  "
                "Install all dependencies with:  pip install -r requirements.txt"
            ) from exc

        logger.info("Loading YAMNet from TF Hub (first run may take a moment)…")
        self._model = hub.load(_YAMNET_URL)

        # Resolve the AudioSet class index for "Baby cry, infant cry"
        import tensorflow as tf  # noqa: PLC0415

        class_map_path = self._model.class_map_path().numpy().decode()
        with tf.io.gfile.GFile(class_map_path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if _CRY_CLASS_SUBSTRING in row["display_name"].lower():
                    self._cry_class_index = int(row["index"])
                    logger.debug(
                        "Baby cry class → index %d ('%s')",
                        self._cry_class_index,
                        row["display_name"],
                    )
                    break

        if self._cry_class_index is None:
            raise RuntimeError(
                f"Could not find a class containing '{_CRY_CLASS_SUBSTRING}' "
                "in the YAMNet class map."
            )

    def _load_audio(self, path: Path) -> np.ndarray:
        """
        Load and resample audio to 16 kHz mono (required by YAMNet).
        Supports any format recognised by *librosa* (wav, mp3, flac, ogg …).
        """
        try:
            import librosa  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "librosa is required.  "
                "Install with:  pip install -r requirements.txt"
            ) from exc

        logger.info("Loading '%s'…", path.name)
        waveform, _ = librosa.load(str(path), sr=16_000, mono=True)
        logger.info(
            "  Duration: %.1f s  |  Samples: %d", len(waveform) / 16_000, len(waveform)
        )
        return waveform.astype(np.float32)

    def _run_yamnet(self, waveform: np.ndarray) -> np.ndarray:
        """Run YAMNet inference and return the scores array (N × 521)."""
        self._ensure_model_loaded()

        import tensorflow as tf  # noqa: PLC0415

        logger.info("Running YAMNet inference…")
        scores, _embeddings, _spectrogram = self._model(
            tf.constant(waveform, dtype=tf.float32)
        )
        return scores.numpy()

    def _build_episodes(self, cry_scores: np.ndarray) -> List[CryEpisode]:
        """
        Convert a per-frame confidence array into merged :class:`CryEpisode`
        objects.

        Strategy
        --------
        1. Mark every frame whose score ≥ threshold.
        2. Convert frame indices to timestamps.
        3. Merge frames separated by ≤ merge_gap seconds.
        4. Discard episodes shorter than min_duration seconds.
        """
        flagged_times = [
            (i * _YAMNET_HOP_SECONDS, float(cry_scores[i]))
            for i in range(len(cry_scores))
            if cry_scores[i] >= self.threshold
        ]

        if not flagged_times:
            return []

        # Group into raw episodes
        raw_episodes: List[List[tuple]] = []
        current: List[tuple] = [flagged_times[0]]

        for t, score in flagged_times[1:]:
            if t - current[-1][0] <= self.merge_gap + _YAMNET_HOP_SECONDS:
                current.append((t, score))
            else:
                raw_episodes.append(current)
                current = [(t, score)]
        raw_episodes.append(current)

        # Convert to CryEpisode objects, apply min_duration filter
        episodes: List[CryEpisode] = []
        for group in raw_episodes:
            times = [t for t, _ in group]
            confs = [s for _, s in group]
            start = times[0]
            # Episode ends at the *last* flagged frame + one frame duration (0.96 s)
            end = times[-1] + 0.96

            if (end - start) < self.min_duration:
                continue

            episodes.append(
                CryEpisode(
                    start=start,
                    end=end,
                    peak_confidence=max(confs),
                    mean_confidence=float(np.mean(confs)),
                    frame_confidences=confs,
                )
            )

        return episodes
