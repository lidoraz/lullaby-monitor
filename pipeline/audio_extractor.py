"""
Extract audio track from a video file to a temporary 16 kHz mono WAV.

Uses ``ffmpeg`` via subprocess — no Python bindings required.
The caller is responsible for deleting the temp file when done
(or use the context manager helper provided here).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000   # Hz — required by YAMNet


def _ffmpeg_executable() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError(
            "ffmpeg not found on PATH.  "
            "Install with:  brew install ffmpeg  (macOS)"
        )
    return exe


def extract_audio(
    video_path: Path,
    output_wav: Path,
    sample_rate: int = _SAMPLE_RATE,
) -> Path:
    """
    Extract the audio stream from *video_path* and write a mono WAV at
    *sample_rate* Hz to *output_wav*.

    SAFETY: *video_path* is passed to ffmpeg as a **read-only input** (-i).
    *output_wav* must not reside inside the same directory as *video_path*;
    an assertion enforces this so the source directory can never be written to.

    Returns *output_wav* on success; raises ``subprocess.CalledProcessError``
    on ffmpeg failure.
    """
    # --- WRITE-PROTECTION GUARD -------------------------------------------
    # Verify the output file will NOT be placed inside the source video's
    # directory.  This is a safety net; in normal operation output_wav always
    # goes into a temporary directory created by tempfile.TemporaryDirectory.
    video_dir  = video_path.resolve().parent
    output_dir = output_wav.resolve().parent
    if output_dir == video_dir or str(output_dir).startswith(str(video_dir) + "/"):
        raise ValueError(
            f"SAFETY VIOLATION: output_wav '{output_wav}' would be written "
            f"inside the source directory '{video_dir}'. "
            "The source directory must never be modified."
        )
    # ----------------------------------------------------------------------
    cmd = [
        _ffmpeg_executable(),
        "-y",                          # overwrite without prompting
        "-i", str(video_path),
        "-vn",                         # no video
        "-ac", "1",                    # mono
        "-ar", str(sample_rate),       # sample rate
        "-f", "wav",
        str(output_wav),
    ]
    logger.debug("ffmpeg cmd: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    logger.debug("Audio extracted → %s", output_wav)
    return output_wav


@contextmanager
def temporary_audio(
    video_path: Path,
    sample_rate: int = _SAMPLE_RATE,
) -> Generator[Path, None, None]:
    """
    Context manager that extracts audio to a temp WAV and deletes it on exit.

    Usage::

        with temporary_audio(Path("video.mp4")) as wav_path:
            waveform, sr = librosa.load(wav_path, sr=16000)
    """
    suffix = video_path.stem + "_audio.wav"
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / suffix
        extract_audio(video_path, wav_path, sample_rate=sample_rate)
        yield wav_path
