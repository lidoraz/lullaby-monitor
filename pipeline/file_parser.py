"""
Parse Xiaomi camera filenames and filter by weekday / working hours.

Xiaomi filename format
----------------------
  video_0282_0_10_20260224194418_20260224200356.mp4
  └─────┘ └──┘└┘└─┘ └──────────────┘ └──────────────┘
  prefix  id  ?  ?   start datetime    end datetime
                       YYYYMMDDHHmmss

The key information we extract is:
  • recording start  (aware datetime, assumed local timezone)
  • recording end    (aware datetime)
  • device / channel id

The module is intentionally generic so that a different filename scheme can
be supported by subclassing or monkey-patching `parse_filename`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# Days 0=Monday … 6=Sunday  (Python weekday())
# "Sunday through Thursday" = weekdays 6, 0, 1, 2, 3
_DEFAULT_WORK_DAYS: set[int] = {6, 0, 1, 2, 3}   # Sun–Thu

_DEFAULT_WORK_HOURS_START = time(0, 0)   # midnight – analyse whole day by default
_DEFAULT_WORK_HOURS_END   = time(23, 59)

# ---------------------------------------------------------------------------
# Exact Xiaomi filename pattern
# ---------------------------------------------------------------------------
# Matches ONLY files that look like:
#   video_<id>_<n>_<n>_YYYYMMDDHHmmss_YYYYMMDDHHmmss.mp4
# Example:
#   video_0282_0_10_20260224194418_20260224200356.mp4
#
# Files that do NOT match this pattern are silently skipped —
# they are NEVER opened, modified, moved, or deleted.
_XIAOMI_RE = re.compile(
    r"^video_(\w+)_\d+_\d+_(\d{14})_(\d{14})\.mp4$",
    re.IGNORECASE,
)

_DT_FMT = "%Y%m%d%H%M%S"

# Only look at .mp4 — Xiaomi always records in mp4; keeps the walk fast
# and avoids accidentally touching documents, images, or other media.
_DEFAULT_EXTENSIONS: set[str] = {".mp4"}


@dataclass
class ScanReport:
    """Detailed summary of what the scanner found (and ignored) in a directory."""
    source:          Path
    total_files:     int = 0      # all files in dir with matching extension
    matched:         int = 0      # passed filename pattern
    skipped_pattern: int = 0      # failed Xiaomi regex
    skipped_weekday: int = 0      # right pattern, wrong weekday
    skipped_hours:   int = 0      # right pattern, outside working hours
    accepted:        List[str] = field(default_factory=list)
    ignored:         List[str] = field(default_factory=list)  # non-matching files

    def log(self) -> None:
        logger.info(
            "Scan of '%s':  %d file(s) found  |  %d matched Xiaomi pattern  "
            "|  %d skipped (pattern)  |  %d skipped (weekday)  "
            "|  %d skipped (hours)  |  %d accepted for processing",
            self.source,
            self.total_files,
            self.matched,
            self.skipped_pattern,
            self.skipped_weekday,
            self.skipped_hours,
            len(self.accepted),
        )
        if self.ignored:
            logger.debug(
                "  Non-Xiaomi files ignored (read-only, untouched): %s",
                ", ".join(self.ignored[:20])
                + (f" … (+{len(self.ignored)-20} more)" if len(self.ignored) > 20 else ""),
            )


@dataclass
class VideoFile:
    path: Path
    device_id: str          # extracted from filename
    start: datetime         # recording start (local time, tz-naive)
    end: datetime           # recording end   (local time, tz-naive)

    @property
    def duration_seconds(self) -> float:
        return (self.end - self.start).total_seconds()

    @property
    def date_label(self) -> str:
        """YYYY-MM-DD string for grouping in the dashboard."""
        return self.start.strftime("%Y-%m-%d")

    @property
    def weekday(self) -> int:
        """Python weekday: 0=Mon … 6=Sun."""
        return self.start.weekday()

    def __repr__(self) -> str:
        return (
            f"VideoFile({self.path.name!r}  "
            f"{self.start:%Y-%m-%d %H:%M}→{self.end:%H:%M}  "
            f"device={self.device_id!r})"
        )


# ---------------------------------------------------------------------------
# Filename parsers
# ---------------------------------------------------------------------------

def parse_xiaomi(path: Path) -> Optional[VideoFile]:
    """
    Parse a Xiaomi camera filename.
    Returns ``None`` if the filename does NOT match the exact pattern.
    The source file is never opened or modified by this function.
    """
    m = _XIAOMI_RE.match(path.name)
    if not m:
        return None

    device_id = m.group(1)
    try:
        start = datetime.strptime(m.group(2), _DT_FMT)
        end   = datetime.strptime(m.group(3), _DT_FMT)
    except ValueError:
        # Digits present but not a valid date — skip silently
        return None

    return VideoFile(path=path, device_id=device_id, start=start, end=end)


# ---------------------------------------------------------------------------
# Directory / file scanner
# ---------------------------------------------------------------------------

class FileScanner:
    """
    Scan a directory (or accept a single file) and return the filtered list
    of :class:`VideoFile` objects.

    SAFETY GUARANTEE
    ----------------
    This class is **strictly read-only** with respect to the source directory:
      - It only calls ``Path.stat()`` and reads directory listings.
      - It never opens, writes, renames, moves, or deletes any file.
      - Files that do not match the Xiaomi pattern are logged at DEBUG level
        and completely ignored — they are never touched.

    Parameters
    ----------
    work_days : set[int]
        Python weekday numbers.  Default: {6, 0, 1, 2, 3}  (Sunday–Thursday).
    work_hours_start : time
        Earliest start time to include.  Default: 00:00 (all day).
    work_hours_end : time
        Latest start time to include.  Default: 23:59 (all day).
    parsers : list of callables
        Each takes a ``Path``, returns ``VideoFile | None``.
    extensions : set[str]
        Extensions to consider (lower-case with dot).  Default: {'.mp4'}.
    """

    def __init__(
        self,
        work_days: set[int] = _DEFAULT_WORK_DAYS,
        work_hours_start: time = _DEFAULT_WORK_HOURS_START,
        work_hours_end: time   = _DEFAULT_WORK_HOURS_END,
        parsers: Optional[List[Callable[[Path], Optional[VideoFile]]]] = None,
        extensions: Optional[set[str]] = None,
    ) -> None:
        self.work_days         = work_days
        self.work_hours_start  = work_hours_start
        self.work_hours_end    = work_hours_end
        self.parsers           = parsers or [parse_xiaomi]
        # Default to mp4 only — Xiaomi always records in mp4
        self.extensions        = extensions or _DEFAULT_EXTENSIONS

    def scan(self, source: str | Path) -> List[VideoFile]:
        """
        Return a sorted list of accepted :class:`VideoFile` objects.
        Also logs a :class:`ScanReport` at INFO level.

        ``source`` may be a single file or a directory (scanned recursively).
        The source directory / file is NEVER modified.
        """
        source = Path(source).resolve()
        candidates: List[Path] = []

        if source.is_file():
            candidates = [source]
        elif source.is_dir():
            # Walk only known video extensions — everything else is skipped
            # entirely and never opened.
            for ext in self.extensions:
                candidates.extend(source.rglob(f"*{ext}"))
        else:
            raise FileNotFoundError(f"Not a file or directory: {source}")

        candidates = sorted(candidates)
        report = ScanReport(source=source, total_files=len(candidates))

        results: List[VideoFile] = []
        for path in candidates:
            vf = self._parse(path)
            if vf is None:
                report.skipped_pattern += 1
                report.ignored.append(path.name)
                logger.debug("  SKIP (no Xiaomi pattern match): %s", path.name)
                continue

            report.matched += 1

            if vf.weekday not in self.work_days:
                report.skipped_weekday += 1
                logger.debug("  SKIP (weekday %d not in work days): %s", vf.weekday, path.name)
                continue

            t = vf.start.time()
            if t < self.work_hours_start or t > self.work_hours_end:
                report.skipped_hours += 1
                logger.debug("  SKIP (time %s outside %s–%s): %s",
                             t, self.work_hours_start, self.work_hours_end, path.name)
                continue

            report.accepted.append(path.name)
            results.append(vf)

        report.log()
        return sorted(results, key=lambda v: v.start)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse(self, path: Path) -> Optional[VideoFile]:
        for parser in self.parsers:
            try:
                result = parser(path)
            except Exception:  # noqa: BLE001
                result = None
            if result is not None:
                return result
        return None
