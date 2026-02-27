"""
SQLite persistence layer.

Schema
------
  recordings   — one row per processed video file
  events       — one row per detected audio event (FK → recordings)

The DB file defaults to  crybaby/data/crybaby.db
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional

from pipeline.processor import ProcessingResult

_DEFAULT_DB = Path(__file__).parent.parent / "data" / "crybaby.db"

_DDL = """
CREATE TABLE IF NOT EXISTS recordings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path       TEXT    NOT NULL UNIQUE,
    device_id       TEXT    NOT NULL DEFAULT 'unknown',
    rec_start       TEXT    NOT NULL,   -- ISO-8601
    rec_end         TEXT    NOT NULL,
    date_label      TEXT    NOT NULL,   -- YYYY-MM-DD
    status          TEXT    NOT NULL,   -- ok | error | skipped
    duration        REAL    NOT NULL DEFAULT 0,
    silent_fraction REAL    NOT NULL DEFAULT 0,
    silent_regions  TEXT    NOT NULL DEFAULT '[]',  -- JSON
    active_regions  TEXT    NOT NULL DEFAULT '[]',  -- JSON
    processed_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    error_message   TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id    INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,

    -- Absolute wall-clock time of the event
    abs_start       TEXT    NOT NULL,   -- ISO-8601 (rec_start + offset)
    abs_end         TEXT    NOT NULL,

    -- Offset in seconds within the audio file
    offset_start    REAL    NOT NULL,
    offset_end      REAL    NOT NULL,

    event_type      TEXT    NOT NULL,   -- baby_cry | yell | loud_noise | abuse
    severity        TEXT    NOT NULL,   -- LOW | MEDIUM | HIGH
    peak_conf       REAL    NOT NULL,
    mean_conf       REAL    NOT NULL,
    peak_secondary  REAL
);

CREATE INDEX IF NOT EXISTS idx_recordings_date ON recordings(date_label);
CREATE INDEX IF NOT EXISTS idx_events_rec      ON events(recording_id);
CREATE INDEX IF NOT EXISTS idx_events_type     ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_abs      ON events(abs_start);
"""


class Database:
    def __init__(self, path: Path = _DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        # Re-create the directory in case it was deleted while the server runs
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode = WAL")
        # Always ensure the schema exists (CREATE TABLE IF NOT EXISTS → no-op
        # when tables are present; re-bootstraps when the DB file was deleted).
        con.executescript(_DDL)
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init(self) -> None:
        with self._conn() as con:
            pass  # schema is now applied inside _conn()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_result(self, result: ProcessingResult) -> int:
        """
        Upsert a :class:`ProcessingResult` and all its events.
        Returns the recording row id.
        """
        vf  = result.video
        sm  = result.silence_map
        d   = result.to_dict()

        with self._conn() as con:
            cur = con.execute(
                """
                INSERT INTO recordings
                    (file_path, device_id, rec_start, rec_end, date_label,
                     status, duration, silent_fraction, silent_regions,
                     active_regions, processed_at, error_message)
                VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'),?)
                ON CONFLICT(file_path) DO UPDATE SET
                    status          = excluded.status,
                    duration        = excluded.duration,
                    silent_fraction = excluded.silent_fraction,
                    silent_regions  = excluded.silent_regions,
                    active_regions  = excluded.active_regions,
                    processed_at    = excluded.processed_at,
                    error_message   = excluded.error_message
                """,
                (
                    str(vf.path),
                    vf.device_id,
                    vf.start.isoformat(),
                    vf.end.isoformat(),
                    vf.date_label,
                    result.status,
                    result.duration,
                    round(sm.silent_fraction, 4) if sm else 0.0,
                    json.dumps(d["silent_regions"]),
                    json.dumps(d["active_regions"]),
                    result.error_message,
                ),
            )
            rec_id = cur.lastrowid

            # Delete existing events for this recording then re-insert
            con.execute("DELETE FROM events WHERE recording_id = ?", (rec_id,))

            for ev in result.events:
                # Compute absolute wall-clock timestamps
                from datetime import timedelta
                abs_start = vf.start + timedelta(seconds=ev.start)
                abs_end   = vf.start + timedelta(seconds=ev.end)
                con.execute(
                    """
                    INSERT INTO events
                        (recording_id, abs_start, abs_end,
                         offset_start, offset_end,
                         event_type, severity,
                         peak_conf, mean_conf, peak_secondary)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        rec_id,
                        abs_start.isoformat(),
                        abs_end.isoformat(),
                        round(ev.start, 3),
                        round(ev.end,   3),
                        ev.type.value,
                        ev.severity.value,
                        round(ev.peak_confidence, 4),
                        round(ev.mean_confidence, 4),
                        round(ev.peak_secondary, 4) if ev.peak_secondary else None,
                    ),
                )

        return rec_id

    def get_cached_paths(self) -> set:
        """Return the set of file paths already stored (status=ok)."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT file_path FROM recordings WHERE status = 'ok'"
            ).fetchall()
        return {row["file_path"] for row in rows}

    # ------------------------------------------------------------------
    # Read — used by the dashboard API
    # ------------------------------------------------------------------

    def get_dates(self) -> List[str]:
        """Return distinct date labels (YYYY-MM-DD) with at least one recording."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT DISTINCT date_label FROM recordings ORDER BY date_label"
            ).fetchall()
        return [r["date_label"] for r in rows]

    def get_recordings_for_date(self, date: str) -> List[dict]:
        """Return all recordings for a given date, with their events embedded."""
        with self._conn() as con:
            recs = con.execute(
                """
                SELECT * FROM recordings
                WHERE date_label = ?
                ORDER BY rec_start
                """,
                (date,),
            ).fetchall()

            result = []
            for rec in recs:
                rec_dict = dict(rec)
                rec_dict["silent_regions"] = json.loads(rec_dict["silent_regions"])
                rec_dict["active_regions"] = json.loads(rec_dict["active_regions"])

                events = con.execute(
                    """
                    SELECT * FROM events
                    WHERE recording_id = ?
                    ORDER BY offset_start
                    """,
                    (rec_dict["id"],),
                ).fetchall()

                rec_dict["events"] = [dict(e) for e in events]
                result.append(rec_dict)

        return result

    def get_all_events_for_date(self, date: str) -> List[dict]:
        """Return all events for a given date (joined with recording info)."""
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT
                    e.*,
                    r.file_path, r.rec_start, r.rec_end, r.device_id, r.duration
                FROM events e
                JOIN recordings r ON r.id = e.recording_id
                WHERE r.date_label = ?
                ORDER BY e.abs_start
                """,
                (date,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """High-level statistics for the dashboard summary card."""
        with self._conn() as con:
            total_recs = con.execute(
                "SELECT COUNT(*) FROM recordings WHERE status='ok'"
            ).fetchone()[0]
            total_events = con.execute(
                "SELECT COUNT(*) FROM events"
            ).fetchone()[0]
            by_type = con.execute(
                "SELECT event_type, COUNT(*) as cnt FROM events GROUP BY event_type"
            ).fetchall()
            dates_count = con.execute(
                "SELECT COUNT(DISTINCT date_label) FROM recordings"
            ).fetchone()[0]

        return {
            "total_recordings": total_recs,
            "total_events":     total_events,
            "dates_count":      dates_count,
            "by_type":          {r["event_type"]: r["cnt"] for r in by_type},
        }
