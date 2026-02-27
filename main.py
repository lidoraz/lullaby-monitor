"""
Baby Cry Detector — command-line interface.

Usage
-----
    python main.py <audio_file> [options]

Examples
--------
    # Analyse a recording with default settings
    python main.py baby_monitor.wav

    # Lower the threshold to catch quieter cries
    python main.py recording.mp3 --threshold 0.15

    # Tighten the threshold and save the log to a file
    python main.py recording.wav --threshold 0.35 --output results.txt

    # Show full per-frame details
    python main.py recording.wav --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from detector import BabyCryDetector


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _format_report(
    audio_path: Path,
    episodes,
    threshold: float,
    verbose: bool,
) -> str:
    lines: list[str] = []
    sep = "─" * 60

    lines.append(sep)
    lines.append(f"  Baby Cry Detection Report")
    lines.append(sep)
    lines.append(f"  File      : {audio_path}")
    lines.append(f"  Threshold : {threshold:.2f}")
    lines.append(sep)

    if not episodes:
        lines.append("  Result    : No baby cry detected.")
    else:
        lines.append(f"  Result    : {len(episodes)} cry episode(s) found.\n")
        for idx, ep in enumerate(episodes, start=1):
            lines.append(f"  Episode {idx:>2} : {ep}")
            if verbose:
                frame_list = "  ".join(f"{c:.2f}" for c in ep.frame_confidences)
                lines.append(f"             frames: {frame_list}")

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect baby cry events in an audio file using YAMNet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "audio_file",
        help="Path to an audio file (wav, mp3, flac, ogg, …).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        metavar="FLOAT",
        help=(
            "Confidence threshold [0–1] for flagging a frame as a cry. "
            "Lower = more sensitive, higher = more strict. (default: 0.25)"
        ),
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=1.5,
        metavar="SECONDS",
        help=(
            "Maximum silence gap (seconds) between flagged frames that are "
            "still merged into the same episode. (default: 1.5)"
        ),
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help=(
            "Discard episodes shorter than this many seconds (noise filter). "
            "(default: 0.5)"
        ),
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write the report to FILE in addition to printing it.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-frame confidence scores and debug logs.",
    )

    args = parser.parse_args()
    _configure_logging(args.verbose)

    audio_path = Path(args.audio_file)

    detector = BabyCryDetector(
        threshold=args.threshold,
        merge_gap=args.merge_gap,
        min_duration=args.min_duration,
    )

    try:
        episodes = detector.detect(audio_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error: {exc}", file=sys.stderr)
        logging.exception("Traceback:")
        sys.exit(1)

    report = _format_report(audio_path, episodes, args.threshold, args.verbose)
    print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(report, encoding="utf-8")
        print(f"\nReport saved to: {out_path}")

    # Exit code: 0 if no cry detected, 1 if cry(s) detected
    # (useful for scripting: `if python main.py baby.wav; then echo "quiet"; fi`)
    sys.exit(0 if not episodes else 1)


if __name__ == "__main__":
    main()
