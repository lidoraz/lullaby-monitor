"""
Baby-cry + caregiver-yelling co-occurrence detector — CLI.

⚠  DISCLAIMER: This is an assistive screening tool only.  Audio pattern
   recognition has inherent limitations and will produce false positives
   and false negatives.  It is NOT a definitive assessment of abuse.
   Always involve qualified professionals when child safety is at risk.

Usage
-----
    python main_abuse.py <audio_file> [options]

Examples
--------
    # Analyse a recording with default settings
    python main_abuse.py baby_monitor.wav

    # Lower thresholds for a quieter/muffled recording
    python main_abuse.py recording.mp3 --cry-threshold 0.15 --yell-threshold 0.15

    # Save the report
    python main_abuse.py recording.wav --output concern_report.txt

    # Show per-frame scores
    python main_abuse.py recording.wav --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from abuse_detector import AbuseDetector, Severity

_DISCLAIMER = """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DISCLAIMER: This tool is an ASSISTIVE SCREENING AID only.
  Audio pattern recognition produces false positives and
  false negatives.  It is NOT a definitive diagnostic of
  abuse.  Always consult qualified professionals.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )


_SEVERITY_LABEL = {
    Severity.LOW:    "LOW    — low-confidence overlap; treat as background noise "
                     "unless repeated.",
    Severity.MEDIUM: "MEDIUM — one signal is strong; review this segment.",
    Severity.HIGH:   "HIGH   — both cry and yelling are strong; URGENT review "
                     "recommended.",
}


def _format_report(
    audio_path: Path,
    episodes,
    cry_threshold: float,
    yell_threshold: float,
    verbose: bool,
) -> str:
    sep  = "─" * 64
    sep2 = "═" * 64
    lines: list[str] = []

    lines.append(sep2)
    lines.append("  Baby-Cry + Caregiver-Yelling Co-occurrence Report")
    lines.append(sep2)
    lines.append(f"  File           : {audio_path}")
    lines.append(f"  Cry threshold  : {cry_threshold:.2f}")
    lines.append(f"  Yell threshold : {yell_threshold:.2f}")
    lines.append(sep)

    if not episodes:
        lines.append("  Result  : No concerning co-occurrences detected.")
        lines.append("            (baby cry and yelling did not overlap)")
    else:
        high   = [e for e in episodes if e.severity == Severity.HIGH]
        medium = [e for e in episodes if e.severity == Severity.MEDIUM]
        low    = [e for e in episodes if e.severity == Severity.LOW]

        lines.append(
            f"  Result  : {len(episodes)} episode(s) found  "
            f"[ HIGH: {len(high)}  MEDIUM: {len(medium)}  LOW: {len(low)} ]"
        )
        lines.append("")

        # Print HIGH first, then MEDIUM, then LOW
        ordered = high + medium + low
        for idx, ep in enumerate(ordered, start=1):
            lines.append(f"  Episode {idx:>2} : {ep}")
            lines.append(
                f"             {_SEVERITY_LABEL[ep.severity]}"
            )
            if verbose:
                cry_str  = "  ".join(f"{c:.2f}" for c in ep.cry_frame_confidences)
                yell_str = "  ".join(f"{c:.2f}" for c in ep.yell_frame_confidences)
                lines.append(f"             cry  frames : {cry_str}")
                lines.append(f"             yell frames : {yell_str}")
            lines.append("")

        # Urgent notice if any HIGH episodes found
        if high:
            lines.append(sep)
            lines.append(
                "  ⚠  HIGH-severity episodes detected.  Please review "
                "immediately and"
            )
            lines.append(
                "     involve appropriate professionals if needed."
            )

    lines.append(sep2)
    lines.append(_DISCLAIMER)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect co-occurrence of baby crying and caregiver yelling "
            "in an audio recording."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "audio_file",
        help="Path to an audio file (wav, mp3, flac, ogg, …).",
    )
    parser.add_argument(
        "--cry-threshold",
        type=float,
        default=0.20,
        metavar="FLOAT",
        help="Baby-cry confidence threshold (0–1). Default: 0.20.",
    )
    parser.add_argument(
        "--yell-threshold",
        type=float,
        default=0.20,
        metavar="FLOAT",
        help="Adult-yelling confidence threshold (0–1). Default: 0.20.",
    )
    parser.add_argument(
        "--co-window",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help=(
            "Maximum gap (seconds) between a cry frame and a yell frame "
            "that still counts as co-occurring. Default: 2.0."
        ),
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Silence gap (seconds) within which episodes are merged. Default: 2.0.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help="Discard episodes shorter than this (noise filter). Default: 0.5.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Also write the report to FILE.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-frame scores and debug logs.",
    )

    args = parser.parse_args()
    _configure_logging(args.verbose)

    audio_path = Path(args.audio_file)

    detector = AbuseDetector(
        cry_threshold=args.cry_threshold,
        yell_threshold=args.yell_threshold,
        co_window=args.co_window,
        merge_gap=args.merge_gap,
        min_duration=args.min_duration,
    )

    try:
        episodes = detector.detect(audio_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:          # noqa: BLE001
        print(f"Unexpected error: {exc}", file=sys.stderr)
        logging.exception("Traceback:")
        sys.exit(1)

    report = _format_report(
        audio_path,
        episodes,
        args.cry_threshold,
        args.yell_threshold,
        args.verbose,
    )
    print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(report, encoding="utf-8")
        print(f"\nReport saved to: {out_path}")

    # Exit codes:
    #   0  = no concern detected
    #   1  = LOW / MEDIUM episodes found
    #   2  = HIGH episodes found (useful for scripting urgent alerts)
    from abuse_detector import Severity as S
    if any(e.severity == S.HIGH for e in episodes):
        sys.exit(2)
    elif episodes:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
