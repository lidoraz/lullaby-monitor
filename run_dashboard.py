#!/usr/bin/env python
"""
Convenience launcher for the crybaby dashboard.

    python run_dashboard.py [--port 7860] [--host 0.0.0.0]
"""
import argparse
import webbrowser
import threading
import time

import uvicorn


def _open_browser(url: str, delay: float = 1.2) -> None:
    def _go():
        time.sleep(delay)
        webbrowser.open(url)
    threading.Thread(target=_go, daemon=True).start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Lullaby Monitor dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print(f"\n�  Lullaby Monitor → {url}\n")

    if not args.no_browser:
        _open_browser(url)

    uvicorn.run(
        "dashboard.app:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )
