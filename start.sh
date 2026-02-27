#!/usr/bin/env bash
# =============================================================
#  crybaby — start the dashboard
#  Run:  bash start.sh
#  Run on a custom port:  bash start.sh --port 8080
# =============================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'

info()  { echo -e "${CYAN}▸ $*${RESET}"; }
error() { echo -e "${RED}✗ $*${RESET}"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"

# ── Check setup was run ────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  error "Virtual environment not found.\n  Please run setup first:  bash setup.sh"
fi

# ── Activate ───────────────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── Parse optional --port argument ────────────────────────
PORT=7860
while [[ $# -gt 0 ]]; do
  case $1 in
    --port|-p) PORT="$2"; shift 2 ;;
    *) shift ;;
  esac
done

# ── Banner ─────────────────────────────────────────────────
echo ""
echo -e "${BOLD}🍼  crybaby Dashboard${RESET}"
echo "────────────────────────────────────────"
echo -e "  Open your browser at: ${GREEN}${BOLD}http://localhost:${PORT}${RESET}"
echo "  Press  Ctrl+C  to stop."
echo "────────────────────────────────────────"
echo ""

# ── Launch ─────────────────────────────────────────────────
python run_dashboard.py --port "$PORT"
