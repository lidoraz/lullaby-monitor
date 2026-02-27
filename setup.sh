#!/usr/bin/env bash
# =============================================================
#  crybaby — one-shot setup script
#  Run once:  bash setup.sh
# =============================================================
set -euo pipefail

# ── Colours ────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}▸ $*${RESET}"; }
success() { echo -e "${GREEN}✓ $*${RESET}"; }
warn()    { echo -e "${YELLOW}⚠ $*${RESET}"; }
error()   { echo -e "${RED}✗ $*${RESET}"; exit 1; }

# ── Banner ─────────────────────────────────────────────────
echo ""
echo -e "${BOLD}🍼  crybaby — Baby Monitor Setup${RESET}"
echo "────────────────────────────────────────"
echo ""

# ── Locate project root ────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
info "Project directory: $SCRIPT_DIR"

# ── Check Python ───────────────────────────────────────────
info "Checking Python …"
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3.9 python3 python; do
  if command -v "$candidate" &>/dev/null; then
    VER=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    MAJOR=$(echo "$VER" | cut -d. -f1)
    MINOR=$(echo "$VER" | cut -d. -f2)
    if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 9 ]; then
      PYTHON="$candidate"
      success "Found $PYTHON  (Python $VER)"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  error "Python 3.9+ is required but was not found.\n  Install it from https://www.python.org/downloads/ and re-run this script."
fi

# ── Check / install ffmpeg ─────────────────────────────────
info "Checking ffmpeg …"
if command -v ffmpeg &>/dev/null; then
  success "ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"
else
  warn "ffmpeg not found."
  if command -v brew &>/dev/null; then
    info "Installing ffmpeg via Homebrew …"
    brew install ffmpeg
    success "ffmpeg installed."
  else
    warn "Homebrew not found.  Please install ffmpeg manually:"
    warn "  macOS:  https://brew.sh  → then: brew install ffmpeg"
    warn "  Linux:  sudo apt install ffmpeg   (Debian/Ubuntu)"
    warn "          sudo dnf install ffmpeg   (Fedora)"
    warn "The dashboard will not work without ffmpeg."
  fi
fi

# ── Create virtual environment ─────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"
if [ -d "$VENV_DIR" ]; then
  info "Virtual environment already exists at .venv — skipping creation."
else
  info "Creating virtual environment at .venv …"
  "$PYTHON" -m venv "$VENV_DIR"
  success "Virtual environment created."
fi

# ── Activate venv ──────────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
success "Virtual environment activated."

# ── Upgrade pip ────────────────────────────────────────────
info "Upgrading pip …"
pip install --quiet --upgrade pip

# ── Detect Apple Silicon ───────────────────────────────────
ARCH=$(uname -m)
REQ_FILE="requirements.txt"

if [ "$ARCH" = "arm64" ]; then
  warn "Apple Silicon (M-series) detected."
  warn "We will use tensorflow-macos + tensorflow-metal instead of tensorflow."

  # Create a temporary patched requirements file
  TMP_REQ="$SCRIPT_DIR/.requirements_macos.txt"
  sed \
    -e 's/^tensorflow>=[^[:space:]]*/tensorflow-macos>=2.13/' \
    -e '/^# .*tensorflow-macos/d' \
    "$REQ_FILE" > "$TMP_REQ"

  # Add tensorflow-metal if not already present
  if ! grep -q "tensorflow-metal" "$TMP_REQ"; then
    echo "tensorflow-metal>=1.0" >> "$TMP_REQ"
  fi

  REQ_FILE="$TMP_REQ"
  info "Using patched requirements for Apple Silicon."
fi

# ── Install Python packages ────────────────────────────────
info "Installing Python packages (this may take a few minutes) …"
pip install --quiet -r "$REQ_FILE"

# Cleanup temp file if created
[ -f "$SCRIPT_DIR/.requirements_macos.txt" ] && rm -f "$SCRIPT_DIR/.requirements_macos.txt"

success "All Python packages installed."

# ── Create data directory ──────────────────────────────────
mkdir -p "$SCRIPT_DIR/data"
success "Data directory ready."

# ── Done ───────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}Setup complete! 🎉${RESET}"
echo ""
echo -e "To start the dashboard, run:"
echo -e "  ${BOLD}bash start.sh${RESET}"
echo ""
