#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ ! -f "$ROOT_DIR/birefnet_finetuned_toonout.pth" ]]; then
  echo "[INFO] Checkpoint not found. Downloading..."
  python fetch_checkpoint.py --output "$ROOT_DIR/birefnet_finetuned_toonout.pth"
fi

python app.py --checkpoint "$ROOT_DIR/birefnet_finetuned_toonout.pth" --device auto --host 127.0.0.1 --port 7860
