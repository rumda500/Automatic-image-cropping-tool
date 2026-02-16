#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

show_menu() {
  cat <<'EOF'

=========== BiRefNet-WebUI Launcher ===========
  1) Web UIを起動
  2) CLIヘルプを表示
  q) 終了
===============================================

EOF
}

while true; do
  show_menu
  read -rp "選択してください: " choice
  case "$choice" in
    1)
      ./run_ui.sh
      ;;
    2)
      if [[ -x ".venv/bin/python" ]]; then
        .venv/bin/python cutout_tool.py --help
      else
        python3 cutout_tool.py --help
      fi
      ;;
    q|Q)
      echo "終了します。"
      exit 0
      ;;
    *)
      echo "[WARN] 無効な入力です: $choice"
      ;;
  esac
done
