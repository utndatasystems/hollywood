#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$BASE_DIR/.venv-linux}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"

"$PYTHON_BOOTSTRAP" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$BASE_DIR/requirements.txt"
python - <<'PY'
import pandas, pyarrow, numpy, requests, dotenv
print("Linux pipeline environment OK")
PY
