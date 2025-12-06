#!/usr/bin/env bash
set -euo pipefail

# Usage: ./set_up_env.sh [config_path]

CONFIG_PATH=${1:-config.yaml}
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: config file not found at $CONFIG_PATH" >&2
  exit 1
fi

PROBLEM_DIR=$(pwd)
# Use execution_env inside resources directory (matching test_local.sh)
EXEC_ROOT="$PROBLEM_DIR/resources/execution_env"
mkdir -p "$EXEC_ROOT"

UV_PROJECT_REL=$(
  python3 - <<'PY' "$CONFIG_PATH"
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
print(cfg.get("dependencies", {}).get("uv_project", ""))
PY
)

if ! command -v uv >/dev/null 2>&1; then
  echo "[cloudcast setup] Installing uv..."
  pip install --user uv || exit 1
  export PATH="$HOME/.local/bin:$PATH"
fi

VENV_DIR="$EXEC_ROOT/.venv"
echo "[cloudcast setup] Creating venv at $VENV_DIR"
uv venv "$VENV_DIR"
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

if [[ -n "$UV_PROJECT_REL" ]]; then
  UV_PROJECT_PATH="$PROBLEM_DIR/$UV_PROJECT_REL"
  if [[ ! -f "$UV_PROJECT_PATH/pyproject.toml" ]]; then
    echo "Error: uv project path $UV_PROJECT_PATH missing pyproject.toml" >&2
    exit 1
  fi
  echo "[cloudcast setup] Syncing dependencies from $UV_PROJECT_PATH"
  uv --project "$UV_PROJECT_PATH" sync --active
else
  echo "[cloudcast setup] No uv project specified; installing core requirements"
  uv pip install "numpy>=1.24" "networkx>=3.0"
fi

echo "[cloudcast setup] Ensuring runtime dependencies are available"
uv pip install "networkx>=3.0" "numpy>=1.24" "colorama>=0.4.6" "pandas>=1.5" "graphviz>=0.20"

echo "[cloudcast setup] Environment setup completed successfully"
