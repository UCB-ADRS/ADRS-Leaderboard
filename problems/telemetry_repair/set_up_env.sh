#!/usr/bin/env bash
set -euo pipefail

# Usage: ./set_up_env.sh [config_path]

CONFIG_PATH=${1:-config.yaml}
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: config file not found at $CONFIG_PATH" >&2
  exit 1
fi

PROBLEM_DIR=$(pwd)
# Create venv inside resources directory (not at root execution_env)
# For container execution, execution_env is created by run_in_container.sh
RESOURCES_DIR="$PROBLEM_DIR/resources"
VENV_DIR="$RESOURCES_DIR/.venv"

# Parse config: first line uv_project (may be empty), subsequent lines dataset JSON objects.
CONFIG_LINES=()
while IFS= read -r line; do
    CONFIG_LINES+=("$line")
done < <(python3 - <<'PY' "$CONFIG_PATH"
import json, sys
from pathlib import Path
cfg_path = Path(sys.argv[1])
try:
    data = json.load(cfg_path.open())
except json.JSONDecodeError as e:
    raise SystemExit(f"Failed to parse {cfg_path}: {e}")
print(data.get("dependencies", {}).get("uv_project", ""))
for dataset in data.get("datasets", []):
    print(json.dumps(dataset))
PY
)

if [[ ${#CONFIG_LINES[@]} -eq 0 ]]; then
  echo "Error: config ${CONFIG_PATH} is empty or invalid" >&2
  exit 1
fi

UV_PROJECT_REL=${CONFIG_LINES[0]}
DATASET_LINES=("${CONFIG_LINES[@]:1}")

echo "[prepare_env] Creating/updating venv at $VENV_DIR"
uv venv "$VENV_DIR"
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

if [[ -n "$UV_PROJECT_REL" ]]; then
  UV_PROJECT_PATH=$(python3 - <<'PY' "$UV_PROJECT_REL"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)
  if [[ ! -f "$UV_PROJECT_PATH/pyproject.toml" ]]; then
    echo "Error: uv project path $UV_PROJECT_REL (resolved to $UV_PROJECT_PATH) missing pyproject.toml" >&2
    exit 1
  fi
  echo "[prepare_env] uv sync project=$UV_PROJECT_PATH"
  uv --project "$UV_PROJECT_PATH" sync --active
else
  echo "[prepare_env] No uv project specified; skipping dependency sync"
fi

echo "[prepare_env] Completed."
