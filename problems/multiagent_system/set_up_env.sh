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
# Remove existing venv if it exists to avoid interactive prompt
if [[ -d "$VENV_DIR" ]]; then
  rm -rf "$VENV_DIR"
fi
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
  echo "[prepare_env] No uv project specified; installing from requirements.txt"
  if [[ -f "$PROBLEM_DIR/requirements.txt" ]]; then
    echo "[prepare_env] Installing requirements from $PROBLEM_DIR/requirements.txt"
    uv pip install -r "$PROBLEM_DIR/requirements.txt"
  else
    echo "[prepare_env] WARNING: requirements.txt not found; installing minimal deps"
    uv pip install aiohttp pydantic openai pyyaml numpy tqdm
  fi
fi

# Helper to decode dataset json fields
decode_dataset_field() {
  local json_input="$1"
  local key="$2"
  python3 - <<'PY' "$json_input" "$key"
import json, sys
obj = json.loads(sys.argv[1])
print(obj.get(sys.argv[2], ""))
PY
}

# Only iterate if DATASET_LINES is not empty
if [[ ${#DATASET_LINES[@]} -gt 0 ]]; then
  for dataset_json in "${DATASET_LINES[@]}"; do
    [[ -z "$dataset_json" ]] && continue
  dataset_type=$(decode_dataset_field "$dataset_json" "type")
  dataset_path_rel=$(decode_dataset_field "$dataset_json" "path")
  target_rel=$(decode_dataset_field "$dataset_json" "target")
  expected_glob_pattern=$(decode_dataset_field "$dataset_json" "expected_glob")

  case "$dataset_type" in
    git_clone)
      # For git_clone, path can be empty (repo URL is in download_datasets.sh), but target is required
      if [[ -z "$target_rel" ]]; then
        echo "Error: dataset entry missing target: $dataset_json" >&2
        exit 1
      fi
      # For git_clone type, we download to resources directory
      # Check if dataset is already in resources/datasets
      LOCAL_DATASETS="$RESOURCES_DIR/datasets"
      if [[ -d "$LOCAL_DATASETS" ]] && [[ -n $(ls -A "$LOCAL_DATASETS" 2>/dev/null) ]]; then
        if [[ -n "$expected_glob_pattern" ]]; then
          if compgen -G "$LOCAL_DATASETS/$expected_glob_pattern" >/dev/null 2>&1; then
            echo "[prepare_env] Dataset already present at $LOCAL_DATASETS"
            continue
          fi
        else
          echo "[prepare_env] Dataset already present at $LOCAL_DATASETS"
          continue
        fi
      fi
      
      # Check if dataset is available in mounted /datasets folder (for containers)
      MOUNTED_DATASETS="/datasets/multiagent_system/openevolve-mast/example_mas/programdev"
      if [[ -d "$MOUNTED_DATASETS" ]] && [[ -n $(ls -A "$MOUNTED_DATASETS" 2>/dev/null) ]]; then
        echo "[prepare_env] Using pre-downloaded dataset from $MOUNTED_DATASETS"
        mkdir -p "$LOCAL_DATASETS"
        cp -rf "$MOUNTED_DATASETS"/* "$LOCAL_DATASETS/" 2>/dev/null || true
        continue
      fi
      
      # Run download_datasets.sh if it exists
      if [[ -f "$PROBLEM_DIR/download_datasets.sh" ]]; then
        echo "[prepare_env] Running download_datasets.sh to fetch dataset..."
        bash "$PROBLEM_DIR/download_datasets.sh"
      else
        echo "Error: download_datasets.sh not found and dataset not available" >&2
        exit 1
      fi
      ;;
    *)
      echo "Error: unsupported dataset type '$dataset_type' in $dataset_json" >&2
      exit 1
      ;;
  esac
  done
fi

# For multiagent_system, datasets are in resources/datasets (downloaded by download_datasets.sh)
# (In containers, datasets are mounted at /datasets/multiagent_system)
LOCAL_DATASETS_TARGET="$RESOURCES_DIR/datasets"
MOUNTED_DATASETS="/datasets/multiagent_system/openevolve-mast/example_mas/programdev"

# Check if datasets are already in resources/datasets
if [[ -d "$LOCAL_DATASETS_TARGET" ]] && [[ -n $(ls -A "$LOCAL_DATASETS_TARGET" 2>/dev/null) ]]; then
  echo "[prepare_env] Datasets already present in $LOCAL_DATASETS_TARGET"
else
  # If datasets not in resources/datasets, try mounted location (for containers)
  if [[ -d "$MOUNTED_DATASETS" ]] && [[ -n $(ls -A "$MOUNTED_DATASETS" 2>/dev/null) ]]; then
    echo "[prepare_env] Copying datasets from mounted location $MOUNTED_DATASETS to $LOCAL_DATASETS_TARGET"
    mkdir -p "$LOCAL_DATASETS_TARGET"
    cp -rf "$MOUNTED_DATASETS"/* "$LOCAL_DATASETS_TARGET/" 2>/dev/null || true
    echo "[prepare_env] Datasets copied successfully to $LOCAL_DATASETS_TARGET"
  else
    echo "[prepare_env] WARNING: No datasets found in $LOCAL_DATASETS_TARGET or mounted location ($MOUNTED_DATASETS)" >&2
    echo "[prepare_env] Please run download_datasets.sh first" >&2
  fi
fi

echo "[prepare_env] Completed."
