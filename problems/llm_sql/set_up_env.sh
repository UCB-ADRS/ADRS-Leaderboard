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
    local_tar)
      if [[ -z "$dataset_path_rel" || -z "$target_rel" ]]; then
        echo "Error: dataset entry missing path or target: $dataset_json" >&2
        exit 1
      fi
      TAR_PATH="$PROBLEM_DIR/$dataset_path_rel"
      TARGET_DIR="$PROBLEM_DIR/$target_rel"
      mkdir -p "$TARGET_DIR"
      has_expected=false
      if [[ -n "$expected_glob_pattern" ]]; then
        if compgen -G "$TARGET_DIR/$expected_glob_pattern" >/dev/null 2>&1; then
          has_expected=true
        fi
      else
        # If no glob provided, check directory non-empty
        if [[ -n $(ls -A "$TARGET_DIR" 2>/dev/null) ]]; then
          has_expected=true
        fi
      fi
      if $has_expected; then
        echo "[prepare_env] Dataset already present at $TARGET_DIR"
        continue
      fi
      
      # Check if dataset is available in mounted /datasets folder
      MOUNTED_DATASETS="/datasets/llm_sql"
      if [[ -d "$MOUNTED_DATASETS" ]] && [[ -n $(ls -A "$MOUNTED_DATASETS" 2>/dev/null) ]]; then
        echo "[prepare_env] Using pre-downloaded dataset from $MOUNTED_DATASETS"
        # Create symlinks to mounted datasets instead of copying
        ln -sf "$MOUNTED_DATASETS"/* "$TARGET_DIR/"
        continue
      fi
      
      if [[ ! -f "$TAR_PATH" ]]; then
        echo "Error: dataset tarball missing at $TAR_PATH" >&2
        exit 1
      fi
      echo "[prepare_env] Extracting $TAR_PATH → $TARGET_DIR"
      tar -xzf "$TAR_PATH" -C "$TARGET_DIR" 2>/dev/null || true
      if [[ -n "$expected_glob_pattern" ]] && ! compgen -G "$TARGET_DIR/$expected_glob_pattern" >/dev/null 2>&1; then
        echo "Error: expected files matching $TARGET_DIR/$expected_glob_pattern not found after extraction" >&2
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

# For llm_sql, copy datasets from downloaded location to resources/datasets for local use
# (In containers, datasets are mounted at /datasets/llm_sql)
LOCAL_DATASETS_TARGET="$RESOURCES_DIR/datasets"
BASE_DIR=$(cd "$PROBLEM_DIR/../../.." && pwd)
DOWNLOADED_DATASETS_DIR="$BASE_DIR/datasets/llm_sql"
MOUNTED_DATASETS="/datasets/llm_sql"

# Check if datasets are already in resources/datasets
if [[ -d "$LOCAL_DATASETS_TARGET" ]] && [[ -n $(ls -A "$LOCAL_DATASETS_TARGET" 2>/dev/null) ]]; then
  # Verify all required datasets are present
  all_present=true
  for dataset in "movies.csv" "beer.csv" "BIRD.csv" "PDMX.csv" "products.csv"; do
    if [[ ! -f "$LOCAL_DATASETS_TARGET/$dataset" ]]; then
      all_present=false
      break
    fi
  done
  if [[ "$all_present" == true ]]; then
    echo "[prepare_env] Datasets already present in $LOCAL_DATASETS_TARGET"
  else
    all_present=false
  fi
else
  all_present=false
fi

# If datasets not in resources/datasets, copy from downloaded location or mounted location
if [[ "$all_present" != true ]]; then
  mkdir -p "$LOCAL_DATASETS_TARGET"
  
  # First try mounted location (for containers)
  if [[ -d "$MOUNTED_DATASETS" ]] && [[ -n $(ls -A "$MOUNTED_DATASETS" 2>/dev/null) ]]; then
    echo "[prepare_env] Copying datasets from mounted location $MOUNTED_DATASETS to $LOCAL_DATASETS_TARGET"
    cp -f "$MOUNTED_DATASETS"/*.csv "$LOCAL_DATASETS_TARGET/" 2>/dev/null || true
  # Then try downloaded location (for local execution)
  elif [[ -d "$DOWNLOADED_DATASETS_DIR" ]] && [[ -n $(ls -A "$DOWNLOADED_DATASETS_DIR" 2>/dev/null) ]]; then
    echo "[prepare_env] Copying datasets from downloaded location $DOWNLOADED_DATASETS_DIR to $LOCAL_DATASETS_TARGET"
    cp -f "$DOWNLOADED_DATASETS_DIR"/*.csv "$LOCAL_DATASETS_TARGET/" 2>/dev/null || true
  else
    echo "[prepare_env] WARNING: No datasets found in mounted location ($MOUNTED_DATASETS) or downloaded location ($DOWNLOADED_DATASETS_DIR)" >&2
    echo "[prepare_env] Please run download_datasets.sh first" >&2
  fi
  
  # Verify datasets were copied
  all_present=true
  for dataset in "movies.csv" "beer.csv" "BIRD.csv" "PDMX.csv" "products.csv"; do
    if [[ ! -f "$LOCAL_DATASETS_TARGET/$dataset" ]]; then
      echo "[prepare_env] WARNING: Dataset $dataset not found in $LOCAL_DATASETS_TARGET" >&2
      all_present=false
    fi
  done
  
  if [[ "$all_present" == true ]]; then
    echo "[prepare_env] All datasets copied successfully to $LOCAL_DATASETS_TARGET"
  fi
fi

echo "[prepare_env] Completed."

