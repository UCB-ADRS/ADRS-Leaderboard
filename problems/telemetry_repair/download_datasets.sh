#!/usr/bin/env bash
set -euo pipefail

# Downloads datasets for telemetry_repair problem (copies from local resources)

PROBLEM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Go up from problem_dir to problems/, then to repo root
BASE_DIR=$(cd "$PROBLEM_DIR/../.." && pwd)
DATASETS_DIR="$BASE_DIR/datasets/telemetry_repair"

mkdir -p "$DATASETS_DIR"

echo "[telemetry_repair download] Checking for datasets..."

# Check if dataset already exists
if [[ -d "$DATASETS_DIR" ]] && [[ -n $(ls -A "$DATASETS_DIR" 2>/dev/null) ]]; then
  echo "[telemetry_repair download] Dataset already exists at $DATASETS_DIR"
  exit 0
fi

# Copy datasets from problem resources
SRC_DIR="$PROBLEM_DIR/resources/datasets"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "[telemetry_repair download] No external datasets required for this problem"
  echo "[telemetry_repair download] Test data is generated programmatically by the evaluator"
  exit 0
fi

echo "[telemetry_repair download] Copying datasets..."
cp -r "$SRC_DIR"/* "$DATASETS_DIR/"

echo "[telemetry_repair download] Dataset ready at $DATASETS_DIR"
