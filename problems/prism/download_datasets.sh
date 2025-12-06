#!/usr/bin/env bash
set -euo pipefail

# Downloads datasets for prism problem (none required)

PROBLEM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Go up from problem_dir to problems/, then to repo root
BASE_DIR=$(cd "$PROBLEM_DIR/../.." && pwd)
DATASETS_DIR="$BASE_DIR/datasets/prism"

mkdir -p "$DATASETS_DIR"

echo "[prism download] Checking for datasets..."
echo "[prism download] No external datasets required for this problem"
echo "[prism download] Test data is generated programmatically by the evaluator"

