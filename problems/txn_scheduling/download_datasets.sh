#!/usr/bin/env bash
set -euo pipefail

# Downloads datasets for txn_scheduling problem (none required)

PROBLEM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Go up from problem_dir to problems/, then to repo root
BASE_DIR=$(cd "$PROBLEM_DIR/../.." && pwd)
DATASETS_DIR="$BASE_DIR/datasets/txn_scheduling"

mkdir -p "$DATASETS_DIR"

echo "[txn_scheduling download] Checking for datasets..."
echo "[txn_scheduling download] No external datasets required for this problem"
echo "[txn_scheduling download] Test data is generated programmatically by the evaluator"
