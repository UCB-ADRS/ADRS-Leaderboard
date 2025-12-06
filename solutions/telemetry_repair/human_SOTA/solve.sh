#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# For telemetry_repair, execution_env is in problem resources directory
# If PROBLEM_NAME is set, use it; otherwise fall back to old location for compatibility
if [[ -n "${PROBLEM_NAME:-}" ]]; then
    # Go up 3 levels from solutions/{problem}/{baseline} to repo root, then to problems/
    TARGET_DIR="$SCRIPT_DIR/../../../problems/${PROBLEM_NAME}/resources/execution_env/solution_env"
else
    # Fallback: go up 3 levels to repo root, then to execution_env
    TARGET_DIR="$SCRIPT_DIR/../../../execution_env/solution_env"
fi
mkdir -p "$TARGET_DIR"
cp "$SCRIPT_DIR/resources/solution.py" "$TARGET_DIR/solution.py"
echo "[TELEMETRY_REPAIR human_SOTA] solution.py staged"
