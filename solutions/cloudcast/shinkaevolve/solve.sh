#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# For cloudcast, execution_env is in problem resources directory
# If PROBLEM_NAME is set, use it; otherwise fall back to old location for compatibility
if [[ -n "${PROBLEM_NAME:-}" ]]; then
    # Go up 3 levels from solutions/{problem}/{method} to repo root, then to problems/
    TARGET_DIR="$SCRIPT_DIR/../../../problems/${PROBLEM_NAME}/resources/execution_env/solution_env"
else
    # Fallback: go up 3 levels to repo root, then to execution_env
    TARGET_DIR="$SCRIPT_DIR/../../../problems/cloudcast/resources/execution_env/solution_env"
fi
mkdir -p "$TARGET_DIR"
# Copy all Python files from resources directory (solution.py, solver.py, utils.py, etc.)
cp "$SCRIPT_DIR/resources/"*.py "$TARGET_DIR/"
echo "[cloudcast shinka] All Python files staged"

