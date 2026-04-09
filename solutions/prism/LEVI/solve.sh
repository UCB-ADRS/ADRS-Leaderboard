#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -n "${PROBLEM_NAME:-}" ]]; then
    TARGET_DIR="$SCRIPT_DIR/../../../problems/${PROBLEM_NAME}/resources/execution_env/solution_env"
else
    TARGET_DIR="$SCRIPT_DIR/../../../execution_env/solution_env"
fi
mkdir -p "$TARGET_DIR"
cp "$SCRIPT_DIR/resources/solution.py" "$TARGET_DIR/solution.py"
echo "[LEVI] solution.py staged"
