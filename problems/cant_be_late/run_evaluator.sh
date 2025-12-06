#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBLEM_NAME=$(basename "$SCRIPT_DIR")
BASE_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

# Use execution_env inside resources directory
EXEC_ROOT="$SCRIPT_DIR/resources/execution_env"
VENV_DIR="$SCRIPT_DIR/resources/.venv"

PYBIN="$VENV_DIR/bin/python"
if [[ ! -x "$PYBIN" ]]; then
  echo "Error: venv python not found at $PYBIN. Did you run set_up_env.sh?" >&2
  exit 1
fi

SOLUTION_PATH="$EXEC_ROOT/solution_env/solution.py"
if [[ ! -f "$SOLUTION_PATH" ]]; then
  echo "Error: solution.py not found at $SOLUTION_PATH" >&2
  exit 1
fi

# Determine solution name and create results directory
SOLUTION_NAME="${SOLUTION_NAME:-unknown_solution}"
BASELINE_NAME=$(basename "$SOLUTION_NAME")
RESULTS_DIR="$BASE_DIR/results/$PROBLEM_NAME/$BASELINE_NAME"
mkdir -p "$RESULTS_DIR"

RESULTS_JSON="$RESULTS_DIR/results.json"
SPEC_PATH="$SCRIPT_DIR/resources/submission_spec.json"

OUTPUT_JSON=$(CBL_LOG_LEVEL=WARNING "$PYBIN" "$SCRIPT_DIR/evaluator.py" --solution "$SOLUTION_PATH" --spec "$SPEC_PATH")
SCORE=$(python3 - <<'PY' "$OUTPUT_JSON"
import json, sys
payload = json.loads(sys.argv[1])
print(payload.get("score", 0))
PY
)

echo "$OUTPUT_JSON" > "$RESULTS_JSON"
echo "$SCORE"
