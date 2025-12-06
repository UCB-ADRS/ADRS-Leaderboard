#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBLEM_NAME=$(basename "$SCRIPT_DIR")
BASE_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

# Use execution_env inside resources directory (problem-specific, cleaned up after)
EXEC_ROOT="$SCRIPT_DIR/resources/execution_env"
mkdir -p "$EXEC_ROOT/solution_env"

if [[ ! -f "$EXEC_ROOT/solution_env/solution.py" ]]; then
  echo "Error: Missing $EXEC_ROOT/solution_env/solution.py" >&2
  exit 1
fi

# Determine solution name from environment (set by main.sh or test_local.sh)
SOLUTION_NAME="${SOLUTION_NAME:-unknown_solution}"

# Extract baseline name from solution path (e.g., "llm_sql/baseline" -> "baseline")
BASELINE_NAME=$(basename "$SOLUTION_NAME")

RESULTS_DIR="$BASE_DIR/results/$PROBLEM_NAME/$BASELINE_NAME"
mkdir -p "$RESULTS_DIR"

# Run evaluator
"$SCRIPT_DIR/run_evaluator.sh"
EVAL_EXIT_CODE=$?

# Clean up execution_env after evaluation
if [[ -d "$EXEC_ROOT" ]]; then
  echo "[evaluate] Cleaning up execution_env..." >&2
  rm -rf "$EXEC_ROOT"
fi

# Exit with the evaluation exit code
exit $EVAL_EXIT_CODE

