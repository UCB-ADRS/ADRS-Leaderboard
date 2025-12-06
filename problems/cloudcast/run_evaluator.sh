#!/usr/bin/env bash
set -euo pipefail

# run_evaluator.sh for cloudcast

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Use venv from execution_env (created by set_up_env.sh)
EXEC_ROOT="$SCRIPT_DIR/resources/execution_env"
VENV_DIR="$EXEC_ROOT/.venv"

echo "[run_evaluator] Current directory: $(pwd)" >&2
echo "[run_evaluator] EXEC_ROOT: $EXEC_ROOT" >&2
echo "[run_evaluator] VENV_DIR: $VENV_DIR" >&2

# Solution path
SOLUTION_PATH="$EXEC_ROOT/solution_env/solution.py"
echo "[run_evaluator] Looking for solution at: $SOLUTION_PATH" >&2
if [[ ! -f "$SOLUTION_PATH" ]]; then
  echo "[run_evaluator] ERROR: solution.py not found at $SOLUTION_PATH" >&2
  echo "[run_evaluator] Contents of $EXEC_ROOT:" >&2
  ls -la "$EXEC_ROOT" >&2 || true
  if [[ -d "$EXEC_ROOT/solution_env" ]]; then
    echo "[run_evaluator] Contents of $EXEC_ROOT/solution_env:" >&2
    ls -la "$EXEC_ROOT/solution_env" >&2 || true
  fi
  exit 1
fi

echo "[run_evaluator] Solution found at: $SOLUTION_PATH" >&2

# Determine solution name and create results directory
PROBLEM_NAME=$(basename "$SCRIPT_DIR")
BASE_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
SOLUTION_NAME="${SOLUTION_NAME:-unknown_solution}"

# Extract baseline name from solution path (e.g., "cloudcast/baseline" -> "baseline")
BASELINE_NAME=$(basename "$SOLUTION_NAME")

RESULTS_DIR="$BASE_DIR/results/$PROBLEM_NAME/$BASELINE_NAME"
mkdir -p "$RESULTS_DIR"

RESULTS_JSON="$RESULTS_DIR/results.json"
EVAL_LOG="$RESULTS_DIR/evaluation.log"

# Check if venv exists and activate it
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  echo "[run_evaluator] Activating venv..." >&2
  source "$VENV_DIR/bin/activate"
  PYBIN="${VENV_DIR}/bin/python"
else
  echo "[run_evaluator] WARNING: venv not found at $VENV_DIR, using system Python" >&2
  PYBIN="python3"
fi

echo "[run_evaluator] Running Cloudcast evaluator..." >&2
echo "[run_evaluator] Using Python: $PYBIN" >&2
echo "[run_evaluator] Evaluator script: $SCRIPT_DIR/evaluator.py" >&2
echo "[run_evaluator] Solution: $SOLUTION_PATH" >&2
echo "[run_evaluator] Output: $RESULTS_JSON" >&2
echo "----------------------------------------" >&2

# Run evaluator - it outputs JSON to stdout
# Capture both stdout (JSON) and stderr (logs)
EVAL_OUTPUT=$("$PYBIN" "$SCRIPT_DIR/evaluator.py" \
  --solution "$SOLUTION_PATH" \
  --spec "$SCRIPT_DIR/resources/submission_spec.json" \
  --out "$RESULTS_JSON" \
  2>&1 | tee "$EVAL_LOG")
EVAL_EXIT_CODE=$?

if [[ $EVAL_EXIT_CODE -ne 0 ]]; then
  echo "----------------------------------------" >&2
  echo "[run_evaluator] ERROR: evaluator.py failed!" >&2
  if [[ -f "$EVAL_LOG" ]]; then
    echo "[run_evaluator] Last 20 lines of evaluation log:" >&2
    tail -20 "$EVAL_LOG" >&2
  fi
  echo "0"  # Output 0 score on error (main.sh expects a number)
  exit 1
fi

echo "----------------------------------------" >&2
echo "[run_evaluator] Results written to $RESULTS_JSON" >&2
echo "[run_evaluator] Log written to $EVAL_LOG" >&2

# Extract score from results JSON and output it
# main.sh expects the score as the last line of stdout
if [[ -f "$RESULTS_JSON" ]]; then
  echo "[run_evaluator] Evaluation results:" >&2
  SCORE=$(python3 - <<'PY' "$RESULTS_JSON"
import json
import sys
try:
    with open(sys.argv[1], 'r') as f:
        results = json.load(f)
    score = results.get('score', 0)
    print(f"  Score: {score}", file=sys.stderr)
    print(f"  Runs Successfully: {results.get('runs_successfully', 'N/A')}", file=sys.stderr)
    if 'combined_score' in results:
        print(f"  Combined Score: {results['combined_score']:.4f}", file=sys.stderr)
    if 'total_cost' in results:
        print(f"  Total Cost: {results['total_cost']:.4f}", file=sys.stderr)
    if 'total_transfer_time' in results:
        print(f"  Total Transfer Time: {results['total_transfer_time']:.4f}", file=sys.stderr)
    if 'error' in results:
        print(f"  Error: {results['error']}", file=sys.stderr)
    print(score)
except Exception as e:
    print(f"  Could not parse results: {e}", file=sys.stderr)
    print(0)
PY
)
  # Output score as last line (main.sh expects this format)
  echo "$SCORE"
else
  echo "[run_evaluator] ERROR: Results file not found!" >&2
  echo "0"
  exit 1
fi
