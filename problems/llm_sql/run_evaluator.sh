#!/usr/bin/env bash
set -euo pipefail

# run_evaluator.sh for llm_sql

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Use venv from resources directory (following cant_be_late format)
VENV_DIR="$SCRIPT_DIR/resources/.venv"

# execution_env is only for solution staging, not for venv
# Use execution_env inside resources directory (set by evaluate.sh)
EXEC_ROOT="$SCRIPT_DIR/resources/execution_env"

echo "[run_evaluator] Current directory: $(pwd)" >&2
echo "[run_evaluator] EXEC_ROOT: $EXEC_ROOT" >&2
echo "[run_evaluator] VENV_DIR: $VENV_DIR" >&2

# Check if uv is available
if ! command -v uv >/dev/null 2>&1; then
  echo "[run_evaluator] ERROR: uv is not installed or not in PATH" >&2
  echo "[run_evaluator] Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

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

# Check if venv exists
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[run_evaluator] WARNING: venv not found at $VENV_DIR" >&2
  echo "[run_evaluator] Creating venv..." >&2
  uv venv "$VENV_DIR"
fi

# Determine solution name and create results directory
PROBLEM_NAME=$(basename "$SCRIPT_DIR")
BASE_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
SOLUTION_NAME="${SOLUTION_NAME:-unknown_solution}"

# Extract baseline name from solution path (e.g., "llm_sql/baseline" -> "baseline")
BASELINE_NAME=$(basename "$SOLUTION_NAME")

RESULTS_DIR="$BASE_DIR/results/$PROBLEM_NAME/$BASELINE_NAME"
mkdir -p "$RESULTS_DIR"

RESULTS_JSON="$RESULTS_DIR/results.json"
EVAL_LOG="$RESULTS_DIR/evaluation.log"

# Check if venv exists and has python
PYBIN="${VENV_DIR}/bin/python"
if [[ ! -x "$PYBIN" ]]; then
  echo "[run_evaluator] ERROR: Python not found in venv at $PYBIN" >&2
  echo "[run_evaluator] Please run set_up_env.sh first" >&2
  exit 1
fi

echo "[run_evaluator] Running LLM_SQL evaluator..." >&2
echo "[run_evaluator] Using Python: $PYBIN" >&2
echo "[run_evaluator] Evaluator script: $SCRIPT_DIR/evaluator.py" >&2
echo "[run_evaluator] Solution: $SOLUTION_PATH" >&2
echo "[run_evaluator] Output: $RESULTS_JSON" >&2
echo "----------------------------------------" >&2

# Run evaluator - it outputs JSON to stdout
# Capture both stdout (JSON) and stderr (logs)
EVAL_OUTPUT=$("$PYBIN" "$SCRIPT_DIR/evaluator.py" \
  --solution "$SOLUTION_PATH" \
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
    if 'avg_hit_rate' in results:
        print(f"  Average Hit Rate: {results['avg_hit_rate']:.2f}%", file=sys.stderr)
    if 'avg_runtime' in results:
        print(f"  Average Runtime: {results['avg_runtime']:.4f}s", file=sys.stderr)
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
