#!/usr/bin/env bash
set -euo pipefail

# run_evaluator.sh for multiagent_system

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Use venv from resources directory (following llm_sql format)
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

# Extract baseline name from solution path (e.g., "multiagent_system/baseline" -> "baseline")
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

echo "[run_evaluator] Running multiagent_system evaluator..." >&2
echo "[run_evaluator] Using Python: $PYBIN" >&2
echo "[run_evaluator] Evaluator script: $SCRIPT_DIR/evaluator.py" >&2
echo "[run_evaluator] Solution: $SOLUTION_PATH" >&2
echo "[run_evaluator] Output: $RESULTS_JSON" >&2
echo "----------------------------------------" >&2

# Run evaluator - it outputs JSON to stdout
# Capture both stdout (JSON) and stderr (logs)
# Write logs incrementally so we can see progress
EVAL_OUTPUT=$("$PYBIN" - <<'PY' "$SOLUTION_PATH" "$RESULTS_JSON" "$EVAL_LOG" "$SCRIPT_DIR"
import json, sys, traceback
from pathlib import Path
from datetime import datetime

solution_path = Path(sys.argv[1]).resolve()
results_json = Path(sys.argv[2])
eval_log = Path(sys.argv[3])
script_dir = Path(sys.argv[4])

def log_write(msg):
    """Write message to log file immediately (flush)"""
    with eval_log.open('a') as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")
        f.flush()
    print(msg, file=sys.stderr)

# Initialize log file
eval_log.write_text("=== Multi-Agent System Evaluation Log ===\n\n")
log_write(f"Starting evaluation at {datetime.now().isoformat()}")
log_write(f"Solution Path: {solution_path}")
log_write(f"Script Directory: {script_dir}")
log_write(f"Results JSON: {results_json}")

try:
    log_write("Step 1: Adding script directory to Python path...")
    # Add script directory to path to import evaluator
    sys.path.insert(0, str(script_dir))
    log_write(f"Python path updated. sys.path[0] = {sys.path[0]}")
    
    log_write("Step 2: Importing evaluator module...")
    # Import evaluator from current working directory (problem dir)
    import evaluator
    log_write("Evaluator module imported successfully")
    
    log_write("Step 3: Validating solution file...")
    # Ensure solution_path is absolute and exists
    if not solution_path.exists():
        raise FileNotFoundError(f"Solution file not found at: {solution_path}")
    log_write(f"Solution file exists: {solution_path}")
    log_write(f"Solution file size: {solution_path.stat().st_size} bytes")
    
    log_write("Step 4: Starting evaluation (this may take a while)...")
    program_file = str(solution_path)
    log_write(f"Calling evaluator.evaluate_stage2({program_file})")
    stage2 = evaluator.evaluate_stage2(program_file)
    log_write("Evaluation completed successfully")
    
    log_write("Step 5: Processing evaluation results...")
    # Extract score and ensure runs_successfully is present
    score = float(stage2.get('score', 0.0))
    runs_successfully = float(stage2.get('runs_successfully', 0.0))
    log_write(f"Extracted score: {score}, runs_successfully: {runs_successfully}")
    
    # Create results payload matching llm_sql format
    payload = {
        'score': score,
        'runs_successfully': runs_successfully
    }
    
    # Add additional fields if present
    if 'avg_failures_per_task' in stage2:
        payload['avg_failures_per_task'] = stage2['avg_failures_per_task']
        log_write(f"avg_failures_per_task: {stage2['avg_failures_per_task']}")
    if 'total_failures' in stage2:
        payload['total_failures'] = stage2['total_failures']
        log_write(f"total_failures: {stage2['total_failures']}")
    if 'successful_runs' in stage2:
        payload['successful_runs'] = stage2['successful_runs']
        log_write(f"successful_runs: {stage2['successful_runs']}")
    if 'error' in stage2:
        payload['error'] = stage2['error']
        log_write(f"Error in results: {stage2['error']}")
    
    log_write("Step 6: Writing results JSON...")
    # Write results JSON (filtered payload for main.sh)
    results_json.write_text(json.dumps(payload, indent=2))
    log_write(f"Results JSON written to: {results_json}")
    
    log_write("Step 7: Writing detailed log...")
    # Append detailed log with evaluation context
    with eval_log.open('a') as f:
        f.write("\n=== Evaluation Summary ===\n")
        f.write(f"Score: {score}\n")
        f.write(f"Runs Successfully: {runs_successfully}\n")
        if 'avg_failures_per_task' in stage2:
            f.write(f"Average Failures per Task: {stage2['avg_failures_per_task']}\n")
        if 'total_failures' in stage2:
            f.write(f"Total Failures: {stage2['total_failures']}\n")
        if 'successful_runs' in stage2:
            f.write(f"Successful Runs: {stage2['successful_runs']}\n")
        f.write("\n=== Full Evaluation Result (JSON) ===\n")
        f.write(json.dumps(stage2, indent=2))
        f.write("\n")
    log_write("Evaluation log completed")
    
    print(f"[multiagent_system run_evaluator] Score: {score}")
    print(json.dumps(payload))
except Exception as e:
    error_msg = f"ERROR: {e}\n{traceback.format_exc()}"
    log_write(f"ERROR occurred: {e}")
    log_write(f"Traceback:\n{traceback.format_exc()}")
    with eval_log.open('a') as f:
        f.write("\n=== Evaluation Error ===\n")
        f.write(error_msg)
    results_json.write_text(json.dumps({'score': 0.0, 'runs_successfully': 0.0, 'error': str(e)}))
    print(f"[multiagent_system run_evaluator] ERROR during evaluation: {e}", file=sys.stderr)
    print(json.dumps({'score': 0.0, 'runs_successfully': 0.0, 'error': str(e)}))
    sys.exit(1)
PY
)
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
    if 'avg_failures_per_task' in results:
        print(f"  Average Failures per Task: {results['avg_failures_per_task']:.2f}", file=sys.stderr)
    if 'total_failures' in results:
        print(f"  Total Failures: {results['total_failures']}", file=sys.stderr)
    if 'successful_runs' in results:
        print(f"  Successful Runs: {results['successful_runs']}", file=sys.stderr)
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
