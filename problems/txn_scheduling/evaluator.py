#!/usr/bin/env python3
"""
Evaluator for the txn_scheduling problem.
Evaluates transaction scheduling algorithms by measuring makespan.
Uses 0-100 scoring with baseline (sequential) and adjusted optimal.
"""
import argparse
import functools
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import numpy as np

HERE = Path(__file__).resolve().parent
RESOURCES = HERE / "resources"
SPEC_PATH = RESOURCES / "submission_spec.json"
OUTPUT_PROGRAM = HERE / "output_program.py"
BASELINE_CACHE_FILE = HERE / "baseline_cache.json"

sys.path.insert(0, str(RESOURCES))

from txn_simulator import Workload  # noqa: E402
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3  # noqa: E402

# Workloads list for iteration
WORKLOADS = [WORKLOAD_1, WORKLOAD_2, WORKLOAD_3]


class TimeoutError(Exception):
    pass


@functools.cache
def compute_baseline_makespan() -> float:
    """
    Compute baseline makespan using sequential ordering (0-point reference).
    Sequential ordering: [0, 1, 2, ..., n-1] - the naive approach.
    """
    # Try to load from cache first
    if BASELINE_CACHE_FILE.exists():
        try:
            cache = json.loads(BASELINE_CACHE_FILE.read_text(encoding="utf-8"))
            if "baseline_makespan" in cache and "optimal_makespan" in cache:
                return cache["baseline_makespan"]
        except Exception:
            pass
    
    # Compute baseline: sequential ordering for each workload
    total = 0
    for workload_data in WORKLOADS:
        w = Workload(workload_data)
        sequential = list(range(w.num_txns))
        total += w.get_opt_seq_cost(sequential)
    
    return float(total)


@functools.cache
def compute_optimal_makespan() -> float:
    """
    Compute theoretical optimal makespan (100-point reference, unreachable).
    Lower bound: sum of the longest transaction length per workload.
    This represents the absolute minimum time if there were no conflicts.
    """
    # Try to load from cache first
    if BASELINE_CACHE_FILE.exists():
        try:
            cache = json.loads(BASELINE_CACHE_FILE.read_text(encoding="utf-8"))
            if "optimal_makespan" in cache:
                return cache["optimal_makespan"]
        except Exception:
            pass
    
    # Compute optimal: theoretical minimum (sum of max transaction lengths)
    total = 0
    for workload_data in WORKLOADS:
        w = Workload(workload_data)
        # Each transaction's length is stored in txn[0][3]
        max_txn_len = max(txn[0][3] for txn in w.txns)
        total += max_txn_len
    
    return float(total)


def save_baseline_cache():
    """Save computed baseline and optimal values to cache file."""
    try:
        cache = {
            "baseline_makespan": compute_baseline_makespan(),
            "optimal_makespan": compute_optimal_makespan(),
        }
        BASELINE_CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[WARNING] Failed to save baseline cache: {e}", file=sys.stderr)


# Optimal shift factor: moves the 100-point reference closer to baseline,
# making high scores more achievable and spreading out top performers.
# 0.0 = use theoretical minimum (original), 1.0 = use baseline (everyone gets 100)
# 0.10 provides good balance for top differentiation
OPTIMAL_SHIFT_FACTOR = 0.10


def calculate_score(actual_makespan: float, valid: bool) -> float:
    """
    Calculate 0-100 score based on makespan.
    
    Score = 0: Makespan equals baseline (sequential ordering)
    Score = 100: Makespan equals adjusted optimal (achievable target)
    
    The optimal is adjusted to be more achievable than the theoretical minimum,
    which spreads out scores at the top and makes the scoring more meaningful.
    
    Formula:
        effective_optimal = optimal + OPTIMAL_SHIFT_FACTOR * (baseline - optimal)
        score = ((baseline - actual) / (baseline - effective_optimal)) * 100
        score = clamp(score, 0, 100)
    """
    if not valid:
        return 0.0
    
    baseline = compute_baseline_makespan()
    optimal = compute_optimal_makespan()
    
    # Adjust optimal to be more achievable (shift toward baseline)
    # This makes 100 points reachable and spreads out top scores
    effective_optimal = optimal + OPTIMAL_SHIFT_FACTOR * (baseline - optimal)
    
    # If actual is worse than or equal to baseline, score is 0
    if actual_makespan >= baseline:
        return 0.0
    
    # If actual is at or better than effective optimal, score is 100
    if actual_makespan <= effective_optimal:
        return 100.0
    
    # Linear interpolation between baseline (0) and effective optimal (100)
    score = ((baseline - actual_makespan) / (baseline - effective_optimal)) * 100
    
    # Clamp to 0-100 range
    return max(0.0, min(100.0, score))


def load_solution_module(solution_path: Path) -> ModuleType:
    """Load a solution module from a file path."""
    if not solution_path.exists():
        raise FileNotFoundError(f"solution.py not found at {solution_path}")
    spec = importlib.util.spec_from_file_location("submitted_solution", solution_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for {solution_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materialize_program(result: Any) -> Path:
    """Convert a Solution.solve() result into a program file path."""
    if isinstance(result, dict):
        if "program_path" in result:
            candidate = Path(result["program_path"]).expanduser()
            if not candidate.exists():
                raise FileNotFoundError(f"Provided program_path does not exist: {candidate}")
            return candidate
        if "code" in result:
            OUTPUT_PROGRAM.write_text(result["code"], encoding="utf-8")
            return OUTPUT_PROGRAM
    if isinstance(result, str):
        # treat as code snippet
        OUTPUT_PROGRAM.write_text(result, encoding="utf-8")
        return OUTPUT_PROGRAM
    raise TypeError("Solution.solve must return dict with 'code' or 'program_path', or a raw code string.")


def validate_schedule(txn_seq: list, num_txns: int) -> bool:
    """Validate that a schedule contains all transactions exactly once."""
    if len(txn_seq) != num_txns:
        return False
    for i in range(num_txns):
        if i not in txn_seq:
            return False
    return True


def run_with_timeout(program_path: str, timeout_seconds: int = 600) -> tuple:
    """
    Run the program in a separate process with timeout.

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        (makespan, schedule, time_taken) tuple from the program
    """
    sched_dir = str(RESOURCES)
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))
# Also add the resources directory for importing sibling modules
sys.path.insert(0, r'{sched_dir}')

# Debugging info
print(f"Running in subprocess, Python version: {{sys.version}}")
print(f"Program path: {program_path}")

try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Run the scheduling function
    print("Calling get_random_costs()...")
    makespan, schedule, time_taken = program.get_random_costs()
    print(f"get_random_costs() returned successfully: makespan = {{makespan}}, time_taken = {{time_taken}}")

    # Save results to a file
    results = {{
        'makespan': makespan,
        'schedule': schedule,
        'time_taken': time_taken,
    }}

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {temp_file.name}.results")
    
except Exception as e:
    print(f"Error in subprocess: {{str(e)}}")
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
    print(f"Error saved to {temp_file.name}.results")
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            print(f"Subprocess stdout: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")

            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")

            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")

                return results["makespan"], results["schedule"], results["time_taken"]
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def evaluate_program(program_path: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a scheduling program.

    Args:
        program_path: Path to the program file
        spec: Submission specification

    Returns:
        Dictionary of metrics
    """
    timeout_seconds = spec.get("timeout_seconds", 600)

    # Get baseline and optimal for reporting
    baseline_makespan = compute_baseline_makespan()
    optimal_makespan = compute_optimal_makespan()
    # Effective optimal used for scoring (adjusted to be more achievable)
    effective_optimal = optimal_makespan + OPTIMAL_SHIFT_FACTOR * (baseline_makespan - optimal_makespan)

    try:
        start_time = time.time()

        makespan, schedule, time_taken = run_with_timeout(
            str(program_path), timeout_seconds=timeout_seconds
        )

        end_time = time.time()
        eval_time = end_time - start_time

        # Validate schedules
        valid = True
        for i, s in enumerate(schedule):
            workload = Workload(WORKLOADS[i])
            if not validate_schedule(s, workload.num_txns):
                valid = False
                break

        validity = 1.0 if valid else 0.0

        # Calculate 0-100 score
        score = calculate_score(makespan, valid)

        print(f"Evaluation: valid={valid}, makespan={makespan}, score={score:.2f}")
        print(f"  Baseline: {baseline_makespan}, Theoretical optimal: {optimal_makespan}, Effective optimal: {effective_optimal:.1f}")

        return {
            "makespan": float(makespan),
            "baseline_makespan": float(baseline_makespan),
            "optimal_makespan": float(optimal_makespan),
            "effective_optimal": float(effective_optimal),
            "num_schedules": float(len(schedule)),
            "time_taken": float(time_taken),
            "validity": float(validity),
            "eval_time": float(eval_time),
            "score": float(score),
            "runs_successfully": 1.0,
        }

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        traceback.print_exc()
        return {
            "makespan": 0.0,
            "baseline_makespan": float(baseline_makespan),
            "optimal_makespan": float(optimal_makespan),
            "effective_optimal": float(effective_optimal),
            "num_schedules": 0.0,
            "time_taken": 0.0,
            "validity": 0.0,
            "eval_time": 0.0,
            "score": 0.0,
            "runs_successfully": 0.0,
            "error": str(e),
        }


class Evaluator:
    """Evaluator class for txn_scheduling problem."""

    def __init__(self):
        """Initialize evaluator with spec from resources."""
        self.spec_path = SPEC_PATH
        self.output_program = OUTPUT_PROGRAM

        # Load spec
        self.spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
        
        # Pre-compute and cache baseline values
        save_baseline_cache()

    def evaluate(self, solution) -> Dict[str, Any]:
        """
        Evaluate a solution.

        Args:
            solution: Solution instance with solve() method

        Returns:
            Dict with score and other metrics
        """
        result = solution.solve(str(self.spec_path))
        program_path = materialize_program(result)
        metrics = evaluate_program(program_path, self.spec)
        return metrics


def evaluate(solution_path: Path, spec_path: Path) -> Dict[str, Any]:
    """Legacy function for backward compatibility."""
    solution_module = load_solution_module(solution_path)
    if not hasattr(solution_module, "Solution"):
        raise AttributeError("solution.py must define a Solution class with a solve method")
    solution_obj = solution_module.Solution()
    if not hasattr(solution_obj, "solve"):
        raise AttributeError("Solution class must define a solve(spec_path: str) method")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    result = solution_obj.solve(str(spec_path))
    program_path = materialize_program(result)
    metrics = evaluate_program(program_path, spec)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate txn_scheduling optimizer")
    parser.add_argument("--solution", default="../../execution_env/solution_env/solution.py")
    parser.add_argument("--spec", default=str(SPEC_PATH))
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()

    solution_path = Path(args.solution).resolve()
    spec_path = Path(args.spec).resolve()
    out_path = Path(args.out).resolve()

    try:
        module = load_solution_module(solution_path)

        # Use new Solution class format
        solution_class = getattr(module, "Solution", None)
        if solution_class is None:
            raise AttributeError("Solution class not found in solution.py")

        print("[evaluator] Using Solution class format", file=sys.stderr)
        evaluator = Evaluator()
        solution = solution_class()
        payload = evaluator.evaluate(solution)

        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
    except Exception as exc:
        error_payload = {"score": 0.0, "error": str(exc)}
        out_path.write_text(json.dumps(error_payload, indent=2), encoding="utf-8")
        print(json.dumps(error_payload))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
