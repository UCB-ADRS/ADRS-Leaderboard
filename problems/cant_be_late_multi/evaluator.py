#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import sys
import subprocess
import logging
import re
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Union

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "resources/cant-be-late-simulator")

# Import pricing from simulator
sys.path.insert(0, PROJECT_ROOT)
from sky_spot.utils import COSTS, ClusterType

MAIN_SIMULATOR_PATH = os.path.join(PROJECT_ROOT, 'main.py')
DATA_PATH = os.path.join(PROJECT_ROOT, "data/converted_multi_region_aligned")

TASK_DURATION_HOURS = 48
DEADLINE_HOURS = 52
RESTART_OVERHEAD_HOURS = 0.2
TIMEOUT_SECONDS = 300
WORST_POSSIBLE_SCORE = -1e9

HERE = Path(__file__).resolve().parent
DEFAULT_SPEC = HERE / "resources" / "submission_spec.json"
ARTIFACT_PATH = Path("./output_ans").resolve()

# Full test scenarios for the final evaluation stage
FULL_TEST_SCENARIOS = [
    # Original Scenarios (more traces)
    {"name": "2_zones_same_region", "regions": ["us-east-1a_v100_1", "us-east-1c_v100_1"], "traces": [f"{i}.json" for i in range(8)]},
    {"name": "2_regions_east_west", "regions": ["us-east-2a_v100_1", "us-west-2a_v100_1"], "traces": [f"{i}.json" for i in range(8)]},
    {"name": "3_regions_diverse", "regions": ["us-east-1a_v100_1", "us-east-2b_v100_1", "us-west-2c_v100_1"], "traces": [f"{i}.json" for i in range(6)]},
    
    # New Scenarios inspired by benchmark script
    {"name": "3_zones_same_region", "regions": ["us-east-1a_v100_1", "us-east-1c_v100_1", "us-east-1d_v100_1"], "traces": [f"{i}.json" for i in range(6)]},
    {"name": "5_regions_high_diversity", "regions": ["us-east-1a_v100_1", "us-east-1f_v100_1", "us-west-2a_v100_1", "us-west-2b_v100_1", "us-east-2b_v100_1"], "traces": [f"{i}.json" for i in range(4)]},
    {"name": "all_9_regions", "regions": ["us-east-2a_v100_1", "us-west-2c_v100_1", "us-east-1d_v100_1", "us-east-2b_v100_1", "us-west-2a_v100_1", "us-east-1f_v100_1", "us-east-1a_v100_1", "us-west-2b_v100_1", "us-east-1c_v100_1"], "traces": [f"{i}.json" for i in range(2)]}
]

# A single, simple scenario for the quick first-stage evaluation
STAGE_1_SCENARIO = {
    "name": "stage_1_quick_check", 
    "regions": ["us-east-1a_v100_1", "us-east-1c_v100_1"], 
    "traces": ["0.json"]
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_solution_module(solution_path: Path) -> ModuleType:
    if not solution_path.exists():
        raise FileNotFoundError(f"solution.py not found at {solution_path}")
    spec = importlib.util.spec_from_file_location("submitted_solution", solution_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for {solution_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def materialize_artifact(result: Any, solution_path: Path) -> Path:
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result, dict):
        with ARTIFACT_PATH.open("w", encoding="utf-8") as fout:
            json.dump(result, fout)
        return ARTIFACT_PATH
    if isinstance(result, str):
        candidate = Path(result)
        if candidate.is_file():
            with ARTIFACT_PATH.open("w", encoding="utf-8") as fout:
                json.dump({"program_path": str(candidate.resolve())}, fout)
            return ARTIFACT_PATH
        with ARTIFACT_PATH.open("w", encoding="utf-8") as fout:
            fout.write(result)
        return ARTIFACT_PATH
    raise TypeError(
        "Solution.solve() must return a dict/path-string/code-string; got "
        f"{type(result)!r}."
    )

def run_simulation(program_path: str, trace_files: List[str]) -> Dict[str, Union[float, str, None]]:
    """
    Runs the main.py simulation and returns a result dictionary.
    """
    cmd = [
        sys.executable,
        os.path.basename(MAIN_SIMULATOR_PATH),
        f"--strategy-file={program_path}",
        "--env=multi_trace",
        f"--task-duration-hours={TASK_DURATION_HOURS}",
        f"--deadline-hours={DEADLINE_HOURS}",
        f"--restart-overhead-hours={RESTART_OVERHEAD_HOURS}",
        "--trace-files",
    ] + trace_files

    try:
        # Set environment to disable wandb
        env = os.environ.copy()
        env['WANDB_MODE'] = 'disabled'
        
        # Using subprocess.run to execute the simulation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True, # Will raise CalledProcessError for non-zero exit codes
            timeout=TIMEOUT_SECONDS,
            cwd=PROJECT_ROOT,
            env=env,
        )

        output = result.stdout + result.stderr
        match = re.search(r"mean:\s*([\d.]+)", output)
        
        if match:
            return {"status": "success", "cost": float(match.group(1)), "output": output}
        
        error_msg = f"Could not parse 'mean:' cost from simulation output."
        return {"status": "failure", "error": error_msg, "output": output}

    except subprocess.CalledProcessError as e:
        error_msg = f"Simulation failed with exit code {e.returncode}."
        return {"status": "failure", "error": error_msg, "stdout": e.stdout, "stderr": e.stderr}
    except subprocess.TimeoutExpired as e:
        error_msg = f"Simulation timed out after {TIMEOUT_SECONDS}s."
        return {"status": "failure", "error": error_msg, "stdout": e.stdout, "stderr": e.stderr}
    except Exception:
        # Catch any other unexpected errors during simulation execution
        error_msg = "An unexpected error occurred during simulation execution."
        return {"status": "failure", "error": error_msg, "traceback": traceback.format_exc()}

def evaluate_stage1(program_path: str) -> Dict[str, Union[float, str]]:
    """
    First-stage evaluation: A quick check to see if the program can run a single,
    simple scenario without crashing. This filters out basic syntax and runtime errors.
    """
    absolute_program_path = os.path.abspath(program_path)
    logger.info(f"--- Stage 1: Quick Check for {os.path.basename(program_path)} ---")

    try:
        trace_files = [os.path.join(DATA_PATH, region, STAGE_1_SCENARIO["traces"][0]) for region in STAGE_1_SCENARIO["regions"]]
        
        
        if not all(os.path.exists(p) for p in trace_files):
            return {"runs_successfully": 0.0, "combined_score": WORST_POSSIBLE_SCORE, "error": f"Missing trace files for Stage 1 {trace_files}."}

        sim_result = run_simulation(absolute_program_path, trace_files)

        if sim_result["status"] == "success":
            logger.info("Stage 1 PASSED.")
            # IMPORTANT: Only return the metric that is being checked by the pass_metric config.
            # The framework's _passes_threshold function incorrectly averages all numeric metrics.
            # By returning only this, we ensure the average is 1.0, passing the check correctly.
            return {"runs_successfully": 1.0}
        else:
            logger.warning(f"Stage 1 FAILED. Reason: {sim_result.get('error')}")
            return {
                "runs_successfully": 0.0,
                "combined_score": WORST_POSSIBLE_SCORE,
                "error": sim_result.get("error"),
                "stdout": sim_result.get("stdout"),
                "stderr": sim_result.get("stderr"),
                "traceback": sim_result.get("traceback"),
            }

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Stage 1 evaluator itself failed: {tb}")
        return {"runs_successfully": 0.0, "combined_score": WORST_POSSIBLE_SCORE, "error": "Evaluator script failure", "traceback": tb}

def evaluate_stage2(program_path: str) -> Dict[str, Union[float, str]]:
    """
    Second-stage evaluation: The full, comprehensive evaluation across all test scenarios.
    This is only run for programs that have passed Stage 1.
    """
    absolute_program_path = os.path.abspath(program_path)
    logger.info(f"--- Stage 2: Full Evaluation for {os.path.basename(program_path)} ---")
    
    scenario_costs = []
    last_error = "No scenarios were successfully evaluated in Stage 2."

    for scenario in FULL_TEST_SCENARIOS:
        scenario_name = scenario["name"]
        total_scenario_cost = 0
        successful_runs_in_scenario = 0
        
        logger.info(f"--- Evaluating Scenario: {scenario_name} ---")

        for trace_file_name in scenario["traces"]:
            trace_files = [os.path.join(DATA_PATH, region, trace_file_name) for region in scenario["regions"]]
            
            if not all(os.path.exists(p) for p in trace_files):
                last_error = f"Missing trace files for {scenario_name}, trace {trace_file_name}."
                logger.warning(last_error)
                continue

            sim_result = run_simulation(absolute_program_path, trace_files)

            if sim_result["status"] == "failure":
                last_error = f"Error in scenario '{scenario_name}': {sim_result.get('error')}"
                break 
            
            total_scenario_cost += sim_result.get("cost", 0.0)
            successful_runs_in_scenario += 1
        
        if successful_runs_in_scenario > 0:
            average_scenario_cost = total_scenario_cost / successful_runs_in_scenario
            scenario_costs.append(average_scenario_cost)
            logger.info(f"Scenario '{scenario_name}' Average Cost: ${average_scenario_cost:.2f}")
        else:
            scenario_costs.append(float('inf'))
            logger.warning(f"Scenario '{scenario_name}' failed completely. Last error: {last_error}")

    valid_costs = [c for c in scenario_costs if c != float('inf')]
    if not valid_costs:
        logger.error(f"All Stage 2 evaluation scenarios failed. Last error: {last_error}")
        # This return is for the database, which correctly uses the combined_score.
        return {"runs_successfully": 1.0, "cost": float('inf'), "combined_score": WORST_POSSIBLE_SCORE, "error": last_error}

    final_average_cost = sum(valid_costs) / len(valid_costs)
    score = -final_average_cost

    logger.info(f"--- Evaluation Summary ---")
    logger.info(f"Final Average Cost across all scenarios: ${final_average_cost:.2f}")

    # Normalized scoring (same as single-region cant_be_late)
    od_anchor = COSTS[ClusterType.ON_DEMAND] * TASK_DURATION_HOURS
    spot_anchor = COSTS[ClusterType.SPOT] * TASK_DURATION_HOURS

    denom = od_anchor - spot_anchor
    normalized = (od_anchor - final_average_cost) / denom
    score = max(0.0, min(1.0, normalized)) * 100

    logger.info(f"Final Normalized Score: {score:.2f}")

    # This full set of metrics is for the database, which will correctly prioritize combined_score.
    return {"runs_successfully": 1.0, "score": round(score, 2), "avg_cost": final_average_cost, "od_anchor": od_anchor, "spot_anchor": spot_anchor}

def evaluate(solution_path: Path, spec_path: Path) -> dict:
    module = load_solution_module(solution_path)
    if not hasattr(module, "Solution"):
        raise AttributeError("solution.py must define a 'Solution' class")
    SolutionCls = module.Solution  # type: ignore[attr-defined]
    solution_obj = SolutionCls()
    if not hasattr(solution_obj, "solve"):
        raise AttributeError("Solution class must define a 'solve' method")
    solve_fn = getattr(solution_obj, "solve")
    result = solve_fn(str(spec_path))
    artifact_path = materialize_artifact(result, solution_path)
    
    # Run the simulation with the artifact
    if os.path.exists(artifact_path):
        with open(artifact_path, 'r') as f:
            artifact_data = json.load(f)
        
        if isinstance(artifact_data, dict) and 'program_path' in artifact_data:
            program_path = artifact_data['program_path']
        elif isinstance(artifact_data, dict) and 'code' in artifact_data:
            # Write the code to a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                tmp_file.write(artifact_data['code'])
                program_path = tmp_file.name
        else:
            # Fallback: treat the artifact path as the program path
            program_path = str(artifact_path)
    else:
        program_path = str(artifact_path)
    
    # Run the cascade evaluation
    stage1_result = evaluate_stage1(program_path)
    if stage1_result.get("runs_successfully", 0.0) > 0:
        stage2_result = evaluate_stage2(program_path)
        return stage2_result
    else:
        return stage1_result

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cant-be-late-multi solution module")
    parser.add_argument(
        "--solution",
        default="../../execution_env/solution_env/solution.py",
        help="Path to contestant solution.py",
    )
    parser.add_argument(
        "--spec",
        default=str(DEFAULT_SPEC),
        help="Path to submission spec (passed to Solution.solve)",
    )
    args = parser.parse_args()

    solution_path = Path(args.solution).resolve()
    spec_path = Path(args.spec).resolve()
    try:
        payload = evaluate(solution_path, spec_path)
    except Exception as e:
        print(json.dumps({"error": str(e), "score": 0}))
        raise
    else:
        print(json.dumps(payload))

if __name__ == "__main__":
    main()