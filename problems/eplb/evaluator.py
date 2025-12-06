import argparse
import functools
import importlib.util
import json
import os
import sys
import time
import traceback
from typing import TypedDict, Any, Dict
from types import ModuleType
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OUTPUT_PROGRAM = HERE / "output_program.py"

# Check for workload file in multiple locations
# 1. Docker mounted path (for container execution)
# 2. Local datasets directory (for local testing)
def _find_workload_path() -> str:
    # Docker container path
    docker_path = "/datasets/eplb/expert-load.json"
    if os.path.exists(docker_path):
        return docker_path
    
    # Local datasets directory (relative to problem dir)
    local_path = HERE.parent.parent / "datasets" / "eplb" / "expert-load.json"
    if local_path.exists():
        return str(local_path)
    
    # Fallback to docker path (will show proper error message)
    return docker_path

WORKLOAD_PATH = _find_workload_path()
REBALANCE_INTERVAL = 100

NUM_REPLICAS = 288
NUM_GROUPS = 8
NUM_GPUS = 32
NUM_NODES = 4


@functools.cache
def load_workloads(path: str) -> list[torch.Tensor]: 
    with open(path, "r") as f:
        data = json.load(f)

    total_len = len(data['load_history'])
    workloads = []
    for i in range(0, total_len, REBALANCE_INTERVAL):
        start = i
        end = min(start + REBALANCE_INTERVAL, total_len)

        load = torch.tensor([x['logical_expert_load'] for x in data['load_history'][start:end]]).sum(dim=0)
        workloads.append(load)

    return workloads


class EvaluationResult(TypedDict, total=False):
    balancedness_score_gpu: float
    balancedness_score_expert: float
    times_algorithm: float
    times_inference: float
    balancedness_score: float
    speed_score: float
    score: float
    error: str


def simulate_inference(
        log2phy: torch.Tensor,
        logcnt: torch.Tensor,
        workload: torch.Tensor,
    ) -> tuple[float, float]:
    '''
    Simulate a MoE inference with the given expert mapping, and return the balancedness factor.
    '''
    # workload shape: (num_layers, num_logical_experts) - load per logical expert per layer
    num_layers, num_logical_experts = workload.shape
    
    # Initialize physical expert load accumulator
    num_physical_experts = NUM_REPLICAS
    total_physical_load = torch.zeros(num_layers, num_physical_experts, dtype=torch.float, device=workload.device)
    
    # For each logical expert, distribute load to its physical replicas
    for layer_id in range(num_layers):
        for logical_id in range(num_logical_experts):
            # Get load for this logical expert
            logical_load = workload[layer_id][logical_id].item()
            
            # Skip zero load
            if logical_load <= 0:
                continue
                
            num_replicas = int(logcnt[layer_id][logical_id].item())
            # Skip zero replicas
            if num_replicas <= 0:
                continue
            # Get physical expert mapping
            physical_ids = log2phy[layer_id][logical_id][:num_replicas]
                
            # Calculate load per replica (based on effective replica count)
            replica_load = logical_load / num_replicas
            
            # Distribute load to valid physical experts
            total_physical_load[layer_id, physical_ids] += replica_load
    
    # Calculate balancedness
    total_load = total_physical_load.sum()
    if total_load == 0:
        return 0.0, 0.0
    
    # Compute expert load
    expert_layer_avg = total_physical_load.mean(dim=1).sum().item()
    expert_layer_max = total_physical_load.max(dim=1).values.sum().item()
    balancedness_expert = expert_layer_avg / expert_layer_max if expert_layer_max > 0 else 0.0

    # Compute GPU load
    gpu_load = total_physical_load.view(num_layers, NUM_GPUS, -1).sum(dim=2)
    
    # Calculate per-layer average and max load, then sum
    layer_avg = gpu_load.mean(dim=1)  # (num_layers,)
    layer_max = gpu_load.max(dim=1).values  # (num_layers,)
    
    avg_load = layer_avg.sum().item()
    max_load = layer_max.sum().item()
    
    # Calculate balancedness: avg_load / max_load
    balancedness_gpu = avg_load / max_load if max_load > 0 else 0.0
    
    return balancedness_gpu, balancedness_expert


def load_solution_module(solution_path: Path) -> ModuleType:
    """Load the solution module from solution.py"""
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
        OUTPUT_PROGRAM.write_text(result, encoding="utf-8")
        return OUTPUT_PROGRAM
    raise TypeError("Solution.solve must return dict with 'code' or 'program_path', or a raw code string.")


def load_program_module(program_path: Path) -> ModuleType:
    """Load the candidate program module"""
    spec = importlib.util.spec_from_file_location("candidate_program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load candidate program from {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_program(program_module: ModuleType) -> EvaluationResult:
    """Evaluate the rebalance_experts algorithm"""
    # Check workload file exists at evaluation time (not import time)
    if not os.path.exists(WORKLOAD_PATH):
        return {
            "balancedness_score_gpu": 0.0,
            "balancedness_score_expert": 0.0,
            "times_algorithm": 0.0,
            "times_inference": 0.0,
            "balancedness_score": 0.0,
            "speed_score": 0.0,
            "score": 0.0,
            "error": f"Workload file {WORKLOAD_PATH} not found. Please download the workload file.",
        }

    workloads = load_workloads(WORKLOAD_PATH)

    if not hasattr(program_module, "rebalance_experts"):
        return {
            "balancedness_score_gpu": 0.0,
            "balancedness_score_expert": 0.0,
            "times_algorithm": 0.0,
            "times_inference": 0.0,
            "balancedness_score": 0.0,
            "speed_score": 0.0,
            "score": 0.0,
            "error": "Missing `rebalance_experts` function",
        }
    
    balancedness_scores_gpu = []
    balancedness_scores_expert = []
    times_algorithm = []
    times_inference = []
    
    for i in range(len(workloads) - 1):
        start_time = time.perf_counter()
        _, log2phy, logcnt = program_module.rebalance_experts(
            workloads[i],
            NUM_REPLICAS,
            NUM_GROUPS,
            NUM_NODES,
            NUM_GPUS,
        )
        end_time_algorithm = time.perf_counter()
        balancedness_score_gpu, balancedness_score_expert = simulate_inference(log2phy, logcnt, workloads[i + 1])
        end_time = time.perf_counter()
        
        balancedness_scores_gpu.append(balancedness_score_gpu)
        balancedness_scores_expert.append(balancedness_score_expert)
        print(f'time_algorithm: {end_time_algorithm - start_time}, time_inference: {end_time - start_time}')
        times_algorithm.append(end_time_algorithm - start_time)
        times_inference.append(end_time - start_time)

    avg_balancedness_score_gpu = sum(balancedness_scores_gpu) / len(balancedness_scores_gpu)
    avg_balancedness_score_expert = sum(balancedness_scores_expert) / len(balancedness_scores_expert)
    avg_time_algorithm = sum(times_algorithm) / len(times_algorithm)
    avg_time_inference = sum(times_inference) / len(times_inference)
    
    # Balancedness score: scale 0-1 to 0-90 points (primary metric)
    balancedness_score = avg_balancedness_score_gpu * 90
    
    # Speed score: linear scale based on algorithm time (0-10 points)
    # Formula: speed_score = min(0.002 / time, 2) * 5
    # - 1ms or faster = 10 points (cap)
    # - Slower algorithms get proportionally less, no floor
    speed_raw = 0.002 / avg_time_algorithm if avg_time_algorithm > 0 else 2.0
    speed_capped = min(speed_raw, 2.0)
    speed_score = speed_capped * 5
    
    # Slow penalty: penalize algorithms slower than 10ms
    # penalty = min(time_seconds * 20, 20) for time > 10ms
    if avg_time_algorithm > 0.01:  # > 10ms
        slow_penalty = min(avg_time_algorithm * 20, 20)
    else:
        slow_penalty = 0
    
    print(f'avg_time_algorithm: {avg_time_algorithm}, avg_time_inference: {avg_time_inference}')
    print(f'balancedness_score: {balancedness_score}, speed_score: {speed_score}, slow_penalty: {slow_penalty}')
    
    # Total score = balancedness_score + speed_score - slow_penalty
    score = balancedness_score + speed_score - slow_penalty

    return {
        "balancedness_score_gpu": float(avg_balancedness_score_gpu),
        "balancedness_score_expert": float(avg_balancedness_score_expert),
        "times_algorithm": float(avg_time_algorithm),
        "times_inference": float(avg_time_inference),
        "balancedness_score": float(balancedness_score),
        "speed_score": float(speed_score),
        "score": float(score),
    }


class Evaluator:
    """Evaluator class for EPLB solutions"""
    def __init__(self):
        self.output_program = OUTPUT_PROGRAM

    def evaluate(self, solution) -> Dict[str, Any]:
        """
        Evaluate the solution
        Args:
            solution: Solution instance with solve() method
        Returns:
            Dict with score and other metrics
        """
        try:
            result = solution.solve()
            program_path = materialize_program(result)
            program_module = load_program_module(program_path)
            metrics = evaluate_program(program_module)
            return metrics
        except Exception as e:
            traceback.print_exc()
            print(f'Error during evaluation: {str(e)}', file=sys.stderr)
            return {
                "balancedness_score_gpu": 0.0,
                "balancedness_score_expert": 0.0,
                "times_algorithm": 0.0,
                "times_inference": 0.0,
                "balancedness_score": 0.0,
                "speed_score": 0.0,
                "score": 0.0,
                "error": str(e),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Expert Parallelism Load Balancer")
    parser.add_argument("--solution", default="resources/execution_env/solution_env/solution.py")
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()

    solution_path = Path(args.solution).resolve()
    out_path = Path(args.out).resolve()

    try:
        module = load_solution_module(solution_path)
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
        print(f"[evaluator] ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        error_payload = {"score": 0.0, "error": str(exc)}
        out_path.write_text(json.dumps(error_payload, indent=2), encoding="utf-8")
        print(json.dumps(error_payload))
        raise


if __name__ == "__main__":
    main()
