#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
RESOURCES = HERE / "resources"
OUTPUT_PROGRAM = HERE / "output_program.py"

GPU_MEM_SIZE = 80  # GB
MIN_INT = float('-inf')


@dataclass
class Model:
    model_name: str
    model_size: int
    req_rate: int
    slo: int
    cur_gpu_id: int


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=30):
    """Run a function with a timeout using concurrent.futures"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")


def safe_float(value):
    """Convert a value to float safely"""
    try:
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def verify_gpu_mem_constraint(placement_data: dict) -> bool:
    """Verify whether models can fit into GPU memory"""
    if placement_data is None:
        return False
    for gpu_id, models in placement_data.items():
        if sum(model.model_size for model in models) > GPU_MEM_SIZE:
            return False
    return True


def calculate_kvcache_pressure(placement_data: dict) -> float:
    """Calculate the maximum KVCache pressure across all GPUs"""
    max_kvpr = MIN_INT
    for gpu_id, models in placement_data.items():
        total_model_size = sum(model.model_size for model in models)
        total_weighted_req_rate = sum(model.req_rate / model.slo for model in models)
        if GPU_MEM_SIZE - total_model_size > 0:
            kvpr = total_weighted_req_rate / (GPU_MEM_SIZE - total_model_size)
        else:
            kvpr = 1000000
        max_kvpr = max(max_kvpr, kvpr)
    return max_kvpr


def round_robin_placement(gpu_num: int, models: List[Model]) -> dict:
    """
    Baseline placement: simple round-robin assignment.
    This is the 0-point reference (naive strategy).
    """
    placement = {i: [] for i in range(gpu_num)}
    for i, model in enumerate(models):
        placement[i % gpu_num].append(model)
    return placement


def compute_theoretical_optimal_kvpr(gpu_num: int, models: List[Model]) -> float:
    """
    Compute the theoretical optimal (minimum possible) max KVPR.
    This assumes perfect load balancing where all GPUs have equal KVPR.
    This is the 100-point reference (impossible to beat).
    
    Theoretical optimal: if we could perfectly distribute load,
    each GPU would have equal KVPR = total_weighted_req / (gpu_num * GPU_MEM_SIZE - total_model_size)
    """
    total_weighted_req = sum(m.req_rate / m.slo for m in models)
    total_model_size = sum(m.model_size for m in models)
    
    # Available memory across all GPUs
    total_available_mem = gpu_num * GPU_MEM_SIZE - total_model_size
    
    if total_available_mem > 0:
        # Perfect balance: evenly distribute load across available memory
        optimal_kvpr = total_weighted_req / total_available_mem
    else:
        optimal_kvpr = 0.0  # Edge case: no memory available
    
    return optimal_kvpr


def generate_test_gpu_models(num_tests=50):
    """Generate multiple test cases with different characteristics"""
    test_cases = []
    np.random.seed(42)

    for i in range(num_tests):
        gpu_num = np.random.randint(5, 10)
        gpu_models = []
        for j in range(gpu_num * 2):
            model_size = np.random.randint(10, 30)
            req_rate = np.random.randint(1, 10)
            slo = np.random.randint(5, 10)
            gpu_models.append(Model(
                model_name=f"model_{j}",
                model_size=model_size,
                req_rate=req_rate,
                slo=slo,
                cur_gpu_id=j
            ))
        test_cases.append((gpu_num, gpu_models))

    return test_cases


def load_solution_module(solution_path: Path) -> ModuleType:
    if not solution_path.exists():
        raise FileNotFoundError(f"solution.py not found at {solution_path}")
    spec = importlib.util.spec_from_file_location("submitted_solution", solution_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for {solution_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materialize_program(result: Any) -> Path:
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
    spec = importlib.util.spec_from_file_location("candidate_program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load candidate program from {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_placement_algorithm(program_module: ModuleType) -> Dict[str, Any]:
    """Evaluate the model placement algorithm with 0-100 scoring"""
    def invalid_result(message: str) -> Dict[str, Any]:
        return {
            "score": 0.0,
            "avg_kvpr": 0.0,
            "baseline_kvpr": 0.0,
            "optimal_kvpr": 0.0,
            "success_rate": 0.0,
            "runs_successfully": 0.0,
            "error": message,
        }

    if not hasattr(program_module, "compute_model_placement"):
        return invalid_result("Missing compute_model_placement function")

    test_gpu_models = generate_test_gpu_models()

    all_solution_kvpr = []
    all_baseline_kvpr = []
    all_optimal_kvpr = []
    all_scores = []
    all_metrics = []
    successful_runs = 0

    for i, (gpu_num, gpu_models) in enumerate(test_gpu_models):
        try:
            # Compute baseline (round-robin) KVPR
            baseline_placement = round_robin_placement(gpu_num, gpu_models)
            baseline_kvpr = calculate_kvcache_pressure(baseline_placement)
            
            # Compute theoretical optimal KVPR
            optimal_kvpr = compute_theoretical_optimal_kvpr(gpu_num, gpu_models)
            
            # Run the solution algorithm
            start_time = time.time()
            result = run_with_timeout(
                program_module.compute_model_placement,
                kwargs={'gpu_num': gpu_num, 'models': gpu_models},
                timeout_seconds=10
            )
            execution_time = time.time() - start_time

            if not isinstance(result, dict):
                return invalid_result(f"Expected dict, got {type(result).__name__}")

            placed_models = []
            for gpu_id, assigned_models in result.items():
                if not isinstance(assigned_models, list):
                    return invalid_result(
                        f"GPU {gpu_id} value must be list, got {type(assigned_models).__name__}"
                    )
                placed_models.extend(assigned_models)

            if len(placed_models) != len(gpu_models):
                return invalid_result(f"Not all models placed: {len(placed_models)}/{len(gpu_models)}")

            placed_ids = [id(m) for m in placed_models]
            if len(set(placed_ids)) != len(placed_ids):
                duplicates = len(placed_ids) - len(set(placed_ids))
                return invalid_result(f"Duplicate models detected: {duplicates} duplicates")

            original_ids = {id(m) for m in gpu_models}
            if set(placed_ids) != original_ids:
                return invalid_result("Placed models don't match input models (missing or foreign models)")

            for gpu_id, assigned_models in result.items():
                total_size = sum(model.model_size for model in assigned_models)
                if total_size > GPU_MEM_SIZE:
                    return invalid_result(
                        f"GPU {gpu_id} exceeds memory: {total_size}GB > {GPU_MEM_SIZE}GB"
                    )

            solution_kvpr = calculate_kvcache_pressure(result)
            
            # Calculate score for this test case (0-100 scale) with sqrt scaling
            # Sqrt scaling gives diminishing returns as solutions approach optimal,
            # providing more credit for initial improvements over baseline
            if baseline_kvpr > optimal_kvpr:
                raw_ratio = (baseline_kvpr - solution_kvpr) / (baseline_kvpr - optimal_kvpr)
                # Clamp ratio to [0, 1] then apply sqrt scaling
                clamped_ratio = max(0.0, min(1.0, raw_ratio))
                test_score = 100.0 * (clamped_ratio ** 0.5)
            else:
                # Edge case: baseline equals or is better than optimal (shouldn't happen)
                test_score = 100.0 if solution_kvpr <= optimal_kvpr else 0.0

            metrics = {
                'solution_kvpr': safe_float(solution_kvpr),
                'baseline_kvpr': safe_float(baseline_kvpr),
                'optimal_kvpr': safe_float(optimal_kvpr),
                'test_score': safe_float(test_score),
                'execution_time': safe_float(execution_time),
            }

            all_solution_kvpr.append(safe_float(solution_kvpr))
            all_baseline_kvpr.append(safe_float(baseline_kvpr))
            all_optimal_kvpr.append(safe_float(optimal_kvpr))
            all_scores.append(safe_float(test_score))
            all_metrics.append(metrics)
            successful_runs += 1

        except TimeoutError:
            print(f"Placement {i}: Timeout", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Placement {i}: Error - {str(e)}", file=sys.stderr)
            continue

    if successful_runs == 0:
        return {
            "score": 0.0,
            "avg_kvpr": 0.0,
            "baseline_kvpr": 0.0,
            "optimal_kvpr": 0.0,
            "success_rate": 0.0,
            "runs_successfully": 0.0,
            "error": "All test cases failed"
        }

    # Calculate aggregate metrics
    avg_solution_kvpr = np.mean(all_solution_kvpr)
    avg_baseline_kvpr = np.mean(all_baseline_kvpr)
    avg_optimal_kvpr = np.mean(all_optimal_kvpr)
    avg_execution_time = np.mean([m['execution_time'] for m in all_metrics])
    success_rate = successful_runs / len(test_gpu_models)
    
    # Final score: average of per-test scores
    final_score = np.mean(all_scores)

    return {
        "score": safe_float(final_score),
        "avg_kvpr": safe_float(avg_solution_kvpr),
        "baseline_kvpr": safe_float(avg_baseline_kvpr),
        "optimal_kvpr": safe_float(avg_optimal_kvpr),
        "execution_time": safe_float(avg_execution_time),
        "success_rate": safe_float(success_rate),
        "successful_runs": successful_runs,
        "total_tests": len(test_gpu_models),
        "runs_successfully": 1.0,
    }


class Evaluator:
    def __init__(self):
        """Initialize evaluator"""
        self.output_program = OUTPUT_PROGRAM

    def evaluate(self, solution):
        """
        Evaluate the solution
        Args:
            solution: Solution instance with solve() method
        Returns:
            Dict with score and other metrics
        """
        result = solution.solve()
        program_path = materialize_program(result)
        program_module = load_program_module(program_path)
        metrics = evaluate_placement_algorithm(program_module)
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PRISM model placement optimizer")
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
        print(traceback.format_exc(), file=sys.stderr)
        error_payload = {"score": 0.0, "runs_successfully": 0.0, "error": str(exc)}
        out_path.write_text(json.dumps(error_payload, indent=2), encoding="utf-8")
        print(json.dumps(error_payload))
        raise


if __name__ == "__main__":
    main()
