import argparse
import json
import os
import sys
import traceback
import importlib.util
import pandas as pd
import gc
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial


def _process_baseline_dataset(args_tuple):
    """Helper function for parallel baseline calculation - must be at module level for pickling"""
    csv_path, merge_spec, resources_dir, idx, total = args_tuple
    
    # Ensure resources_dir is in path for utils import
    if resources_dir not in sys.path:
        sys.path.insert(0, resources_dir)
    
    from utils import evaluate_df_prefix_hit_cnt
    
    dataset_name = os.path.basename(csv_path)
    print(f"[evaluator] Baseline [{idx}/{total}] Processing {dataset_name}...", file=sys.stderr, flush=True)
    
    try:
        # Load dataset
        print(f"[evaluator] Loading {dataset_name}...", file=sys.stderr, flush=True)
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"[evaluator] Loaded {dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns", file=sys.stderr, flush=True)
        
        # Apply column merges if needed
        if merge_spec:
            print(f"[evaluator] Applying column merges to {dataset_name}...", file=sys.stderr, flush=True)
            for col_to_merge in merge_spec:
                if all(col in df.columns for col in col_to_merge):
                    merged_name = "_".join(col_to_merge)
                    df[merged_name] = df[col_to_merge].apply(
                        lambda x: "".join([f"{val}" for val in x]), axis=1
                    )
                    df = df.drop(columns=col_to_merge)
            print(f"[evaluator] Column merges applied", file=sys.stderr, flush=True)
        
        # Evaluate baseline with original column order (no optimization)
        print(f"[evaluator] Evaluating prefix hit rate for {dataset_name} ({df.shape[0]} rows, this may take a while)...", file=sys.stderr, flush=True)
        start_time = time.time()
        if df.shape[0] > 10000:
            print(f"[evaluator] Large dataset detected, processing {df.shape[0]} rows...", file=sys.stderr, flush=True)
        _, hit_rate_percent = evaluate_df_prefix_hit_cnt(df)
        eval_time = time.time() - start_time
        hit_rate = hit_rate_percent / 100.0
        print(f"[evaluator] Baseline [{idx}/{total}] {dataset_name} hit rate: {hit_rate_percent:.2f}% (took {eval_time:.2f}s)", file=sys.stderr, flush=True)
        
        # Release memory
        del df
        gc.collect()
        print(f"[evaluator] Baseline [{idx}/{total}] {dataset_name} completed", file=sys.stderr, flush=True)
        
        return hit_rate
    except Exception as e:
        print(f"[evaluator] ERROR processing {dataset_name}: {e}", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 0.0


def _process_evaluation_dataset(args_tuple):
    """Helper function for parallel evaluation - must be at module level for pickling"""
    csv_path, merge_spec, solution_module_path, resources_dir, idx, total = args_tuple
    
    # Ensure resources_dir is in path for utils import
    if resources_dir not in sys.path:
        sys.path.insert(0, resources_dir)
    
    # Add solution_env directory to sys.path so solution can import solver, utils, etc.
    solution_env_dir = os.path.dirname(solution_module_path)
    if solution_env_dir not in sys.path:
        sys.path.insert(0, solution_env_dir)
    
    from utils import evaluate_df_prefix_hit_cnt
    
    dataset_name = os.path.basename(csv_path)
    print(f"[evaluator] [{idx}/{total}] Processing dataset: {dataset_name}", file=sys.stderr, flush=True)
    
    try:
        # Load solution module in this process
        spec = importlib.util.spec_from_file_location("solution", solution_module_path)
        solution = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution)
        
        if not hasattr(solution, "Solution"):
            print(f"[evaluator] ERROR: Missing Solution class in {dataset_name}", file=sys.stderr, flush=True)
            return None, 0.0
        
        solver = solution.Solution()
        
        # Load dataset
        print(f"[evaluator] Loading dataset from {csv_path}...", file=sys.stderr, flush=True)
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"[evaluator] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns", file=sys.stderr, flush=True)
        
        # Run solver
        print(f"[evaluator] Running solver...", file=sys.stderr, flush=True)
        start = time.time()
        reordered = solver.solve(
            df,
            early_stop=100000,
            row_stop=4,
            col_stop=2,
            col_merge=merge_spec,
            one_way_dep=[],
            distinct_value_threshold=0.7,
            parallel=True,
        )
        runtime = time.time() - start
        print(f"[evaluator] Solver completed in {runtime:.2f}s", file=sys.stderr, flush=True)
        
        # Evaluate prefix hit rate
        print(f"[evaluator] Evaluating prefix hit rate...", file=sys.stderr, flush=True)
        _, hit_rate_percent = evaluate_df_prefix_hit_cnt(reordered)
        hit_rate = hit_rate_percent / 100.0
        print(f"[evaluator] Hit rate: {hit_rate_percent:.2f}%", file=sys.stderr, flush=True)
        
        # Release memory
        del df
        del reordered
        gc.collect()
        print(f"[evaluator] Completed dataset {idx}/{total}", file=sys.stderr, flush=True)
        
        return hit_rate, runtime
    except Exception as e:
        print(f"[evaluator] ERROR processing {dataset_name}: {e}", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return None, 0.0


class Evaluator:
    def __init__(self, problem_dir: str):
        print(f"[evaluator] Initializing Evaluator with problem_dir: {problem_dir}", file=sys.stderr, flush=True)
        self.problem_dir = problem_dir
        self.resources_dir = os.path.join(problem_dir, "resources")
        print(f"[evaluator] Resources directory: {self.resources_dir}", file=sys.stderr, flush=True)
        
        # Check mounted datasets directory first (from main repo datasets folder)
        mounted_datasets_dir = "/datasets/llm_sql"
        if os.path.exists(mounted_datasets_dir) and os.listdir(mounted_datasets_dir):
            self.datasets_dir = mounted_datasets_dir
            print(f"[evaluator] Using mounted datasets directory: {self.datasets_dir}", file=sys.stderr, flush=True)
        else:
            # Fallback to resources/datasets if mounted directory doesn't exist
            self.datasets_dir = os.path.join(self.resources_dir, "datasets")
            print(f"[evaluator] Using local datasets directory: {self.datasets_dir}", file=sys.stderr, flush=True)
        
        # All datasets from openevolve: movies, beer, BIRD, PDMX, products
        ordered_names = ["movies.csv", "beer.csv", "BIRD.csv", "PDMX.csv", "products.csv"]
        self.trace_files = [
            os.path.join(self.datasets_dir, name)
            for name in ordered_names
            if os.path.exists(os.path.join(self.datasets_dir, name))
        ]
        print(f"[evaluator] Found {len(self.trace_files)} dataset files", file=sys.stderr, flush=True)
        for tf in self.trace_files:
            print(f"[evaluator]   - {tf}", file=sys.stderr, flush=True)

        # Provide per-dataset column merge specs (from original LLM_SQL tests)
        # Order matches: movies, beer, BIRD, PDMX, products
        self.col_merges = [
            [["movieinfo", "movietitle", "rottentomatoeslink"]],
            [["beer/beerId", "beer/name"]],
            [["PostId", "Body"]],
            [["path", "metadata"], ["hasmetadata", "isofficial", "isuserpublisher", "isdraft", "hasannotations", "subsetall"]],
            [["product_title", "parent_asin"]],
        ]

        # Ensure local resources import
        if self.resources_dir not in sys.path:
            sys.path.insert(0, self.resources_dir)
        
        print("[evaluator] Importing utils...", file=sys.stderr, flush=True)
        from utils import evaluate_df_prefix_hit_cnt  # verify utils import
        self._eval_prefix = evaluate_df_prefix_hit_cnt
        print("[evaluator] Utils imported successfully", file=sys.stderr, flush=True)
        
        # Cache file path for baseline results
        self.baseline_cache_path = os.path.join(self.resources_dir, "baseline_result.json")
    
    def _get_baseline_cache_path(self) -> str:
        """Get the path to the baseline cache file"""
        return self.baseline_cache_path
    
    def _load_baseline_from_cache(self) -> float:
        """Load baseline hit rate from cache if it exists and is valid"""
        cache_path = self._get_baseline_cache_path()
        
        if not os.path.exists(cache_path):
            print(f"[evaluator] Baseline cache not found at {cache_path}", file=sys.stderr, flush=True)
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            # Validate cache: check if datasets match
            cached_datasets = set(cache_data.get("datasets", []))
            current_datasets = set([os.path.basename(tf) for tf in self.trace_files])
            
            if cached_datasets != current_datasets:
                print(f"[evaluator] Baseline cache invalid: datasets changed", file=sys.stderr, flush=True)
                print(f"[evaluator]   Cached: {sorted(cached_datasets)}", file=sys.stderr, flush=True)
                print(f"[evaluator]   Current: {sorted(current_datasets)}", file=sys.stderr, flush=True)
                return None
            
            baseline_hit = cache_data.get("baseline_hit_rate")
            if baseline_hit is None:
                print(f"[evaluator] Baseline cache invalid: missing baseline_hit_rate", file=sys.stderr, flush=True)
                return None
            
            print(f"[evaluator] Loaded baseline hit rate from cache: {baseline_hit:.4f}", file=sys.stderr, flush=True)
            return float(baseline_hit)
        except Exception as e:
            print(f"[evaluator] Error loading baseline cache: {e}", file=sys.stderr, flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            return None
    
    def _save_baseline_to_cache(self, baseline_hit: float):
        """Save baseline hit rate to cache"""
        cache_path = self._get_baseline_cache_path()
        
        try:
            cache_data = {
                "baseline_hit_rate": baseline_hit,
                "datasets": [os.path.basename(tf) for tf in self.trace_files],
                "timestamp": time.time()
            }
            
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            print(f"[evaluator] Saved baseline hit rate to cache: {cache_path}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[evaluator] Error saving baseline cache: {e}", file=sys.stderr, flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)
    
    def _calculate_baseline_hit_rate(self) -> float:
        """Calculate the baseline hit rate using original column order (0-point anchor).
        Uses cache if available, otherwise calculates and caches the result."""
        # Try to load from cache first
        cached_baseline = self._load_baseline_from_cache()
        if cached_baseline is not None:
            return cached_baseline
        
        print("[evaluator] Calculating baseline hit rate in parallel...", file=sys.stderr, flush=True)
        
        if not self.trace_files:
            return 0.0
        
        # Prepare arguments for parallel processing
        args_list = [
            (csv_path, merge_spec, self.resources_dir, idx, len(self.trace_files))
            for idx, (csv_path, merge_spec) in enumerate(
                zip(self.trace_files, self.col_merges[: len(self.trace_files)]), 1
            )
        ]
        
        # Use ProcessPoolExecutor for parallel processing
        # Use min to avoid creating more processes than datasets
        max_workers = min(len(self.trace_files), os.cpu_count() or 1)
        print(f"[evaluator] Using {max_workers} worker processes for baseline calculation", file=sys.stderr, flush=True)
        
        baseline_hit_rates = [0.0] * len(self.trace_files)  # Pre-allocate to maintain order
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and track their indices
            future_to_idx = {
                executor.submit(_process_baseline_dataset, args): args[3] - 1  # Convert to 0-based index
                for args in args_list
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    hit_rate = future.result()
                    baseline_hit_rates[idx] = hit_rate
                except Exception as e:
                    print(f"[evaluator] ERROR in baseline calculation for dataset {idx + 1}: {e}", file=sys.stderr, flush=True)
                    print(traceback.format_exc(), file=sys.stderr, flush=True)
                    baseline_hit_rates[idx] = 0.0
        
        avg_baseline_hit = sum(baseline_hit_rates) / len(self.trace_files) if baseline_hit_rates else 0.0
        
        print(f"[evaluator] Baseline hit rate calculation completed: {avg_baseline_hit:.4f}", file=sys.stderr, flush=True)
        
        # Save to cache for future use
        self._save_baseline_to_cache(avg_baseline_hit)
        
        return avg_baseline_hit

    def evaluate(self, solution_module_path: str) -> dict:
        # Add solution_env directory to sys.path so solution can import solver, utils, etc.
        solution_env_dir = os.path.dirname(solution_module_path)
        if solution_env_dir not in sys.path:
            sys.path.insert(0, solution_env_dir)
        
        # Validate solution module exists and has Solution class (fail fast)
        print(f"[evaluator] Validating solution module: {solution_module_path}", file=sys.stderr, flush=True)
        spec = importlib.util.spec_from_file_location("solution", solution_module_path)
        if spec is None or spec.loader is None:
            return {"score": 0.0, "runs_successfully": 0.0, "error": "Cannot load solution module"}
        
        solution = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution)
        print("[evaluator] Solution module validated", file=sys.stderr, flush=True)

        if not hasattr(solution, "Solution"):
            return {"score": 0.0, "runs_successfully": 0.0, "error": "Missing Solution class"}
        
        # Calculate baseline hit rate
        print("[evaluator] Calculating baseline hit rate (this may take a while)...", file=sys.stderr, flush=True)
        baseline_hit = self._calculate_baseline_hit_rate()
        print(f"[evaluator] Baseline hit rate calculated: {baseline_hit}", file=sys.stderr, flush=True)

        print(f"[evaluator] Processing {len(self.trace_files)} datasets in parallel...", file=sys.stderr, flush=True)
        
        # Prepare arguments for parallel processing
        args_list = [
            (csv_path, merge_spec, solution_module_path, self.resources_dir, idx, len(self.trace_files))
            for idx, (csv_path, merge_spec) in enumerate(
                zip(self.trace_files, self.col_merges[: len(self.trace_files)]), 1
            )
        ]
        
        # Use ProcessPoolExecutor for parallel processing
        max_workers = min(len(self.trace_files), os.cpu_count() or 1)
        print(f"[evaluator] Using {max_workers} worker processes for evaluation", file=sys.stderr, flush=True)
        
        hit_rates = [0.0] * len(self.trace_files)  # Pre-allocate to maintain order
        runtimes = [0.0] * len(self.trace_files)  # Pre-allocate to maintain order
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and track their indices
            future_to_idx = {
                executor.submit(_process_evaluation_dataset, args): args[4] - 1  # Convert to 0-based index
                for args in args_list
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    hit_rate, runtime = future.result()
                    if hit_rate is not None:
                        hit_rates[idx] = hit_rate
                        runtimes[idx] = runtime
                    else:
                        print(f"[evaluator] WARNING: Failed to process dataset {idx + 1}", file=sys.stderr, flush=True)
                        hit_rates[idx] = 0.0
                        runtimes[idx] = 0.0
                except Exception as e:
                    print(f"[evaluator] ERROR in evaluation for dataset {idx + 1}: {e}", file=sys.stderr, flush=True)
                    print(traceback.format_exc(), file=sys.stderr, flush=True)
                    hit_rates[idx] = 0.0
                    runtimes[idx] = 0.0
        
        total_runtime = sum(runtimes)

        if not self.trace_files:
            return {"score": 0.0, "runs_successfully": 0.0, "error": "No datasets found"}

        avg_runtime = total_runtime / len(self.trace_files)
        avg_hit_rate = sum(hit_rates) / len(self.trace_files)
        
        # Calculate normalized hit rate component (0-100 scale)
        # This normalizes based on baseline, where baseline = 0 points
        if baseline_hit >= 1.0:
            # Edge case: baseline is perfect, so any improvement is bonus
            normalized_hit_score = 100.0 if avg_hit_rate >= 1.0 else 0.0
        else:
            normalized_hit_score = ((avg_hit_rate - baseline_hit) / (1.0 - baseline_hit)) * 100
            normalized_hit_score = max(0, min(100, normalized_hit_score))
        
        # Calculate runtime component (0-100 scale)
        # Runtime component: 100 if runtime is 0, decreases linearly to 0 at 10 seconds
        # Similar to openevolve: (10 - min(10, avg_runtime)) / 10 * 100
        runtime_component = (10.0 - min(10.0, avg_runtime)) / 10.0 * 100
        
        # Combined score: 95% weight on hit rate, 5% weight on runtime
        # This matches the openevolve approach while maintaining 0-100 scale
        score = 0.95 * normalized_hit_score + 0.05 * runtime_component
        
        return {
            "score": score,
            "runs_successfully": 1.0,
            "avg_hit_rate": avg_hit_rate * 100 if hit_rates else 0.0,
            "normalized_hit_score": normalized_hit_score,
            "runtime_component": runtime_component,
            "total_runtime": total_runtime,
            "avg_runtime": avg_runtime,
            "runtime_threshold": 10.0
        }


def main():
    print("[evaluator] Starting evaluator...", file=sys.stderr, flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    print(f"[evaluator] Arguments: solution={args.solution}, out={args.out}", file=sys.stderr, flush=True)

    try:
        problem_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"[evaluator] Problem directory: {problem_dir}", file=sys.stderr, flush=True)
        print("[evaluator] Creating Evaluator instance...", file=sys.stderr, flush=True)
        evaluator = Evaluator(problem_dir)
        print(f"[evaluator] Found {len(evaluator.trace_files)} datasets", file=sys.stderr, flush=True)
        print(f"[evaluator] Dataset directory: {evaluator.datasets_dir}", file=sys.stderr, flush=True)
        print("[evaluator] Starting evaluation...", file=sys.stderr, flush=True)
        result = evaluator.evaluate(args.solution)
    except Exception as e:
        print(f"[evaluator] ERROR: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        result = {"score": 0.0, "runs_successfully": 0.0, "error": str(e)}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

