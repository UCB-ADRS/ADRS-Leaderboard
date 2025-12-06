#!/usr/bin/env python3
"""
Evaluator for the telemetry_repair problem.
Evaluates network telemetry repair algorithms by measuring repair accuracy and confidence calibration.
Uses 0-100 scoring with passthrough baseline (0 points) and perfect repair (100 points).
"""
import argparse
import ast
import csv
import importlib.util
import json
import os
import random
import sys
import tempfile
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
RESOURCES = HERE / "resources"
SPEC_PATH = RESOURCES / "submission_spec.json"
OUTPUT_PROGRAM = HERE / "output_program.py"

# Scoring constants for 0-100 piecewise linear scoring
# Uses piecewise linear scaling with knee point at midpoint:
#   - Baseline (0 pts): passthrough algorithm, combined_score ~0.80
#   - Knee point (50 pts): combined_score = 0.82
#   - Optimal (100 pts): perfect repair, combined_score = 1.0 (unreachable)
BASELINE_COMBINED_SCORE = 0.80
KNEE_COMBINED_SCORE = 0.82
OPTIMAL_COMBINED_SCORE = 1.0


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
        OUTPUT_PROGRAM.write_text(result, encoding="utf-8")
        return OUTPUT_PROGRAM
    raise TypeError("Solution.solve must return dict with 'code' or 'program_path', or a raw code string.")


def load_network_data(base_dir: Path) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Load network data from CSV and topology files.
    
    Args:
        base_dir: Base directory containing the problem
        
    Returns:
        Tuple of (csv_rows, topology_dict)
    """
    # Check mounted datasets directory first (from main repo datasets folder)
    mounted_datasets_dir = Path("/datasets/telemetry_repair")
    if mounted_datasets_dir.exists() and any(mounted_datasets_dir.iterdir()):
        datasets_dir = mounted_datasets_dir
    else:
        # Fallback to repo-level datasets folder
        repo_datasets_dir = base_dir.parent.parent / "datasets" / "telemetry_repair"
        if repo_datasets_dir.exists() and any(repo_datasets_dir.iterdir()):
            datasets_dir = repo_datasets_dir
        else:
            # Final fallback to resources/datasets
            datasets_dir = base_dir / "resources" / "datasets"
    
    data_file = datasets_dir / "evaluation_data.csv"
    topology_file = datasets_dir / "topology.json"
    
    csv_data = []
    if data_file.exists():
        with open(data_file, 'r') as f:
            reader = csv.DictReader(f)
            csv_data = list(reader)
    else:
        print(f"Data file not found: {data_file}")
    
    topology = {}
    if topology_file.exists():
        with open(topology_file, 'r') as f:
            topology = json.load(f)
    
    return csv_data, topology


def convert_csv_row_to_interfaces(data_row: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Adapter function: Convert a CSV row to the interface format expected by repair algorithms.
    
    Args:
        data_row: Single row from the evaluation CSV
        
    Returns:
        Dictionary in interface format
    """
    interfaces = {}
    
    for col_name, col_value in data_row.items():
        if col_name.startswith('low_') and ('_egress_to_' in col_name or '_ingress_from_' in col_name):
            try:
                telemetry_data = ast.literal_eval(col_value)
                
                if 'ground_truth' not in telemetry_data:
                    continue
                
                if '_egress_to_' in col_name:
                    parts = col_name.replace('low_', '').split('_egress_to_')
                    if len(parts) != 2:
                        continue
                    source, dest = parts
                    
                    if_id = f"{source}_to_{dest}"
                    if if_id not in interfaces:
                        interfaces[if_id] = {
                            'interface_status': 'up',
                            'rx_rate': 0.0,
                            'tx_rate': 0.0,
                            'connected_to': f"{dest}_to_{source}",
                            'local_router': source,
                            'remote_router': dest
                        }
                    
                    tx_rate = telemetry_data.get('perturbed') if telemetry_data.get('perturbed') is not None else telemetry_data['ground_truth']
                    interfaces[if_id]['tx_rate'] = float(tx_rate) if tx_rate is not None else 0.0
                    interfaces[if_id]['_ground_truth_tx'] = float(telemetry_data['ground_truth'])
                    
                elif '_ingress_from_' in col_name:
                    parts = col_name.replace('low_', '').split('_ingress_from_')
                    if len(parts) != 2:
                        continue
                    dest, source = parts
                    
                    if_id = f"{dest}_to_{source}"
                    if if_id not in interfaces:
                        interfaces[if_id] = {
                            'interface_status': 'up',
                            'rx_rate': 0.0,
                            'tx_rate': 0.0,
                            'connected_to': f"{source}_to_{dest}",
                            'local_router': dest,
                            'remote_router': source
                        }
                    
                    rx_rate = telemetry_data.get('perturbed') if telemetry_data.get('perturbed') is not None else telemetry_data['ground_truth']
                    interfaces[if_id]['rx_rate'] = float(rx_rate) if rx_rate is not None else 0.0
                    interfaces[if_id]['_ground_truth_rx'] = float(telemetry_data['ground_truth'])
                
            except (ValueError, SyntaxError, KeyError):
                continue
    
    return interfaces


def apply_test_perturbations(interfaces: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Apply perturbations for testing and keep ground truth.
    
    Args:
        interfaces: Original interface data
        topology: Dictionary where key is router_id and value contains a list of interface_ids
        
    Returns:
        Tuple of (perturbed_interfaces, ground_truth_interfaces)
    """
    perturbed = {}
    ground_truth = {}
    
    bug_type = random.choice(['dropped', 'scaled', 'correlated_dropped', 'correlated_scaled'])
    
    for if_id, if_data in interfaces.items():
        ground_truth[if_id] = {
            'rx_rate': if_data.get('_ground_truth_rx', if_data['rx_rate']),
            'tx_rate': if_data.get('_ground_truth_tx', if_data['tx_rate']),
            'interface_status': if_data['interface_status'],
        }
        
        perturbed_data = if_data.copy()
        
        # Add some noise to all measurements
        perturbed_data['rx_rate'] *= random.uniform(0.98, 1.02)
        perturbed_data['tx_rate'] *= random.uniform(0.98, 1.02)
        
        if bug_type == 'dropped':
            if random.random() < 0.20:
                perturbed_data['rx_rate'] = 0
                perturbed_data['tx_rate'] = 0
        elif bug_type == 'scaled':
            if random.random() < 0.20:
                perturbed_data['rx_rate'] *= random.choice([random.uniform(0.5, 0.8), random.uniform(1.2, 1.5)])
                perturbed_data['tx_rate'] *= random.choice([random.uniform(0.5, 0.8), random.uniform(1.2, 1.5)])
        
        perturbed_data.pop('_ground_truth_rx', None)
        perturbed_data.pop('_ground_truth_tx', None)
        
        perturbed[if_id] = perturbed_data
    
    # Correlated case
    for _, if_ids in topology.items():
        if random.random() < 0.80:
            continue
        if bug_type == 'correlated_dropped':
            for if_id in if_ids:
                if if_id in interfaces:
                    perturbed_data = interfaces[if_id].copy()
                    perturbed_data['rx_rate'] = 0
                    perturbed_data['tx_rate'] = 0
                    perturbed_data.pop('_ground_truth_rx', None)
                    perturbed_data.pop('_ground_truth_tx', None)
                    perturbed[if_id] = perturbed_data
        elif bug_type == 'correlated_scaled':
            for if_id in if_ids:
                if if_id in interfaces:
                    perturbed_data = interfaces[if_id].copy()
                    perturbed_data['rx_rate'] *= random.choice([random.uniform(0.5, 0.8), random.uniform(1.2, 1.5)])
                    perturbed_data['tx_rate'] *= random.choice([random.uniform(0.5, 0.8), random.uniform(1.2, 1.5)])
                    perturbed_data.pop('_ground_truth_rx', None)
                    perturbed_data.pop('_ground_truth_tx', None)
                    perturbed[if_id] = perturbed_data
    
    return perturbed, ground_truth


def calculate_counter_repair_quality(repaired_interfaces: Dict[str, Dict], 
                                     ground_truth: Dict[str, Dict[str, float]]) -> float:
    """Calculate how well the repair algorithm restored ground truth values."""
    total_error = 0.0
    num_measurements = 0
    
    for if_id in ground_truth:
        if if_id not in repaired_interfaces:
            continue
            
        gt = ground_truth[if_id]
        repaired = repaired_interfaces[if_id]
        
        if 'rx_rate' in gt:
            gt_rx = gt['rx_rate']
            if isinstance(repaired.get('rx_rate'), tuple) and len(repaired['rx_rate']) >= 2:
                rep_rx = repaired['rx_rate'][1]
            else:
                rep_rx = repaired.get('rx_rate', 0.0)
            
            if gt_rx > 0:
                error = abs(rep_rx - gt_rx) / gt_rx
                total_error += min(error, 1.0)
                num_measurements += 1
        
        if 'tx_rate' in gt:
            gt_tx = gt['tx_rate']
            if isinstance(repaired.get('tx_rate'), tuple) and len(repaired['tx_rate']) >= 2:
                rep_tx = repaired['tx_rate'][1]
            else:
                rep_tx = repaired.get('tx_rate', 0.0)
            
            if gt_tx > 0:
                error = abs(rep_tx - gt_tx) / gt_tx
                total_error += min(error, 1.0)
                num_measurements += 1
    
    if num_measurements == 0:
        return 0.0
    
    avg_error = total_error / num_measurements
    repair_quality = 1.0 - avg_error
    return max(0.0, repair_quality)


def calculate_status_repair_quality(repaired_interfaces: Dict[str, Dict], 
                                    ground_truth: Dict[str, Dict[str, float]]) -> float:
    """Calculate how well the repair algorithm restored interface status values."""
    correct_status_repairs = 0
    total_status_measurements = 0
    
    for if_id in ground_truth:
        if if_id not in repaired_interfaces:
            continue
            
        gt = ground_truth[if_id]
        repaired = repaired_interfaces[if_id]
        
        if 'interface_status' in gt:
            gt_status = gt['interface_status']
            
            if isinstance(repaired.get('interface_status'), tuple) and len(repaired['interface_status']) >= 2:
                rep_status = repaired['interface_status'][1]
            else:
                rep_status = repaired.get('interface_status', 'unknown')
            
            total_status_measurements += 1
            if rep_status == gt_status:
                correct_status_repairs += 1
    
    if total_status_measurements == 0:
        return 1.0
    
    return correct_status_repairs / total_status_measurements


def calculate_confidence_calibration(repaired_interfaces: Dict[str, Dict],
                                     ground_truth: Dict[str, Dict[str, float]]) -> float:
    """Evaluate how well confidence scores reflect repair accuracy."""
    total_score = 0.0
    total_measurements = 0
    
    for if_id in ground_truth:
        if if_id not in repaired_interfaces:
            continue
            
        gt = ground_truth[if_id]
        repaired = repaired_interfaces[if_id]
        
        for rate_key in ['rx_rate', 'tx_rate']:
            if rate_key in gt:
                gt_val = gt[rate_key]
                
                if isinstance(repaired.get(rate_key), tuple) and len(repaired[rate_key]) >= 3:
                    _, rep_val, confidence = repaired[rate_key]
                    
                    max_val = max(abs(gt_val), abs(rep_val), 1.0)
                    repair_error = abs(gt_val - rep_val) / max_val
                    repair_accuracy = 1.0 - repair_error
                    
                    if repair_accuracy > 0.8:
                        if confidence > 0.7:
                            score = 1.0
                        else:
                            score = 0.5 + (confidence - 0.3) * 1.25
                    else:
                        if confidence < 0.3:
                            score = 0.8
                        else:
                            overconfidence_penalty = confidence * (1.0 - repair_accuracy)
                            score = max(0.0, 0.8 - overconfidence_penalty * 2.0)
                    
                    total_score += max(0.0, min(1.0, score))
                    total_measurements += 1
    
    if total_measurements == 0:
        return 1.0
    
    return total_score / total_measurements


def evaluate_interface_scenario(program, interfaces: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate repair algorithm on a single interface scenario.
    
    Args:
        program: The imported repair program module
        interfaces: Interface data in the expected format
        
    Returns:
        Dictionary with scenario metrics
    """
    try:
        topology = {}
        for interface_id, telemetry in interfaces.items():
            local_router = telemetry.get('local_router')
            if local_router not in topology:
                topology[local_router] = []
            topology[local_router].append(interface_id)
        
        perturbed_interfaces, ground_truth = apply_test_perturbations(interfaces, topology)
        
        # Run the repair algorithm
        repaired_interfaces = program.run_repair(perturbed_interfaces, topology)
        
        # Calculate metrics
        counter_repair_score = calculate_counter_repair_quality(repaired_interfaces, ground_truth)
        status_repair_score = calculate_status_repair_quality(repaired_interfaces, ground_truth)
        confidence_calibration_score = calculate_confidence_calibration(repaired_interfaces, ground_truth)
        
        # Combined score: 75% counter, 5% status, 20% confidence
        combined_score = counter_repair_score * 0.75 + status_repair_score * 0.05 + confidence_calibration_score * 0.2
        
        return {
            'combined_score': combined_score,
            'counter_repair_accuracy': counter_repair_score,
            'status_repair_accuracy': status_repair_score,
            'confidence_calibration': confidence_calibration_score
        }
        
    except Exception as e:
        print(f"Interface scenario evaluation failed: {e}")
        traceback.print_exc()
        return {'combined_score': 0.0, 'counter_repair_accuracy': 0.0, 'status_repair_accuracy': 0.0, 'confidence_calibration': 0.0}


def evaluate_program(program_path: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a repair program.
    
    Args:
        program_path: Path to the program file
        spec: Submission specification
        
    Returns:
        Dictionary of metrics
    """
    try:
        # Import the program
        spec_import = importlib.util.spec_from_file_location("program", str(program_path))
        program = importlib.util.module_from_spec(spec_import)
        spec_import.loader.exec_module(program)
        
        # Load network data
        csv_data, _ = load_network_data(HERE)
        
        if not csv_data:
            print("No data loaded from CSV")
            return {"score": 0.0, "runs_successfully": 0.0, "error": "No data loaded"}
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Test on sample of rows
        total_score = 0.0
        num_tests = 0
        test_results = []
        
        test_rows = random.sample(csv_data, min(30, len(csv_data)))
        
        for row in test_rows:
            interfaces = convert_csv_row_to_interfaces(row)
            
            if not interfaces:
                continue
            
            score_dict = evaluate_interface_scenario(program, interfaces)
            test_results.append(score_dict)
            total_score += score_dict.get('combined_score', 0.0)
            num_tests += 1
        
        avg_score = total_score / num_tests if num_tests > 0 else 0.0
        
        # Calculate aggregate metrics
        counter_repair_accuracy = sum(r.get('counter_repair_accuracy', 0) for r in test_results) / len(test_results) if test_results else 0.0
        status_repair_accuracy = sum(r.get('status_repair_accuracy', 0) for r in test_results) / len(test_results) if test_results else 0.0
        confidence_calibration = sum(r.get('confidence_calibration', 0) for r in test_results) / len(test_results) if test_results else 0.0
        
        # Calculate 0-100 score using two-part piecewise linear scaling:
        # - 0 points = passthrough baseline (combined_score = 0.80)
        # - 50 points = knee point (combined_score = 0.82)
        # - 100 points = perfect repair (combined_score = 1.0, unreachable)
        if avg_score <= KNEE_COMBINED_SCORE:
            # Below or at knee: linear 0-50
            raw_score = ((avg_score - BASELINE_COMBINED_SCORE) / (KNEE_COMBINED_SCORE - BASELINE_COMBINED_SCORE)) * 50.0
        else:
            # Above knee: linear 50-100
            raw_score = 50.0 + ((avg_score - KNEE_COMBINED_SCORE) / (OPTIMAL_COMBINED_SCORE - KNEE_COMBINED_SCORE)) * 50.0
        score = max(0.0, min(100.0, raw_score))  # Clamp to [0, 100]
        
        return {
            'score': score,
            'combined_score': avg_score,
            'baseline_combined_score': BASELINE_COMBINED_SCORE,
            'counter_repair_accuracy': counter_repair_accuracy,
            'status_repair_accuracy': status_repair_accuracy,
            'confidence_calibration': confidence_calibration,
            'num_tests': num_tests,
            'runs_successfully': 1.0
        }
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        traceback.print_exc()
        return {
            'score': 0.0,
            'combined_score': 0.0,
            'baseline_combined_score': BASELINE_COMBINED_SCORE,
            'counter_repair_accuracy': 0.0,
            'status_repair_accuracy': 0.0,
            'confidence_calibration': 0.0,
            'num_tests': 0,
            'runs_successfully': 0.0,
            'error': str(e)
        }


class Evaluator:
    """Evaluator class for telemetry_repair problem."""
    
    def __init__(self):
        """Initialize evaluator with spec from resources."""
        self.spec_path = SPEC_PATH
        self.output_program = OUTPUT_PROGRAM
        
        # Load spec if it exists
        if self.spec_path.exists():
            self.spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
        else:
            self.spec = {"timeout_seconds": 60}
    
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate telemetry_repair algorithm")
    parser.add_argument("--solution", default="../../execution_env/solution_env/solution.py")
    parser.add_argument("--spec", default=str(SPEC_PATH))
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
        error_payload = {"score": 0.0, "error": str(exc)}
        out_path.write_text(json.dumps(error_payload, indent=2), encoding="utf-8")
        print(json.dumps(error_payload))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

