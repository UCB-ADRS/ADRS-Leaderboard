import random
import math
import time

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3


def get_best_schedule(workload, num_seqs):
    """
    Computes an optimal transaction schedule to minimize makespan using 
    Simulated Annealing initialized with LPT and Heuristic construction.
    """
    
    # --- Configuration ---
    # Heuristics Parameters
    GREEDY_SAMPLE_SIZE = 10  # Number of candidates to evaluate in greedy phase
    
    # Simulated Annealing Parameters
    SA_ITER_MULTIPLIER = 120 # Iterations per transaction (scales with size)
    START_TEMP = 10.0        # Initial temperature
    FINAL_TEMP = 0.01        # Stopping temperature
    SHIFT_PROB = 0.8         # Probability of using SHIFT over SWAP mutation
    
    # --- Pre-computation ---
    num_txns = workload.num_txns
    all_txns = list(range(num_txns))
    
    # 1. Calculate 'weights' (cost in isolation)
    # Heavier transactions are harder to schedule, used for LPT and tie-breaking.
    txn_costs = {t: workload.get_opt_seq_cost([t]) for t in all_txns}
    sorted_by_weight = sorted(all_txns, key=lambda t: txn_costs[t], reverse=True)
    
    # --- Phase 1: High-Quality Initialization ---
    # We generate two candidate schedules and pick the best one to start SA.
    
    # Candidate A: Longest Processing Time (LPT)
    # Often effective for parallel workloads (heavy items first hide latency of small items)
    sched_lpt = list(sorted_by_weight)
    cost_lpt = workload.get_opt_seq_cost(sched_lpt)
    
    best_schedule = sched_lpt
    best_makespan = cost_lpt
    
    # Candidate B: Minimum Latency Greedy
    # Construct a schedule by iteratively picking the txn that adds minimal time
    sched_greedy = []
    remaining = set(all_txns)
    current_greedy_cost = 0.0
    
    while remaining:
        # Optimization: Only look at the heaviest remaining items to reduce complexity
        # and prevent checking 'light' items that don't drive makespan.
        candidates = []
        count = 0
        for t in sorted_by_weight:
            if t in remaining:
                candidates.append(t)
                count += 1
                if count >= GREEDY_SAMPLE_SIZE:
                    break
        
        best_cand = -1
        best_cand_delta = float('inf')
        best_cand_new_cost = float('inf')
        
        for t in candidates:
            # Check cost of appending t
            new_cost = workload.get_opt_seq_cost(sched_greedy + [t])
            delta = new_cost - current_greedy_cost
            
            # We want the smallest delta (most parallelism)
            # Tie-breaker: If deltas are equal, prefer heavier transaction (handled by sort order)
            if delta < best_cand_delta:
                best_cand_delta = delta
                best_cand_new_cost = new_cost
                best_cand = t
                
                # Optimization: If delta is effectively 0 (perfect parallelism),
                # take it immediately and stop searching.
                if delta < 1e-5:
                    break
        
        sched_greedy.append(best_cand)
        remaining.remove(best_cand)
        current_greedy_cost = best_cand_new_cost

    # Compare candidates
    if current_greedy_cost < best_makespan:
        best_makespan = current_greedy_cost
        best_schedule = sched_greedy
        
    # --- Phase 2: Simulated Annealing Optimization ---
    
    curr_schedule = list(best_schedule)
    curr_cost = best_makespan
    
    # Determine iteration count based on problem size
    # We use num_seqs loosely as a budget multiplier
    max_iter = num_txns * SA_ITER_MULTIPLIER * max(1, num_seqs // 2)
    
    # Calculate cooling rate to reach FINAL_TEMP from START_TEMP in max_iter
    if max_iter > 0:
        alpha = math.pow(FINAL_TEMP / START_TEMP, 1 / max_iter)
    else:
        alpha = 0.99
        
    temp = START_TEMP
    
    for _ in range(max_iter):
        # 1. Generate Neighbor (Mutation)
        # We copy the list. Slicing is fast enough for typical transaction counts.
        candidate = list(curr_schedule)
        
        # Mutation Strategy:
        # SHIFT (Insert): Moves a transaction from idx i to j. 
        # Excellent for fixing dependency orderings (topology).
        if random.random() < SHIFT_PROB:
            idx_src = random.randint(0, num_txns - 1)
            idx_dest = random.randint(0, num_txns - 1)
            if idx_src == idx_dest: continue
            
            item = candidate.pop(idx_src)
            candidate.insert(idx_dest, item)
            
        # SWAP: Exchanges two positions. 
        # Good for general reordering without shifting the whole tail.
        else:
            idx_a = random.randint(0, num_txns - 1)
            idx_b = random.randint(0, num_txns - 1)
            if idx_a == idx_b: continue
            candidate[idx_a], candidate[idx_b] = candidate[idx_b], candidate[idx_a]
            
        # 2. Evaluate
        new_cost = workload.get_opt_seq_cost(candidate)
        delta = new_cost - curr_cost
        
        # 3. Acceptance Criteria (Metropolis-Hastings)
        # Always accept improvement (delta < 0)
        # Sometimes accept degradation based on Temp
        if delta < 0 or random.random() < math.exp(-delta / temp):
            curr_schedule = candidate
            curr_cost = new_cost
            
            # Update Global Best if strictly better
            if curr_cost < best_makespan:
                best_makespan = curr_cost
                best_schedule = list(curr_schedule)
                
        # 4. Cool Down
        temp *= alpha
        
        # Early exit if temp is negligible
        if temp < 1e-6:
            break

    return best_makespan, best_schedule


def get_random_costs():
    """Entry point called by the evaluator."""
    start_time = time.time()

    workload1 = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload1, 10)
    cost1 = workload1.get_opt_seq_cost(schedule1)

    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 10)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 10)
    cost3 = workload3.get_opt_seq_cost(schedule3)

    return cost1 + cost2 + cost3, [schedule1, schedule2, schedule3], time.time() - start_time


# Solution class wrapper for evaluator compatibility
from pathlib import Path
from typing import Dict, Any

class Solution:
    """GEPA solution for txn_scheduling."""

    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        # Read this file and return the code (everything before the Solution class)
        code = Path(__file__).read_text(encoding="utf-8")
        lines = code.split('\n')
        end_idx = next(i for i, line in enumerate(lines) if 'class Solution:' in line)
        program_code = '\n'.join(lines[:end_idx])
        return {"code": program_code}