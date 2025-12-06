Txn Scheduling Problem
======================

Problem Setting
---------------
Optimize transaction scheduling for database workloads. Given a set of transactions with read and write operations on data items, find the optimal ordering that minimizes the total makespan (execution time).

Transactions conflict when they access the same data items - read-write and write-write conflicts create dependencies that affect scheduling efficiency.

Target
------
- **Primary**: Minimize total makespan (lower is better)
- **Constraint**: All transactions must be present in the schedule exactly once

API Specification
-----------------
Implement a `Solution` class that returns a scheduling algorithm:

```python
class Solution:
    def solve(self, spec_path: str = None) -> dict:
        """
        Returns a dict with either:
        - {"code": "python_code_string"}
        - {"program_path": "path/to/algorithm.py"}
        """
        pass
```

Your algorithm code must implement:

```python
from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

def get_best_schedule(workload: Workload, num_seqs: int) -> tuple:
    """
    Find optimal transaction schedule.
    
    Args:
        workload: Workload object containing transactions
        num_seqs: Number of sequences to consider
    
    Returns:
        Tuple of (makespan, schedule) where:
        - makespan: Total execution time
        - schedule: List of transaction indices in execution order
    """
    pass

def get_random_costs() -> tuple:
    """
    Entry point for evaluation.
    
    Returns:
        Tuple of (total_makespan, schedules, time_taken) where:
        - total_makespan: Sum of makespans across all workloads
        - schedules: List of schedules for each workload
        - time_taken: Wall-clock time for computation
    """
    pass
```

**Workload Format**:
- Transactions are JSON objects: `{"txn0": "w-17 r-5 w-3 r-4", "txn1": "r-17 r-280 w-10"}`
- Operations: `r-{key}` (read), `w-{key}` (write)
- Conflicts: read-write and write-write on same key create dependencies
- Read-read on same key: no conflict (shared access)

**Evaluation Process**:
1. Your algorithm produces schedules for 3 predefined workloads
2. Each schedule's makespan is computed using conflict-aware simulation
3. Total makespan is the sum across all workloads

Scoring (0-100)
---------------
```
baseline_makespan = Makespan from sequential ordering [0, 1, 2, ..., n-1]
theoretical_optimal = Sum of max transaction lengths per workload (theoretical minimum)
optimal_shift_factor = 0.10 (makes 100-point more achievable)
effective_optimal = theoretical_optimal + optimal_shift_factor × (baseline - theoretical_optimal)
actual_makespan = Your solution's total makespan

score = ((baseline - actual) / (baseline - effective_optimal)) × 100
score = clamp(score, 0, 100)
```

The effective optimal is adjusted to be 10% closer to the baseline than the theoretical minimum.
This makes high scores more achievable and provides better differentiation among top solutions.

**Scoring Examples**:
- actual_makespan = baseline_makespan → Score = 0 (no improvement over naive)
- actual_makespan = effective_optimal → Score = 100 (achievable target)
- Scores above 100 (beating effective optimal) are clamped to 100

Implementation Notes
--------------------
- Use `workload.get_opt_seq_cost(txn_seq)` to compute makespan for a sequence
- Schedule must include all transaction indices [0, num_txns-1]
- The algorithm is evaluated on 3 predefined workloads (100 transactions each)
- Time limit: 600 seconds total
- Use `workload.num_txns` to get number of transactions in a workload
