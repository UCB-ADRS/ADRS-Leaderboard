PRISM Problem
=============

Problem Setting
---------------
Optimize the placement of machine learning models across GPUs to minimize the maximum KV Cache Pressure (KVPR). Given a set of models with varying sizes, request rates, and SLO requirements, determine the optimal assignment of models to GPUs while respecting memory constraints.

KVPR (KV Cache Pressure) measures how crowded a GPU is:
```
KVPR = sum(model.req_rate / model.slo for model in gpu_models) / (GPU_MEM_SIZE - sum(model.model_size for model in gpu_models))
```

Lower maximum KVPR across all GPUs is better.

Target
------
- **Primary**: Minimize maximum KVPR across all GPUs (lower is better)
- **Hard Constraint**: Models must fit within GPU memory (80GB per GPU)
- **Secondary**: Maximize successful placement rate across test cases

API Specification
-----------------
Implement a `Solution` class that returns a model placement algorithm:

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
from dataclasses import dataclass

GPU_MEM_SIZE = 80  # GB

@dataclass
class Model:
    model_name: str
    model_size: int   # GB
    req_rate: int     # requests per second
    slo: int          # service level objective (latency target)
    cur_gpu_id: int   # current GPU assignment (can be ignored)

def compute_model_placement(gpu_num: int, models: list[Model]) -> dict[int, list[Model]]:
    """
    Compute optimal model placement across GPUs.
    
    Args:
        gpu_num: Number of available GPUs (typically 5-10)
        models: List of Model objects to place
    
    Returns:
        dict mapping gpu_id (int) to list of Models assigned to that GPU
        Example: {0: [model_a, model_b], 1: [model_c], 2: [model_d, model_e]}
    
    Constraints:
        - Each model must be assigned to exactly one GPU
        - Total model_size on each GPU must not exceed GPU_MEM_SIZE (80GB)
    
    Goal:
        Minimize max(KVPR) across all GPUs
    """
    pass
```

Scoring (0-100)
---------------
```
baseline_kvpr = Average max-KVPR using round-robin placement
optimal_kvpr = Theoretical minimum KVPR with perfect load balance
solution_kvpr = Your solution's average max-KVPR across all test cases

For each test case:
    raw_ratio = (baseline_kvpr - solution_kvpr) / (baseline_kvpr - optimal_kvpr)
    clamped_ratio = clamp(raw_ratio, 0, 1)
    test_score = 100 × sqrt(clamped_ratio)

final_score = Average of individual test scores
```

The sqrt scaling provides diminishing returns as solutions approach optimal, giving more credit for initial improvements over the baseline.

**Scoring Examples** (with sqrt scaling):
- raw_ratio = 0.00 → Score = 0.0   (no improvement over baseline)
- raw_ratio = 0.25 → Score = 50.0  (25% of the way to optimal)
- raw_ratio = 0.50 → Score = 70.7  (50% of the way to optimal)
- raw_ratio = 0.75 → Score = 86.6  (75% of the way to optimal)
- raw_ratio = 1.00 → Score = 100.0 (achieved theoretical optimal)

Implementation Notes
--------------------
- GPU memory: 80 GB per GPU
- Model sizes: 10-30 GB
- Request rates: 1-10 requests/second
- SLO targets: 5-10 (latency units)
- Number of models per test: 2× gpu_num
- Number of GPUs per test: 5-10
- Each test case has a 10-second timeout
- 50 test cases are evaluated
- Test cases use a fixed random seed (42) for reproducibility
