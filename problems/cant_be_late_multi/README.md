# Cant Be Late Multi-Region Problem

## Critical Requirements

**Your Strategy class MUST include these to avoid evaluation errors:**

1. **NAME attribute**: Add a unique string identifier
2. **_from_args classmethod**: Required for evaluator instantiation
3. **__init__ method**: Must accept args parameter and call super().__init__(args)

See implementation details in the API specification below.

## Problem Setting

Design a multi-region scheduling policy for cloud compute tasks across multiple AWS regions where:
- **Spot instances**: Cheap but preemptible, causing job restart
- **On-demand instances**: Guaranteed but expensive
- **Multi-region**: Can switch between regions to find better spot availability

Given Spot instance traces across 9 AWS regions, create a scheduling strategy that minimizes cost by dynamically switching regions based on availability.

## Target

Minimize total cost across diverse region configurations while meeting deadlines.

## API Specification

Implement a `Solution` class:

```python
class Solution:
    def solve(self, spec_path: str = None) -> str | dict:
        """
        Returns a multi-region scheduling strategy.
        
        Returns one of:
        - Python code string implementing a MultiRegionStrategy subclass
        - {"code": "python_code_string"}
        - {"program_path": "path/to/strategy.py"}
        """
        # Your implementation
        pass
```

Your strategy code must implement:

```python
from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

class YourStrategy(MultiRegionStrategy):
    NAME = "your_strategy_name"  # REQUIRED: Add unique identifier
    
    def __init__(self, args=None):  # REQUIRED: Must accept args parameter
        super().__init__(args)
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        
        Available attributes:
        - self.env.get_current_region(): Get current region index (0-8)
        - self.env.switch_region(idx): Switch to region by index
        - self.env.elapsed_seconds: Current time elapsed
        - self.task_duration: Total task duration needed
        - self.task_done_time: List of completed work segments
        - self.deadline: Deadline time

        Returns: ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        # Implement your decision logic
        pass
    
    @classmethod
    def _from_args(cls, parser):  # REQUIRED: For evaluator instantiation
        args, _ = parser.parse_known_args()
        return cls(args)
```

## Scoring (0-100)

```
OD_anchor = Cost of running fully on-demand (baseline upper bound) = 3.06 × 48 = 146.88
SPOT_anchor = Cost of running fully on spot (baseline lower bound) = 0.9731 × 48 = 46.71
AvgCost = Your strategy's average cost across all scenarios

normalized_score = (OD_anchor - AvgCost) / (OD_anchor - SPOT_anchor)
score = clip(normalized_score, 0, 1) × 100
```

- 0 = Cost of full on-demand
- 100 = Cost of optimal spot strategy (theoretical best)

## Evaluation Details

**Stage 1**: Quick check on 2-region scenario (must pass to proceed)  
**Stage 2**: Full evaluation on 6 scenarios:
- 2 zones same region (8 traces)
- 2 regions east-west (8 traces)
- 3 regions diverse (6 traces)
- 3 zones same region (6 traces)
- 5 regions high diversity (4 traces)
- All 9 regions (2 traces)

Task: 48 hours duration, 52 hour deadline (4-hour slack)

## Implementation Notes

**Required Elements (Missing these will cause evaluation failures):**
- `NAME` attribute must be defined on your Strategy class
- `_from_args` classmethod must be implemented
- `__init__` method must accept args parameter and call super().__init__(args)

