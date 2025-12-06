# Cant-Be-Late Scheduling Problem

## Critical Requirements

**Your Strategy class MUST include these to avoid evaluation errors:**

1. **NAME attribute**: Add a unique string identifier
2. **_from_args classmethod**: Required for evaluator instantiation
3. **Careful NONE handling**: Returning NONE when work remains can cause simulation errors

See implementation details in the API specification below.

## Problem Setting

Design a scheduling policy for running compute tasks on cloud infrastructure where:
- **Spot instances**: Cheap but can be preempted at any time, causing job restart
- **On-demand instances**: Guaranteed availability but expensive

Given historical Spot instance traces and task requirements, create a dynamic scheduling strategy that meets deadlines while minimizing cost.

## Target

Minimize total cost while ensuring tasks complete before their deadline.

## API Specification

Implement a `Solution` class:

```python
class Solution:
    def solve(self, spec_path: str = None) -> str | dict:
        """
        Returns a scheduling strategy implementation.
        
        Returns one of:
        - Python code string implementing a Strategy subclass
        - {"code": "python_code_string"}
        - {"program_path": "path/to/strategy.py"}
        """
        # Your implementation
        pass
```

Your strategy code must implement:

```python
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class YourStrategy(Strategy):
    NAME = "your_strategy_name"  # REQUIRED: Add unique identifier
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Return next cluster type to use.
        
        Available attributes:
        - self.env.elapsed_seconds: Current time elapsed
        - self.env.gap_seconds: Time step size
        - self.env.cluster_type: Current cluster type
        - self.task_duration: Total task duration needed
        - self.task_done_time: List of completed work segments
        - self.deadline: Deadline time
        - self.restart_overhead: Time overhead when restarting
        
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
OD_anchor = Cost of running fully on-demand (baseline upper bound)
SPOT_anchor = Cost of running fully on spot (baseline lower bound)
AvgCost = Your strategy's average cost

normalized_score = (OD_anchor - AvgCost) / (OD_anchor - SPOT_anchor)
score = clip(normalized_score, 0, 1) × 100
```

## Evaluation Details

- Tested on 1080 evaluations across 6 cloud environments:
  - us-west-2a_k80_8, us-west-2b_k80_1, us-west-2b_k80_8
  - us-west-2a_v100_1, us-west-2a_v100_8, us-west-2b_v100_1
- 30 traces per environment, 2 deadline configs, 3 restart overhead configs
- Task duration: 48 hours
- Deadlines: 52 hours and 70 hours
- Restart overhead: 0.02, 0.05, 0.1 hours
- 0 = Cost of full on-demand
- 100 = Cost of optimal spot strategy

## Data Setup

Before running evaluations, extract the trace data:

```bash
cd resources/cant-be-late-simulator
tar -xzf ../real_traces.tar.gz
mkdir -p data && mv real data/
```

## Implementation Notes

**Required Elements (Missing these will cause evaluation failures):**
- `NAME` attribute must be defined on your Strategy class (avoids "Name abstract already exists" error)
- `_from_args` classmethod must be implemented (avoids JSON decode errors)
- Ensure proper handling of ClusterType.NONE return values (avoids "Timestamp X out of range X" error)

