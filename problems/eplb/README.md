# EPLB Problem

Optimize expert rearrangement for MoE models in distributed inference. Rearrange and replicate logical experts across physical GPU slots for optimal load balancing.

**Target**: Maximize GPU-level balancedness (avg_load / max_load), minimize execution time

Evaluates on real workload traces from vLLM server logs.

## API

```python
import torch

def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass
```

- `weight`: [layers, num_logical_experts], load statistics
- `num_replicas`: 288 (physical experts)
- `num_groups`: 8
- `num_nodes`: 4
- `num_gpus`: 32

Returns:
- `physical_to_logical_map`: [layers, num_replicas]
- `logical_to_physical_map`: [layers, num_logical_experts, X]
- `expert_count`: [layers, num_logical_experts]

## Scoring

```
balancedness_score_gpu = Average (avg_load / max_load) across all layers and workloads
avg_time_algorithm = Average algorithm execution time (seconds)

balancedness_score = balancedness_score_gpu × 90
speed_score = min(0.002 / time, 2) × 5
slow_penalty = min(time_seconds × 20, 20) if time > 10ms else 0

final_score = balancedness_score + speed_score - slow_penalty
```

Speed: ≤1ms = 10 points, 2ms = 5 points, 10ms = 1 point. Algorithms > 10ms receive penalty (up to 20 points).

Mapping from step i is evaluated against workload from step i+1.
