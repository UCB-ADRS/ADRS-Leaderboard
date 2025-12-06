# Cloudcast Broadcast Optimization

Optimize broadcast topology for multi-cloud data distribution. Find optimal paths from a source to multiple destinations across AWS, Azure, and GCP to minimize transfer cost and time.

**Target**: Minimize total transfer cost ($/GB)

## API

```python
class Solution:
    def solve(self, spec_path: str = None) -> dict:
        # Returns {"code": "python_code_string"} or {"program_path": "path/to/algorithm.py"}
        pass

def search_algorithm(src: str, dsts: list[str], G: nx.DiGraph, num_partitions: int) -> BroadCastTopology:
    pass

class BroadCastTopology:
    def __init__(self, src: str, dsts: list[str], num_partitions: int):
        self.src = src
        self.dsts = dsts
        self.num_partitions = int(num_partitions)
        self.paths = {dst: {str(i): None for i in range(self.num_partitions)} for dst in dsts}
    
    def append_dst_partition_path(self, dst: str, partition: int, path: list, graph=None):
        pass
    
    def set_num_partitions(self, num_partitions: int):
        pass
```

Graph `G` has edge attributes: `cost` ($/GB) and `throughput` (Gbps).

## Scoring

```
LOWER_COST = 1199.00
UPPER_COST = 626.24
cost_clamped = max(min(total_cost, LOWER_COST), UPPER_COST)
normalized_cost = (LOWER_COST - cost_clamped) / (LOWER_COST - UPPER_COST)
score = normalized_cost * 100
```

Tested on 5 network configurations: intra-AWS, intra-Azure, intra-GCP, inter-AGZ, inter-GAZ2. Default: 2 VMs per region.

