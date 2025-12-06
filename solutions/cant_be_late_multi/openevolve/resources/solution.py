# EVOLVE-BLOCK-START

import configargparse
import logging
import typing
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:
    from sky_spot import env, task

logger = logging.getLogger(__name__)

class AdaptivePacingStrategy(MultiRegionStrategy):
    """
    Adaptive Strategy with Dynamic Tolerance and Safety Buffers.
    
    This strategy improves upon fixed-pacing strategies by making the pacing 
    tolerance and safety buffers dynamic.
    
    Key Innovations:
    1. **Dynamic Safety Buffer**: The safety margin scales with `remaining_work`.
       Longer remaining tasks need a larger buffer to absorb accumulated overheads
       and variances, preventing the catastrophic system override.
    2. **Adaptive Pacing Tolerance**: The allowed lag (tolerance) is calculated based
       on the cost to scan all regions (`num_regions * 5s`). This ensures that even
       with tight slack, the strategy tries to reserve enough time for at least one
       full exploration cycle before panicking into On-Demand.
    3. **Smart Exploration**: Uses a freshness metric to prevent thrashing (rapid switching).
       It waits in the current region if global information is fresh, allowing time
       to capitalize on local Spot appearances without incurring switching costs.
    """
    NAME = 'adaptive_pacing_v1'

    def __init__(self, args: configargparse.Namespace):
        super().__init__(args)
        # Cache: region_idx -> last_visited_timestamp
        self.region_last_visited: typing.Dict[int, float] = {}
        self.initialized = False

    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)
        self.initialized = True
        # Initialize with -inf so any check is considered "old" initially
        for i in range(self.env.get_num_regions()):
            self.region_last_visited[i] = float('-inf')
        logger.info(f"{self.NAME} reset. D={self.deadline}, T={self.task_duration}")

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self.initialized or self.task.is_done:
            return ClusterType.NONE

        current_time = self.env.elapsed_seconds
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        
        # 1. Update Cache
        self.region_last_visited[current_region] = current_time

        # 2. Priority 1: Spot (Global Optimum)
        if has_spot:
            return ClusterType.SPOT

        # 3. Calculate Metrics
        done_work = sum(self.task_done_time)
        remaining_work = max(0.0, self.task_duration - done_work)
        remaining_time = max(0.001, self.deadline - current_time)
        
        # Slack: The absolute time buffer we have.
        slack = remaining_time - remaining_work
        
        # Total Slack: The initial total buffer available at start.
        total_slack = max(0.0, self.deadline - self.task_duration)

        # 4. Priority 2: Dynamic Safety Net
        # We need a buffer to prevent the system override (remaining_task >= remaining_time).
        # We use a base buffer (15s) plus a small fraction (1%) of remaining work.
        # This scales the safety net: if we have a lot of work left, we are more conservative.
        safety_buffer = 15.0 + (0.01 * remaining_work)
        
        if slack < safety_buffer:
            return ClusterType.ON_DEMAND

        # 5. Priority 3: Adaptive Pacing with Tolerance
        # Calculate Linear Target
        if self.deadline > 0:
            linear_ratio = current_time / self.deadline
            linear_ratio = max(0.0, min(1.0, linear_ratio))
            linear_target = self.task_duration * linear_ratio
        else:
            linear_target = self.task_duration

        # Calculate Adaptive Tolerance
        # We want to allow enough lag to scan all regions at least once.
        # Estimate 5.0s per region switch/check.
        scan_cost = num_regions * 5.0
        
        # Tolerance Logic:
        # Base: 20% of Total Slack.
        # Floor: scan_cost (Ensure we can scan).
        # Cap: 40% of Total Slack (Don't risk too much).
        
        ideal_tolerance = max(total_slack * 0.20, scan_cost)
        tolerance = min(ideal_tolerance, total_slack * 0.40)
        
        # Ensure tolerance is non-negative
        tolerance = max(0.0, tolerance)
        
        min_acceptable_progress = linear_target - tolerance

        if done_work < min_acceptable_progress:
            # We are behind schedule beyond our calculated tolerance.
            # Must catch up using On-Demand to preserve agility for later.
            return ClusterType.ON_DEMAND

        # 6. Priority 4: Smart Exploration (Freshness)
        # We are safe and within tolerance. Check if we should switch.
        
        oldest_visit_time = min(self.region_last_visited.values())
        data_age = current_time - oldest_visit_time
        
        # Freshness Threshold
        # If data is younger than a full scan cycle + buffer (10s), we consider it fresh.
        # This prevents switching loops (thrashing).
        freshness_threshold = scan_cost + 10.0
        
        is_data_fresh = data_age < freshness_threshold

        if is_data_fresh:
            # All regions checked recently. No Spot found.
            # Waiting in current region is better than thrashing or OD.
            # We essentially "spend" slack to wait for Spot in current region.
            return ClusterType.NONE
        
        # Data is Stale. Switch to the least recently visited region.
        best_region = self._get_lru_region(current_region)
        
        if best_region != current_region:
            self.env.switch_region(best_region)
            # Return NONE to allow switch to happen (transit time)
            return ClusterType.NONE
        
        # Fallback (e.g. single region)
        return ClusterType.NONE

    def _get_lru_region(self, current_region: int) -> int:
        """Finds the least recently visited region."""
        num_regions = self.env.get_num_regions()
        if num_regions <= 1:
            return current_region

        best_region = (current_region + 1) % num_regions
        oldest_time = float('inf')

        for r in range(num_regions):
            if r == current_region:
                continue
            
            last_visit = self.region_last_visited.get(r, float('-inf'))
            if last_visit < oldest_time:
                oldest_time = last_visit
                best_region = r
        
        return best_region

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)


# Solution class wrapper for evaluator compatibility
from pathlib import Path
from typing import Any, Dict

class Solution:
    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        code = Path(__file__).read_text(encoding="utf-8")
        lines = code.split("\n")
        end_idx = next(i for i, line in enumerate(lines) if "class Solution:" in line)
        program_code = "\n".join(lines[:end_idx])
        return {"code": program_code}
