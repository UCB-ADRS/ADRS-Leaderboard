# EVOLVE-BLOCK-START

import configargparse
import logging
import typing

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:
    from sky_spot import env, task

logger = logging.getLogger(__name__)

class EvolutionaryStrategy(MultiRegionStrategy):
    NAME = 'evolutionary_robust_starter'

    def __init__(self, args: configargparse.Namespace):
        super().__init__(args)
        # Tracks how many regions we have checked consecutively without finding Spot
        self.consecutive_searches = 0

    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)
        self.consecutive_searches = 0
        logger.info(f"{self.NAME} strategy has been reset.")

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decides on the next action: request SPOT, switch region (NONE), or fallback to ON_DEMAND.
        Prioritizes Spot usage but switches to On-Demand if the deadline is threatened.
        """
        # 1. If Spot is available, greedily take it.
        if has_spot:
            self.consecutive_searches = 0
            return ClusterType.SPOT

        # 2. No Spot available. Prepare to search or fallback.
        self.consecutive_searches += 1
        
        # Calculate progress and deadlines
        work_done = sum(self.task_done_time)
        remaining_work = max(0, self.task_duration - work_done)
        
        # If work is basically done, just wait/return NONE (system will handle completion)
        if remaining_work <= 1e-6:
            return ClusterType.NONE

        remaining_time = max(0, self.deadline - self.env.elapsed_seconds)
        num_regions = self.env.get_num_regions()

        # Calculate Safety Ratio: Available Time / Required Work
        # We need a buffer > 1.0 to account for region switching delays and restart overheads.
        safety_ratio = (remaining_time / remaining_work) if remaining_work > 0 else 999.0

        # Heuristic Thresholds:
        # We lower the critical ratio to 1.15 (vs previous 1.25) to be more aggressive 
        # with Spot searching. We also removed the cap on max search loops, allowing 
        # the strategy to persist in searching as long as the deadline allows.
        # This minimizes premature OD fallback in long-deadline/sparse-availability scenarios.
        CRITICAL_RATIO = 1.15
        
        # Ensure we have scanned the entire fleet at least once before considering fallback
        checked_all_regions = self.consecutive_searches >= num_regions

        # Decision Logic:
        # Only fallback if we have looked everywhere AND we are critically short on time.
        if checked_all_regions and safety_ratio < CRITICAL_RATIO:
            logger.info(f"Fallback to OD: Safety ratio {safety_ratio:.2f} < {CRITICAL_RATIO}")
            return ClusterType.ON_DEMAND

        # 3. Exploration Strategy: Round Robin
        # Switch to the next region and return NONE to continue the search loop.
        current_region_idx = self.env.get_current_region()
        next_region = (current_region_idx + 1) % num_regions
        self.env.switch_region(next_region)
        
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)

# EVOLVE-BLOCK-END


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
