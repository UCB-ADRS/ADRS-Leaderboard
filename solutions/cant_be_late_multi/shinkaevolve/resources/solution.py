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

class AdaptiveHibernationStrategy(MultiRegionStrategy):
    NAME = 'adaptive_hibernation_strategy'

    def __init__(self, args: configargparse.Namespace):
        super().__init__(args)
        # Region State
        self.region_stats: typing.Dict[int, typing.Dict[str, float]] = {}
        self.initialized = False

        # Drought / Backoff State
        self.consecutive_dry_switches = 0
        self.is_waiting = False
        self.stand_down_until = -1.0
        self.current_backoff_duration = 2.0  # Starts at 2s
        self.max_backoff = 32.0              # Deep hibernation cap

        # Hyperparameters
        self.ema_alpha = 0.2
        self.prob_weight = 60.0       # High weight to trust EMA
        self.switch_penalty = 20.0    # Explicit penalty for switching
        self.drought_threshold = 4    # Set in reset based on topology
        self.start_buffer = 0.25      # 25% initial slack
        self.base_panic_urgency = 0.95

    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)
        num_regions = self.env.get_num_regions()
        # Topology-Adaptive Drought Thresholds
        self.drought_threshold = max(2, num_regions)

        for i in range(num_regions):
            self.region_stats[i] = {
                'prob': 0.5,
                'last_checked': -300.0
            }
        self.initialized = True
        self.consecutive_dry_switches = 0
        self.is_waiting = False
        self.stand_down_until = -1.0
        self.current_backoff_duration = 2.0
        logger.info(f"{self.NAME} reset. Deadline: {self.deadline}")

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self.initialized or self.task.is_done:
            return ClusterType.NONE

        current_region = self.env.get_current_region()
        now = self.env.elapsed_seconds

        # 1. Handle Stand-Down / Waiting Logic
        if self.is_waiting:
            if now < self.stand_down_until:
                return ClusterType.NONE

            # Wake up from stand-down
            self.is_waiting = False

            if has_spot:
                # Success!
                self.current_backoff_duration = 2.0
            else:
                # Failure after wait. Apply softer penalty to avoid thrashing.
                self.region_stats[current_region]['prob'] *= 0.8
                # Extended Hibernation Cap
                self.current_backoff_duration = min(self.current_backoff_duration * 2.0, self.max_backoff)

        # 2. Update EMA Beliefs
        stats = self.region_stats[current_region]
        obs = 1.0 if has_spot else 0.0
        stats['prob'] = stats['prob'] * (1.0 - self.ema_alpha) + obs * self.ema_alpha
        stats['last_checked'] = now

        if has_spot:
            # Found a spot (either immediately or after wait)
            self.consecutive_dry_switches = 0
            self.stand_down_until = -1.0
            self.current_backoff_duration = 2.0 # Reset backoff
            return ClusterType.SPOT

        # 3. Panic Checks (Cubic Funnel + Continuous Modulated Urgency)
        work_rem = self.task_duration - sum(self.task_done_time)
        time_rem = self.deadline - now

        if time_rem <= 1e-9:
            return ClusterType.ON_DEMAND

        # Continuous Panic Modulation:
        # Relax panic threshold if we are switching often (desperate) or waking from deep sleep.
        relaxation = 0.01 * self.consecutive_dry_switches
        if self.current_backoff_duration > 2.0:
            relaxation += 0.03

        panic_threshold = min(0.99, self.base_panic_urgency + relaxation)

        urgency = work_rem / time_rem
        if urgency > panic_threshold:
            return ClusterType.ON_DEMAND

        # Cubic Funnel
        time_ratio = now / self.deadline
        completion_ratio = sum(self.task_done_time) / self.task_duration

        # Buffer decreases cubically: 0.25 -> 0.0
        current_buffer = max(0.0, self.start_buffer * (1.0 - (time_ratio ** 3)))
        required_ratio = time_ratio - current_buffer

        if completion_ratio < required_ratio:
            return ClusterType.ON_DEMAND

        # 4. Smart Exploration & Switching
        best_region = -1
        best_score = -float('inf')

        # Deadline-Decaying Exploration (Staleness Damping)
        staleness_damping = max(0.0, 1.0 - (time_ratio ** 4))

        for r_idx, r_stats in self.region_stats.items():
            if r_idx == current_region:
                staleness = 0.0
                switch_cost = 0.0
            else:
                raw_staleness = now - r_stats['last_checked']
                if r_stats['last_checked'] < 0:
                    raw_staleness = 1000.0

                staleness = raw_staleness * staleness_damping
                switch_cost = self.switch_penalty

            score = (r_stats['prob'] * self.prob_weight) + staleness - switch_cost

            if score > best_score:
                best_score = score
                best_region = r_idx

        # Decision
        if best_region != -1 and best_region != current_region:
            self.consecutive_dry_switches += 1

            if self.consecutive_dry_switches >= self.drought_threshold:
                # Enter Stand-Down
                self.is_waiting = True
                self.stand_down_until = now + self.current_backoff_duration
                self.consecutive_dry_switches = 0
                return ClusterType.NONE
            else:
                self.env.switch_region(best_region)

        return ClusterType.NONE

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
