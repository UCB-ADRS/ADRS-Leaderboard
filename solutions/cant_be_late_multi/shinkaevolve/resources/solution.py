# EVOLVE-BLOCK-START

import configargparse
import json
import logging
import math
import typing

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:
    from sky_spot import env, task

logger = logging.getLogger(__name__)

class EvolutionaryStrategy(MultiRegionStrategy):
    """
    A robust, stateful, and well-structured strategy for multi-region environments.
    This initial program serves as a strong and safe starting point for evolution.
    It correctly handles the object lifecycle, provides a basic caching mechanism,
    and implements a sound, urgency-based heuristic.
    """
    NAME = 'evolutionary_robust_starter'

    def __init__(self, args: configargparse.Namespace):
        super().__init__(args)
        # --- Framework Lifecycle Note ---
        # `self.env` and `self.task` are NOT available in `__init__`.
        # They are initialized later by the framework via the `reset()` method.
        # Therefore, any attributes that depend on them must be initialized here
        # as None or empty, and populated in `reset()`.

        # --- State Variables ---
        self.initialized: bool = False
        self.region_cache: typing.Dict[int, typing.Dict[str, typing.Any]] = {}
        self.next_exploration_target_idx: int = 0
        self.consecutive_failures: int = 0

    def reset(self, env: 'env.Env', task: 'task.Task'):
        """Called by the framework to initialize environment-dependent state."""
        super().reset(env, task)
        # Initialize the cache for all known regions
        for i in range(self.env.get_num_regions()):
            self.region_cache[i] = {
                'has_spot': None,
                'last_checked': -1,
                'success_count': 0
            }
        self.consecutive_failures = 0
        self.initialized = True
        logger.info(f"{self.NAME} strategy has been reset and initialized.")

    def _get_urgency(self) -> float:
        """
        Calculates the urgency as (Work Remaining / Time Remaining).
        Returns a float. > 1.0 means impossible to finish without restart overheads.
        """
        if self.task_done:
            return 0.0

        work_remaining = self.task_duration - sum(self.task_done_time)
        time_remaining = self.deadline - self.env.elapsed_seconds

        if time_remaining <= 0.001:
            return 100.0  # Max urgency

        return work_remaining / time_remaining

    def _choose_exploration_target(self) -> int:
        """
        Selects the best region to explore based on cached spot availability, recency, and historical success.
        """
        current_idx = self.env.get_current_region()
        num_regions = self.env.get_num_regions()

        if num_regions <= 1:
            return current_idx

        best_idx = (current_idx + 1) % num_regions
        best_score = -float('inf')

        for i in range(num_regions):
            if i == current_idx:
                continue

            stats = self.region_cache.get(i)
            if not stats:
                continue

            score = 0.0

            # 1. Base Preference
            if stats['has_spot'] is True:
                score += 2000.0
            elif stats['has_spot'] is False:
                score -= 1000.0

            # 2. Weighted Staleness
            # Recency is weighted by historical success to revisit good regions faster.
            last_checked = stats['last_checked']
            if last_checked < 0:
                staleness = 5000.0  # High priority for unvisited
            else:
                staleness = self.env.elapsed_seconds - last_checked
                # Cap staleness to prevent it from infinitely overriding the "no spot" penalty
                # unless a significant amount of time has passed (e.g., 20 mins).
                # 1200s * 1.0 = 1200 > 1000 (overcomes penalty).
                if staleness > 1200.0:
                    staleness = 1200.0

            success_count = stats.get('success_count', 0)
            if success_count > 0:
                # Logarithmic boost: 10 successes -> ~3.4x multiplier
                staleness *= (1.0 + math.log(1 + success_count))

            score += staleness

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Adaptive strategy step using urgency, cached region info, and failure tracking.
        """
        if not self.initialized or self.task_done:
            return ClusterType.NONE

        current_region_idx = self.env.get_current_region()

        # Update cache
        self.region_cache[current_region_idx]['has_spot'] = has_spot
        self.region_cache[current_region_idx]['last_checked'] = self.env.elapsed_seconds

        if has_spot:
            self.consecutive_failures = 0
            self.region_cache[current_region_idx]['success_count'] = \
                self.region_cache[current_region_idx].get('success_count', 0) + 1
        else:
            self.consecutive_failures += 1

        urgency = self._get_urgency()

        # Dynamic Panic Threshold
        # Standard threshold is 0.95 (close to deadline but safe).
        # If we have cycled all regions without finding spot, the environment is saturated.
        # Lower threshold to 0.85 to stop burning time on futile exploration.
        panic_threshold = 0.95
        if self.consecutive_failures >= self.env.get_num_regions():
            panic_threshold = 0.85

        if urgency > panic_threshold:
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND

        # Normal Operation
        if has_spot:
            return ClusterType.SPOT
        else:
            # Explore
            target = self._choose_exploration_target()
            self.env.switch_region(target)
            return ClusterType.NONE

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
