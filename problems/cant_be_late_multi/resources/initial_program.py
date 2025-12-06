#!/usr/bin/env python3
"""
Simple greedy multi-region strategy for evolution starting point.
Uses basic greedy decisions across multiple regions.
"""

import argparse
import math
import typing

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task


# EVOLVE-BLOCK START
class SimpleGreedyMultiRegionStrategy(MultiRegionStrategy):
    """
    Simple greedy multi-region strategy.
    Tries SPOT in current region first, switches regions if SPOT unavailable,
    falls back to ON_DEMAND when deadline is critical.
    """

    NAME = 'evolved_greedy_multi'

    def __init__(self, args):
        super().__init__(args)
        self.next_region_to_try = 0

    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)
        self.next_region_to_try = 0

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """Simple greedy decision: use SPOT if available, otherwise try other regions."""
        env = self.env

        # Calculate remaining work
        remaining_task_time = self.task_duration - sum(self.task_done_time)
        if remaining_task_time <= 1e-3:
            return ClusterType.NONE

        # Calculate deadline pressure
        remaining_time = math.floor(
            (self.deadline - env.elapsed_seconds) / env.gap_seconds
        ) * env.gap_seconds

        total_task_remaining = math.ceil(
            (remaining_task_time + self.restart_overhead) / env.gap_seconds
        ) * env.gap_seconds

        total_task_remaining_2d = math.ceil(
            (remaining_task_time + 2 * self.restart_overhead) / env.gap_seconds
        ) * env.gap_seconds

        # Critical deadline check - must use ON_DEMAND
        if total_task_remaining >= remaining_time:
            # If we're on a working SPOT with no overhead remaining, risk staying
            if last_cluster_type == ClusterType.SPOT and self.remaining_restart_overhead < 1e-3:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # Near deadline - prefer ON_DEMAND unless already on working SPOT
        if total_task_remaining_2d >= remaining_time:
            if last_cluster_type == ClusterType.SPOT and has_spot:
                return ClusterType.SPOT
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND

        # Normal operation - try SPOT
        if has_spot:
            return ClusterType.SPOT

        # No SPOT in current region, try switching regions
        num_regions = env.get_num_regions()
        current_region = env.get_current_region()

        for i in range(num_regions):
            next_region = (current_region + 1 + i) % num_regions
            if next_region != current_region:
                env.switch_region(next_region)
                # Return NONE this tick, will try SPOT in new region next tick
                return ClusterType.NONE

        # No other regions to try, just wait
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser) -> 'SimpleGreedyMultiRegionStrategy':
        args, _ = parser.parse_known_args()
        return cls(args)
# EVOLVE-BLOCK END
