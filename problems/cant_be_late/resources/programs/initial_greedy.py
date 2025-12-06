import argparse
import math

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class GreedySafetyStrategy(Strategy):
    """Baseline strategy: stay on spot until deadline pressure dictates on-demand."""

    NAME = "greedy_safety_seed"

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = self.env
        gap = env.gap_seconds

        work_left = self.task_duration - sum(self.task_done_time)
        if work_left <= 1e-9:
            return ClusterType.NONE

        left_ticks = max(0, math.floor((self.deadline - env.elapsed_seconds) / gap))
        need1d = math.ceil((work_left + self.restart_overhead) / gap)
        need2d = math.ceil((work_left + 2 * self.restart_overhead) / gap)

        if need1d >= left_ticks:
            return ClusterType.ON_DEMAND

        if need2d >= left_ticks:
            if env.cluster_type == ClusterType.SPOT and has_spot:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        return ClusterType.SPOT if has_spot else ClusterType.NONE

    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser) -> "GreedySafetyStrategy":
        # Accept the default arguments; return an instance seeded with them.
        parser.add_argument_group("GreedySafetyStrategy")
        args, _ = parser.parse_known_args()
        return cls(args)
