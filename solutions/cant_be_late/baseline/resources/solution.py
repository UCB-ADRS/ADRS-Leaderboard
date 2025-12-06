from pathlib import Path
from typing import Dict, Any


class Solution:
    """Baseline submission for cant_be_late: returns the initial greedy strategy."""

    def __init__(self) -> None:
        # When COPIED to execution_env/solution_env/solution.py and run from there:
        # parents[0]=solution_env, [1]=execution_env, [2]=resources, [3]=cant_be_late, [4]=problems, [5]=repo_root
        self._default_strategy_path = (
            Path(__file__).resolve()
            .parents[5]
            / "problems"
            / "cant_be_late"
            / "resources"
            / "programs"
            / "initial_greedy.py"
        )

    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        try:
            code = self._default_strategy_path.read_text(encoding="utf-8")
            return {"code": code}
        except FileNotFoundError:
            # Fallback minimal strategy if the reference file is missing.
            return {
                "code": (
                    "import math\n"
                    "from sky_spot.strategies.strategy import Strategy\n"
                    "from sky_spot.utils import ClusterType\n\n"
                    "class GreedySafetyStrategy(Strategy):\n"
                    "    NAME = 'greedy_safety_seed'\n\n"
                    "    def _step(self, last_cluster_type, has_spot):\n"
                    "        env = self.env\n"
                    "        gap = env.gap_seconds\n"
                    "        work_left = self.task_duration - sum(self.task_done_time)\n"
                    "        if work_left <= 1e-9:\n"
                    "            return ClusterType.NONE\n"
                    "        left_ticks = max(0, math.floor((self.deadline - env.elapsed_seconds) / gap))\n"
                    "        need1d = math.ceil((work_left + self.restart_overhead) / gap)\n"
                    "        need2d = math.ceil((work_left + 2 * self.restart_overhead) / gap)\n"
                    "        if need1d >= left_ticks:\n"
                    "            return ClusterType.ON_DEMAND\n"
                    "        if need2d >= left_ticks:\n"
                    "            if env.cluster_type == ClusterType.SPOT and has_spot:\n"
                    "                return ClusterType.SPOT\n"
                    "            return ClusterType.ON_DEMAND\n"
                    "        return ClusterType.SPOT if has_spot else ClusterType.NONE\n"
                )
            }
