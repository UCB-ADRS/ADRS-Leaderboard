from pathlib import Path
from typing import Any, Dict


class Solution:
    """Human SOTA submission for prism: returns the initial program."""

    def __init__(self) -> None:
        # When copied to execution_env, the path is:
        # problems/prism/resources/execution_env/solution_env/solution.py
        # parents[0] = solution_env/
        # parents[1] = execution_env/
        # parents[2] = resources/
        # parents[3] = problems/prism/  <- initial_program.py is here
        # parents[4] = problems/
        current = Path(__file__).resolve()
        
        # Try multiple possible locations
        possible_paths = [
            current.parents[3] / "initial_program.py",  # From execution_env: problems/prism/
            current.parents[4] / "initial_program.py",  # Fallback
            current.parents[2] / "problems" / "prism" / "initial_program.py",  # Original structure
        ]
        
        self._default_program = None
        for path in possible_paths:
            if path.exists():
                self._default_program = path
                break

    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        """
        Solve the prism GPU model placement optimization problem.
        Args:
            spec_path: Path to specification file (optional)
        Returns:
            Dict with code or program_path
        """
        if self._default_program and self._default_program.exists():
            code = self._default_program.read_text(encoding="utf-8")
            return {"code": code}
        else:
            # Fallback minimal program if the reference file is missing
            return {
                "code": (
                    "GPU_MEM_SIZE = 80\n\n"
                    "def compute_model_placement(gpu_num, models):\n"
                    "    '''Simple baseline that assigns models round-robin.'''\n"
                    "    placement = {i: [] for i in range(gpu_num)}\n"
                    "    for i, model in enumerate(models):\n"
                    "        placement[i % gpu_num].append(model)\n"
                    "    return placement\n"
                )
            }

