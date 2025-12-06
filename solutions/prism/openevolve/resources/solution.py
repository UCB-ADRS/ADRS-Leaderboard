GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    data = [(m, m.req_rate/m.slo, m.model_size, i) for i, m in enumerate(models)]
    
    def fit(k):
        keys = [lambda x: x[1] + k*x[2], lambda x: x[2], lambda x: x[1],
                lambda x: (x[1] + k*x[2]) * ((x[3] * 2654435761 % 1000) / 500.0 + 0.5),
                lambda x: (x[1] + k*x[2]) * ((x[3] * 0x5bd1e995 % 1000) / 500.0 + 0.5)]
        for key in keys:
            rem = [k * GPU_MEM_SIZE] * gpu_num
            ans = [[] for _ in range(gpu_num)]
            for m, l, s, _ in sorted(data, key=key, reverse=True):
                w = l + k*s
                bi, min_r = -1, float('inf')
                for i in range(gpu_num):
                    if rem[i] >= w and rem[i] < min_r:
                        bi, min_r = i, rem[i]
                if bi == -1: break
                rem[bi] -= w
                ans[bi].append(m)
            else: return {i: ans[i] for i in range(gpu_num)}
        return None

    low, high = 0.0, 1.0
    while not (best := fit(high)):
        low, high = high, high * 2
        if high > 1e9: break
    
    for _ in range(32):
        mid = (low + high) / 2
        if p := fit(mid): best, high = p, mid
        else: low = mid
    return best

# EVOLVE-BLOCK-END


# Solution class wrapper for evaluator compatibility
from pathlib import Path
from typing import Any, Dict

class Solution:
    """OpenEvolve solution for prism GPU model placement."""

    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        # Read this file and return the code
        code = Path(__file__).read_text(encoding="utf-8")
        # Extract everything up to the Solution class
        lines = code.split('\n')
        end_idx = next(i for i, line in enumerate(lines) if 'class Solution:' in line)
        program_code = '\n'.join(lines[:end_idx])
        return {"code": program_code}
