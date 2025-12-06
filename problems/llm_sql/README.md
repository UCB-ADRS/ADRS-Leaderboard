# LLM SQL Problem

Optimize CSV column ordering to maximize prefix hit rate. Reorder columns so concatenated row values have maximum common prefix overlap between consecutive rows.

**Target**: Maximize prefix hit rate, minimize runtime (up to 10s threshold)

Evaluates on: movies.csv, beer.csv, BIRD.csv, PDMX.csv, products.csv

## API

```python
import pandas as pd

class Solution:
    def solve(
        self,
        df: pd.DataFrame,
        early_stop: int = 100000,
        row_stop: int = 4,
        col_stop: int = 2,
        col_merge: list = None,
        one_way_dep: list = None,
        distinct_value_threshold: float = 0.7,
        parallel: bool = True,
    ) -> pd.DataFrame:
        pass
```

Returns DataFrame with reordered columns. Rows are concatenated (no spaces) and prefix hit rate is calculated.

## Scoring

```
baseline_hit_rate = Average prefix hit rate using original column order
avg_hit_rate = Your solution's average prefix hit rate
avg_runtime = Average runtime per dataset (seconds)

normalized_hit_score = ((avg_hit_rate - baseline_hit_rate) / (1.0 - baseline_hit_rate)) × 100
normalized_hit_score = clamp(normalized_hit_score, 0, 100)

runtime_component = (10.0 - min(10.0, avg_runtime)) / 10.0 × 100

final_score = 0.95 × normalized_hit_score + 0.05 × runtime_component
```

Runtime component: 100 at 0s, 0 at 10s. Solutions ≥ 10s get 0 runtime component.