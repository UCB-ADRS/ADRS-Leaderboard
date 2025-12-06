# 🏆 ADRS Leaderboard 

Evaluation framework for [AI-Driven Research Systems (ADRS)](https://ucbskyadrs.github.io/about/), a Berkeley Sky Computing Lab initiative. Read the paper: [arXiv:2510.06189](https://arxiv.org/abs/2510.06189).

**🎯 [View Live Leaderboard →](https://ucbskyadrs.github.io/leaderboard) 📊**

> **⚠️ Work in Progress**  
> This repository is actively under development. We are adding more framework-evolved solutions into each problem. Problem specifications and solutions may change. Contributions and feedback are welcome!

## 📚 Problems

- [MoE expert placement](problems/eplb)
- [Transaction scheduling](problems/txn_scheduling)
- [LLM-SQL optimization](problems/llm_sql)
- [Spot instance scheduling (single region)](problems/cant_be_late)
- [Spot instance scheduling (multi-region)](problems/cant_be_late_multi)
- [Multi-region data transfer (Cloudcast)](problems/cloudcast)
- [Multi-agent system design](problems/multiagent_system)
- [ML model placement (PRISM)](problems/prism)
- [Network telemetry repair](problems/telemetry_repair)

Each problem directory contains detailed documentation in its `README.md`.

## 🚀 Quick Start

```bash
./main.sh
# or ./test_local.sh
```

Runs solution-problem pairs in Docker containers and generates results.

**Configuration**: `pairs.txt` specifies pairs to evaluate:
```
{problem_name}/{solution_name}:{problem_name}
```

Example: `llm_sql/baseline:llm_sql`, `llm_sql/openevolve:llm_sql`, `llm_sql/gepa:llm_sql`

Use custom pairs files:
```bash
./test_local.sh cloudcast.txt
PAIRS_FILE=cloudcast.txt ./test_local.sh
```

## 📁 Solution Structure

```
solutions/{problem_name}/{solution_name}/
  resources/solution.py   # Required
  solve.sh                # Required
  prepare_env.sh          # Optional
```

Solutions implement a `Solution` class with a `solve()` method. See each problem's `README.md` for interface details.

## 🔄 Evaluation Flow

```
set_up_env.sh → solve.sh → evaluate.sh → run_evaluator.sh → evaluator.py → score
```

1. **Setup**: Environment and datasets prepared (`set_up_env.sh` runs in Docker)
2. **Solution**: Code staged via `solve.sh` (and optionally `prepare_env.sh`)
3. **Execution**: Solution runs and generates output
4. **Evaluation**: `evaluator.py` computes score (0-100)
5. **Results**: 
   - Summary: `results/{solution_name}_{problem_name}_result.txt`
   - Detailed: `results/{problem_name}/{solution_name}/results.json` and `evaluation.log`

## 🏆 How to Submit to Leaderboard

### Submission Process

1. **Create solution directory** under `solutions/{problem_name}/{your_solution_name}/`
2. **Implement solution**:
   - `resources/solution.py`: Implement `Solution` class with `solve()` method
   - `solve.sh`: Stage solution code into execution environment
   - `prepare_env.sh` (optional): Install solution-specific dependencies
3. **Test locally**:
   ```bash
   ./test_local.sh {solution_name} {problem_name}  # Single pair
   ./test_local.sh                                  # All pairs from pairs.txt
   ```
4. **Add to `pairs.txt`**: `{problem_name}/{your_solution_name}:{problem_name}`
5. **Run evaluation**: `./main.sh`
6. **Verify results**: Check `results/{problem_name}/{your_solution_name}/results.json`
7. **Submit**: Create a pull request with your solution and a brief description

### Submission Requirements

<details>
<summary>📋 Required Assets</summary>
<br>

* **`resources/solution.py`**: Your solution implementation
* **`solve.sh`**: Script to stage your solution (must be executable)
* **`prepare_env.sh`** (optional): Script to install solution-specific dependencies

Your solution must:
- Implement the `Solution` class with a `solve()` method
- Follow the problem-specific interface requirements (see each problem's README)
- Run successfully in an isolated Docker container
- Generate output in the format expected by the evaluator

</details>

### Best Practices

- **Test locally first**: Use `test_local.sh` to verify before submitting
- **Follow the interface**: Match the problem-specific interface (see problem's `README.md`)
- **Keep it self-contained**: Solution should work with provided problem resources and datasets
- **Use custom pairs files**: Create files like `cloudcast.txt` to test specific subsets

## 📊 Scoring

Scoring details are in each problem's `README.md`. Scores range from 0-100, with higher scores indicating better performance.

## Acknowledgement

We reference the submission format from [FrontierCS](https://frontier-cs.org/), more release is coming soon!