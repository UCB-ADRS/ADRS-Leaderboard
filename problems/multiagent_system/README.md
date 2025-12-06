Multi-Agent System Problem
===========================

Problem Setting
---------------
Optimize multi-agent systems to minimize failure modes detected by an LLM judge. Given programming tasks, coordinate multiple agents (coder, tester, reviewer, verifier) to complete tasks while minimizing 14 types of failure modes defined in the MAST taxonomy.

The system evaluates agent interactions on programming tasks from the programdev dataset. An LLM judge analyzes execution traces to detect failure modes across three categories: System Design Issues (1.1-1.5), Inter-Agent Misalignment (2.1-2.6), and Verification Issues (3.1-3.3).

This problem evaluates on multiple programming tasks sampled from the programdev dataset.

Target
------
- **Primary**: Minimize failure modes (lower is better, 0-14 possible per task)
- **Secondary**: Complete tasks successfully
- **Tertiary**: Efficient agent coordination and communication

API Specification
-----------------
Implement a `run_multi_agent_task` function:

```python
async def run_multi_agent_task(
    idea: str,
    n_rounds: int = 4,
    log_file: str = None
) -> str:
    """
    Run a multi-agent task and return the execution trace.
    
    Args:
        idea: The programming task description to complete
        n_rounds: Number of agent interaction rounds (default: 4)
        log_file: Optional path to log file for execution trace (default: None)
    
    Returns:
        Execution trace string containing all agent interactions
    """
    # Your implementation
    pass
```

**Evaluation Process**:
1. Your `run_multi_agent_task()` function executes the multi-agent system on each sampled task
2. Execution traces are captured and analyzed by an LLM judge
3. The judge detects 14 types of failure modes (MAST taxonomy) for each task
4. Each task receives a normalized score (0-100%) based on its failure count
5. The final score is the average of normalized scores across all evaluated tasks

Scoring (0-100)
---------------
The final score is calculated by scoring each task individually, normalizing to 0-100%, and then averaging across all tasks:

**For each task**:
1. Count failure modes detected (0-14 range)
2. Calculate raw score: `raw_score = 1.0 / (1.0 + failures)`
   - Maximum: `1.0` (when failures = 0)
   - Minimum: `1/15` (when failures = 14)
3. Normalize to 0-100%:
   ```
   min_raw_score = 1.0 / 15.0
   max_raw_score = 1.0
   normalized_score = ((raw_score - min_raw_score) / (max_raw_score - min_raw_score)) × 100
   normalized_score = clamp(normalized_score, 0, 100)
   ```

**Final score**: Average of normalized scores across all evaluated tasks

**Scoring Examples** (per task):
- failures = 0 (perfect, no failures):
  - raw_score = 1.0 / (1.0 + 0) = 1.0
  - normalized_score = ((1.0 - 1/15) / (1.0 - 1/15)) × 100 = 100

- failures = 7 (half of possible failures):
  - raw_score = 1.0 / (1.0 + 7) = 0.125
  - normalized_score = ((0.125 - 1/15) / (1.0 - 1/15)) × 100 ≈ 6.25

- failures = 14 (all failures):
  - raw_score = 1.0 / (1.0 + 14) = 1/15 ≈ 0.0667
  - normalized_score = ((1/15 - 1/15) / (1.0 - 1/15)) × 100 = 0

**Note**: Timeouts are penalized with 7 failures, and errors are penalized with 14 failures per task.

Implementation Notes
--------------------
- The system evaluates on 6 randomly sampled tasks from the programdev dataset per evaluation
- Each task runs with the default n_rounds=4 (as specified in the function signature)
- Each task has a 60-second timeout; timeouts are penalized with 7 failures
- Execution traces are captured via log files and analyzed by an LLM judge
- The judge evaluates 14 failure modes:
  - **Individual Agent (1.1-1.5)**: Task/role disobedience, repetition, memory loss, unaware of termination
  - **Inter-Agent (2.1-2.6)**: Conversation reset, fail to ask clarification, task derailment, information withholding, ignored input, action-reasoning mismatch
  - **System-Level (3.1-3.3)**: Premature termination, no/incorrect verification, weak verification
- Solutions should focus on improving agent coordination, communication patterns, error handling, and verification steps
