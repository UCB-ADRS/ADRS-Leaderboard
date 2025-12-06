# Problems Directory

Each problem must have:
- `set_up_env.sh` - Sets up the environment
- `evaluate.sh` - Runs the evaluation
- `download_datasets.sh` - Downloads/extracts datasets
- `README.md` - Problem documentation

## Workflow

1. `download_datasets.sh` - Downloads datasets to cache
2. `set_up_env.sh` - Creates environment (uses cached datasets if available)
3. `evaluate.sh` - Runs evaluation, assumes `solution_env` in `../../execution_env` with `solution.py`
4. `evaluate.sh` returns score (printed to stdout as last line)

Each problem must specify:
1. Environment setup
2. Input/output format for `solution.py`
3. Dataset download and usage

