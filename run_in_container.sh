set -euo pipefail

# Parse arguments
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <problem_name> <solution_name>" >&2
    exit 1
fi

PROBLEM_NAME="$1"
SOLUTION_NAME="$2"

# Export SOLUTION_NAME for evaluate.sh to use
export SOLUTION_NAME

# Helpers & error handling
error_handler() {
    local code=$?
    echo "ERROR: command failed with exit code ${code}" >&2
    if [[ -s /tmp/error.log ]]; then
        echo "---------- Begin captured stderr ----------" >&2
        head -200 /tmp/error.log >&2
        echo "----------- End captured stderr -----------" >&2
    fi
    exit "${code}"
}
trap error_handler ERR
# Duplicate all stderr to /tmp/error.log
exec 2> >(tee /tmp/error.log >&2)

# Prepare workspace
echo "[INFO] Copying repository to writable workspace…" >&2
mkdir -p /work
cp -r /workspace/ADRS /work/
cd /work/ADRS
if ! command -v uv >/dev/null 2>&1; then
  echo "[info] Installing uv..."
  pip install --user uv || exit 1
  export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

mkdir -p "problems/${PROBLEM_NAME}/resources/execution_env"

# Problem set-up
echo "[INFO] Setting up problem environment for '${PROBLEM_NAME}'…" >&2
cd "problems/${PROBLEM_NAME}"

# Set up environment (check if exists, like test_local.sh)
if [[ -f ./set_up_env.sh ]]; then
    chmod +x ./set_up_env.sh
    ./set_up_env.sh 2> /tmp/problem_setup_err.log || {
        echo "ERROR: problem setup failed" >&2
        if [[ -s /tmp/problem_setup_err.log ]]; then
            echo "---------- Problem setup stderr ----------" >&2
            head -200 /tmp/problem_setup_err.log >&2
            echo "--------------- End stderr ---------------" >&2
        fi
        exit 1
    }
else
    echo "[WARNING] No set_up_env.sh found, skipping environment setup" >&2
fi

cd /work/ADRS  # back to repo root

# Solution preparation
echo "[INFO] Preparing solution '${SOLUTION_NAME}'…" >&2
cd "solutions/${SOLUTION_NAME}"
# Run prepare_env.sh if it exists
if [[ -f ./prepare_env.sh ]]; then
    chmod +x ./prepare_env.sh
    ./prepare_env.sh
fi

# Copy the solution's code into the execution environment.
echo "[INFO] Copying solution resources…" >&2
mkdir -p "/work/ADRS/problems/${PROBLEM_NAME}/resources/execution_env/solution_env"

# Check if solution.py exists (like test_local.sh)
if [[ -f "resources/solution.py" ]]; then
    cp "resources/solution.py" \
       "/work/ADRS/problems/${PROBLEM_NAME}/resources/execution_env/solution_env/"
else
    echo "[ERROR] resources/solution.py not found in solution directory" >&2
    exit 1
fi

# Run solution
echo "[INFO] Running solution…" >&2
# Export PROBLEM_NAME for solve.sh to use
export PROBLEM_NAME

# Check if solve.sh exists (like test_local.sh)
if [[ -f ./solve.sh ]]; then
    chmod +x ./solve.sh
    ./solve.sh
else
    echo "[WARNING] No solve.sh found, skipping solution execution" >&2
fi

cd /work/ADRS  # back to repo root

# Evaluation
echo "[INFO] Setting up evaluation…" >&2

echo "[INFO] Running evaluation…" >&2
cd "/work/ADRS/problems/${PROBLEM_NAME}"

# Check if evaluate.sh exists (like test_local.sh)
if [[ -f ./evaluate.sh ]]; then
    chmod +x ./evaluate.sh
    ./evaluate.sh
else
    echo "[ERROR] evaluate.sh not found" >&2
    exit 1
fi

EXEC_ROOT="/work/ADRS/problems/${PROBLEM_NAME}/resources/execution_env"
if [[ -d "$EXEC_ROOT" ]]; then
    echo "[INFO] Cleaning up execution_env..." >&2
    rm -rf "$EXEC_ROOT"
fi

# Also clean up any incorrectly created execution_env in solutions directory
if [[ -d "/work/ADRS/solutions/problems" ]]; then
    echo "[INFO] Cleaning up incorrectly created solutions/problems directory..." >&2
    rm -rf "/work/ADRS/solutions/problems"
fi

# Done
echo "[INFO] Done" >&2