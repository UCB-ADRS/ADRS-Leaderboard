#!/usr/bin/env bash
set -euo pipefail

# Local test script that runs evaluation without Docker
# Usage: 
#   ./test_local.sh                                    # Reads from pairs.txt (default)
#   ./test_local.sh <pairs_file.txt>                  # Reads from specified pairs file
#   ./test_local.sh [solution_name] [problem_name]     # Run single evaluation
#   PAIRS_FILE=<file.txt> ./test_local.sh             # Use environment variable

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$BASE_DIR"

# Check for OPENAI_API_KEY if needed (for multiagent_system problem)
# Users should set this environment variable before running:
#   export OPENAI_API_KEY='your-key-here'

# If not set, warn but continue (some problems don't need it)
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[WARNING] OPENAI_API_KEY not set. Some problems (e.g., multiagent_system) require it." >&2
    echo "[WARNING] Set it with: export OPENAI_API_KEY='your-key-here'" >&2
    echo "[WARNING] Continuing anyway..." >&2
fi

# Function to run evaluation for a single solution-problem pair
run_single_evaluation() {
    local SOLUTION_NAME="$1"
    local PROBLEM_NAME="$2"

    # Ensure we're in BASE_DIR before checking paths
    cd "$BASE_DIR"

    echo "=========================================="
    echo "Local Test: $SOLUTION_NAME -> $PROBLEM_NAME"
    echo "=========================================="
    echo ""

    # Check if problem and solution exist (using absolute paths)
    if [ ! -d "$BASE_DIR/problems/$PROBLEM_NAME" ]; then
        echo "ERROR: Problem '$PROBLEM_NAME' not found" >&2
        return 1
    fi

    if [ ! -d "$BASE_DIR/solutions/$SOLUTION_NAME" ]; then
        echo "ERROR: Solution '$SOLUTION_NAME' not found" >&2
        return 1
    fi

    # Create execution environment inside problem resources
    EXEC_ROOT="$BASE_DIR/problems/${PROBLEM_NAME}/resources/execution_env"
    mkdir -p "$EXEC_ROOT"
    mkdir -p "$EXEC_ROOT/solution_env"

    echo "[INFO] Setting up problem environment for '${PROBLEM_NAME}'..."
    cd "problems/${PROBLEM_NAME}"

    # Download datasets if needed
    if [ -f "./download_datasets.sh" ]; then
        echo "[INFO] Downloading datasets..."
        bash ./download_datasets.sh
    fi

    # Set up environment
    if [ -f "./set_up_env.sh" ]; then
        echo "[INFO] Running set_up_env.sh..."
        chmod +x ./set_up_env.sh
        bash ./set_up_env.sh
    else
        echo "[WARNING] No set_up_env.sh found, skipping environment setup"
    fi

    cd "$BASE_DIR"

    # Prepare solution
    echo "[INFO] Preparing solution '${SOLUTION_NAME}'..."
    cd "solutions/${SOLUTION_NAME}"

    # Run prepare_env.sh if it exists
    if [ -f "./prepare_env.sh" ]; then
        echo "[INFO] Running prepare_env.sh..."
        chmod +x ./prepare_env.sh
        bash ./prepare_env.sh
    fi

    # Copy solution to execution environment
    echo "[INFO] Copying solution resources..."
    if [ -f "resources/solution.py" ]; then
        cp "resources/solution.py" "$EXEC_ROOT/solution_env/solution.py"
        echo "[INFO] Solution copied to $EXEC_ROOT/solution_env/solution.py"
    else
        echo "[ERROR] ERROR: resources/solution.py not found in solution directory"
        return 1
    fi

    cd "$BASE_DIR"

    # Run solution
    echo "[INFO] Running solution..."
    cd "solutions/${SOLUTION_NAME}"
    # Export PROBLEM_NAME for solve.sh to use
    export PROBLEM_NAME
    if [ -f "./solve.sh" ]; then
        chmod +x ./solve.sh
        bash ./solve.sh
    else
        echo "[WARNING] No solve.sh found, skipping solution execution"
    fi

    cd "$BASE_DIR"

    # Run evaluation
    echo "[INFO] Running evaluation..."
    cd "problems/${PROBLEM_NAME}"

    # Export SOLUTION_NAME for evaluate.sh to use
    export SOLUTION_NAME

    if [ -f "./evaluate.sh" ]; then
        chmod +x ./evaluate.sh
        echo "----------------------------------------"
        echo "EVALUATION OUTPUT:"
        echo "----------------------------------------"
        bash ./evaluate.sh
        EVAL_EXIT_CODE=$?
        echo "----------------------------------------"
        
        # Extract baseline name from solution path (e.g., "llm_sql/baseline" -> "baseline")
        BASELINE_NAME=$(basename "$SOLUTION_NAME")
        
        # Read results.json if it exists
        RESULTS_DIR="$BASE_DIR/results/$PROBLEM_NAME/$BASELINE_NAME"
        if [ -f "$RESULTS_DIR/results.json" ]; then
            echo ""
            echo "Results JSON:"
            cat "$RESULTS_DIR/results.json" | python3 -m json.tool 2>/dev/null || cat "$RESULTS_DIR/results.json"
            SCORE=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get('score', 'N/A'))" "$RESULTS_DIR/results.json" 2>/dev/null || echo "N/A")
            echo ""
            echo "Score: $SCORE"
            echo "Results saved to: $RESULTS_DIR/"
        fi
    else
        echo "[ERROR] evaluate.sh not found"
        return 1
    fi

    # Clean up execution_env after evaluation (if it exists)
    if [[ -d "$EXEC_ROOT" ]]; then
        echo "[INFO] Cleaning up execution_env..." >&2
        rm -rf "$EXEC_ROOT"
    fi

    echo ""
    echo "[INFO] Done for $SOLUTION_NAME -> $PROBLEM_NAME"
    echo ""
}

# Check if arguments are provided
if [ $# -ge 2 ]; then
    # Two arguments - treat as solution_name and problem_name (single evaluation)
    SOLUTION_NAME="$1"
    PROBLEM_NAME="$2"
    run_single_evaluation "$SOLUTION_NAME" "$PROBLEM_NAME"
    exit 0
elif [ $# -eq 1 ]; then
    # Single argument - check if it's a .txt file
    if [[ "$1" == *.txt ]]; then
        # It's a pairs file
        PAIRS_FILE="$BASE_DIR/$1"
        if [[ ! -f "$PAIRS_FILE" ]]; then
            echo "ERROR: Pairs file not found: $PAIRS_FILE" >&2
            exit 1
        fi
    else
        echo "ERROR: Invalid argument. Expected pairs file (.txt) or both solution_name and problem_name" >&2
        echo "       Usage: ./test_local.sh [pairs_file.txt] or ./test_local.sh [solution_name] [problem_name]" >&2
        exit 1
    fi
else
    # No arguments provided - use default or environment variable
    if [[ -n "${PAIRS_FILE:-}" ]]; then
        # Use environment variable if set
        if [[ "$PAIRS_FILE" != /* ]]; then
            # Relative path - make it relative to BASE_DIR
            PAIRS_FILE="$BASE_DIR/$PAIRS_FILE"
        fi
    else
        # Default to pairs.txt
        PAIRS_FILE="$BASE_DIR/pairs.txt"
    fi
    
    if [[ ! -f "$PAIRS_FILE" ]]; then
        echo "ERROR: Pairs file not found: $PAIRS_FILE" >&2
        exit 1
    fi
fi

# If we reach here, we're reading from a pairs file
echo "=========================================="
echo "Reading pairs from: $PAIRS_FILE"
echo "=========================================="
echo ""

# Read working pairs into a temporary file (skip empty lines and comments)
TEMP_PAIRS_FILE=$(mktemp)
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments (lines starting with #)
    if [ -n "$line" ] && ! echo "$line" | grep -q '^[[:space:]]*#'; then
        echo "$line" >> "$TEMP_PAIRS_FILE"
    fi
done < "$PAIRS_FILE"

# Check if we have any working pairs
if [ ! -s "$TEMP_PAIRS_FILE" ]; then
    echo "ERROR: No valid pairs found in $PAIRS_FILE" >&2
    rm -f "$TEMP_PAIRS_FILE"
    exit 1
fi

echo "Processing pairs from $PAIRS_FILE..."
echo ""

# Process each pair
PAIR_COUNT=0
while IFS= read -r pair; do
    IFS=':' read -r solution problem <<< "$pair"
    PAIR_COUNT=$((PAIR_COUNT + 1))
    
    echo "=========================================="
    echo "Pair $PAIR_COUNT: $solution -> $problem"
    echo "=========================================="
    echo ""
    
    run_single_evaluation "$solution" "$problem"
done < "$TEMP_PAIRS_FILE"

# Clean up temporary file
rm -f "$TEMP_PAIRS_FILE"

echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="

