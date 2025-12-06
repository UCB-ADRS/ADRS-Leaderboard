#!/bin/bash
set -euo pipefail

BASE_DIR=$(pwd)

if [[ "${OSTYPE:-linux}" != "darwin"* ]] && [[ "${OSTYPE:-linux}" != "msys"* ]] && [[ "${OSTYPE:-linux}" != "cygwin"* ]] && [[ "${OSTYPE:-linux}" != "win32"* ]]; then
    if ! groups | grep -q '\bdocker\b'; then
        echo "Adding current user to docker group..."
        # Use $USER if available, otherwise try to get username from whoami
        username="${USER:-$(whoami)}"
        if command -v sudo >/dev/null 2>&1; then
            sudo usermod -aG docker "$username"
            echo "User added to docker group. You may need to log out and back in for changes to take effect."
        else
            echo "Warning: sudo not available, skipping Docker group modification"
        fi
        echo ""
    fi
fi

# Read working pairs from file
WORKING_PAIRS_FILE="$BASE_DIR/pairs.txt"
if [[ ! -f "$WORKING_PAIRS_FILE" ]]; then
    echo "ERROR: Pairs file not found: $WORKING_PAIRS_FILE"
    exit 1
fi

# Read working pairs into a temporary file
TEMP_PAIRS_FILE=$(mktemp)
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments (lines starting with #)
    if [ -n "$line" ] && ! echo "$line" | grep -q '^[[:space:]]*#'; then
        echo "$line" >> "$TEMP_PAIRS_FILE"
    fi
done < "$WORKING_PAIRS_FILE"

# Check if we have any working pairs
if [ ! -s "$TEMP_PAIRS_FILE" ]; then
    echo "ERROR: No valid pairs found in $WORKING_PAIRS_FILE"
    rm -f "$TEMP_PAIRS_FILE"
    exit 1
fi

RESULTS_DIR="$BASE_DIR/results"
mkdir -p "$RESULTS_DIR"

DATASETS_DIR="$BASE_DIR/datasets"
mkdir -p "$DATASETS_DIR"

echo "Starting containerized evaluation..."
echo "Working pairs: $(cat "$TEMP_PAIRS_FILE" | tr '\n' ' ')"
echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo "Datasets will be cached in: $DATASETS_DIR"
echo ""

# Download datasets once at the beginning (skip if already downloaded)
echo "=========================================="
echo "Downloading datasets (if not already present)..."
echo "=========================================="

# Get unique list of problems from pairs
UNIQUE_PROBLEMS=""
while IFS= read -r pair; do
    IFS=':' read -r solution problem <<< "$pair"
    # Check if problem is already in the list
    found=false
    for existing in $UNIQUE_PROBLEMS; do
        if [ "$existing" = "$problem" ]; then
            found=true
            break
        fi
    done
    if [ "$found" = false ]; then
        UNIQUE_PROBLEMS="$UNIQUE_PROBLEMS $problem"
    fi
done < "$TEMP_PAIRS_FILE"

# Download datasets for each unique problem
for problem_name in $UNIQUE_PROBLEMS; do
    # Convert problem notation: imagenet_pareto/1m -> imagenet_pareto/1m or imagenet_pareto (if no slash)
    # Check both nested (with slash) and flat (without slash) formats
    problem_dir=""
    if [[ "$problem_name" == *"/"* ]]; then
        # Nested variant: imagenet_pareto/1m
        problem_dir="problems/$problem_name"
    else
        # Flat: imagenet_pareto
        problem_dir="problems/$problem_name"
    fi

    if [ ! -d "$problem_dir" ]; then
        echo "Warning: Problem '$problem_name' not found at $problem_dir, skipping dataset download..." >&2
        continue
    fi

    download_script="$BASE_DIR/$problem_dir/download_datasets.sh"

    if [ -f "$download_script" ]; then
        echo "[Download] Preparing datasets for $problem_name..." >&2
        # Redirect all output to stderr so it doesn't get captured
        if ! bash "$download_script" >&2; then
            echo "Warning: Dataset download failed for $problem_name, continuing anyway..." >&2
        fi
    fi
done

echo "=========================================="
echo "Dataset preparation complete."
echo "=========================================="
echo ""

# Function to run a single solution-problem pair in a container
run_solution_problem_pair() {
    local solution_name="$1"
    local problem_name="$2"
    # Sanitize container name by replacing slashes with underscores
    local sanitized_problem=$(echo "$problem_name" | tr '/' '_')
    local container_name="eval_${solution_name}_${sanitized_problem}_$(date +%s)"

    echo "==========================================" >&2
    echo "Running: $solution_name -> $problem_name" >&2
    echo "Container: $container_name" >&2
    echo "==========================================" >&2

    # Create a temporary directory for this evaluation
    local temp_dir=$(mktemp -d)

    local result_file="$RESULTS_DIR/${solution_name}_${sanitized_problem}_result.txt"

    # Create minimal directory structure and copy only necessary files
    mkdir -p "$temp_dir/ADRS/problems"
    mkdir -p "$temp_dir/ADRS/solutions"

    # For nested problem paths, create parent directories
    if [[ "$problem_name" == *"/"* ]]; then
        # Create full parent directory path for nested variant
        mkdir -p "$temp_dir/ADRS/problems/$(dirname "$problem_name")"
    fi

    # For nested solution paths, create parent directories
    if [[ "$solution_name" == *"/"* ]]; then
        # Create full parent directory path for nested solution (e.g., cant_be_late/baseline)
        mkdir -p "$temp_dir/ADRS/solutions/$(dirname "$solution_name")"
    fi

    # Copy only the specific problem and solution directories
    # For nested problems like imagenet_pareto/1m, this preserves the structure
    # Use tar to preserve directory structure properly
    (cd "$BASE_DIR/problems" && tar cf - "$problem_name" | tar xf - -C "$temp_dir/ADRS/problems") || \
        cp -r "$BASE_DIR/problems/$problem_name" "$temp_dir/ADRS/problems/"
    # Use tar to preserve nested solution structure (e.g., cant_be_late/baseline)
    (cd "$BASE_DIR/solutions" && tar cf - "$solution_name" | tar xf - -C "$temp_dir/ADRS/solutions") || \
        cp -r "$BASE_DIR/solutions/$solution_name" "$temp_dir/ADRS/solutions/"

    local docker_image="python:3.13-slim-trixie"
    local gpu_flags=""
    
    local base_problem=$(echo "$problem_name" | cut -d'/' -f1)
    if [ "$base_problem" = "eplb" ]; then
        docker_image="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime"
    fi

    # Check if docker is available
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: Docker is not installed or not in PATH." >&2
        rm -rf "$temp_dir"
        exit 1
    fi
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        echo "ERROR: Docker daemon is not running or not accessible." >&2
        rm -rf "$temp_dir"
        exit 1
    fi

    # Initialize exit code
    local exit_code=0

    docker run --rm \
        --name "$container_name" \
        $gpu_flags \
        -e OPENAI_API_KEY \
        -v "$temp_dir:/workspace:ro" \
        -v "$DATASETS_DIR:/datasets:ro" \
        -v "$BASE_DIR/run_in_container.sh:/run_in_container.sh:ro" \
        -w "/work" \
        "$docker_image" \
        bash -c "bash /run_in_container.sh \"$problem_name\" \"$solution_name\"" > "$result_file" 2>&1 || exit_code=$?

    # Extract score from result file and handle errors
    local score
    # Try to extract the last line as score, filtering out log messages
    local last_line=$(tail -1 "$result_file" 2>/dev/null || echo "")

    # Check if last line looks like a number (score) and is not a log message
    if echo "$last_line" | grep -q '^-?[0-9]\+\.\?[0-9]*$' && ! echo "$last_line" | grep -q '\[INFO\]'; then
        score="$last_line"
    elif [ $exit_code -eq 0 ]; then
        # Success case but no numeric score - look for any numeric line that's not a log message
        local numeric_line=$(grep -E '^-?[0-9]+\.?[0-9]*$' "$result_file" | grep -v '\[INFO\]' | tail -1)
        if [ -n "$numeric_line" ]; then
            score="$numeric_line"
        else
            score="$last_line"
        fi
    else
        # Error case - extract detailed error message
        local error_msg=""

        if grep -Eq "[A-Za-z]+Error:" "$result_file"; then
            error_msg=$(grep -E "[A-Za-z]+Error:" "$result_file" | head -1)
        elif grep -q "ERROR:" "$result_file"; then
            error_msg=$(grep "ERROR:" "$result_file" | tail -1)
        else
            # Look for actual error patterns in the last 50 lines
            error_msg=$(tail -50 "$result_file" | grep -iE "(error|failed|exception|traceback)" | tail -1)
        fi

        # Clean up the error message
        if [ -n "$error_msg" ]; then
            score="ERROR: $error_msg"
        else
            score="ERROR: Container execution failed (exit code: $exit_code)"
        fi
    fi

    echo "Full output saved to: $result_file" >&2
    echo "" >&2

    # Clean up temp directory
    rm -rf "$temp_dir"

    # Return the score for summary
    echo "$score"
}

# Initialize results summary
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "Evaluation Summary - $(date)" > "$SUMMARY_FILE"
echo "=================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Initialize CSV file
CSV_FILE="$RESULTS_DIR/results.csv"
echo "solution,problem,score,status,timestamp" > "$CSV_FILE"

# Function to escape CSV fields (handles commas, quotes, newlines)
escape_csv_field() {
    local field="$1"
    # Replace quotes with double quotes and wrap in quotes if needed
    if [[ "$field" == *[\",\n]* ]]; then
        field="${field//\"/\"\"}"
        echo "\"$field\""
    else
        echo "$field"
    fi
}

# Run only the working solution-problem pairs
while IFS= read -r pair; do
    IFS=':' read -r solution problem <<< "$pair"

    # Check if problem and solution exist
    if [ ! -d "problems/$problem" ]; then
        echo "ERROR: Problem '$problem' not found, skipping..."
        echo "$solution -> $problem: ERROR (problem not found)" >> "$SUMMARY_FILE"
        # Write to CSV
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "$solution,$problem,N/A,ERROR: problem not found,$timestamp" >> "$CSV_FILE"
        continue
    fi

    if [ ! -d "solutions/$solution" ]; then
        echo "ERROR: Solution '$solution' not found, skipping..."
        echo "$solution -> $problem: ERROR (solution not found)" >> "$SUMMARY_FILE"
        # Write to CSV
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "$solution,$problem,N/A,ERROR: solution not found,$timestamp" >> "$CSV_FILE"
        continue
    fi

    # Run the evaluation
    score=$(run_solution_problem_pair "$solution" "$problem")
    echo "$solution -> $problem:" >> "$SUMMARY_FILE"
    echo "Score: $score" >> "$SUMMARY_FILE"

    # Write to CSV
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    if [[ "$score" == ERROR:* ]]; then
        # Extract just the error message for the status column
        error_msg="${score#ERROR: }"
        escaped_error=$(escape_csv_field "$error_msg")
        echo "$solution,$problem,N/A,\"ERROR: $escaped_error\",$timestamp" >> "$CSV_FILE"
    else
        # Numeric score - treat as success
        escaped_score=$(escape_csv_field "$score")
        echo "$solution,$problem,$escaped_score,SUCCESS,$timestamp" >> "$CSV_FILE"
    fi
done < "$TEMP_PAIRS_FILE"

# Clean up temporary file
rm -f "$TEMP_PAIRS_FILE"

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Summary saved to: $SUMMARY_FILE"
echo "CSV results saved to: $CSV_FILE"
echo ""
echo "Full summary:"
cat "$SUMMARY_FILE"
echo ""
echo "CSV preview (first 10 lines):"
head -10 "$CSV_FILE"
