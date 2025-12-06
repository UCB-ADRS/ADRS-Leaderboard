#!/usr/bin/env bash
set -euo pipefail

# Downloads datasets for llm_sql problem from GitHub if not present locally

PROBLEM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Go up from problem_dir to problems/, then to repo root
BASE_DIR=$(cd "$PROBLEM_DIR/../../.." && pwd)
DATASETS_DIR="$BASE_DIR/datasets/llm_sql"
LOCAL_DATASETS_DIR="$PROBLEM_DIR/resources/datasets"

# GitHub repository URL for raw files
GITHUB_BASE_URL="https://raw.githubusercontent.com/UCB-ADRS/ADRS/main/openevolve/examples/ADRS/llm_sql/datasets"

# List of datasets to download
DATASETS=("movies.csv" "beer.csv" "BIRD.csv" "PDMX.csv" "products.csv")

mkdir -p "$DATASETS_DIR"
mkdir -p "$LOCAL_DATASETS_DIR"

echo "[llm_sql download] Checking for datasets..."

# Check if datasets already exist in the main datasets folder
if [[ -d "$DATASETS_DIR" ]] && [[ -n $(ls -A "$DATASETS_DIR" 2>/dev/null) ]]; then
  # Verify all required datasets are present
  all_present=true
  for dataset in "${DATASETS[@]}"; do
    if [[ ! -f "$DATASETS_DIR/$dataset" ]]; then
      all_present=false
      break
    fi
  done
  
  if [[ "$all_present" == true ]]; then
    echo "[llm_sql download] All datasets already exist at $DATASETS_DIR"
    # Also copy to local resources/datasets for immediate use
    if [[ ! -d "$LOCAL_DATASETS_DIR" ]] || [[ -z $(ls -A "$LOCAL_DATASETS_DIR" 2>/dev/null) ]]; then
      echo "[llm_sql download] Copying datasets to local resources/datasets for immediate use..."
      mkdir -p "$LOCAL_DATASETS_DIR"
      cp -f "$DATASETS_DIR"/*.csv "$LOCAL_DATASETS_DIR/" 2>/dev/null || true
    fi
    exit 0
  fi
fi

# First, try to copy from local resources if they exist
if [[ -d "$LOCAL_DATASETS_DIR" ]] && [[ -n $(ls -A "$LOCAL_DATASETS_DIR" 2>/dev/null) ]]; then
  echo "[llm_sql download] Copying datasets from local resources..."
  cp -r "$LOCAL_DATASETS_DIR"/* "$DATASETS_DIR/" 2>/dev/null || true
  
  # Verify all datasets are present after copy
  all_present=true
  for dataset in "${DATASETS[@]}"; do
    if [[ ! -f "$DATASETS_DIR/$dataset" ]]; then
      all_present=false
      break
    fi
  done
  
  if [[ "$all_present" == true ]]; then
    echo "[llm_sql download] Datasets copied successfully from local resources"
    exit 0
  fi
fi

# If local copy failed or incomplete, download from GitHub
echo "[llm_sql download] Downloading datasets from GitHub..."
echo "[llm_sql download] Source: $GITHUB_BASE_URL"

# Check if curl or wget is available
if command -v curl >/dev/null 2>&1; then
  DOWNLOAD_CMD="curl"
elif command -v wget >/dev/null 2>&1; then
  DOWNLOAD_CMD="wget"
else
  echo "Error: Neither curl nor wget is available. Please install one of them." >&2
  exit 1
fi

# Download each dataset
for dataset in "${DATASETS[@]}"; do
  dataset_url="${GITHUB_BASE_URL}/${dataset}"
  output_path="$DATASETS_DIR/$dataset"
  
  if [[ -f "$output_path" ]]; then
    echo "[llm_sql download] $dataset already exists, skipping..."
    continue
  fi
  
  echo "[llm_sql download] Downloading $dataset..."
  
  if [[ "$DOWNLOAD_CMD" == "curl" ]]; then
    if curl -L -f -o "$output_path" "$dataset_url" 2>/dev/null; then
      echo "[llm_sql download] Successfully downloaded $dataset"
    else
      echo "[llm_sql download] ERROR: Failed to download $dataset from $dataset_url" >&2
      exit 1
    fi
  elif [[ "$DOWNLOAD_CMD" == "wget" ]]; then
    if wget -O "$output_path" "$dataset_url" 2>/dev/null; then
      echo "[llm_sql download] Successfully downloaded $dataset"
    else
      echo "[llm_sql download] ERROR: Failed to download $dataset from $dataset_url" >&2
      exit 1
    fi
  fi
done

# Verify all datasets were downloaded
echo "[llm_sql download] Verifying downloaded datasets..."
all_present=true
for dataset in "${DATASETS[@]}"; do
  if [[ ! -f "$DATASETS_DIR/$dataset" ]]; then
    echo "[llm_sql download] ERROR: $dataset is missing!" >&2
    all_present=false
  else
    size=$(stat -f%z "$DATASETS_DIR/$dataset" 2>/dev/null || stat -c%s "$DATASETS_DIR/$dataset" 2>/dev/null || echo "0")
    echo "[llm_sql download]   ✓ $dataset ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo "${size} bytes"))"
  fi
done

if [[ "$all_present" != true ]]; then
  echo "[llm_sql download] ERROR: Some datasets are missing!" >&2
  exit 1
fi

# Also copy to local resources/datasets for immediate use
echo "[llm_sql download] Copying datasets to local resources/datasets for immediate use..."
mkdir -p "$LOCAL_DATASETS_DIR"
cp -f "$DATASETS_DIR"/*.csv "$LOCAL_DATASETS_DIR/" 2>/dev/null || true

echo "[llm_sql download] All datasets ready at $DATASETS_DIR"
echo "[llm_sql download] Datasets also available at $LOCAL_DATASETS_DIR"

