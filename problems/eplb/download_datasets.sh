#!/usr/bin/env bash
set -euo pipefail

# Downloads datasets for eplb problem from HuggingFace if not present locally

PROBLEM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Go up from problem_dir to problems/, then to repo root
BASE_DIR=$(cd "$PROBLEM_DIR/../.." && pwd)
DATASETS_DIR="$BASE_DIR/datasets/eplb"

mkdir -p "$DATASETS_DIR"

echo "[eplb download] Checking for datasets..."

# Check if dataset already exists
if [[ -f "$DATASETS_DIR/expert-load.json" ]]; then
  echo "[eplb download] Dataset already exists at $DATASETS_DIR/expert-load.json"
  exit 0
fi

# Check if curl or wget is available
if command -v curl >/dev/null 2>&1; then
  DOWNLOAD_CMD="curl"
elif command -v wget >/dev/null 2>&1; then
  DOWNLOAD_CMD="wget"
else
  echo "Error: Neither curl nor wget is available. Please install one of them." >&2
  exit 1
fi

# Download dataset
dataset_url="https://huggingface.co/datasets/abmfy/eplb-openevolve/resolve/main/expert-load.json"
output_path="$DATASETS_DIR/expert-load.json"

echo "[eplb download] Downloading expert-load.json..."
echo "[eplb download] Source: $dataset_url"

if [[ "$DOWNLOAD_CMD" == "curl" ]]; then
  if curl -L -f -o "$output_path" "$dataset_url" 2>/dev/null; then
    echo "[eplb download] Successfully downloaded expert-load.json"
  else
    echo "[eplb download] ERROR: Failed to download dataset from $dataset_url" >&2
    rm -f "$output_path"
    exit 1
  fi
elif [[ "$DOWNLOAD_CMD" == "wget" ]]; then
  if wget -O "$output_path" "$dataset_url" 2>/dev/null; then
    echo "[eplb download] Successfully downloaded expert-load.json"
  else
    echo "[eplb download] ERROR: Failed to download dataset from $dataset_url" >&2
    rm -f "$output_path"
    exit 1
  fi
fi

# Verify download
echo "[eplb download] Verifying downloaded datasets..."
if [[ -f "$output_path" ]]; then
  size=$(stat -f%z "$output_path" 2>/dev/null || stat -c%s "$output_path" 2>/dev/null || echo "0")
  echo "[eplb download]   ✓ expert-load.json ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo "${size} bytes"))"
else
  echo "[eplb download] ERROR: Dataset file not found after download!" >&2
  exit 1
fi

echo "[eplb download] All datasets ready at $DATASETS_DIR"

