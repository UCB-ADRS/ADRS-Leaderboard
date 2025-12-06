#!/usr/bin/env bash
set -euo pipefail

# Downloads external OpenEvolve MAST dataset for multiagent_system problem.
# Clones https://github.com/mert-cemri/openevolve-mast into resources/openevolve-mast
# Idempotent: if repo already exists, pulls latest.

PROBLEM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESOURCES_DIR="$PROBLEM_DIR/resources"
REPO_DIR="$RESOURCES_DIR/openevolve-mast"
REPO_URL="https://github.com/mert-cemri/openevolve-mast"
DATASETS_DIR="$RESOURCES_DIR/datasets"

mkdir -p "$RESOURCES_DIR"

echo "[multiagent_system download] Using resources directory: $RESOURCES_DIR" >&2

# Check if repository directory already exists (with or without .git)
if [[ -d "$REPO_DIR" ]]; then
  if [[ -d "$REPO_DIR/.git" ]]; then
    echo "[multiagent_system download] Repository already exists, attempting git pull..." >&2
    (cd "$REPO_DIR" && git pull --ff-only || echo "[multiagent_system download] WARNING: git pull failed; keeping existing copy" >&2)
  else
    echo "[multiagent_system download] Directory $REPO_DIR already exists (not a git repo), skipping clone" >&2
  fi
else
  echo "[multiagent_system download] Cloning repository $REPO_URL ..." >&2
  git clone --depth 1 "$REPO_URL" "$REPO_DIR" || {
    echo "Error: Failed to clone $REPO_URL" >&2
    exit 1
  }
fi

PROGRAMDEV_DIR="$REPO_DIR/example_mas/programdev"
if [[ -d "$PROGRAMDEV_DIR" ]]; then
  echo "[multiagent_system download] programdev dataset ready at $PROGRAMDEV_DIR" >&2
else
  echo "Error: programdev directory missing at $PROGRAMDEV_DIR" >&2
  exit 1
fi

# Copy datasets to resources/datasets for immediate use
if [[ -d "$PROGRAMDEV_DIR" ]] && [[ -n $(ls -A "$PROGRAMDEV_DIR" 2>/dev/null) ]]; then
  echo "[multiagent_system download] Copying datasets to resources/datasets for immediate use..." >&2
  mkdir -p "$DATASETS_DIR"
  cp -rf "$PROGRAMDEV_DIR"/* "$DATASETS_DIR/" 2>/dev/null || true
  echo "[multiagent_system download] Datasets available at $DATASETS_DIR" >&2
fi

echo "[multiagent_system download] Done" >&2
