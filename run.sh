#!/bin/bash
# Helper script to run the winner's solution without Docker
# Usage: ./run.sh <command> [--debug]
# Example: ./run.sh create-candidates --debug
# Example: ./run.sh create-features --debug
# Example: ./run.sh create-datasets --debug

# Change to script directory
cd "$(dirname "$0")"

# Set PYTHONPATH to project root
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

# Run invoke with all arguments
python -m invoke "$@"
