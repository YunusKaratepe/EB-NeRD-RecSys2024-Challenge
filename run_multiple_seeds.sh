#!/bin/bash
# Run experiment with multiple seeds for significance analysis
# Usage: ./run_multiple_seeds.sh <exp_name> <seed1> <seed2> <seed3>
# Example: ./run_multiple_seeds.sh medium067_001 7 42 123

if [ -z "$1" ]; then
    echo "Usage: ./run_multiple_seeds.sh <exp_name> <seed1> <seed2> <seed3>"
    echo "Example: ./run_multiple_seeds.sh medium067_001 7 42 123"
    exit 1
fi

EXP_NAME="$1"
SEED1="${2:-7}"
SEED2="${3:-42}"
SEED3="${4:-123}"

echo "========================================"
echo "Running $EXP_NAME with seeds: $SEED1, $SEED2, $SEED3"
echo "========================================"

echo ""
echo "[1/3] Running with seed=$SEED1"
./run.sh train --exp="$EXP_NAME" --seed="$SEED1"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed with seed=$SEED1"
    exit 1
fi

echo ""
echo "[2/3] Running with seed=$SEED2"
./run.sh train --exp="$EXP_NAME" --seed="$SEED2"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed with seed=$SEED2"
    exit 1
fi

echo ""
echo "[3/3] Running with seed=$SEED3"
./run.sh train --exp="$EXP_NAME" --seed="$SEED3"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed with seed=$SEED3"
    exit 1
fi

echo ""
echo "========================================"
echo "All seeds completed successfully!"
echo "Check output directories for results with different seeds"
echo "========================================"
