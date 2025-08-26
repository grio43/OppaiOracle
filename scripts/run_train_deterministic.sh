#!/bin/bash
# Script to run training with deterministic settings for reproducible results
# This ensures CUBLAS operations are deterministic across runs

# Set CUBLAS workspace configuration for deterministic matrix multiplication
# :4096:8 allocates 4096 workspaces of 8 bytes each
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Optional: Set other deterministic environment variables
# Disable hash randomization in Python
export PYTHONHASHSEED=0

# Optional: Force single-threaded operations for complete determinism
# Uncomment if you need exact reproducibility (will be slower)
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1

# Color output for better visibility
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running training with deterministic settings...${NC}"
echo -e "${BLUE}CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}${NC}"
echo -e "${BLUE}PYTHONHASHSEED=${PYTHONHASHSEED}${NC}"

# Check if Python script exists
if [ ! -f "train_direct.py" ]; then
    echo "Error: train_direct.py not found in current directory"
    exit 1
fi

# Run the training script with all arguments passed to this script
python train_direct.py "$@"
