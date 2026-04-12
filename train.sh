#!/usr/bin/env bash

set -e

CONFIG_PATH=${1:-configs/train.yaml}

echo "======================================"
echo "Start training with config: ${CONFIG_PATH}"
echo "Current directory: $(pwd)"
echo "Python: $(which python)"
echo "======================================"

if command -v nvidia-smi &> /dev/null; then
    echo "GPU info:"
    nvidia-smi
    echo "======================================"
fi

python scripts/train.py --config "${CONFIG_PATH}"

echo "======================================"
echo "Training finished successfully."
echo "======================================"