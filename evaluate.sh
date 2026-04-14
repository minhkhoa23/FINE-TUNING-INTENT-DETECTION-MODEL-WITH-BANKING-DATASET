#!/usr/bin/env bash

set -e

CONFIG_PATH=${1:-configs/inference.yaml}
TEST_PATH=${2:-sample_data/processed/test.csv}
OUTPUT_DIR=${3:-outputs/eval_results}

python scripts/evaluate.py \
  --config "${CONFIG_PATH}" \
  --test_path "${TEST_PATH}" \
  --output_dir "${OUTPUT_DIR}"