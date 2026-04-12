#!/usr/bin/env bash

set -e

CONFIG_PATH=${1:-configs/inference.yaml}
INPUT_TEXT=${2:-"I lost my card"}

python scripts/inference.py --config "${CONFIG_PATH}" --text "${INPUT_TEXT}"