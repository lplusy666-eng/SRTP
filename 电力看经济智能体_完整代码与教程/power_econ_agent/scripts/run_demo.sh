#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/demo.yaml}"
power-econ demo -c "$CONFIG"
python scripts/verify_all.py --config "$CONFIG"
