#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/demo.yaml}"
power-econ dashboard -c "$CONFIG"
