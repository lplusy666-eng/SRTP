#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/demo.yaml}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
power-econ serve -c "$CONFIG" --host "$HOST" --port "$PORT"
