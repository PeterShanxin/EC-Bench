#!/bin/bash
set -euo pipefail

STAGE="${1:-full}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ECBENCH_ROOT="${ECBENCH_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
FYP_ROOT="${FYP_ROOT:-/home/svu/e0969321/FYP-fewshotlearn}"
PROTOEC_SCRATCH_ROOT="${PROTOEC_SCRATCH_ROOT:-/scratch/e0969321/ecbench_hopper_h100_task1_runtime_20260313/protoec}"
PROTOEC_CONFIG="${PROTOEC_CONFIG:-configs/config.ecbench.yaml}"
PROTOEC_PROTOCOL="${PROTOEC_PROTOCOL:-A}"
PROTOEC_THRESHOLDS="${PROTOEC_THRESHOLDS:-100}"

exec python "${FYP_ROOT}/scripts/protoec_ecbench_adapter.py" \
  --stage "${STAGE}" \
  --protocol "${PROTOEC_PROTOCOL}" \
  --ecbench-root "${ECBENCH_ROOT}" \
  --fyp-root "${FYP_ROOT}" \
  --scratch-root "${PROTOEC_SCRATCH_ROOT}" \
  --config "${PROTOEC_CONFIG}" \
  --thresholds "${PROTOEC_THRESHOLDS}" \
  "$@"
