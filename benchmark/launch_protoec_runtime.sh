#!/bin/bash
set -euo pipefail

ECBENCH_ROOT="${ECBENCH_ROOT:-$(pwd)}"
FYP_ROOT="${FYP_ROOT:-/home/svu/e0969321/FYP-fewshotlearn}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/e0969321/ecbench_hopper_h100_task1_runtime_20260313}"
OUT_ROOT="${OUT_ROOT:-/home/svu/e0969321/FYP-fewshotlearn/results/ecbench_hopper_h100_task1_runtime_20260313}"
PREPARED_ECBENCH_ROOT="${PREPARED_ECBENCH_ROOT:-${SCRATCH_ROOT}/data_prep/ecbench_worktree}"
DATA_PREP_MANIFEST="${DATA_PREP_MANIFEST:-${SCRATCH_ROOT}/data_prep/data_prep_manifest.json}"
PROTOEC_CONFIG="${PROTOEC_CONFIG:-configs/config.ecbench.yaml}"
PROTOEC_THRESHOLDS="${PROTOEC_THRESHOLDS:-100}"
BENCHMARK_CONTAINER_IMAGE="${BENCHMARK_CONTAINER_IMAGE:-}"
BENCHMARK_REQUIRE_H100="${BENCHMARK_REQUIRE_H100:-0}"
BENCHMARK_REQUIRE_GPU_MODEL="${BENCHMARK_REQUIRE_GPU_MODEL:-}"

if [ ! -d "${PREPARED_ECBENCH_ROOT}" ]; then
  echo "[launch_protoec_runtime][error] prepared EC-Bench root not found: ${PREPARED_ECBENCH_ROOT}" >&2
  exit 1
fi

export PYTHONPATH="${ECBENCH_ROOT}:${FYP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export ECBENCH_ROOT="${PREPARED_ECBENCH_ROOT}"
export PROTOEC_SCRATCH_ROOT="${SCRATCH_ROOT}/protoec"
export PROTOEC_CONFIG
export PROTOEC_THRESHOLDS
export FYP_ROOT
export BENCHMARK_CONTAINER_IMAGE

MEASUREMENTS_DIR="${SCRATCH_ROOT}/measurements"
LOGS_DIR="${SCRATCH_ROOT}/logs"
mkdir -p "${MEASUREMENTS_DIR}" "${LOGS_DIR}"

PRECHECK_ARGS=(--out "${SCRATCH_ROOT}/hardware_preflight.json")
if [ "${BENCHMARK_REQUIRE_H100}" = "1" ]; then
  PRECHECK_ARGS+=(--require-h100)
fi
if [ -n "${BENCHMARK_REQUIRE_GPU_MODEL}" ]; then
  PRECHECK_ARGS+=(--require-gpu-model "${BENCHMARK_REQUIRE_GPU_MODEL}")
fi
python -m benchmark.preflight_hopper "${PRECHECK_ARGS[@]}"

bash "${ECBENCH_ROOT}/ProtoEC/run_model.sh" prepare --force-prepare

read -r QUERY_COUNT EPISODE_COUNT <<EOF
$(python - <<'PY'
import json
import os
from pathlib import Path
import yaml

cfg_path = Path(os.environ["FYP_ROOT"]) / os.environ["PROTOEC_CONFIG"]
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
thresholds = [int(item.strip()) for item in os.environ["PROTOEC_THRESHOLDS"].split(",") if item.strip()]
episodes_train = int((cfg.get("episodes") or {}).get("train", 0) or 0)
query_total = 0
scratch_root = Path(os.environ["SCRATCH_ROOT"])
for threshold in thresholds:
    split_path = scratch_root / "protoec" / "mirror" / "splits" / f"id{threshold}" / "fold-1" / "test.jsonl"
    if not split_path.exists():
        continue
    with split_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            query_total += len(payload.get("accessions", []))
print(query_total, episodes_train * len(thresholds))
PY
)
EOF

python -m benchmark.measure_command \
  --model-id protoec \
  --phase full \
  --protocol A \
  --cwd "${ECBENCH_ROOT}" \
  --json-out "${MEASUREMENTS_DIR}/protoec_protocolA_full.json" \
  --log-file "${LOGS_DIR}/protoec_protocolA_full.log" \
  --query-count "${QUERY_COUNT}" \
  -- \
  bash "${ECBENCH_ROOT}/ProtoEC/run_model.sh" full --protocol A --force-prepare --force-embed --force-train --force-test

python -m benchmark.measure_command \
  --model-id protoec \
  --phase train \
  --protocol B \
  --cwd "${ECBENCH_ROOT}" \
  --json-out "${MEASUREMENTS_DIR}/protoec_protocolB_train.json" \
  --log-file "${LOGS_DIR}/protoec_protocolB_train.log" \
  --training-unit episode \
  --unit-count "${EPISODE_COUNT}" \
  -- \
  bash "${ECBENCH_ROOT}/ProtoEC/run_model.sh" train --protocol B --force-train

python -m benchmark.measure_command \
  --model-id protoec \
  --phase test \
  --protocol B \
  --cwd "${ECBENCH_ROOT}" \
  --json-out "${MEASUREMENTS_DIR}/protoec_protocolB_test.json" \
  --log-file "${LOGS_DIR}/protoec_protocolB_test.log" \
  --query-count "${QUERY_COUNT}" \
  -- \
  bash "${ECBENCH_ROOT}/ProtoEC/run_model.sh" test --protocol B --force-test

bash "${ECBENCH_ROOT}/ProtoEC/run_model.sh" manifest

RUN_BENCHMARK_ARGS=(
  --ecbench-root "${PREPARED_ECBENCH_ROOT}"
  --scratch-root "${SCRATCH_ROOT}"
  --out-root "${OUT_ROOT}"
  --protoec-manifest "${SCRATCH_ROOT}/protoec/adapter_manifest.json"
  --data-prep-manifest "${DATA_PREP_MANIFEST}"
)

if [ -n "${CLEAN_BUNDLE_ROOT:-}" ]; then
  RUN_BENCHMARK_ARGS+=(--clean-bundle-root "${CLEAN_BUNDLE_ROOT}")
fi

if [ -n "${BENCHMARK_EXTRA_MEASUREMENT_DIRS:-}" ]; then
  OLD_IFS="${IFS}"
  IFS=:
  read -r -a EXTRA_MEASUREMENT_DIRS <<< "${BENCHMARK_EXTRA_MEASUREMENT_DIRS}"
  IFS="${OLD_IFS}"
  for extra_dir in "${EXTRA_MEASUREMENT_DIRS[@]}"; do
    if [ -n "${extra_dir}" ]; then
      RUN_BENCHMARK_ARGS+=(--measurement-dir "${extra_dir}")
    fi
  done
fi

python -m benchmark.run_benchmark "${RUN_BENCHMARK_ARGS[@]}"
