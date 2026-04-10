#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  submit_clean_id30_supconH_chain.sh --model-name NAME --train-epochs N [options]

Queue-safe chained submission helper for the resume-safe CLEAN ID30 SupCon-Hard
wrapper. This does NOT run chunks in parallel; it submits a dependency chain so
each chunk starts only after the previous one finishes.

Required:
  --model-name NAME            Shared CLEAN model name/state prefix
  --train-epochs N             Global target epoch count (for example 1875)

Options:
  --training-data NAME         EC-Bench training split name (default: train_ec_30)
  --epochs-per-chunk N         Max epochs per PBS job (default: 90)
  --job-name-prefix NAME       PBS job-name prefix (default: cleanch)
  --measurement-prefix NAME    Measurement basename prefix (default: MODEL_NAME)
  --walltime HH:MM:SS          Walltime for each chunk (default: 24:00:00)
  --scratch-root PATH          Shared scratch root (default: /scratch/e0969321/ecbench_clean_id30_retrain_20260325)
  --expected-gpu-model MODEL   GPU model check (default: H200)
  --stage-local-ssd 0|1        Stage esm_data to /local_ssd if available (default: 1)
  --esm-cache-size N           CLEAN embedding cache size (default: 300000)
  --seed N                     CLEAN training seed (default: 1234)
  --node-local-cache-mode MODE
                               Node-local cache mode forwarded to the wrapper
                               (default: shared)
  --depend-mode MODE           PBS dependency mode (default: afterany)
  --start-chunk N              First chunk index to submit (default: 1)
  --initial-dependency JOBID   Optional dependency for the first submitted chunk
  --wrapper PATH               Wrapper PBS script
  --manifest-out PATH          Optional TSV manifest of planned/submitted chunks
  --pbs-log-dir PATH           Optional per-chunk PBS outer-log directory
  --submit                     Actually submit jobs (default is dry-run)
  --dry-run                    Print planned qsub commands only

Examples:
  Dry-run a full 1875-epoch chain:
    submit_clean_id30_supconH_chain.sh \
      --model-name train_ec_30_supconH_full \
      --train-epochs 1875

  Submit the chain:
    submit_clean_id30_supconH_chain.sh \
      --model-name train_ec_30_supconH_full \
      --train-epochs 1875 \
      --submit
EOF
}

MODEL_NAME=""
TRAIN_EPOCHS=""
TRAINING_DATA="train_ec_30"
EPOCHS_PER_CHUNK=90
JOB_NAME_PREFIX="cleanch"
MEASUREMENT_PREFIX=""
WALLTIME="24:00:00"
SCRATCH_ROOT="/scratch/e0969321/ecbench_clean_id30_retrain_20260325"
EXPECTED_GPU_MODEL="H200"
STAGE_LOCAL_SSD="1"
ESM_CACHE_SIZE="300000"
SEED="1234"
NODE_LOCAL_CACHE_MODE="shared"
DEPEND_MODE="afterany"
START_CHUNK=1
INITIAL_DEPENDENCY=""
WRAPPER="/home/svu/e0969321/EC-Bench/benchmark/run_clean_id30_supconH_train.pbs"
MANIFEST_OUT=""
PBS_LOG_DIR=""
DO_SUBMIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --train-epochs)
      TRAIN_EPOCHS="$2"
      shift 2
      ;;
    --epochs-per-chunk)
      EPOCHS_PER_CHUNK="$2"
      shift 2
      ;;
    --training-data)
      TRAINING_DATA="$2"
      shift 2
      ;;
    --job-name-prefix)
      JOB_NAME_PREFIX="$2"
      shift 2
      ;;
    --measurement-prefix)
      MEASUREMENT_PREFIX="$2"
      shift 2
      ;;
    --walltime)
      WALLTIME="$2"
      shift 2
      ;;
    --scratch-root)
      SCRATCH_ROOT="$2"
      shift 2
      ;;
    --expected-gpu-model)
      EXPECTED_GPU_MODEL="$2"
      shift 2
      ;;
    --stage-local-ssd)
      STAGE_LOCAL_SSD="$2"
      shift 2
      ;;
    --esm-cache-size)
      ESM_CACHE_SIZE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --node-local-cache-mode)
      NODE_LOCAL_CACHE_MODE="$2"
      shift 2
      ;;
    --depend-mode)
      DEPEND_MODE="$2"
      shift 2
      ;;
    --start-chunk)
      START_CHUNK="$2"
      shift 2
      ;;
    --initial-dependency)
      INITIAL_DEPENDENCY="$2"
      shift 2
      ;;
    --wrapper)
      WRAPPER="$2"
      shift 2
      ;;
    --manifest-out)
      MANIFEST_OUT="$2"
      shift 2
      ;;
    --pbs-log-dir)
      PBS_LOG_DIR="$2"
      shift 2
      ;;
    --submit)
      DO_SUBMIT=1
      shift
      ;;
    --dry-run)
      DO_SUBMIT=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[submit_clean_chain][error] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${MODEL_NAME}" || -z "${TRAIN_EPOCHS}" ]]; then
  echo "[submit_clean_chain][error] --model-name and --train-epochs are required" >&2
  usage >&2
  exit 2
fi

if ! [[ "${TRAIN_EPOCHS}" =~ ^[0-9]+$ && "${EPOCHS_PER_CHUNK}" =~ ^[0-9]+$ && "${START_CHUNK}" =~ ^[0-9]+$ ]]; then
  echo "[submit_clean_chain][error] epoch values must be integers" >&2
  exit 2
fi

if [[ "${EPOCHS_PER_CHUNK}" -le 0 || "${START_CHUNK}" -le 0 ]]; then
  echo "[submit_clean_chain][error] --epochs-per-chunk and --start-chunk must be > 0" >&2
  exit 2
fi

if [[ -z "${MEASUREMENT_PREFIX}" ]]; then
  MEASUREMENT_PREFIX="${MODEL_NAME}"
fi

if [[ ! -f "${WRAPPER}" ]]; then
  echo "[submit_clean_chain][error] wrapper not found: ${WRAPPER}" >&2
  exit 2
fi

TOTAL_CHUNKS=$(( (TRAIN_EPOCHS + EPOCHS_PER_CHUNK - 1) / EPOCHS_PER_CHUNK ))
if [[ "${START_CHUNK}" -gt "${TOTAL_CHUNKS}" ]]; then
  echo "[submit_clean_chain][error] --start-chunk (${START_CHUNK}) exceeds total_chunks (${TOTAL_CHUNKS})" >&2
  exit 2
fi

PREV_JOB_ID="${INITIAL_DEPENDENCY}"

if [[ -n "${MANIFEST_OUT}" ]]; then
  mkdir -p "$(dirname "${MANIFEST_OUT}")"
  printf "chunk_index\tjob_name\tmeasurement_basename\tdependency\tjob_id_or_status\n" > "${MANIFEST_OUT}"
fi

if [[ -n "${PBS_LOG_DIR}" ]]; then
  mkdir -p "${PBS_LOG_DIR}"
fi

echo "[submit_clean_chain] mode=$([[ "${DO_SUBMIT}" -eq 1 ]] && echo submit || echo dry-run)"
echo "[submit_clean_chain] model_name=${MODEL_NAME}"
echo "[submit_clean_chain] train_epochs=${TRAIN_EPOCHS}"
echo "[submit_clean_chain] training_data=${TRAINING_DATA}"
echo "[submit_clean_chain] epochs_per_chunk=${EPOCHS_PER_CHUNK}"
echo "[submit_clean_chain] total_chunks=${TOTAL_CHUNKS}"
echo "[submit_clean_chain] start_chunk=${START_CHUNK}"
echo "[submit_clean_chain] initial_dependency=${INITIAL_DEPENDENCY:-none}"
echo "[submit_clean_chain] node_local_cache_mode=${NODE_LOCAL_CACHE_MODE}"
echo "[submit_clean_chain] depend_mode=${DEPEND_MODE}"

for (( chunk=START_CHUNK; chunk<=TOTAL_CHUNKS; chunk++ )); do
  chunk_label=$(printf "chunk%03d" "${chunk}")
  job_name=$(printf "%s%03d" "${JOB_NAME_PREFIX}" "${chunk}")
  measurement_basename="${MEASUREMENT_PREFIX}_${chunk_label}"
  dependency_id="${PREV_JOB_ID:-}"

  vlist="MODEL_NAME=${MODEL_NAME},TRAINING_DATA=${TRAINING_DATA},TRAIN_EPOCHS=${TRAIN_EPOCHS},MAX_EPOCHS_THIS_RUN=${EPOCHS_PER_CHUNK},MEASUREMENT_BASENAME=${measurement_basename},SCRATCH_ROOT=${SCRATCH_ROOT},EXPECTED_GPU_MODEL=${EXPECTED_GPU_MODEL},STAGE_ESM_TO_LOCAL_SSD=${STAGE_LOCAL_SSD},CLEAN_ESM_CACHE_SIZE=${ESM_CACHE_SIZE},CLEAN_SEED=${SEED},NODE_LOCAL_CACHE_MODE=${NODE_LOCAL_CACHE_MODE}"
  qsub_args=(qsub -N "${job_name}" -l "walltime=${WALLTIME}")
  if [[ -n "${dependency_id}" ]]; then
    qsub_args+=(-W "depend=${DEPEND_MODE}:${dependency_id}")
  fi
  if [[ -n "${PBS_LOG_DIR}" ]]; then
    qsub_args+=(-o "${PBS_LOG_DIR}/${measurement_basename}.pbs.log")
  fi
  qsub_args+=(-v "${vlist}" "${WRAPPER}")

  if [[ "${DO_SUBMIT}" -eq 1 ]]; then
    job_id="$("${qsub_args[@]}")"
    echo "[submit_clean_chain] submitted ${chunk_label} -> ${job_id} (job_name=${job_name}, measurement=${measurement_basename}, dependency=${dependency_id:-none})"
    PREV_JOB_ID="${job_id}"
    status_field="${job_id}"
  else
    echo "[submit_clean_chain][dry-run] ${qsub_args[*]}"
    status_field="DRY_RUN"
    PREV_JOB_ID="JOBID_${chunk_label}"
  fi

  if [[ -n "${MANIFEST_OUT}" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\n" \
      "${chunk}" \
      "${job_name}" \
      "${measurement_basename}" \
      "${dependency_id:-none}" \
      "${status_field}" >> "${MANIFEST_OUT}"
  fi
done
