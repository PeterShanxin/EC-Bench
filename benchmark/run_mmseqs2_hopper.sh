#!/bin/bash
set -euo pipefail

DATA_DIR="${1:-data}"
MMSEQS_BIN="${MMSEQS_BIN:-}"
MMSEQS_THREADS="${MMSEQS_THREADS:-${PBS_NCPUS:-12}}"

if [ -z "${MMSEQS_BIN}" ] && command -v mmseqs >/dev/null 2>&1; then
  MMSEQS_BIN="$(command -v mmseqs)"
fi

if [ -z "${MMSEQS_BIN}" ]; then
  source /app1/ebapps/ebenv_hopper.sh >/dev/null 2>&1 || true
  module load MMseqs2 >/dev/null 2>&1 || true
  if command -v mmseqs >/dev/null 2>&1; then
    MMSEQS_BIN="$(command -v mmseqs)"
  fi
fi

if [ -z "${MMSEQS_BIN}" ] && [ -x "/app1/ebapps/arches/flat-avx2/software/MMseqs2/17-b804f-gompi-2024a/bin/mmseqs" ]; then
  MMSEQS_BIN="/app1/ebapps/arches/flat-avx2/software/MMseqs2/17-b804f-gompi-2024a/bin/mmseqs"
fi

if [ -z "${MMSEQS_BIN}" ]; then
  echo "[run_mmseqs2_hopper][error] mmseqs not found on PATH and no fallback binary was found." >&2
  exit 1
fi

case "${MMSEQS_THREADS}" in
  ''|*[!0-9]*)
    MMSEQS_THREADS=12
    ;;
esac
if [ "${MMSEQS_THREADS}" -le 0 ]; then
  MMSEQS_THREADS=12
fi

echo "[run_mmseqs2_hopper] using mmseqs=${MMSEQS_BIN}" >&2
echo "[run_mmseqs2_hopper] using threads=${MMSEQS_THREADS}" >&2

cd "${DATA_DIR}"

capture_outputs() {
  local target_dir=$1
  mkdir -p "${target_dir}"
  if [ -f clusterRes_cluster.tsv ]; then
    cp -f clusterRes_cluster.tsv "${target_dir}/clusterEns_cluster.tsv"
  fi
  if [ -f clusterRes_rep_seq.fasta ]; then
    cp -f clusterRes_rep_seq.fasta "${target_dir}/clusterRes_rep_seq.fasta"
  fi
  if [ -f clusterRes_all_seqs.fasta ]; then
    cp -f clusterRes_all_seqs.fasta "${target_dir}/clusterRes_all_seqs.fasta"
  fi
}

cleanup_clusterres() {
  rm -rf clusterRes clusterRes_* || true
}

for required in pretrain.fasta train.fasta test.fasta price-149.fasta ensemble.fasta; do
  if [ ! -f "${required}" ]; then
    echo "[run_mmseqs2_hopper][error] missing required FASTA: ${DATA_DIR}/${required}" >&2
    exit 1
  fi
done

mkdir -p cluster-100 cluster-90 cluster-70 cluster-50 cluster-30
awk 1 pretrain.fasta train.fasta test.fasta price-149.fasta ensemble.fasta > all.fasta

cleanup_clusterres
"${MMSEQS_BIN}" easy-cluster all.fasta clusterRes cluster-100/ --min-seq-id 1.0 -c 1.0 --cov-mode 1 --threads "${MMSEQS_THREADS}"
capture_outputs cluster-100
python - <<'PY'
from pathlib import Path

src = Path("cluster-100/clusterRes_rep_seq.fasta")
dst = Path("cluster-100/clusterRes_rep_seq.gt11.fasta")
seq_id = None
chunks = []

def flush(handle, current_id, current_chunks):
    if current_id is None:
        return
    seq = "".join(current_chunks)
    if len(seq) >= 11:
        handle.write(f">{current_id}\n")
        for start in range(0, len(seq), 80):
            handle.write(seq[start:start+80] + "\n")

with src.open("r", encoding="utf-8") as in_handle, dst.open("w", encoding="utf-8") as out_handle:
    for raw in in_handle:
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush(out_handle, seq_id, chunks)
            seq_id = line[1:].strip()
            chunks = []
        else:
            chunks.append(line)
    flush(out_handle, seq_id, chunks)
PY
cleanup_clusterres
"${MMSEQS_BIN}" easy-cluster cluster-100/clusterRes_rep_seq.gt11.fasta clusterRes cluster-90/ --min-seq-id 0.9 -c 0.8 --cov-mode 1 --threads "${MMSEQS_THREADS}"
capture_outputs cluster-90
cleanup_clusterres
"${MMSEQS_BIN}" easy-cluster cluster-90/clusterRes_rep_seq.fasta clusterRes cluster-70/ --min-seq-id 0.7 -c 0.8 --cov-mode 1 --threads "${MMSEQS_THREADS}"
capture_outputs cluster-70
cleanup_clusterres
"${MMSEQS_BIN}" easy-cluster cluster-70/clusterRes_rep_seq.fasta clusterRes cluster-50/ --min-seq-id 0.5 -c 0.8 --cov-mode 1 --threads "${MMSEQS_THREADS}"
capture_outputs cluster-50
cleanup_clusterres
"${MMSEQS_BIN}" easy-cluster cluster-50/clusterRes_rep_seq.fasta clusterRes cluster-30/ --min-seq-id 0.3 -c 0.8 --cov-mode 1 --threads "${MMSEQS_THREADS}"
capture_outputs cluster-30
