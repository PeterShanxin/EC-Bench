from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _run(cmd: List[str]) -> Dict[str, object]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
        }


def _parse_gpu_probe() -> List[Dict[str, str]]:
    probe = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,cuda_version",
            "--format=csv,noheader",
        ]
    )
    if not probe["ok"] or not probe["stdout"]:
        return []
    rows: List[Dict[str, str]] = []
    for line in str(probe["stdout"]).splitlines():
        parts = [chunk.strip() for chunk in line.split(",")]
        if len(parts) < 4:
            continue
        rows.append(
            {
                "name": parts[0],
                "memory_total": parts[1],
                "driver_version": parts[2],
                "cuda_version": parts[3],
            }
        )
    return rows


def _container_image() -> str:
    for key in ("BENCHMARK_CONTAINER_IMAGE", "APPTAINER_CONTAINER", "SINGULARITY_CONTAINER"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def _cluster_gpu_summary() -> Dict[str, int]:
    probe = _run(["pbsnodes", "-av"])
    summary = {
        "h100_free": 0,
        "h100_offline": 0,
        "h200_free": 0,
        "h200_offline": 0,
    }
    if not probe["ok"]:
        return summary

    current_model = ""
    current_state = ""
    for raw_line in str(probe["stdout"]).splitlines():
        line = raw_line.strip()
        if line.startswith("state ="):
            current_state = line.split("=", 1)[1].strip()
        elif line.startswith("resources_available.gpu_model ="):
            current_model = line.split("=", 1)[1].strip().upper()
            if current_model == "H100":
                if "free" in current_state:
                    summary["h100_free"] += 1
                elif "offline" in current_state:
                    summary["h100_offline"] += 1
            elif current_model == "H200":
                if "free" in current_state:
                    summary["h200_free"] += 1
                elif "offline" in current_state:
                    summary["h200_offline"] += 1
    return summary


def collect_preflight() -> Dict[str, object]:
    hostname = os.environ.get("BENCHMARK_NODE_NAME", "").strip() or socket.gethostname()
    gpu_rows = _parse_gpu_probe()
    gpu_model_hint = os.environ.get("BENCHMARK_GPU_MODEL_HINT", "").strip()
    if not gpu_rows and gpu_model_hint:
        gpu_rows = [{"name": gpu_model_hint}]
    h100_detected = any("H100" in row.get("name", "").upper() for row in gpu_rows)
    payload: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": hostname,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "user": os.environ.get("USER", ""),
        "pbs_jobid": os.environ.get("PBS_JOBID", ""),
        "pbs_queue": os.environ.get("PBS_QUEUE", ""),
        "pbs_ncpus": os.environ.get("PBS_NCPUS", ""),
        "project_hint": "CFP03-SF-097",
        "container_image": _container_image(),
        "singularity_path": shutil.which("singularity") or "",
        "qsub_path": shutil.which("qsub") or "",
        "python_path": shutil.which("python3") or shutil.which("python") or "",
        "gpu_probe": gpu_rows,
        "cluster_gpu_summary": _cluster_gpu_summary(),
        "h100_detected_on_current_node": h100_detected,
        "gpu_model_hint": gpu_model_hint,
    }
    return payload


def gpu_model_matches(required_gpu_model: str, gpu_rows: List[Dict[str, str]]) -> bool:
    required = str(required_gpu_model or "").strip().upper()
    if not required:
        return True
    for row in gpu_rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "") or "").upper()
        if not name:
            continue
        if required in name:
            return True
        compact = re.sub(r"[^A-Z0-9]+", "", name)
        if required in compact:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect Hopper preflight metadata for the EC-Bench runtime pass.")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    ap.add_argument(
        "--require-h100",
        action="store_true",
        help="Exit non-zero if the current node has visible GPUs and none of them are H100s.",
    )
    ap.add_argument(
        "--require-gpu-model",
        default="",
        help="Exit non-zero if the current node has visible GPUs and none of them match the requested model string.",
    )
    args = ap.parse_args()

    payload = collect_preflight()
    rendered = json.dumps(payload, indent=2, sort_keys=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)

    gpu_rows = payload.get("gpu_probe") or []
    required_gpu_model = str(args.require_gpu_model or "").strip()
    if args.require_h100 and not required_gpu_model:
        required_gpu_model = "H100"
    if required_gpu_model and gpu_rows:
        if not gpu_model_matches(required_gpu_model, [row for row in gpu_rows if isinstance(row, dict)]):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
