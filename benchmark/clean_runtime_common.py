from __future__ import annotations

import os
import socket
import subprocess
from datetime import datetime, timezone
from typing import Dict, List


def _run_capture(cmd: List[str]) -> Dict[str, object]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def resolve_container_image() -> str:
    for key in ("BENCHMARK_CONTAINER_IMAGE", "APPTAINER_CONTAINER", "SINGULARITY_CONTAINER"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def gpu_probe() -> List[Dict[str, str]]:
    probe = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,cuda_version",
            "--format=csv,noheader",
        ]
    )
    rows: List[Dict[str, str]] = []
    if not probe["ok"] or not probe["stdout"]:
        return rows
    for raw_line in str(probe["stdout"]).splitlines():
        parts = [chunk.strip() for chunk in raw_line.split(",")]
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


def torch_probe() -> Dict[str, object]:
    payload: Dict[str, object] = {
        "torch_version": "",
        "torch_cuda_version": "",
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
        "torch_cuda_devices": [],
    }
    try:
        import torch

        payload["torch_version"] = getattr(torch, "__version__", "")
        payload["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", "") or ""
        cuda_available = bool(torch.cuda.is_available())
        payload["torch_cuda_available"] = cuda_available
        if cuda_available:
            count = int(torch.cuda.device_count())
            payload["torch_cuda_device_count"] = count
            payload["torch_cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(count)]
    except Exception as exc:
        payload["torch_probe_error"] = str(exc)
    return payload


def runtime_context() -> Dict[str, object]:
    payload: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": os.environ.get("BENCHMARK_NODE_NAME", "").strip() or socket.gethostname(),
        "pbs_jobid": os.environ.get("PBS_JOBID", ""),
        "pbs_queue": os.environ.get("PBS_QUEUE", ""),
        "pbs_ncpus": os.environ.get("PBS_NCPUS", "") or os.environ.get("NCPUS", ""),
        "gpu_model_hint": os.environ.get("BENCHMARK_GPU_MODEL_HINT", ""),
        "container_image": resolve_container_image(),
        "gpu_probe": gpu_probe(),
    }
    payload.update(torch_probe())
    if not payload["gpu_probe"] and payload["gpu_model_hint"]:
        payload["gpu_probe"] = [{"name": str(payload["gpu_model_hint"])}]
    return payload


def gpu_model_matches(expected_gpu_model: str, rows: List[Dict[str, str]]) -> bool:
    if not expected_gpu_model:
        return True
    expected = expected_gpu_model.strip().upper()
    if not expected:
        return True
    return any(expected in str(row.get("name", "")).upper() for row in rows)
