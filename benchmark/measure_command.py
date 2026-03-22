from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Set

from benchmark._common import ensure_dir, write_json


def _run_capture(cmd: Sequence[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return ""
    return proc.stdout.strip()


def _build_process_tree(root_pid: int) -> Set[int]:
    listing = _run_capture(["ps", "-e", "-o", "pid=", "-o", "ppid="])
    if not listing:
        return {root_pid}
    parent_to_children: Dict[int, List[int]] = {}
    for line in listing.splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        parent_to_children.setdefault(ppid, []).append(pid)

    out: Set[int] = set()
    queue = [root_pid]
    while queue:
        pid = queue.pop()
        if pid in out:
            continue
        out.add(pid)
        queue.extend(parent_to_children.get(pid, []))
    return out


def _read_rss_kib(pid: int) -> int:
    status_path = Path("/proc") / str(pid) / "status"
    if not status_path.exists():
        return 0
    try:
        for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
    except OSError:
        return 0
    return 0


def _sample_tree_rss_gib(root_pid: int) -> float:
    total_kib = 0
    for pid in _build_process_tree(root_pid):
        total_kib += _read_rss_kib(pid)
    return float(total_kib) / (1024.0 * 1024.0)


def _gpu_probe() -> List[Dict[str, str]]:
    raw = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,cuda_version",
            "--format=csv,noheader",
        ]
    )
    out: List[Dict[str, str]] = []
    if not raw:
        return out
    for line in raw.splitlines():
        parts = [chunk.strip() for chunk in line.split(",")]
        if len(parts) < 4:
            continue
        out.append(
            {
                "name": parts[0],
                "memory_total": parts[1],
                "driver_version": parts[2],
                "cuda_version": parts[3],
            }
        )
    return out


def _sample_gpu_mem_gib(root_pid: int) -> float:
    raw = _run_capture(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if not raw:
        return 0.0
    tree = _build_process_tree(root_pid)
    total_mib = 0.0
    for line in raw.splitlines():
        parts = [chunk.strip() for chunk in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            used_mib = float(parts[1])
        except ValueError:
            continue
        if pid in tree:
            total_mib += used_mib
    return total_mib / 1024.0


def _env_from_assignments(items: Sequence[str]) -> Dict[str, str]:
    env = os.environ.copy()
    for item in items:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        env[key] = value
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure wall-clock and memory for a benchmark command.")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--phase", required=True, choices=["setup", "train", "test", "full"])
    ap.add_argument("--protocol", default="A")
    ap.add_argument("--cwd", type=Path, default=Path.cwd())
    ap.add_argument("--json-out", type=Path, required=True)
    ap.add_argument("--log-file", type=Path, required=True)
    ap.add_argument("--query-count", type=int, default=0)
    ap.add_argument("--training-unit", default="")
    ap.add_argument("--unit-count", type=int, default=0)
    ap.add_argument("--sample-seconds", type=float, default=1.0)
    ap.add_argument("--env", action="append", default=[], help="Additional KEY=VALUE env vars.")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    if args.command and args.command[0] == "--":
        command = args.command[1:]
    else:
        command = list(args.command)
    if not command:
        raise SystemExit("measure_command.py requires a command after '--'")

    ensure_dir(args.json_out.parent)
    ensure_dir(args.log_file.parent)

    env = _env_from_assignments(args.env)
    started_utc = datetime.now(timezone.utc).isoformat()
    gpu_probe = _gpu_probe()

    with args.log_file.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"[measure] start_utc={started_utc}\n")
        log_handle.write(f"[measure] cwd={args.cwd}\n")
        log_handle.write(f"[measure] command={' '.join(shlex.quote(part) for part in command)}\n")
        log_handle.flush()

        proc = subprocess.Popen(
            command,
            cwd=str(args.cwd),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            text=True,
        )

        start = time.time()
        max_rss_gib = 0.0
        max_gpu_mem_gib = 0.0
        try:
            while True:
                rc = proc.poll()
                rss_gib = _sample_tree_rss_gib(proc.pid)
                gpu_mem_gib = _sample_gpu_mem_gib(proc.pid)
                if rss_gib > max_rss_gib:
                    max_rss_gib = rss_gib
                if gpu_mem_gib > max_gpu_mem_gib:
                    max_gpu_mem_gib = gpu_mem_gib
                if rc is not None:
                    break
                time.sleep(max(0.2, float(args.sample_seconds)))
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGTERM)
            raise

    end = time.time()
    finished_utc = datetime.now(timezone.utc).isoformat()
    duration_s = float(end - start)
    exit_code = int(proc.returncode or 0)

    memory_usage_gib = max_gpu_mem_gib if max_gpu_mem_gib > 0 else max_rss_gib
    memory_kind = "gpu" if max_gpu_mem_gib > 0 else "cpu_rss"
    per_protein_latency_ms = None
    if args.query_count and args.query_count > 0:
        per_protein_latency_ms = (duration_s * 1000.0) / float(args.query_count)

    run_time_per_epoch_s = None
    run_time_per_episode_s = None
    train_flops_total = None
    train_flops_per_epoch_or_episode = None
    if args.phase == "train" and args.unit_count > 0:
        if args.training_unit == "epoch":
            run_time_per_epoch_s = duration_s / float(args.unit_count)
            train_flops_per_epoch_or_episode = None
        elif args.training_unit == "episode":
            run_time_per_episode_s = duration_s / float(args.unit_count)

    payload = {
        "model_id": args.model_id,
        "phase": args.phase,
        "protocol": args.protocol,
        "cwd": str(args.cwd),
        "command": command,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "duration_s": duration_s,
        "exit_code": exit_code,
        "hostname": socket.gethostname(),
        "query_count": args.query_count,
        "per_protein_latency_ms": per_protein_latency_ms,
        "memory_usage_gib": memory_usage_gib,
        "memory_kind": memory_kind,
        "max_rss_gib": max_rss_gib,
        "max_gpu_mem_gib": max_gpu_mem_gib,
        "gpu_probe": gpu_probe,
        "training_unit": args.training_unit,
        "unit_count": args.unit_count,
        "run_time_per_epoch_s": run_time_per_epoch_s,
        "run_time_per_episode_s": run_time_per_episode_s,
        "train_flops_total": train_flops_total,
        "train_flops_per_epoch_or_episode": train_flops_per_epoch_or_episode,
        "test_runtime_s": duration_s if args.phase == "test" else None,
        "total_training_runtime_s": duration_s if args.phase == "train" else None,
        "container_image": os.environ.get("BENCHMARK_CONTAINER_IMAGE", ""),
        "pbs_jobid": os.environ.get("PBS_JOBID", ""),
        "pbs_queue": os.environ.get("PBS_QUEUE", ""),
    }
    write_json(args.json_out, payload)


if __name__ == "__main__":
    main()
