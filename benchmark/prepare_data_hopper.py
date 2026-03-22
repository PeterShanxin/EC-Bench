from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from benchmark._common import ensure_dir, path_size_mib, write_json


STAGES = (
    "download_data",
    "extract_coordinates",
    "data_preprocessing",
    "run_mmseqs2",
    "create_data",
    "go_creator",
)


def _run(cmd: List[str], *, cwd: Path, log_path: Path) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(cmd)}\n")
        handle.flush()
        subprocess.run(cmd, check=True, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT)


def _sync_worktree(src_root: Path, dst_root: Path) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--delete",
            "--exclude",
            ".git",
            f"{src_root}/",
            str(dst_root),
        ],
        check=True,
    )


def _marker_path(markers_dir: Path, stage: str) -> Path:
    return markers_dir / f"{stage}.done"


def _collect_file_manifest(data_root: Path) -> Dict[str, object]:
    required = {
        "goa_uniprot_all_gaf": data_root / "goa_uniprot_all.gaf",
        "swissprot_coordinates": data_root / "swissprot_coordinates.json",
        "pretrain_csv": data_root / "pretrain.csv",
        "pretrain_go_final": data_root / "pretrain_go_final.csv",
        "train_csv": data_root / "train.csv",
        "test_csv": data_root / "test.csv",
        "cluster_30_train": data_root / "cluster-30" / "train_ec.csv",
        "cluster_30_test": data_root / "cluster-30" / "test_ec.csv",
        "cluster_100_train": data_root / "cluster-100" / "train_ec.csv",
        "cluster_100_test": data_root / "cluster-100" / "test_ec.csv",
        "test_ec_fasta": data_root / "test_ec.fasta",
        "price_149_fasta": data_root / "price-149.fasta",
    }
    files: Dict[str, object] = {}
    for key, path in required.items():
        files[key] = {
            "path": str(path),
            "exists": path.exists(),
            "size_mib": path_size_mib(path),
        }
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the official EC-Bench data-prep flow in scratch on Hopper.")
    ap.add_argument("--ecbench-root", type=Path, default=Path.cwd())
    ap.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("/scratch/e0969321/ecbench_hopper_h100_task1_runtime_20260313"),
    )
    ap.add_argument("--manifest-out", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    ecbench_root = args.ecbench_root.resolve()
    scratch_root = args.scratch_root.resolve()
    work_root = scratch_root / "data_prep"
    worktree_root = work_root / "ecbench_worktree"
    logs_dir = work_root / "logs"
    markers_dir = work_root / "markers"
    ensure_dir(work_root)
    ensure_dir(logs_dir)
    ensure_dir(markers_dir)

    if not args.resume or not worktree_root.exists():
        _sync_worktree(ecbench_root, worktree_root)

    stage_records: List[Dict[str, object]] = []
    stage_commands = {
        "download_data": ["python", "code/download_data.py", "2023", "01", "2018", "02", "2023", "02"],
        "extract_coordinates": ["python", "code/extract_coordinates.py", "data", "swissprot_pdb_v6.tar", "--json_file", "swissprot_coordinates.json"],
        "data_preprocessing": [
            "python",
            "code/data_preprocessing.py",
            "--data_path",
            "data",
            "--pretrain_name",
            "uniprot_trembl2018_02.tsv",
            "--train_name",
            "uniprot_sprot2018_02.tsv",
            "--test_name",
            "uniprot_sprot2023_01.tsv",
            "--price_name",
            "price-149.csv",
            "--ensemble_name",
            "uniprot_sprot2023_02.tsv",
        ],
        "run_mmseqs2": ["bash", "benchmark/run_mmseqs2_hopper.sh", "data"],
        "create_data": [
            "python",
            "code/create_train_test_data.py",
            "--data_path",
            "data",
            "--ensemble_file_name",
            "ensemble.csv",
            "--threshod",
            "30",
        ],
        "go_creator": ["python", "code/go_creator.py", "--data_path", "data"],
    }

    for stage in STAGES:
        marker = _marker_path(markers_dir, stage)
        started = time.time()
        status = "success"
        if args.resume and marker.exists():
            stage_records.append(
                {
                    "stage": stage,
                    "status": "skipped_resume",
                    "duration_s": 0.0,
                    "marker": str(marker),
                }
            )
            continue
        try:
            _run(stage_commands[stage], cwd=worktree_root, log_path=logs_dir / f"{stage}.log")
            if stage == "create_data":
                _run(
                    [
                        "python",
                        "code/create_train_test_data.py",
                        "--data_path",
                        "data",
                        "--ensemble_file_name",
                        "ensemble.csv",
                        "--threshod",
                        "100",
                    ],
                    cwd=worktree_root,
                    log_path=logs_dir / f"{stage}.log",
                )
            marker.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
        except subprocess.CalledProcessError:
            status = "failed"
            raise
        finally:
            stage_records.append(
                {
                    "stage": stage,
                    "status": status,
                    "duration_s": round(time.time() - started, 6),
                    "marker": str(marker),
                    "log_path": str(logs_dir / f"{stage}.log"),
                }
            )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_ecbench_root": str(ecbench_root),
        "prepared_ecbench_root": str(worktree_root),
        "scratch_root": str(scratch_root),
        "stages": stage_records,
        "files": _collect_file_manifest(worktree_root / "data"),
    }
    manifest_out = args.manifest_out.resolve() if args.manifest_out else (work_root / "data_prep_manifest.json")
    write_json(manifest_out, payload)
    print(f"[prepare_data_hopper] wrote manifest to {manifest_out}")


if __name__ == "__main__":
    main()
