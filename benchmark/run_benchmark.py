from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from benchmark._common import (
    BASE_MODEL_COLUMN_MAP,
    OFFICIAL_FILES,
    bytes_to_mib,
    ensure_dir,
    expand_existing_paths,
    path_size_mib,
    read_csv_rows,
    read_json,
    split_labels,
    task1_metrics_from_rows,
    top1_only,
    write_csv,
    write_json,
)
from benchmark.preflight_hopper import collect_preflight


def _load_manifest(path: Path) -> List[Dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _exit_ok(row: Dict[str, object]) -> bool:
    value = row.get("exit_code")
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return False


def _pbs_gpu_model_for_host(hostname: str) -> str:
    if not hostname:
        return ""
    probe = subprocess.run(["pbsnodes", "-av"], capture_output=True, text=True, check=False)
    if probe.returncode != 0:
        return ""
    short = hostname.split(".", 1)[0]
    want = {hostname, short, f"{short}.cm.cluster"}
    inside = False
    for line in probe.stdout.splitlines():
        stripped = line.strip()
        if not line.startswith(" "):
            inside = stripped in want
            continue
        if inside and "resources_available.gpu_model =" in stripped:
            return stripped.split("=", 1)[1].strip()
    return ""


def _resolve_hardware_manifest(scratch_root: Path) -> Dict[str, object]:
    preflight_path = scratch_root / "hardware_preflight.json"
    if preflight_path.exists():
        payload = read_json(preflight_path)
        if isinstance(payload, dict):
            payload = dict(payload)
            gpu_probe = payload.get("gpu_probe")
            hostname = str(payload.get("hostname", "") or "")
            if (not isinstance(gpu_probe, list) or not gpu_probe) and hostname:
                gpu_model = _pbs_gpu_model_for_host(hostname)
                if gpu_model:
                    payload["gpu_probe"] = [{"name": gpu_model}]
            return payload
    payload = collect_preflight()
    hostname = str(payload.get("hostname", "") or "")
    gpu_probe = payload.get("gpu_probe")
    if (not isinstance(gpu_probe, list) or not gpu_probe) and hostname:
        gpu_model = _pbs_gpu_model_for_host(hostname)
        if gpu_model:
            payload["gpu_probe"] = [{"name": gpu_model}]
    return payload


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _data_prep_manifest(ecbench_root: Path) -> Dict[str, object]:
    data_root = ecbench_root / "data"
    required = {
        "test_ec_csv": data_root / "test_ec.csv",
        "test_ec_fasta": data_root / "test_ec.fasta",
        "price_149_csv": data_root / "price-149.csv",
        "price_149_fasta": data_root / "price-149.fasta",
        "cluster_30_train": data_root / "cluster-30" / "train_ec.csv",
        "cluster_30_test": data_root / "cluster-30" / "test_ec.csv",
        "cluster_100_train": data_root / "cluster-100" / "train_ec.csv",
        "cluster_100_test": data_root / "cluster-100" / "test_ec.csv",
        "goa": data_root / "goa_uniprot_all.gaf",
        "swissprot_coordinates": data_root / "swissprot_coordinates.json",
    }
    files = {}
    for key, path in required.items():
        files[key] = {
            "path": str(path),
            "exists": path.exists(),
            "size_mib": path_size_mib(path),
            "rows": _count_csv_rows(path) if path.suffix.lower() == ".csv" else None,
        }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ecbench_root": str(ecbench_root),
        "data_root": str(data_root),
        "release_hints": {
            "pretraining_trembl": "2018-02",
            "training_swissprot": "2018-02",
            "test_swissprot": "2023-01",
            "ensemble_swissprot": "2023-02",
        },
        "files": files,
    }


def _resolve_data_manifest(ecbench_root: Path, explicit_path: Path | None) -> Dict[str, object]:
    if explicit_path is not None and explicit_path.exists():
        payload = read_json(explicit_path)
        if isinstance(payload, dict):
            return payload
    return _data_prep_manifest(ecbench_root)


def _task1_rows_from_official(ecbench_root: Path, manifest_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for threshold, rel_path in OFFICIAL_FILES.items():
        csv_path = ecbench_root / rel_path
        rows = read_csv_rows(csv_path)
        for item in manifest_rows:
            model_id = str(item["model_id"])
            if model_id == "protoec":
                continue
            column = str(item.get("task1_column") or BASE_MODEL_COLUMN_MAP.get(model_id) or "")
            if not column:
                continue
            metrics = task1_metrics_from_rows(rows, column)
            out.append(
                {
                    "threshold": threshold,
                    "model_id": model_id,
                    "display_name": item["display_name"],
                    "source_type": "official_csv",
                    "source_path": str(csv_path),
                    "exact_top1": metrics.exact_top1,
                    "micro_precision": metrics.micro_precision,
                    "micro_recall": metrics.micro_recall,
                    "micro_f1": metrics.micro_f1,
                    "weighted_precision": metrics.weighted_precision,
                    "weighted_recall": metrics.weighted_recall,
                    "weighted_f1": metrics.weighted_f1,
                    "coverage": metrics.coverage,
                    "no_pred_rate": metrics.no_pred_rate,
                    "variant_note": item.get("notes", ""),
                }
            )
    return out


def _task1_rows_from_protoec(protoec_manifest_path: Path) -> List[Dict[str, object]]:
    if not protoec_manifest_path.exists():
        return []
    manifest = read_json(protoec_manifest_path)
    if not isinstance(manifest, dict):
        return []
    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, dict):
        return []
    out: List[Dict[str, object]] = []
    for threshold_raw, obj in thresholds.items():
        if not isinstance(obj, dict):
            continue
        pred_path = Path(str(obj.get("predictions_csv", "")))
        if not pred_path.exists():
            continue
        rows = read_csv_rows(pred_path)
        converted = []
        for row in rows:
            converted.append(
                {
                    "ec_number": row.get("true_ecs", ""),
                    "protoec": row.get("top1_ec", ""),
                }
            )
        metrics = task1_metrics_from_rows(converted, "protoec")
        out.append(
            {
                "threshold": int(threshold_raw),
                "model_id": "protoec",
                "display_name": "ProtoEC",
                "source_type": "protoec_adapter",
                "source_path": str(pred_path),
                "exact_top1": metrics.exact_top1,
                "micro_precision": metrics.micro_precision,
                "micro_recall": metrics.micro_recall,
                "micro_f1": metrics.micro_f1,
                "weighted_precision": metrics.weighted_precision,
                "weighted_recall": metrics.weighted_recall,
                "weighted_f1": metrics.weighted_f1,
                "coverage": metrics.coverage,
                "no_pred_rate": metrics.no_pred_rate,
                "variant_note": obj.get("protocol_note", ""),
            }
        )
    return out


def _load_measurements(measurements_root: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    if not measurements_root.exists():
        return out
    for path in sorted(measurements_root.glob("*.json")):
        payload = read_json(path)
        if isinstance(payload, dict):
            payload = dict(payload)
            payload["measurement_path"] = str(path)
            out.append(payload)
    return out


def _operational_summary(measurements: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in measurements:
        model_id = str(row.get("model_id", ""))
        protocol = str(row.get("protocol", "A"))
        key = (model_id, protocol)
        bucket = grouped.setdefault(
            key,
            {
                "model_id": model_id,
                "protocol": protocol,
                "training_phase_status": "missing",
                "test_phase_status": "missing",
                "memory_usage_gib": None,
                "memory_kind": "",
                "run_time_per_epoch_s": None,
                "run_time_per_episode_s": None,
                "total_training_runtime_s": None,
                "test_runtime_s": None,
                "full_runtime_s": None,
                "per_protein_latency_ms": None,
                "gpu_model": "",
                "node_name": row.get("hostname", ""),
                "container_image": row.get("container_image", ""),
            },
        )
        phase = str(row.get("phase", ""))
        if phase == "train":
            bucket["training_phase_status"] = "ok" if _exit_ok(row) else "failed"
            bucket["total_training_runtime_s"] = row.get("duration_s")
            bucket["run_time_per_epoch_s"] = row.get("run_time_per_epoch_s")
            bucket["run_time_per_episode_s"] = row.get("run_time_per_episode_s")
        elif phase == "test":
            bucket["test_phase_status"] = "ok" if _exit_ok(row) else "failed"
            bucket["test_runtime_s"] = row.get("duration_s")
            bucket["per_protein_latency_ms"] = row.get("per_protein_latency_ms")
        elif phase == "full":
            bucket["full_phase_status"] = "ok" if _exit_ok(row) else "failed"
            bucket["full_runtime_s"] = row.get("duration_s")

        gpu_probe = row.get("gpu_probe") or []
        if isinstance(gpu_probe, list) and gpu_probe:
            first = gpu_probe[0]
            if isinstance(first, dict):
                bucket["gpu_model"] = first.get("name", "")
        bucket["node_name"] = row.get("hostname", bucket["node_name"])
        if not bucket["gpu_model"] and bucket["node_name"]:
            bucket["gpu_model"] = _pbs_gpu_model_for_host(str(bucket["node_name"]))
        bucket["container_image"] = row.get("container_image", bucket["container_image"])

        mem_value = row.get("memory_usage_gib")
        try:
            numeric_mem = float(mem_value) if mem_value is not None else None
        except (TypeError, ValueError):
            numeric_mem = None
        current_mem = bucket.get("memory_usage_gib")
        try:
            current_mem_f = float(current_mem) if current_mem is not None else None
        except (TypeError, ValueError):
            current_mem_f = None
        if numeric_mem is not None and (current_mem_f is None or numeric_mem > current_mem_f):
            bucket["memory_usage_gib"] = numeric_mem
            bucket["memory_kind"] = row.get("memory_kind", "")

    return sorted(grouped.values(), key=lambda item: (str(item["model_id"]), str(item["protocol"])))


def _artifact_inventory(ecbench_root: Path, manifest_rows: List[Dict[str, object]], protoec_manifest_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in manifest_rows:
        model_id = str(item["model_id"])
        patterns = item.get("runtime_artifact_globs") or []
        if not isinstance(patterns, list):
            patterns = []
        for path in expand_existing_paths(ecbench_root, [str(p) for p in patterns]):
            rows.append(
                {
                    "model_id": model_id,
                    "path": str(path),
                    "kind": "runtime_artifact",
                    "size_mib": path_size_mib(path),
                }
            )

    if protoec_manifest_path.exists():
        payload = read_json(protoec_manifest_path)
        if isinstance(payload, dict):
            for key in ("mirror_root", "outputs_root", "runs_root", "embeddings_base"):
                value = payload.get(key)
                if not value:
                    continue
                path = Path(str(value))
                if key == "embeddings_base":
                    for suffix in (".X.npy", ".keys.npy"):
                        actual = Path(str(path) + suffix)
                        if actual.exists():
                            rows.append(
                                {
                                    "model_id": "protoec",
                                    "path": str(actual),
                                    "kind": "embedding_cache",
                                    "size_mib": path_size_mib(actual),
                                }
                            )
                elif path.exists():
                    rows.append(
                        {
                            "model_id": "protoec",
                            "path": str(path),
                            "kind": key,
                            "size_mib": path_size_mib(path),
                        }
                    )
            thresholds = payload.get("thresholds") or {}
            if isinstance(thresholds, dict):
                for obj in thresholds.values():
                    if not isinstance(obj, dict):
                        continue
                    for key in ("predictions_csv", "metrics_json", "checkpoint_path"):
                        value = obj.get(key)
                        if not value:
                            continue
                        path = Path(str(value))
                        if path.exists():
                            rows.append(
                                {
                                    "model_id": "protoec",
                                    "path": str(path),
                                    "kind": key,
                                    "size_mib": path_size_mib(path),
                                }
                            )
    return rows


def _params_storage_summary(
    manifest_rows: List[Dict[str, object]],
    artifact_rows: List[Dict[str, object]],
    protoec_manifest_path: Path,
) -> List[Dict[str, object]]:
    size_by_model: Dict[str, float] = {}
    for row in artifact_rows:
        model_id = str(row["model_id"])
        size_by_model[model_id] = size_by_model.get(model_id, 0.0) + float(row.get("size_mib", 0.0) or 0.0)

    protoec_payload = read_json(protoec_manifest_path) if protoec_manifest_path.exists() else {}
    thresholds = protoec_payload.get("thresholds") if isinstance(protoec_payload, dict) else {}
    checkpoint_size = None
    if isinstance(thresholds, dict):
        for obj in thresholds.values():
            if not isinstance(obj, dict):
                continue
            value = obj.get("checkpoint_size_mib")
            if value is not None:
                checkpoint_size = value
                break

    rows: List[Dict[str, object]] = []
    for item in manifest_rows:
        model_id = str(item["model_id"])
        row = {
            "model_id": model_id,
            "display_name": item["display_name"],
            "trainable_params": "",
            "total_params": "",
            "checkpoint_size_mib": "",
            "runtime_artifact_size_mib": size_by_model.get(model_id, 0.0),
            "embedding_cache_size_mib": "",
            "model_size_mib": size_by_model.get(model_id, 0.0),
            "params_status": "na_for_heterogeneous_baseline",
        }
        if model_id == "protoec" and isinstance(protoec_payload, dict):
            row.update(
                {
                    "trainable_params": protoec_payload.get("trainable_params", ""),
                    "total_params": protoec_payload.get("total_params", ""),
                    "checkpoint_size_mib": checkpoint_size if checkpoint_size is not None else "",
                    "runtime_artifact_size_mib": size_by_model.get(model_id, 0.0),
                    "embedding_cache_size_mib": protoec_payload.get("embedding_cache_size_mib", ""),
                    "model_size_mib": checkpoint_size if checkpoint_size is not None else size_by_model.get(model_id, 0.0),
                    "params_status": "ok",
                }
            )
        rows.append(row)
    return rows


def _flops_summary(manifest_rows: List[Dict[str, object]], protoec_manifest_path: Path) -> List[Dict[str, object]]:
    protoec_payload = read_json(protoec_manifest_path) if protoec_manifest_path.exists() else {}
    rows: List[Dict[str, object]] = []
    for item in manifest_rows:
        model_id = str(item["model_id"])
        row = {
            "model_id": model_id,
            "display_name": item["display_name"],
            "train_flops_total": "",
            "train_flops_per_epoch_or_episode": "",
            "test_flops_total": "",
            "test_flops_per_protein": "",
            "flops_status": "opaque_external_binary_or_not_profiled",
        }
        if model_id == "protoec" and isinstance(protoec_payload, dict):
            row.update(
                {
                    "train_flops_total": protoec_payload.get("train_flops_total", ""),
                    "train_flops_per_epoch_or_episode": protoec_payload.get("train_flops_per_epoch_or_episode", ""),
                    "test_flops_total": protoec_payload.get("test_flops_total", ""),
                    "test_flops_per_protein": protoec_payload.get("test_flops_per_protein", ""),
                    "flops_status": protoec_payload.get("flops_status", "partial_protoec_breakdown"),
                }
            )
        rows.append(row)
    return rows


def _render_report(
    out_root: Path,
    hardware_manifest: Dict[str, object],
    data_manifest: Dict[str, object],
    task1_rows: List[Dict[str, object]],
    operational_rows: List[Dict[str, object]],
    params_rows: List[Dict[str, object]],
    flops_rows: List[Dict[str, object]],
) -> str:
    lines: List[str] = []
    lines.append("# EC-Bench Hopper GPU Runtime Benchmark Report")
    lines.append("")
    lines.append("## Hardware and execution context")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Host: `{hardware_manifest.get('hostname', '')}`")
    lines.append(f"- H100 detected on current node: `{hardware_manifest.get('h100_detected_on_current_node', False)}`")
    gpu_probe = hardware_manifest.get("gpu_probe") or []
    if gpu_probe:
        first = gpu_probe[0]
        if isinstance(first, dict):
            lines.append(f"- Visible GPU: `{first.get('name', '')}` with total memory `{first.get('memory_total', '')}`")
    lines.append("- Project policy: `CFP03-SF-097`")
    lines.append("- Caveat: EC-Bench Table 3 used RTX6000-40GB; absolute time/memory values are not directly comparable.")
    lines.append("")
    lines.append("## Data preparation provenance")
    files = data_manifest.get("files") if isinstance(data_manifest, dict) else {}
    present = 0
    total = 0
    if isinstance(files, dict):
        for obj in files.values():
            if not isinstance(obj, dict):
                continue
            total += 1
            if obj.get("exists"):
                present += 1
    lines.append(f"- Required manifest entries present: `{present}/{total}`")
    lines.append("- Full official data-prep path expected: `download_data -> extract_coordinates -> data_preprocessing -> run_mmseqs2 -> create_data -> go_creator`.")
    lines.append("")
    lines.append("## Task 1 accuracy summary")
    if task1_rows:
        ranked = sorted(task1_rows, key=lambda row: (int(row["threshold"]), float(row["weighted_f1"])), reverse=True)
        for row in ranked[:8]:
            lines.append(
                f"- ID{row['threshold']} `{row['display_name']}`: weighted-F1 `{float(row['weighted_f1']):.4f}`, micro-F1 `{float(row['micro_f1']):.4f}`, top-1 `{float(row['exact_top1']):.4f}`"
            )
    else:
        lines.append("- No Task 1 metrics were available.")
    lines.append("")
    lines.append("## Operational efficiency")
    if operational_rows:
        for row in operational_rows:
            lines.append(
                f"- `{row['model_id']}` / protocol `{row['protocol']}`: full=`{row.get('full_runtime_s', '')}`s, train=`{row.get('total_training_runtime_s', '')}`s, "
                f"test=`{row.get('test_runtime_s', '')}`s, latency=`{row.get('per_protein_latency_ms', '')}` ms/protein, "
                f"memory=`{row.get('memory_usage_gib', '')}` GiB ({row.get('memory_kind', '')})"
            )
    else:
        lines.append("- No measured runtime JSON files were found under the scratch measurement directory yet.")
    lines.append("")
    lines.append("## Measurement gaps")
    gaps = []
    if not gpu_probe:
        gaps.append("hardware preflight did not capture in-container gpu_probe rows; GPU model is currently inferred from PBS host context instead")
    if any(not row.get("gpu_model") for row in operational_rows):
        gaps.append("operational summary rows are missing explicit gpu_model strings even when GPU memory was sampled")
    if any(str(row.get("flops_status", "")).startswith("approx") for row in flops_rows):
        gaps.append("ProtoEC FLOPs are approximate head-only estimates and exclude frozen backbone embedding cost")
    if gaps:
        for item in gaps:
            lines.append(f"- {item}")
    else:
        lines.append("- No major measurement gaps were detected in this pass.")
    lines.append("")
    lines.append("## Parameter and storage efficiency")
    for row in params_rows:
        lines.append(
            f"- `{row['display_name']}`: trainable_params=`{row['trainable_params']}`, total_params=`{row['total_params']}`, "
            f"checkpoint_size_mib=`{row['checkpoint_size_mib']}`, runtime_artifact_size_mib=`{row['runtime_artifact_size_mib']}`"
        )
    lines.append("")
    lines.append("## Theoretical compute efficiency")
    for row in flops_rows:
        lines.append(
            f"- `{row['display_name']}`: train_flops_total=`{row['train_flops_total']}`, test_flops_per_protein=`{row['test_flops_per_protein']}`, status=`{row['flops_status']}`"
        )
    lines.append("")
    lines.append("## ProtoEC full-pipeline vs cached-feature breakdown")
    lines.append("- Protocol A should be interpreted as the full measured path after prepared benchmark artifacts are available.")
    lines.append("- Protocol B is the cached-feature / downstream-only supplementary view and is ProtoEC-specific rather than a roster-wide fairness claim.")
    lines.append("")
    lines.append("## Caveats for heterogeneous baselines")
    lines.append("- Some baselines are search/profile tools or opaque external binaries; trainable parameters and FLOPs can legitimately be `NA`.")
    lines.append("- Official Task 1 CSVs are used here as the stable accuracy surface for the paper-facing base roster.")
    lines.append("- Missing runtime records indicate that the benchmark harness is in place but the corresponding long-running cluster job has not been completed yet.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize the Hopper H100 EC-Bench benchmark outputs.")
    ap.add_argument("--manifest", type=Path, default=Path("benchmark/models.json"))
    ap.add_argument("--ecbench-root", type=Path, default=Path.cwd())
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("/home/svu/e0969321/FYP-fewshotlearn/results/ecbench_hopper_h100_task1_runtime_20260313"),
    )
    ap.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("/scratch/e0969321/ecbench_hopper_h100_task1_runtime_20260313"),
    )
    ap.add_argument(
        "--protoec-manifest",
        type=Path,
        default=None,
        help="Optional explicit ProtoEC adapter manifest path.",
    )
    ap.add_argument(
        "--data-prep-manifest",
        type=Path,
        default=None,
        help="Optional explicit data-prep manifest generated by the Hopper prep workflow.",
    )
    args = ap.parse_args()

    out_root = args.out_root.resolve()
    scratch_root = args.scratch_root.resolve()
    measurements_root = scratch_root / "measurements"
    protoec_manifest_path = args.protoec_manifest or (scratch_root / "protoec" / "adapter_manifest.json")
    ensure_dir(out_root)
    ensure_dir(scratch_root)

    manifest_rows = _load_manifest((args.ecbench_root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest)
    hardware_manifest = _resolve_hardware_manifest(scratch_root)
    hardware_manifest["container_image"] = hardware_manifest.get("container_image") or os.environ.get("BENCHMARK_CONTAINER_IMAGE", "")
    data_manifest = _resolve_data_manifest(args.ecbench_root.resolve(), args.data_prep_manifest)

    task1_rows = _task1_rows_from_official(args.ecbench_root.resolve(), manifest_rows)
    task1_rows.extend(_task1_rows_from_protoec(protoec_manifest_path))
    task1_rows = sorted(task1_rows, key=lambda row: (int(row["threshold"]), str(row["model_id"])))

    measurements = _load_measurements(measurements_root)
    operational_rows = _operational_summary(measurements)
    artifact_rows = _artifact_inventory(args.ecbench_root.resolve(), manifest_rows, protoec_manifest_path)
    params_rows = _params_storage_summary(manifest_rows, artifact_rows, protoec_manifest_path)
    flops_rows = _flops_summary(manifest_rows, protoec_manifest_path)

    raw_scratch_link = out_root / "raw_scratch"
    if raw_scratch_link.exists() or raw_scratch_link.is_symlink():
        raw_scratch_link.unlink()
    raw_scratch_link.symlink_to(scratch_root, target_is_directory=True)

    write_json(out_root / "hardware_manifest.json", hardware_manifest)
    write_json(out_root / "data_prep_manifest.json", data_manifest)
    write_csv(
        out_root / "task1_metrics_summary.csv",
        task1_rows,
        [
            "threshold",
            "model_id",
            "display_name",
            "source_type",
            "source_path",
            "exact_top1",
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
            "coverage",
            "no_pred_rate",
            "variant_note",
        ],
    )
    write_csv(
        out_root / "efficiency_raw.csv",
        measurements,
        sorted({key for row in measurements for key in row.keys()}) if measurements else ["model_id", "phase"],
    )
    write_csv(
        out_root / "efficiency_operational_summary.csv",
        operational_rows,
        [
            "model_id",
            "protocol",
            "training_phase_status",
            "test_phase_status",
            "full_phase_status",
            "memory_usage_gib",
            "memory_kind",
            "run_time_per_epoch_s",
            "run_time_per_episode_s",
            "total_training_runtime_s",
            "test_runtime_s",
            "full_runtime_s",
            "per_protein_latency_ms",
            "gpu_model",
            "node_name",
            "container_image",
        ],
    )
    write_csv(
        out_root / "efficiency_params_storage_summary.csv",
        params_rows,
        [
            "model_id",
            "display_name",
            "trainable_params",
            "total_params",
            "checkpoint_size_mib",
            "runtime_artifact_size_mib",
            "embedding_cache_size_mib",
            "model_size_mib",
            "params_status",
        ],
    )
    write_csv(
        out_root / "efficiency_flops_summary.csv",
        flops_rows,
        [
            "model_id",
            "display_name",
            "train_flops_total",
            "train_flops_per_epoch_or_episode",
            "test_flops_total",
            "test_flops_per_protein",
            "flops_status",
        ],
    )
    write_csv(
        out_root / "artifact_inventory.csv",
        artifact_rows,
        ["model_id", "path", "kind", "size_mib"],
    )
    report = _render_report(out_root, hardware_manifest, data_manifest, task1_rows, operational_rows, params_rows, flops_rows)
    (out_root / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"[run_benchmark] wrote report bundle to {out_root}")


if __name__ == "__main__":
    main()
