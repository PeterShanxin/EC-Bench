from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def _safe_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _task1_scope_fields(
    *,
    coverage: object,
    queries_total: object,
    queries_evaluated: object,
    source_type: str,
) -> Dict[str, object]:
    total = _safe_int(queries_total)
    evaluated = _safe_int(queries_evaluated)
    coverage_value = _safe_float(coverage)

    if total is None and source_type == "official_csv":
        total = evaluated
    if evaluated is None and total is not None and source_type == "official_csv":
        evaluated = total
    if evaluated is None and total is not None and coverage_value is not None and coverage_value >= 0.999999:
        evaluated = total

    if total is not None and evaluated is not None and evaluated < total:
        coverage_scope = "evaluated_subset_of_full_test"
        accuracy_scope = "evaluated_subset_only"
        accuracy_direct_compare_ok = False
    elif total is not None and evaluated is not None:
        coverage_scope = "full_test"
        accuracy_scope = "full_test"
        accuracy_direct_compare_ok = True
    elif source_type == "official_csv":
        coverage_scope = "full_test"
        accuracy_scope = "full_test"
        accuracy_direct_compare_ok = True
    else:
        coverage_scope = ""
        accuracy_scope = ""
        accuracy_direct_compare_ok = None

    missing = ""
    if total is not None and evaluated is not None:
        missing = max(0, total - evaluated)

    return {
        "coverage_total_queries": total if total is not None else "",
        "coverage_evaluated_queries": evaluated if evaluated is not None else "",
        "coverage_missing_queries": missing,
        "coverage_scope": coverage_scope,
        "accuracy_scope": accuracy_scope,
        "accuracy_direct_compare_ok": accuracy_direct_compare_ok,
    }


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
        evaluated_rows = len(rows)
        # EC-Bench release CSVs in OFFICIAL_FILES are `*_go_3.csv` subsets.
        # When the prepared split CSV exists, treat that as the full test size
        # so the report correctly flags these rows as subset-evaluated.
        full_test_rows = _count_csv_rows(ecbench_root / "data" / f"test_ec_{threshold}.csv")
        total_rows = full_test_rows if full_test_rows else evaluated_rows
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
                    **_task1_scope_fields(
                        coverage=metrics.coverage,
                        queries_total=total_rows,
                        queries_evaluated=evaluated_rows,
                        source_type="official_csv",
                    ),
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
        metrics_payload: Dict[str, object] = {}
        metrics_path = Path(str(obj.get("metrics_json", "")))
        if metrics_path.exists():
            payload = read_json(metrics_path)
            if isinstance(payload, dict):
                metrics_payload = payload
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
        coverage = metrics.coverage
        no_pred_rate = metrics.no_pred_rate
        if metrics_payload:
            try:
                coverage = float(metrics_payload.get("coverage_ratio", coverage))
            except (TypeError, ValueError):
                coverage = metrics.coverage
            no_pred_rate = 1.0 - coverage
        queries_total = obj.get("queries_total", "")
        if queries_total in ("", None):
            queries_total = metrics_payload.get("queries_total", "")
        queries_evaluated = obj.get("queries_evaluated", "")
        if queries_evaluated in ("", None):
            queries_evaluated = metrics_payload.get("queries_evaluated", _count_csv_rows(pred_path))
        scope_fields = _task1_scope_fields(
            coverage=coverage,
            queries_total=queries_total,
            queries_evaluated=queries_evaluated,
            source_type="protoec_adapter",
        )
        variant_note = str(obj.get("protocol_note", "") or "").strip()
        if scope_fields.get("accuracy_scope") == "evaluated_subset_only":
            evaluated = scope_fields.get("coverage_evaluated_queries", "")
            total = scope_fields.get("coverage_total_queries", "")
            subset_note = (
                f"Accuracy metrics are evaluated on the covered subset only ({evaluated}/{total} queries); "
                "missing-label queries are excluded from the accuracy denominator."
            )
            variant_note = f"{variant_note} {subset_note}".strip()
        elif metrics_payload:
            extra_note = "Coverage is aligned to the current global_metrics.json for the full-test ProtoEC EC-Bench evaluation path."
            variant_note = f"{variant_note} {extra_note}".strip()
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
                "coverage": coverage,
                "no_pred_rate": no_pred_rate,
                **scope_fields,
                "variant_note": variant_note,
            }
        )
    return out


def _load_measurements(measurement_roots: Sequence[Path]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen: set[str] = set()
    for measurements_root in measurement_roots:
        if not measurements_root.exists():
            continue
        for path in sorted(measurements_root.glob("*.json")):
            payload = read_json(path)
            if isinstance(payload, dict):
                payload = dict(payload)
                payload["measurement_path"] = str(path)
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                out.append(payload)
    return out


def _first_existing(root: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _threshold_from_text(*values: object) -> Optional[int]:
    for value in values:
        text = str(value or "")
        if not text:
            continue
        match = re.search(r"(?:train|test)_ec_(\d+)", text)
        if match:
            return int(match.group(1))
        match = re.search(r"\bid(\d+)\b", text.lower())
        if match:
            return int(match.group(1))
    return None


def _clean_head_params() -> Tuple[int, int]:
    trainable = (
        (1280 * 512) + 512
        + (512 + 512)
        + (512 * 512) + 512
        + (512 + 512)
        + (512 * 256) + 256
    )
    return trainable, 650_000_000 + trainable


def _load_clean_bundle(clean_bundle_root: Optional[Path]) -> Dict[str, object]:
    if clean_bundle_root is None:
        return {}
    root = clean_bundle_root.resolve()
    if not root.exists():
        return {}

    train_manifest_path = _first_existing(root, ["train_ec_*_supconH*.manifest.json"])
    infer_manifest_path = _first_existing(root, ["*infer*_manifest.json"])
    infer_metrics_path = _first_existing(root, ["*infer*_metrics.json"])
    predictions_path = _first_existing(root, ["*maxsep.csv"])
    checkpoint_path = _first_existing(root, ["train_ec_*_supconH*.pth"])

    train_manifest = read_json(train_manifest_path) if train_manifest_path and train_manifest_path.exists() else {}
    infer_manifest = read_json(infer_manifest_path) if infer_manifest_path and infer_manifest_path.exists() else {}
    infer_metrics = read_json(infer_metrics_path) if infer_metrics_path and infer_metrics_path.exists() else {}

    if isinstance(infer_manifest, dict) and infer_manifest.get("predictions_csv"):
        candidate = Path(str(infer_manifest["predictions_csv"]))
        if predictions_path is None and candidate.exists():
            predictions_path = candidate
    if isinstance(infer_manifest, dict) and infer_manifest.get("checkpoint_path"):
        candidate = Path(str(infer_manifest["checkpoint_path"]))
        if checkpoint_path is None and candidate.exists():
            checkpoint_path = candidate
    if isinstance(train_manifest, dict) and train_manifest.get("final_model_out"):
        candidate = Path(str(train_manifest["final_model_out"]))
        if checkpoint_path is None and candidate.exists():
            checkpoint_path = candidate

    trainable_params, total_params = _clean_head_params()
    split_threshold = _threshold_from_text(
        train_manifest.get("training_data") if isinstance(train_manifest, dict) else "",
        infer_manifest.get("train_data") if isinstance(infer_manifest, dict) else "",
        infer_manifest.get("test_data") if isinstance(infer_manifest, dict) else "",
        str(root),
    )
    return {
        "bundle_root": str(root),
        "train_manifest_path": str(train_manifest_path) if train_manifest_path else "",
        "infer_manifest_path": str(infer_manifest_path) if infer_manifest_path else "",
        "infer_metrics_path": str(infer_metrics_path) if infer_metrics_path else "",
        "predictions_path": str(predictions_path) if predictions_path else "",
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "checkpoint_size_mib": path_size_mib(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else "",
        "split_threshold": split_threshold,
        "train_manifest": train_manifest if isinstance(train_manifest, dict) else {},
        "infer_manifest": infer_manifest if isinstance(infer_manifest, dict) else {},
        "infer_metrics": infer_metrics if isinstance(infer_metrics, dict) else {},
        "trainable_params": trainable_params,
        "total_params": total_params,
        "embedding_model": "esm1b_t33_650M_UR50S",
        "params_status": "exact_head_plus_approx_backbone",
    }


def _resolve_optional_json(explicit_path: Path | None, candidates: Sequence[Path]) -> Dict[str, object]:
    if explicit_path is not None and explicit_path.exists():
        payload = read_json(explicit_path)
        if isinstance(payload, dict):
            return dict(payload)
    for candidate in candidates:
        if candidate.exists():
            payload = read_json(candidate)
            if isinstance(payload, dict):
                return dict(payload)
    return {}


def _select_fields(rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> List[Dict[str, object]]:
    keys = list(fieldnames)
    return [{key: row.get(key, "") for key in keys} for row in rows]


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
                "setup_phase_status": "missing",
                "training_phase_status": "missing",
                "test_phase_status": "missing",
                "setup_runtime_s": None,
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
                "split_threshold": row.get("split_threshold", ""),
                "runtime_scope": row.get("runtime_scope", ""),
                "threads_requested": row.get("threads_requested", ""),
                "hardware_class": row.get("hardware_class", ""),
                "notes": row.get("notes", ""),
            },
        )
        phase = str(row.get("phase", ""))
        if phase == "setup":
            bucket["setup_phase_status"] = "ok" if _exit_ok(row) else "failed"
            bucket["setup_runtime_s"] = row.get("duration_s")
        elif phase == "train":
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
        for extra_key in ("split_threshold", "runtime_scope", "threads_requested", "hardware_class", "notes"):
            extra_value = row.get(extra_key)
            if extra_value not in ("", None):
                bucket[extra_key] = extra_value

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


def _artifact_inventory(
    ecbench_root: Path,
    manifest_rows: List[Dict[str, object]],
    protoec_manifest_path: Path,
    clean_bundle: Dict[str, object],
) -> List[Dict[str, object]]:
    def _quick_size(path: Path) -> object:
        if path.is_file():
            return path_size_mib(path)
        return ""

    def _inventory_paths(patterns: Sequence[str]) -> List[Path]:
        resolved: List[Path] = []
        for pattern in patterns:
            if "**/*" in pattern:
                prefix = pattern.split("**/*", 1)[0].rstrip("/")
                if prefix:
                    candidate = ecbench_root / prefix
                    if candidate.exists():
                        resolved.append(candidate)
                    continue
            resolved.extend(expand_existing_paths(ecbench_root, [pattern]))
        deduped = sorted({str(path.resolve()) for path in resolved})
        return [Path(item) for item in deduped]

    rows: List[Dict[str, object]] = []
    for item in manifest_rows:
        model_id = str(item["model_id"])
        patterns = item.get("runtime_artifact_globs") or []
        if not isinstance(patterns, list):
            patterns = []
        for path in _inventory_paths([str(p) for p in patterns]):
            rows.append(
                {
                    "model_id": model_id,
                    "path": str(path),
                    "kind": "runtime_artifact",
                    "size_mib": _quick_size(path),
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
                                    "size_mib": _quick_size(actual),
                                }
                            )
                elif path.exists():
                    rows.append(
                        {
                            "model_id": "protoec",
                            "path": str(path),
                            "kind": key,
                            "size_mib": _quick_size(path),
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
                                    "size_mib": _quick_size(path),
                                }
                            )
    if clean_bundle:
        for key in (
            "train_manifest_path",
            "infer_manifest_path",
            "infer_metrics_path",
            "predictions_path",
            "checkpoint_path",
        ):
            value = clean_bundle.get(key)
            if not value:
                continue
            path = Path(str(value))
            if path.exists():
                rows.append(
                    {
                        "model_id": "clean",
                        "path": str(path),
                        "kind": f"clean_bundle_{key}",
                        "size_mib": _quick_size(path),
                    }
                )
    return rows


def _params_storage_summary(
    manifest_rows: List[Dict[str, object]],
    artifact_rows: List[Dict[str, object]],
    protoec_manifest_path: Path,
    clean_bundle: Dict[str, object],
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
        elif model_id == "clean" and clean_bundle:
            row.update(
                {
                    "trainable_params": clean_bundle.get("trainable_params", ""),
                    "total_params": clean_bundle.get("total_params", ""),
                    "checkpoint_size_mib": clean_bundle.get("checkpoint_size_mib", ""),
                    "params_status": clean_bundle.get("params_status", "ok"),
                }
            )
        rows.append(row)
    return rows


def _flops_summary(
    manifest_rows: List[Dict[str, object]],
    protoec_manifest_path: Path,
    *,
    protoec_profiler_payload: Dict[str, object],
    clean_profiler_payload: Dict[str, object],
    clean_bundle: Dict[str, object],
) -> List[Dict[str, object]]:
    protoec_payload = read_json(protoec_manifest_path) if protoec_manifest_path.exists() else {}
    rows: List[Dict[str, object]] = []
    for item in manifest_rows:
        model_id = str(item["model_id"])
        family = str(item.get("family", "") or "")
        default_status = "opaque_external_binary_or_not_profiled"
        if family in {"search", "rule", "profile"}:
            default_status = "non_neural_search_tool"
        elif family in {"ml"}:
            default_status = "opaque_external_binary_or_traditional_ml"
        elif family in {"neural", "contrastive", "fewshot"}:
            default_status = "not_yet_profiled_model_core"
        row = {
            "model_id": model_id,
            "display_name": item["display_name"],
            "train_flops_total": "",
            "train_flops_per_epoch_or_episode": "",
            "test_flops_total": "",
            "test_flops_per_protein": "",
            "flops_status": default_status,
        }
        if model_id == "protoec" and isinstance(protoec_profiler_payload, dict) and protoec_profiler_payload:
            row.update(
                {
                    "train_flops_total": protoec_profiler_payload.get("train_flops_total", ""),
                    "train_flops_per_epoch_or_episode": protoec_profiler_payload.get("train_flops_per_epoch_or_episode", ""),
                    "test_flops_total": protoec_profiler_payload.get("test_flops_total", ""),
                    "test_flops_per_protein": protoec_profiler_payload.get("test_flops_per_protein", ""),
                    "flops_status": protoec_profiler_payload.get("flops_status", "profiler_model_core"),
                }
            )
        elif model_id == "protoec" and isinstance(protoec_payload, dict):
            row.update(
                {
                    "train_flops_total": protoec_payload.get("train_flops_total", ""),
                    "train_flops_per_epoch_or_episode": protoec_payload.get("train_flops_per_epoch_or_episode", ""),
                    "test_flops_total": protoec_payload.get("test_flops_total", ""),
                    "test_flops_per_protein": protoec_payload.get("test_flops_per_protein", ""),
                    "flops_status": protoec_payload.get("flops_status", "partial_protoec_breakdown"),
                }
            )
        elif model_id == "clean" and isinstance(clean_profiler_payload, dict) and clean_profiler_payload:
            row.update(
                {
                    "train_flops_total": clean_profiler_payload.get("train_flops_total", ""),
                    "train_flops_per_epoch_or_episode": clean_profiler_payload.get("train_flops_per_epoch_or_episode", ""),
                    "test_flops_total": clean_profiler_payload.get("test_flops_total", ""),
                    "test_flops_per_protein": clean_profiler_payload.get("test_flops_per_protein", ""),
                    "flops_status": clean_profiler_payload.get("flops_status", "profiler_model_core"),
                }
            )
        elif model_id == "clean" and clean_bundle:
            row["flops_status"] = "not_yet_profiled_model_core"
        rows.append(row)
    return rows


def _latest_measurement(
    measurements: List[Dict[str, object]],
    *,
    model_id: str,
    phase: str,
    protocol_contains: Sequence[str] = (),
    pbs_jobids: Sequence[str] = (),
) -> Dict[str, object]:
    want_jobids = {str(item) for item in pbs_jobids if str(item)}
    candidates: List[Dict[str, object]] = []
    for row in measurements:
        if str(row.get("model_id", "")) != model_id:
            continue
        if str(row.get("phase", "")) != phase:
            continue
        protocol = str(row.get("protocol", ""))
        if protocol_contains and not all(chunk in protocol for chunk in protocol_contains):
            continue
        if want_jobids and str(row.get("pbs_jobid", "")) not in want_jobids:
            continue
        if not _exit_ok(row):
            continue
        candidates.append(row)
    if not candidates:
        return {}
    candidates.sort(key=lambda item: (str(item.get("finished_utc", "")), str(item.get("started_utc", "")), str(item.get("measurement_path", ""))))
    return candidates[-1]


def _protoec_actual_task1_row(protoec_manifest_path: Path, threshold: int) -> Dict[str, object]:
    for row in _task1_rows_from_protoec(protoec_manifest_path):
        if int(row.get("threshold", -1)) == int(threshold):
            return row
    return {}


def _clean_actual_task1_row(clean_bundle: Dict[str, object]) -> Dict[str, object]:
    metrics = clean_bundle.get("infer_metrics")
    if not isinstance(metrics, dict) or not metrics:
        return {}
    threshold = clean_bundle.get("split_threshold")
    if threshold is None:
        return {}
    return {
        "threshold": int(threshold),
        "model_id": "clean",
        "display_name": "CLEAN",
        "source_type": "clean_retrained_bundle",
        "source_path": str(clean_bundle.get("predictions_path", "") or clean_bundle.get("infer_metrics_path", "")),
        "exact_top1": metrics.get("exact_top1", ""),
        "micro_precision": metrics.get("micro_precision", ""),
        "micro_recall": metrics.get("micro_recall", ""),
        "micro_f1": metrics.get("micro_f1", ""),
        "weighted_precision": metrics.get("weighted_precision", ""),
        "weighted_recall": metrics.get("weighted_recall", ""),
        "weighted_f1": metrics.get("weighted_f1", ""),
        "coverage": metrics.get("coverage", ""),
        "no_pred_rate": metrics.get("no_pred_rate", ""),
        **_task1_scope_fields(
            coverage=metrics.get("coverage", ""),
            queries_total=metrics.get("query_count", ""),
            queries_evaluated=metrics.get("query_count", ""),
            source_type="clean_retrained_bundle",
        ),
        "variant_note": "Executed-split retrained ID30 checkpoint from the durable home bundle.",
    }


def _matched_protoec_clean_summary(
    *,
    ecbench_root: Path,
    measurements: List[Dict[str, object]],
    protoec_manifest_path: Path,
    clean_bundle: Dict[str, object],
) -> Dict[str, object]:
    protoec_manifest = read_json(protoec_manifest_path) if protoec_manifest_path.exists() else {}
    protoec_thresholds = protoec_manifest.get("thresholds") if isinstance(protoec_manifest, dict) else {}
    protoec_threshold = None
    if isinstance(protoec_thresholds, dict) and protoec_thresholds:
        try:
            protoec_threshold = min(int(key) for key in protoec_thresholds.keys())
        except ValueError:
            protoec_threshold = None
    clean_threshold = clean_bundle.get("split_threshold")
    matched_threshold = protoec_threshold if protoec_threshold is not None else clean_threshold
    if matched_threshold is None:
        matched_threshold = clean_threshold

    protoec_full = _latest_measurement(measurements, model_id="protoec", phase="full")
    protoec_train = _latest_measurement(measurements, model_id="protoec", phase="train")
    protoec_test = _latest_measurement(measurements, model_id="protoec", phase="test")
    protoec_task1 = _protoec_actual_task1_row(protoec_manifest_path, int(matched_threshold)) if matched_threshold is not None else {}

    clean_train_manifest = clean_bundle.get("train_manifest") if isinstance(clean_bundle.get("train_manifest"), dict) else {}
    clean_infer_manifest = clean_bundle.get("infer_manifest") if isinstance(clean_bundle.get("infer_manifest"), dict) else {}
    clean_train_jobid = str(((clean_train_manifest.get("runtime_context") or {}).get("pbs_jobid", "")) if isinstance(clean_train_manifest, dict) else "")
    clean_test_jobid = str(((clean_infer_manifest.get("runtime_context") or {}).get("pbs_jobid", "")) if isinstance(clean_infer_manifest, dict) else "")
    clean_distance = _latest_measurement(measurements, model_id="clean", phase="setup", protocol_contains=("distance_map",))
    clean_orphan_embed = _latest_measurement(measurements, model_id="clean", phase="full", protocol_contains=("orphan_embed",), pbs_jobids=[clean_train_jobid])
    clean_train = _latest_measurement(measurements, model_id="clean", phase="train", pbs_jobids=[clean_train_jobid])
    clean_test_embed = _latest_measurement(measurements, model_id="clean", phase="full", protocol_contains=("test_embed",), pbs_jobids=[clean_test_jobid])
    clean_test = _latest_measurement(measurements, model_id="clean", phase="test", pbs_jobids=[clean_test_jobid])
    clean_task1 = _clean_actual_task1_row(clean_bundle)

    protoec_embedding_runtime = ""
    if protoec_full and protoec_train and protoec_test:
        protoec_embedding_runtime = max(
            0.0,
            float(protoec_full.get("duration_s", 0.0) or 0.0)
            - float(protoec_train.get("duration_s", 0.0) or 0.0)
            - float(protoec_test.get("duration_s", 0.0) or 0.0),
        )

    clean_embedding_components = [
        float(row.get("duration_s", 0.0) or 0.0)
        for row in (clean_orphan_embed, clean_test_embed)
        if row
    ]
    clean_embedding_runtime = sum(clean_embedding_components) if clean_embedding_components else ""
    clean_embedding_scope = ""
    if clean_embedding_components:
        clean_embedding_scope = "partial_orphan_plus_test_only_shared_train_embeddings_reused"

    clean_including_runtime = ""
    if clean_embedding_scope == "complete" and clean_embedding_runtime != "" and clean_distance and clean_train and clean_test:
        clean_including_runtime = (
            float(clean_embedding_runtime)
            + float(clean_distance.get("duration_s", 0.0) or 0.0)
            + float(clean_train.get("duration_s", 0.0) or 0.0)
            + float(clean_test.get("duration_s", 0.0) or 0.0)
        )

    clean_embedding_cache_size = ""
    clean_esm_matrix_path = ""
    if isinstance(clean_train_manifest, dict):
        clean_esm_matrix_path = str(clean_train_manifest.get("esm_matrix_path", "") or "")
    if clean_esm_matrix_path:
        esm_matrix_path = Path(clean_esm_matrix_path)
        if esm_matrix_path.exists():
            clean_embedding_cache_size = path_size_mib(esm_matrix_path)

    protoec_row = {
        "model_id": "protoec",
        "display_name": "ProtoEC",
        "split_threshold": matched_threshold if matched_threshold is not None else "",
        "status": (
            "completed_partial_accuracy_scope"
            if protoec_full and protoec_train and protoec_test and protoec_task1 and protoec_task1.get("accuracy_scope") == "evaluated_subset_only"
            else "completed"
            if protoec_full and protoec_train and protoec_test and protoec_task1
            else "missing_or_staged"
        ),
        "accuracy_weighted_f1": protoec_task1.get("weighted_f1", ""),
        "accuracy_micro_f1": protoec_task1.get("micro_f1", ""),
        "accuracy_top1": protoec_task1.get("exact_top1", ""),
        "coverage": protoec_task1.get("coverage", ""),
        "no_pred_rate": protoec_task1.get("no_pred_rate", ""),
        "accuracy_scope": protoec_task1.get("accuracy_scope", ""),
        "accuracy_direct_compare_ok": protoec_task1.get("accuracy_direct_compare_ok", ""),
        "coverage_scope": protoec_task1.get("coverage_scope", ""),
        "coverage_total_queries": protoec_task1.get("coverage_total_queries", ""),
        "coverage_evaluated_queries": protoec_task1.get("coverage_evaluated_queries", ""),
        "coverage_missing_queries": protoec_task1.get("coverage_missing_queries", ""),
        "embedding_backbone": protoec_manifest.get("embedding_model", "") if isinstance(protoec_manifest, dict) else "",
        "embedding_runtime_s": protoec_embedding_runtime,
        "embedding_runtime_scope": "full_protocolA_minus_protocolB_train_test" if protoec_embedding_runtime != "" else "",
        "precompute_runtime_s": "",
        "total_training_runtime_s": protoec_train.get("duration_s", "") if protoec_train else "",
        "training_unit": protoec_train.get("training_unit", "") if protoec_train else "",
        "runtime_per_training_unit_s": protoec_train.get("run_time_per_episode_s", "") if protoec_train else "",
        "test_runtime_s": protoec_test.get("duration_s", "") if protoec_test else "",
        "including_embedding_runtime_s": protoec_full.get("duration_s", "") if protoec_full else "",
        "excluding_embedding_runtime_s": (
            float(protoec_train.get("duration_s", 0.0) or 0.0) + float(protoec_test.get("duration_s", 0.0) or 0.0)
            if protoec_train and protoec_test
            else ""
        ),
        "embedding_memory_gib": protoec_full.get("memory_usage_gib", "") if protoec_full else "",
        "peak_memory_gib": max(
            [
                float(row.get("memory_usage_gib", 0.0) or 0.0)
                for row in (protoec_full, protoec_train, protoec_test)
                if row
            ],
            default=0.0,
        ),
        "trainable_params": protoec_manifest.get("trainable_params", "") if isinstance(protoec_manifest, dict) else "",
        "total_params": protoec_manifest.get("total_params", "") if isinstance(protoec_manifest, dict) else "",
        "checkpoint_size_mib": next(
            (
                obj.get("checkpoint_size_mib", "")
                for obj in (protoec_thresholds or {}).values()
                if isinstance(obj, dict)
            ),
            "",
        ) if isinstance(protoec_thresholds, dict) else "",
        "embedding_cache_size_mib": protoec_manifest.get("embedding_cache_size_mib", "") if isinstance(protoec_manifest, dict) else "",
        "gpu_model": (
            (protoec_full.get("gpu_probe") or [{}])[0].get("name", "")
            if protoec_full and isinstance(protoec_full.get("gpu_probe"), list) and protoec_full.get("gpu_probe")
            else protoec_full.get("gpu_model_hint", "") if protoec_full else ""
        ),
        "node_name": protoec_full.get("hostname", "") if protoec_full else "",
        "container_image": protoec_full.get("container_image", "") if protoec_full else "",
        "pbs_jobids": ",".join(
            [str(row.get("pbs_jobid", "")) for row in (protoec_full, protoec_train, protoec_test) if row and row.get("pbs_jobid")]
        ),
        "notes": (
            "Protocol A is the full measured path; Protocol B isolates cached-feature downstream training and test. "
            "Current accuracy metrics are subset-only because queries without labels in the prototype bank are excluded from evaluation."
            if protoec_task1.get("accuracy_scope") == "evaluated_subset_only"
            else "Protocol A is the full measured path; Protocol B isolates cached-feature downstream training and test."
        ),
    }

    clean_runtime_rows = [row for row in (clean_distance, clean_orphan_embed, clean_train, clean_test_embed, clean_test) if row]
    clean_row = {
        "model_id": "clean",
        "display_name": "CLEAN",
        "split_threshold": clean_threshold if clean_threshold is not None else matched_threshold if matched_threshold is not None else "",
        "status": "completed_with_partial_embedding_accounting" if clean_train and clean_test and clean_task1 else "missing_or_staged",
        "accuracy_weighted_f1": clean_task1.get("weighted_f1", ""),
        "accuracy_micro_f1": clean_task1.get("micro_f1", ""),
        "accuracy_top1": clean_task1.get("exact_top1", ""),
        "coverage": clean_task1.get("coverage", ""),
        "no_pred_rate": clean_task1.get("no_pred_rate", ""),
        "accuracy_scope": clean_task1.get("accuracy_scope", ""),
        "accuracy_direct_compare_ok": clean_task1.get("accuracy_direct_compare_ok", ""),
        "coverage_scope": clean_task1.get("coverage_scope", ""),
        "coverage_total_queries": clean_task1.get("coverage_total_queries", ""),
        "coverage_evaluated_queries": clean_task1.get("coverage_evaluated_queries", ""),
        "coverage_missing_queries": clean_task1.get("coverage_missing_queries", ""),
        "embedding_backbone": clean_bundle.get("embedding_model", ""),
        "embedding_runtime_s": clean_embedding_runtime,
        "embedding_runtime_scope": clean_embedding_scope,
        "precompute_runtime_s": clean_distance.get("duration_s", "") if clean_distance else "",
        "total_training_runtime_s": clean_train.get("duration_s", "") if clean_train else "",
        "training_unit": clean_train.get("training_unit", "") if clean_train else "epoch",
        "runtime_per_training_unit_s": clean_train.get("run_time_per_epoch_s", "") if clean_train else "",
        "test_runtime_s": clean_test.get("duration_s", "") if clean_test else "",
        "including_embedding_runtime_s": clean_including_runtime,
        "excluding_embedding_runtime_s": (
            float(clean_distance.get("duration_s", 0.0) or 0.0)
            + float(clean_train.get("duration_s", 0.0) or 0.0)
            + float(clean_test.get("duration_s", 0.0) or 0.0)
            if clean_distance and clean_train and clean_test
            else ""
        ),
        "embedding_memory_gib": max(
            [float(row.get("memory_usage_gib", 0.0) or 0.0) for row in (clean_orphan_embed, clean_test_embed) if row],
            default=0.0,
        ),
        "peak_memory_gib": max([float(row.get("memory_usage_gib", 0.0) or 0.0) for row in clean_runtime_rows], default=0.0),
        "trainable_params": clean_bundle.get("trainable_params", ""),
        "total_params": clean_bundle.get("total_params", ""),
        "checkpoint_size_mib": clean_bundle.get("checkpoint_size_mib", ""),
        "embedding_cache_size_mib": clean_embedding_cache_size,
        "gpu_model": (
            (clean_train.get("gpu_probe") or [{}])[0].get("name", "")
            if clean_train and isinstance(clean_train.get("gpu_probe"), list) and clean_train.get("gpu_probe")
            else clean_train.get("gpu_model_hint", "") if clean_train else ""
        ),
        "node_name": clean_train.get("hostname", "") if clean_train else "",
        "container_image": clean_train.get("container_image", "") if clean_train else "",
        "pbs_jobids": ",".join(
            [str(row.get("pbs_jobid", "")) for row in clean_runtime_rows if row and row.get("pbs_jobid")]
        ),
        "notes": "Train-set embeddings were reused from the shared cache, so the current including-embedding total is intentionally left blank until a matched train-embed rerun exists.",
    }

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "matched_split_threshold": matched_threshold if matched_threshold is not None else "",
        "hardware_regime": clean_row["gpu_model"] or protoec_row["gpu_model"],
        "rows": [protoec_row, clean_row],
    }


def _render_report(
    out_root: Path,
    hardware_manifest: Dict[str, object],
    data_manifest: Dict[str, object],
    task1_rows: List[Dict[str, object]],
    operational_rows: List[Dict[str, object]],
    params_rows: List[Dict[str, object]],
    flops_rows: List[Dict[str, object]],
    matched_summary: Dict[str, object],
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
        focus_threshold = matched_summary.get("matched_split_threshold", "") if isinstance(matched_summary, dict) else ""
        try:
            focus_threshold_int = int(focus_threshold)
        except (TypeError, ValueError):
            thresholds = sorted({int(row["threshold"]) for row in task1_rows})
            focus_threshold_int = thresholds[0] if thresholds else 30
        lines.append(f"- Accuracy summary below is pinned to `ID{focus_threshold_int}` because the current matched runtime comparison uses that split.")
        focus_rows = [row for row in task1_rows if int(row["threshold"]) == focus_threshold_int]
        comparable_rows = [row for row in focus_rows if row.get("accuracy_scope") == "full_test"]
        partial_rows = [row for row in focus_rows if row.get("accuracy_scope") == "evaluated_subset_only"]
        if comparable_rows:
            ranked = sorted(comparable_rows, key=lambda row: float(row["weighted_f1"]), reverse=True)
            for row in ranked:
                lines.append(
                    f"- ID{row['threshold']} `{row['display_name']}`: weighted-F1 `{float(row['weighted_f1']):.4f}`, micro-F1 `{float(row['micro_f1']):.4f}`, "
                    f"top-1 `{float(row['exact_top1']):.4f}`, coverage `{float(row['coverage']):.4f}` on the full evaluated test surface."
                )
        if partial_rows:
            lines.append("- Partial-coverage rows for the same split are listed separately and are not ranked against the full-test rows:")
            for row in sorted(partial_rows, key=lambda item: (int(item["threshold"]), str(item["model_id"]))):
                lines.append(
                    f"- ID{row['threshold']} `{row['display_name']}`: weighted-F1 `{float(row['weighted_f1']):.4f}`, micro-F1 `{float(row['micro_f1']):.4f}`, "
                    f"top-1 `{float(row['exact_top1']):.4f}`, coverage `{float(row['coverage']):.4f}` with "
                    f"`{row.get('coverage_evaluated_queries', '')}/{row.get('coverage_total_queries', '')}` queries evaluated "
                    f"(`{row.get('accuracy_scope', '')}` / `{row.get('coverage_scope', '')}`)."
                )
    else:
        lines.append("- No Task 1 metrics were available.")
    lines.append("")
    lines.append("## Operational efficiency")
    if operational_rows:
        for row in operational_rows:
            lines.append(
                f"- `{row['model_id']}` / protocol `{row['protocol']}`: setup=`{row.get('setup_runtime_s', '')}`s, full=`{row.get('full_runtime_s', '')}`s, "
                f"train=`{row.get('total_training_runtime_s', '')}`s, test=`{row.get('test_runtime_s', '')}`s, "
                f"latency=`{row.get('per_protein_latency_ms', '')}` ms/protein, memory=`{row.get('memory_usage_gib', '')}` GiB ({row.get('memory_kind', '')}), "
                f"hardware=`{row.get('hardware_class', '') or row.get('gpu_model', '')}`, scope=`{row.get('runtime_scope', '')}`"
            )
    else:
        lines.append("- No measured runtime JSON files were found under the scratch measurement directory yet.")
    lines.append("")
    matched_rows = matched_summary.get("rows") if isinstance(matched_summary, dict) else []
    if isinstance(matched_rows, list) and matched_rows:
        lines.append("## Matched ProtoEC + CLEAN Summary")
        matched_threshold = matched_summary.get("matched_split_threshold", "")
        hardware_regime = matched_summary.get("hardware_regime", "")
        lines.append(f"- Executed/staged matched split: `ID{matched_threshold}`")
        lines.append(f"- Actual GPU regime captured so far: `{hardware_regime}`")
        lines.append("")
        lines.append("### Including Embedding Extraction")
        lines.append("")
        lines.append("| Model | Status | Accuracy scope | Evaluated / total | Coverage | Weighted-F1 | Micro-F1 | Top-1 | Precompute / embedding bucket (s) | Precompute (s) | Train (s) | Test (s) | Total incl. embedding (s) | Peak memory (GiB) | GPU |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in matched_rows:
            lines.append(
                f"| {row['display_name']} | {row['status']} | {row.get('accuracy_scope', '')} | "
                f"{row.get('coverage_evaluated_queries', '')}/{row.get('coverage_total_queries', '')} | {row.get('coverage', '')} | "
                f"{row.get('accuracy_weighted_f1', '')} | {row.get('accuracy_micro_f1', '')} | {row.get('accuracy_top1', '')} | "
                f"{row.get('embedding_runtime_s', '')} | {row.get('precompute_runtime_s', '')} | "
                f"{row.get('total_training_runtime_s', '')} | {row.get('test_runtime_s', '')} | {row.get('including_embedding_runtime_s', '')} | "
                f"{row.get('peak_memory_gib', '')} | {row.get('gpu_model', '')} |"
            )
        lines.append("")
        lines.append("### Excluding Embedding Extraction")
        lines.append("")
        lines.append("| Model | Accuracy scope | Evaluated / total | Training unit | Runtime per unit (s) | Total excl. embedding (s) | Trainable params | Checkpoint (MiB) | Coverage |")
        lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in matched_rows:
            lines.append(
                f"| {row['display_name']} | {row.get('accuracy_scope', '')} | {row.get('coverage_evaluated_queries', '')}/{row.get('coverage_total_queries', '')} | "
                f"{row.get('training_unit', '')} | {row.get('runtime_per_training_unit_s', '')} | "
                f"{row.get('excluding_embedding_runtime_s', '')} | {row.get('trainable_params', '')} | {row.get('checkpoint_size_mib', '')} | "
                f"{row.get('coverage', '')} |"
            )
        lines.append("")
        lines.append("### Embedding Cost Breakdown")
        lines.append("")
        for row in matched_rows:
            if row.get("model_id") == "protoec":
                lines.append(
                    f"- `{row['display_name']}`: backbone=`{row.get('embedding_backbone', '')}`, precompute_or_embedding_bucket_s=`{row.get('embedding_runtime_s', '')}`, "
                    f"embedding_memory_gib=`{row.get('embedding_memory_gib', '')}`, embedding_cache_size_mib=`{row.get('embedding_cache_size_mib', '')}`, "
                    f"scope=`{row.get('embedding_runtime_scope', '')}`. This bucket is derived as `Protocol A full - Protocol B train - Protocol B test`, so it includes embedding plus other full-path adapter/precompute overhead."
                )
            elif row.get("model_id") == "clean":
                lines.append(
                    f"- `{row['display_name']}`: backbone=`{row.get('embedding_backbone', '')}`, partial_embedding_runtime_s=`{row.get('embedding_runtime_s', '')}`, "
                    f"embedding_memory_gib=`{row.get('embedding_memory_gib', '')}`, embedding_cache_size_mib=`{row.get('embedding_cache_size_mib', '')}`, "
                    f"scope=`{row.get('embedding_runtime_scope', '')}`. This currently covers orphan/test-side embedding only; the train-set embeddings were reused from cache."
                )
            else:
                lines.append(
                    f"- `{row['display_name']}`: backbone=`{row.get('embedding_backbone', '')}`, embedding_runtime_s=`{row.get('embedding_runtime_s', '')}`, "
                    f"embedding_memory_gib=`{row.get('embedding_memory_gib', '')}`, embedding_cache_size_mib=`{row.get('embedding_cache_size_mib', '')}`, "
                    f"scope=`{row.get('embedding_runtime_scope', '')}`."
                )
            if row.get("notes"):
                lines.append(f"- `{row['display_name']}` note: {row['notes']}")
        if any(row.get("accuracy_scope") == "evaluated_subset_only" for row in matched_rows):
            lines.append("- Partial-coverage accuracy rows are kept visible for transparency, but they should not be read as direct full-test replacements for the full-coverage comparator.")
        lines.append("")
    lines.append("## Measurement gaps")
    gaps = []
    if not gpu_probe:
        gaps.append("hardware preflight did not capture in-container gpu_probe rows; GPU model is currently inferred from PBS host context instead")
    if any((row.get("memory_kind") == "gpu") and not row.get("gpu_model") for row in operational_rows):
        gaps.append("operational summary rows are missing explicit gpu_model strings even when GPU memory was sampled")
    if any(row.get("accuracy_scope") == "evaluated_subset_only" for row in matched_rows if isinstance(row, dict)):
        gaps.append("ProtoEC currently exposes only subset-evaluated Task 1 accuracy in the matched paper-facing comparison; runtime remains comparable, but accuracy is not yet a direct full-test replacement for CLEAN")
    if any(str(row.get("flops_status", "")).startswith("approx") for row in flops_rows):
        gaps.append("current ProtoEC FLOPs are temporary analytical head-only placeholders; final reporting should prefer profiler-based model-core FLOPs for ProtoEC/CLEAN and keep other baselines as NA")
    if any((str(row.get("model_id", "")) == "clean") and (str(row.get("flops_status", "")) == "not_yet_profiled_model_core") for row in flops_rows):
        gaps.append("CLEAN model-core FLOPs are not yet profiled in this bundle")
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
    ap.add_argument(
        "--measurement-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional measurement directories to merge into the summary alongside scratch_root/measurements.",
    )
    ap.add_argument(
        "--clean-bundle-root",
        type=Path,
        default=None,
        help="Optional durable CLEAN bundle root (for example EC-Bench/local_artifacts/clean_id30_retrained_20260329).",
    )
    ap.add_argument(
        "--protoec-flops-json",
        type=Path,
        default=None,
        help="Optional profiler-based ProtoEC model-core FLOPs JSON. When present, it overrides manifest placeholders.",
    )
    ap.add_argument(
        "--clean-flops-json",
        type=Path,
        default=None,
        help="Optional profiler-based CLEAN model-core FLOPs JSON. When present, it overrides placeholder FLOPs rows.",
    )
    args = ap.parse_args()

    out_root = args.out_root.resolve()
    scratch_root = args.scratch_root.resolve()
    measurements_root = scratch_root / "measurements"
    protoec_manifest_path = args.protoec_manifest or (scratch_root / "protoec" / "adapter_manifest.json")
    ensure_dir(out_root)
    ensure_dir(scratch_root)
    measurement_roots = [measurements_root]
    for extra_root in args.measurement_dir:
        resolved = extra_root.resolve()
        if resolved not in measurement_roots:
            measurement_roots.append(resolved)

    manifest_rows = _load_manifest((args.ecbench_root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest)
    hardware_manifest = _resolve_hardware_manifest(scratch_root)
    hardware_manifest["container_image"] = hardware_manifest.get("container_image") or os.environ.get("BENCHMARK_CONTAINER_IMAGE", "")
    data_manifest = _resolve_data_manifest(args.ecbench_root.resolve(), args.data_prep_manifest)
    clean_bundle = _load_clean_bundle(args.clean_bundle_root)

    task1_rows = _task1_rows_from_official(args.ecbench_root.resolve(), manifest_rows)
    task1_rows.extend(_task1_rows_from_protoec(protoec_manifest_path))
    task1_rows = sorted(task1_rows, key=lambda row: (int(row["threshold"]), str(row["model_id"])))

    measurements = _load_measurements(measurement_roots)
    operational_rows = _operational_summary(measurements)
    artifact_rows = _artifact_inventory(args.ecbench_root.resolve(), manifest_rows, protoec_manifest_path, clean_bundle)
    params_rows = _params_storage_summary(manifest_rows, artifact_rows, protoec_manifest_path, clean_bundle)
    protoec_profiler_payload = _resolve_optional_json(
        args.protoec_flops_json,
        [
            scratch_root / "protoec" / "profiling" / "model_core_flops.json",
            scratch_root / "profiling" / "protoec_model_core_flops.json",
        ],
    )
    clean_profiler_payload = _resolve_optional_json(
        args.clean_flops_json,
        [
            scratch_root / "clean" / "profiling" / "model_core_flops.json",
            scratch_root / "profiling" / "clean_model_core_flops.json",
        ],
    )
    flops_rows = _flops_summary(
        manifest_rows,
        protoec_manifest_path,
        protoec_profiler_payload=protoec_profiler_payload,
        clean_profiler_payload=clean_profiler_payload,
        clean_bundle=clean_bundle,
    )
    matched_summary = _matched_protoec_clean_summary(
        ecbench_root=args.ecbench_root.resolve(),
        measurements=measurements,
        protoec_manifest_path=protoec_manifest_path,
        clean_bundle=clean_bundle,
    )

    raw_scratch_link = out_root / "raw_scratch"
    if raw_scratch_link.exists() or raw_scratch_link.is_symlink():
        raw_scratch_link.unlink()
    raw_scratch_link.symlink_to(scratch_root, target_is_directory=True)

    write_json(out_root / "hardware_manifest.json", hardware_manifest)
    write_json(out_root / "data_prep_manifest.json", data_manifest)
    write_json(out_root / "matched_protoec_clean_summary.json", matched_summary)
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
            "coverage_scope",
            "accuracy_scope",
            "accuracy_direct_compare_ok",
            "coverage_total_queries",
            "coverage_evaluated_queries",
            "coverage_missing_queries",
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
            "setup_phase_status",
            "training_phase_status",
            "test_phase_status",
            "full_phase_status",
            "setup_runtime_s",
            "memory_usage_gib",
            "memory_kind",
            "run_time_per_epoch_s",
            "run_time_per_episode_s",
            "total_training_runtime_s",
            "test_runtime_s",
            "full_runtime_s",
            "per_protein_latency_ms",
            "split_threshold",
            "runtime_scope",
            "threads_requested",
            "hardware_class",
            "notes",
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
    matched_rows = matched_summary.get("rows") if isinstance(matched_summary, dict) else []
    write_csv(
        out_root / "matched_protoec_clean_including_embedding.csv",
        _select_fields(matched_rows if isinstance(matched_rows, list) else [], [
            "model_id",
            "display_name",
            "split_threshold",
            "status",
            "accuracy_weighted_f1",
            "accuracy_micro_f1",
            "accuracy_top1",
            "accuracy_scope",
            "accuracy_direct_compare_ok",
            "coverage",
            "no_pred_rate",
            "coverage_scope",
            "coverage_total_queries",
            "coverage_evaluated_queries",
            "coverage_missing_queries",
            "embedding_backbone",
            "embedding_runtime_s",
            "embedding_runtime_scope",
            "precompute_runtime_s",
            "total_training_runtime_s",
            "training_unit",
            "runtime_per_training_unit_s",
            "test_runtime_s",
            "including_embedding_runtime_s",
            "embedding_memory_gib",
            "peak_memory_gib",
            "gpu_model",
            "node_name",
            "container_image",
            "pbs_jobids",
            "notes",
        ]),
        [
            "model_id",
            "display_name",
            "split_threshold",
            "status",
            "accuracy_weighted_f1",
            "accuracy_micro_f1",
            "accuracy_top1",
            "accuracy_scope",
            "accuracy_direct_compare_ok",
            "coverage",
            "no_pred_rate",
            "coverage_scope",
            "coverage_total_queries",
            "coverage_evaluated_queries",
            "coverage_missing_queries",
            "embedding_backbone",
            "embedding_runtime_s",
            "embedding_runtime_scope",
            "precompute_runtime_s",
            "total_training_runtime_s",
            "training_unit",
            "runtime_per_training_unit_s",
            "test_runtime_s",
            "including_embedding_runtime_s",
            "embedding_memory_gib",
            "peak_memory_gib",
            "gpu_model",
            "node_name",
            "container_image",
            "pbs_jobids",
            "notes",
        ],
    )
    write_csv(
        out_root / "matched_protoec_clean_excluding_embedding.csv",
        _select_fields(matched_rows if isinstance(matched_rows, list) else [], [
            "model_id",
            "display_name",
            "split_threshold",
            "status",
            "accuracy_weighted_f1",
            "accuracy_micro_f1",
            "accuracy_top1",
            "accuracy_scope",
            "accuracy_direct_compare_ok",
            "coverage",
            "coverage_scope",
            "coverage_total_queries",
            "coverage_evaluated_queries",
            "coverage_missing_queries",
            "trainable_params",
            "total_params",
            "checkpoint_size_mib",
            "embedding_cache_size_mib",
            "training_unit",
            "runtime_per_training_unit_s",
            "excluding_embedding_runtime_s",
            "gpu_model",
            "node_name",
            "container_image",
            "pbs_jobids",
            "notes",
        ]),
        [
            "model_id",
            "display_name",
            "split_threshold",
            "status",
            "accuracy_weighted_f1",
            "accuracy_micro_f1",
            "accuracy_top1",
            "accuracy_scope",
            "accuracy_direct_compare_ok",
            "coverage",
            "coverage_scope",
            "coverage_total_queries",
            "coverage_evaluated_queries",
            "coverage_missing_queries",
            "trainable_params",
            "total_params",
            "checkpoint_size_mib",
            "embedding_cache_size_mib",
            "training_unit",
            "runtime_per_training_unit_s",
            "excluding_embedding_runtime_s",
            "gpu_model",
            "node_name",
            "container_image",
            "pbs_jobids",
            "notes",
        ],
    )
    report = _render_report(out_root, hardware_manifest, data_manifest, task1_rows, operational_rows, params_rows, flops_rows, matched_summary)
    (out_root / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"[run_benchmark] wrote report bundle to {out_root}")


if __name__ == "__main__":
    main()
