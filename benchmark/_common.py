from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


OFFICIAL_FILES: Dict[int, str] = {
    30: "data/results_test_30_go_3.csv",
    100: "data/results_test_100_go_3.csv",
}

# Canonical base-roster mapping used by this benchmark harness.
# The EC-Bench release CSV contains several variant columns; we pin one column
# per paper-facing base model so the runtime report has a stable 10-model roster.
BASE_MODEL_COLUMN_MAP: Dict[str, str] = {
    "blastp": "blastp",
    "catfam": "catfam",
    "priam": "priam",
    "deepec": "deepec",
    "deepectransformer": "deepectransformer-regular",
    "ecpred": "ecpred",
    "ecrecer": "ecrecer",
    "proteinbert": "proteinbert-regular",
    "enzbert": "enzbert-regular",
    "clean": "clean-supcon",
}


@dataclass(frozen=True)
class EvalResult:
    exact_top1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    coverage: float
    no_pred_rate: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def split_labels(raw: object) -> List[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text == "-" or text.lower() == "nan":
        return []
    out: List[str] = []
    for piece in text.replace(",", ";").split(";"):
        token = piece.strip()
        if not token or token == "-":
            continue
        if ":" in token:
            token = token.split(":", 1)[0].strip()
        if token:
            out.append(token)
    return out


def top1_only(raw: object) -> List[str]:
    labels = split_labels(raw)
    return labels[:1]


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num / den)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def evaluate_multilabel_top1(true_sets: Sequence[List[str]], pred_sets: Sequence[List[str]]) -> EvalResult:
    if len(true_sets) != len(pred_sets):
        raise ValueError("true_sets and pred_sets must have equal length")
    n_rows = len(true_sets)
    if n_rows == 0:
        return EvalResult(*(0.0 for _ in range(12)))

    classes = sorted({label for labels in true_sets for label in labels} | {label for labels in pred_sets for label in labels})
    if not classes:
        return EvalResult(*(0.0 for _ in range(12)))

    per_class = []
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    total_support = 0
    exact_hits = 0
    covered = 0

    true_sets_as_set = [set(labels) for labels in true_sets]
    pred_sets_as_set = [set(labels) for labels in pred_sets]

    for pred in pred_sets_as_set:
        if pred:
            covered += 1

    for truth, pred in zip(true_sets_as_set, pred_sets):
        if pred and pred[0] in truth:
            exact_hits += 1

    for label in classes:
        tp = 0
        fp = 0
        fn = 0
        support = 0
        for truth, pred in zip(true_sets_as_set, pred_sets_as_set):
            in_true = label in truth
            in_pred = label in pred
            if in_true:
                support += 1
            if in_true and in_pred:
                tp += 1
            elif (not in_true) and in_pred:
                fp += 1
            elif in_true and (not in_pred):
                fn += 1
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _f1(precision, recall)
        per_class.append(
            {
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        total_support += support

    micro_precision = _safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = _safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = _f1(micro_precision, micro_recall)

    macro_precision = float(sum(item["precision"] for item in per_class) / len(per_class))
    macro_recall = float(sum(item["recall"] for item in per_class) / len(per_class))
    macro_f1 = float(sum(item["f1"] for item in per_class) / len(per_class))

    weighted_precision = _safe_div(sum(item["precision"] * item["support"] for item in per_class), total_support)
    weighted_recall = _safe_div(sum(item["recall"] * item["support"] for item in per_class), total_support)
    weighted_f1 = _safe_div(sum(item["f1"] * item["support"] for item in per_class), total_support)

    coverage = _safe_div(covered, n_rows)
    no_pred_rate = 1.0 - coverage
    exact_top1 = _safe_div(exact_hits, n_rows)

    return EvalResult(
        exact_top1=exact_top1,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
        coverage=coverage,
        no_pred_rate=no_pred_rate,
    )


def task1_metrics_from_rows(rows: Sequence[Dict[str, str]], pred_column: str) -> EvalResult:
    true_sets = [split_labels(row.get("ec_number", "")) for row in rows]
    pred_sets = [top1_only(row.get(pred_column, "")) for row in rows]
    return evaluate_multilabel_top1(true_sets, pred_sets)


def bytes_to_mib(size_bytes: int) -> float:
    return round(float(size_bytes) / (1024.0 * 1024.0), 6)


def file_size_mib(path: Path) -> float:
    if not path.exists():
        return 0.0
    return bytes_to_mib(path.stat().st_size)


def path_size_mib(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return file_size_mib(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            if fp.exists():
                total += fp.stat().st_size
    return bytes_to_mib(total)


def expand_existing_paths(root: Path, patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for pattern in patterns:
        for candidate in root.glob(pattern):
            if candidate.exists():
                out.append(candidate)
    deduped = sorted({str(path.resolve()) for path in out})
    return [Path(item) for item in deduped]


def round_or_blank(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{value:.{digits}f}"
