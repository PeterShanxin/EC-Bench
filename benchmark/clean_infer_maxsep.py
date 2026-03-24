from __future__ import annotations

import argparse
import csv
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from benchmark._common import EvalResult, file_size_mib, evaluate_multilabel_top1, write_json
from benchmark.clean_runtime_common import runtime_context
from CLEAN.app.src.CLEAN.distance_map import get_dist_map_test
from CLEAN.app.src.CLEAN.evaluate import write_max_sep_choices
from CLEAN.app.src.CLEAN.model import LayerNormNet
from CLEAN.app.src.CLEAN.utils import get_ec_id_dict


def _format_embedding(payload: object) -> torch.Tensor:
    if isinstance(payload, dict):
        if "mean_representations" in payload:
            return payload["mean_representations"][33]
        if "representations" in payload:
            return payload["representations"][33].mean(0)
    if isinstance(payload, torch.Tensor):
        return payload
    raise TypeError(f"Unsupported embedding payload type: {type(payload)!r}")


def _load_embedding(path: Path) -> torch.Tensor:
    return _format_embedding(torch.load(path, map_location="cpu")).unsqueeze(0)


def _load_test_embeddings(embed_dir: Path, id_ec_test: Dict[str, List[str]], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    missing: List[str] = []
    for entry in id_ec_test.keys():
        path = embed_dir / f"{entry}.pt"
        if not path.exists():
            if len(missing) < 20:
                missing.append(entry)
            continue
        tensors.append(_load_embedding(path))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)}+ test embeddings under {embed_dir}; examples: {missing}")
    return torch.cat(tensors).to(device=device, dtype=dtype)


def _parse_prediction_row(row: List[str]) -> List[str]:
    out: List[str] = []
    for token in row[1:]:
        token = token.strip()
        if not token:
            continue
        label = token.split("/", 1)[0]
        if label.startswith("EC:"):
            label = label[3:]
        if label:
            out.append(label)
    return out


def _metrics_payload(result: EvalResult) -> Dict[str, float]:
    return {
        "exact_top1": result.exact_top1,
        "micro_precision": result.micro_precision,
        "micro_recall": result.micro_recall,
        "micro_f1": result.micro_f1,
        "macro_precision": result.macro_precision,
        "macro_recall": result.macro_recall,
        "macro_f1": result.macro_f1,
        "weighted_precision": result.weighted_precision,
        "weighted_recall": result.weighted_recall,
        "weighted_f1": result.weighted_f1,
        "coverage": result.coverage,
        "no_pred_rate": result.no_pred_rate,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run CLEAN max-separation inference against EC-Bench test data.")
    ap.add_argument("--ecbench-root", type=Path, required=True)
    ap.add_argument("--train-data", default="train_ec_100")
    ap.add_argument("--test-data", default="test_ec_100")
    ap.add_argument("--embedding-dir", type=Path, required=True)
    ap.add_argument("--esm-matrix-path", type=Path, required=True)
    ap.add_argument("--checkpoint-path", type=Path, required=True)
    ap.add_argument("--predictions-prefix", type=Path, required=True)
    ap.add_argument("--metrics-out", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--out-dim", type=int, default=256)
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    predictions_csv = args.predictions_prefix.with_name(args.predictions_prefix.name + "_maxsep.csv")
    if args.resume and predictions_csv.exists() and args.metrics_out.exists():
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "skipped_resume",
            "predictions_csv": str(predictions_csv),
            "metrics_out": str(args.metrics_out),
            "runtime_context": runtime_context(),
        }
        write_json(args.manifest_out, payload)
        print(json.dumps(payload, indent=2))
        return

    use_cuda = torch.cuda.is_available()
    if args.require_cuda and not use_cuda:
        raise SystemExit("CUDA is required for CLEAN inference but torch.cuda.is_available() is false")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32

    train_csv = args.ecbench_root / "data" / f"{args.train_data}.csv"
    test_csv = args.ecbench_root / "data" / f"{args.test_data}.csv"
    id_ec_train, ec_id_dict_train = get_ec_id_dict(str(train_csv))
    id_ec_test, _ = get_ec_id_dict(str(test_csv))

    model = LayerNormNet(args.hidden_dim, args.out_dim, device, dtype)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    esm_emb = pickle.load(args.esm_matrix_path.open("rb"))
    if not isinstance(esm_emb, torch.Tensor):
        raise TypeError(f"Expected tensor in {args.esm_matrix_path}, got {type(esm_emb)!r}")
    esm_emb = esm_emb.to(device=device, dtype=dtype)
    emb_train = model(esm_emb)
    emb_test = model(_load_test_embeddings(args.embedding_dir, id_ec_test, device, dtype))
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    eval_df = pd.DataFrame.from_dict(eval_dist)

    args.predictions_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_max_sep_choices(eval_df, str(args.predictions_prefix))

    true_by_id: Dict[str, List[str]] = {}
    with test_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            true_by_id[row[0]] = [item.strip() for item in row[2].split(",") if item.strip()]

    pred_by_id: Dict[str, List[str]] = {}
    with predictions_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            pred_by_id[row[0]] = _parse_prediction_row(row)

    ordered_ids = list(true_by_id.keys())
    true_sets = [true_by_id[entry] for entry in ordered_ids]
    pred_sets = [pred_by_id.get(entry, []) for entry in ordered_ids]
    metrics = _metrics_payload(evaluate_multilabel_top1(true_sets, pred_sets))
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.metrics_out,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "query_count": len(ordered_ids),
            "checkpoint_path": str(args.checkpoint_path),
            "predictions_csv": str(predictions_csv),
            **metrics,
        },
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "train_data": args.train_data,
        "test_data": args.test_data,
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_size_mib": file_size_mib(args.checkpoint_path),
        "predictions_csv": str(predictions_csv),
        "metrics_out": str(args.metrics_out),
        "query_count": len(ordered_ids),
        "runtime_context": runtime_context(),
    }
    write_json(args.manifest_out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
