from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch

from benchmark._common import write_json
from benchmark.clean_runtime_common import runtime_context
from CLEAN.app.src.CLEAN.distance_map import get_dist_map
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


def _embedding_matrix(embed_dir: Path, ec_id_dict: Dict[str, object], missing_limit: int = 20) -> tuple[torch.Tensor, List[str], int]:
    missing: List[str] = []
    tensors: List[torch.Tensor] = []
    duplicated_rows = 0
    for ec in ec_id_dict.keys():
        ids_for_query = list(ec_id_dict[ec])
        duplicated_rows += len(ids_for_query)
        for entry in ids_for_query:
            path = embed_dir / f"{entry}.pt"
            if not path.exists():
                if len(missing) < missing_limit:
                    missing.append(entry)
                continue
            tensors.append(_load_embedding(path))
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)}+ base train embeddings under {embed_dir}; examples: {missing}"
        )
    return torch.cat(tensors), missing, duplicated_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute CLEAN distance-map and ESM matrix artifacts for EC-Bench retraining.")
    ap.add_argument("--ecbench-root", type=Path, required=True)
    ap.add_argument("--training-data", default="train_ec_100")
    ap.add_argument("--embedding-dir", type=Path, required=True)
    ap.add_argument("--distance-map-out", type=Path, required=True)
    ap.add_argument("--esm-matrix-out", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--require-cuda", action="store_true")
    args = ap.parse_args()

    if args.resume and args.distance_map_out.exists() and args.esm_matrix_out.exists():
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "skipped_resume",
            "ecbench_root": str(args.ecbench_root),
            "training_data": args.training_data,
            "embedding_dir": str(args.embedding_dir),
            "distance_map_out": str(args.distance_map_out),
            "esm_matrix_out": str(args.esm_matrix_out),
            "runtime_context": runtime_context(),
        }
        write_json(args.manifest_out, payload)
        print(json.dumps(payload, indent=2))
        return

    use_cuda = torch.cuda.is_available()
    if args.require_cuda and not use_cuda:
        raise SystemExit("CUDA is required for CLEAN distance-map precompute but torch.cuda.is_available() is false")

    started = time.time()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec, ec_id_dict_raw = get_ec_id_dict(str(args.ecbench_root / "data" / f"{args.training_data}.csv"))
    ec_id_dict = {key: list(ec_id_dict_raw[key]) for key in ec_id_dict_raw.keys()}
    esm_emb, _missing, duplicated_rows = _embedding_matrix(args.embedding_dir, ec_id_dict)
    esm_emb = esm_emb.to(device=device, dtype=dtype)
    dist_map = get_dist_map(ec_id_dict, esm_emb, device, dtype)

    args.distance_map_out.parent.mkdir(parents=True, exist_ok=True)
    args.esm_matrix_out.parent.mkdir(parents=True, exist_ok=True)
    with args.distance_map_out.open("wb") as handle:
        pickle.dump(dist_map, handle)
    with args.esm_matrix_out.open("wb") as handle:
        pickle.dump(esm_emb.detach().cpu(), handle)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "ecbench_root": str(args.ecbench_root),
        "training_data": args.training_data,
        "embedding_dir": str(args.embedding_dir),
        "distance_map_out": str(args.distance_map_out),
        "esm_matrix_out": str(args.esm_matrix_out),
        "train_sequence_count": len(id_ec),
        "unique_ec_count": len(ec_id_dict),
        "embedding_matrix_rows": duplicated_rows,
        "device": str(device),
        "duration_s": time.time() - started,
        "distance_map_bytes": args.distance_map_out.stat().st_size,
        "esm_matrix_bytes": args.esm_matrix_out.stat().st_size,
        "runtime_context": runtime_context(),
    }
    write_json(args.manifest_out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
