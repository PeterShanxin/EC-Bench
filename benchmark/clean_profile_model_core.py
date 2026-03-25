from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from benchmark.clean_runtime_common import runtime_context
from CLEAN.app.src.CLEAN.losses import SupConHardLoss
from CLEAN.app.src.CLEAN.model import LayerNormNet
from benchmark.clean_train_supconh_resume import _get_dataloader
from CLEAN.app.src.CLEAN.utils import get_ec_id_dict


def _profiler_activities() -> List[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def _profile_callable(fn, trace_path: Optional[Path] = None) -> Tuple[int, List[Dict[str, object]], object]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=_profiler_activities(),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))
    events = prof.key_averages()
    total_flops = int(sum(int(getattr(evt, "flops", 0) or 0) for evt in events))
    top_ops: List[Dict[str, object]] = []
    for evt in sorted(events, key=lambda item: int(getattr(item, "flops", 0) or 0), reverse=True)[:20]:
        top_ops.append(
            {
                "key": str(getattr(evt, "key", "")),
                "count": int(getattr(evt, "count", 0) or 0),
                "flops": int(getattr(evt, "flops", 0) or 0),
                "cpu_time_total_us": float(getattr(evt, "cpu_time_total", 0.0) or 0.0),
                "cuda_time_total_us": float(getattr(evt, "cuda_time_total", 0.0) or 0.0),
            }
        )
    return total_flops, top_ops, result


def _format_embedding(payload: object) -> torch.Tensor:
    if isinstance(payload, dict):
        if "mean_representations" in payload:
            return payload["mean_representations"][33]
        if "representations" in payload:
            return payload["representations"][33].mean(0)
    if isinstance(payload, torch.Tensor):
        return payload
    raise TypeError(f"Unsupported embedding payload type: {type(payload)!r}")


def _load_test_embeddings(embed_dir: Path, ids: Sequence[str], limit: int) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    for entry in list(ids)[: max(1, int(limit))]:
        payload = torch.load(embed_dir / f"{entry}.pt", map_location="cpu")
        tensors.append(_format_embedding(payload).unsqueeze(0))
    return torch.cat(tensors)


def _load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _profile_train_core(args: argparse.Namespace, device: torch.device, dtype: torch.dtype, trace_dir: Optional[Path]) -> Dict[str, object]:
    os.chdir(args.ecbench_root)
    id_ec, ec_id_dict_raw = get_ec_id_dict(str(args.ecbench_root / "data" / f"{args.training_data}.csv"))
    ec_id = {key: list(ec_id_dict_raw[key]) for key in ec_id_dict_raw.keys()}
    dist_map = _load_pickle(args.distance_map_path)

    model = LayerNormNet(args.hidden_dim, args.out_dim, device, dtype)
    if args.checkpoint_path and Path(args.checkpoint_path).exists():
        state = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(state, strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    train_loader = _get_dataloader(dist_map, id_ec, ec_id, args.n_pos, args.n_neg)
    loader_len = len(train_loader)
    batch = next(iter(train_loader))

    def _train_step() -> None:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        model_emb = model(batch.to(device=device, dtype=dtype))
        loss = SupConHardLoss(model_emb, args.temp, args.n_pos)
        loss.backward()
        optimizer.step()

    train_trace = None if trace_dir is None else (trace_dir / "clean_train_core.trace.json")
    total_flops, top_ops, _ = _profile_callable(_train_step, trace_path=train_trace)
    per_epoch_flops = int(total_flops * max(1, loader_len))
    target_epochs = int(args.epoch_count or 0)
    return {
        "phase": "train",
        "training_unit": "epoch",
        "profiled_batches": 1,
        "batches_per_epoch": int(loader_len),
        "train_flops_per_epoch_or_episode": per_epoch_flops,
        "train_flops_total": int(per_epoch_flops * target_epochs) if target_epochs > 0 else "",
        "target_epoch_count": target_epochs,
        "flops_status": "profiler_model_core_train_scaled_from_profiled_batch_excludes_refresh",
        "top_ops": top_ops,
    }


def _profile_test_core(args: argparse.Namespace, device: torch.device, dtype: torch.dtype, trace_dir: Optional[Path]) -> Dict[str, object]:
    os.chdir(args.ecbench_root)
    _id_ec_train, ec_id_dict_train = get_ec_id_dict(str(args.ecbench_root / "data" / f"{args.training_data}.csv"))
    id_ec_test, _ = get_ec_id_dict(str(args.ecbench_root / "data" / f"{args.test_data}.csv"))

    model = LayerNormNet(args.hidden_dim, args.out_dim, device, dtype)
    state = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    esm_emb_train = _load_pickle(args.esm_matrix_path)
    if not isinstance(esm_emb_train, torch.Tensor):
        raise TypeError(f"Expected tensor in {args.esm_matrix_path}, got {type(esm_emb_train)!r}")
    total_train_rows = int(esm_emb_train.shape[0])
    train_batch = esm_emb_train[: max(1, min(int(args.train_projection_batch_size), total_train_rows))].to(device=device, dtype=dtype)

    test_ids = list(id_ec_test.keys())
    total_test_queries = len(test_ids)
    test_batch_cpu = _load_test_embeddings(args.embedding_dir, test_ids, int(args.test_batch_size))
    test_batch = test_batch_cpu.to(device=device, dtype=dtype)

    class_count = len(ec_id_dict_train)
    center_lookup = torch.randn((class_count, args.out_dim), device=device, dtype=dtype)

    def _train_projection_step() -> None:
        with torch.no_grad():
            _ = model(train_batch)

    def _test_projection_and_score_step() -> None:
        with torch.no_grad():
            test_emb = model(test_batch)
            for row in range(test_emb.shape[0]):
                current = test_emb[row].unsqueeze(0)
                _ = (current - center_lookup).norm(dim=1, p=2)

    train_proj_trace = None if trace_dir is None else (trace_dir / "clean_test_train_projection.trace.json")
    train_proj_flops, train_proj_ops, _ = _profile_callable(_train_projection_step, trace_path=train_proj_trace)
    per_train_row_flops = float(train_proj_flops) / float(max(1, train_batch.shape[0]))
    total_train_projection = int(per_train_row_flops * total_train_rows)

    score_trace = None if trace_dir is None else (trace_dir / "clean_test_query_score.trace.json")
    score_flops, score_ops, _ = _profile_callable(_test_projection_and_score_step, trace_path=score_trace)
    per_query_flops = float(score_flops) / float(max(1, test_batch.shape[0]))
    total_test = int(total_train_projection + (per_query_flops * total_test_queries))

    return {
        "phase": "test",
        "profiled_queries": int(test_batch.shape[0]),
        "queries_total": int(total_test_queries),
        "train_rows_total": int(total_train_rows),
        "class_count": int(class_count),
        "train_projection_flops_total": int(total_train_projection),
        "test_flops_per_protein": int(total_test / max(1, total_test_queries)),
        "test_flops_total": int(total_test),
        "flops_status": "profiler_model_core_test_scaled_from_projection_and_distance_batches",
        "top_ops": {
            "train_projection": train_proj_ops,
            "test_projection_and_distance": score_ops,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile CLEAN model-core FLOPs with torch.profiler.")
    ap.add_argument("--phase", choices=("train", "test", "both"), default="both")
    ap.add_argument("--ecbench-root", type=Path, required=True)
    ap.add_argument("--training-data", default="train_ec_100")
    ap.add_argument("--test-data", default="test_ec_100")
    ap.add_argument("--distance-map-path", type=Path, required=True)
    ap.add_argument("--esm-matrix-path", type=Path, required=True)
    ap.add_argument("--embedding-dir", type=Path, required=True)
    ap.add_argument("--checkpoint-path", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, required=True)
    ap.add_argument("--trace-dir", type=Path, default=None)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--epoch-count", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--n-pos", type=int, default=9)
    ap.add_argument("--n-neg", type=int, default=30)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--out-dim", type=int, default=256)
    ap.add_argument("--train-projection-batch-size", type=int, default=4096)
    ap.add_argument("--test-batch-size", type=int, default=512)
    ap.add_argument("--require-cuda", action="store_true")
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    if args.require_cuda and not use_cuda:
        raise SystemExit("CUDA is required for CLEAN model-core profiling but torch.cuda.is_available() is false")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32

    payload: Dict[str, object] = {
        "model_id": "clean",
        "flops_source": "torch_profiler_with_flops",
        "scope": "model_core_only",
        "runtime_context": runtime_context(),
        "checkpoint_path": str(args.checkpoint_path),
    }
    if args.phase in {"train", "both"}:
        payload.update(_profile_train_core(args, device, dtype, args.trace_dir))
    if args.phase in {"test", "both"}:
        test_payload = _profile_test_core(args, device, dtype, args.trace_dir)
        if args.phase == "test":
            payload.update(test_payload)
        else:
            payload["test"] = test_payload
            payload["test_flops_total"] = test_payload.get("test_flops_total", 0)
            payload["test_flops_per_protein"] = test_payload.get("test_flops_per_protein", 0)
            payload["phase"] = "both"
            payload["flops_status"] = "profiler_model_core_train_and_test"
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
