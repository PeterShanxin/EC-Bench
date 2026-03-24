from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch

from benchmark._common import file_size_mib, write_json
from benchmark.clean_runtime_common import runtime_context
from CLEAN.app.src.CLEAN.dataloader import MultiPosNeg_dataset_with_mine_EC, mine_hard_negative
from CLEAN.app.src.CLEAN.distance_map import get_dist_map
from CLEAN.app.src.CLEAN.losses import SupConHardLoss
from CLEAN.app.src.CLEAN.model import LayerNormNet
from CLEAN.app.src.CLEAN.utils import get_ec_id_dict, seed_everything


def _get_dataloader(dist_map: Dict[str, Dict[str, float]], id_ec: Dict[str, List[str]], ec_id: Dict[str, List[str]], n_pos: int, n_neg: int) -> torch.utils.data.DataLoader:
    params = {
        "batch_size": 6000,
        "shuffle": True,
    }
    negative = mine_hard_negative(dist_map, 100)
    train_data = MultiPosNeg_dataset_with_mine_EC(id_ec, ec_id, negative, n_pos, n_neg)
    return torch.utils.data.DataLoader(train_data, **params)


def _train_epoch(
    model: LayerNormNet,
    learning_rate: float,
    temp: float,
    n_pos: int,
    epoch: int,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dtype: torch.dtype,
    verbose: bool,
) -> float:
    model.train()
    total_loss = 0.0
    start_time = time.time()
    for batch, data in enumerate(train_loader):
        optimizer.zero_grad()
        model_emb = model(data.to(device=device, dtype=dtype))
        loss = SupConHardLoss(model_emb, temp, n_pos)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if verbose:
            ms_per_batch = (time.time() - start_time) * 1000.0
            print(
                f"| epoch {epoch:4d} | {batch:5d}/{len(train_loader):5d} batches | "
                f"lr {learning_rate:02.4f} | ms/batch {ms_per_batch:8.4f} | loss {total_loss:7.4f}"
            )
            start_time = time.time()
    return total_loss / float(batch + 1)


def _load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _save_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _save_state(
    state_out: Path,
    epoch: int,
    best_loss: float,
    best_epoch: int,
    cumulative_train_runtime_s: float,
    model: LayerNormNet,
    optimizer: torch.optim.Optimizer,
    dist_map_path: Path,
    config: Dict[str, object],
) -> None:
    state_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "last_completed_epoch": epoch,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "cumulative_train_runtime_s": cumulative_train_runtime_s,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "dist_map_path": str(dist_map_path),
            "config": config,
        },
        state_out,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Resume-safe CLEAN SupCon-Hard training wrapper for EC-Bench ID100.")
    ap.add_argument("--ecbench-root", type=Path, required=True)
    ap.add_argument("--training-data", default="train_ec_100")
    ap.add_argument("--distance-map-path", type=Path, required=True)
    ap.add_argument("--esm-matrix-path", type=Path, required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--final-model-out", type=Path, required=True)
    ap.add_argument("--state-out", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--max-epochs-this-run", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--n-pos", type=int, default=9)
    ap.add_argument("--n-neg", type=int, default=30)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--out-dim", type=int, default=256)
    ap.add_argument("--adaptive-rate", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    use_cuda = torch.cuda.is_available()
    if args.require_cuda and not use_cuda:
        raise SystemExit("CUDA is required for CLEAN SupCon-Hard training but torch.cuda.is_available() is false")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32

    os.chdir(args.ecbench_root)

    id_ec, ec_id_dict_raw = get_ec_id_dict(str(args.ecbench_root / "data" / f"{args.training_data}.csv"))
    ec_id = {key: list(ec_id_dict_raw[key]) for key in ec_id_dict_raw.keys()}

    model = LayerNormNet(args.hidden_dim, args.out_dim, device, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    start_epoch = 1
    best_loss = float("inf")
    best_epoch = 0
    cumulative_train_runtime_s = 0.0
    active_dist_map_path = args.distance_map_path

    if args.resume and args.state_out.exists():
        state = torch.load(args.state_out, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state.get("last_completed_epoch", 0)) + 1
        best_loss = float(state.get("best_loss", float("inf")))
        best_epoch = int(state.get("best_epoch", 0))
        cumulative_train_runtime_s = float(state.get("cumulative_train_runtime_s", 0.0))
        active_dist_map_path = Path(str(state.get("dist_map_path") or args.distance_map_path))
        if args.final_model_out.exists() and int(state.get("last_completed_epoch", 0)) >= int(args.epoch):
            payload = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": "skipped_resume_final_exists",
                "model_name": args.model_name,
                "target_epoch": int(args.epoch),
                "final_model_out": str(args.final_model_out),
                "final_model_size_mib": file_size_mib(args.final_model_out),
                "runtime_context": runtime_context(),
            }
            write_json(args.manifest_out, payload)
            print(json.dumps(payload, indent=2))
            return

    target_end_epoch = int(args.epoch)
    if args.max_epochs_this_run and args.max_epochs_this_run > 0:
        target_end_epoch = min(target_end_epoch, start_epoch + int(args.max_epochs_this_run) - 1)

    if start_epoch > int(args.epoch):
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "already_completed",
            "model_name": args.model_name,
            "target_epoch": int(args.epoch),
            "start_epoch": start_epoch,
            "final_model_out": str(args.final_model_out),
            "runtime_context": runtime_context(),
        }
        write_json(args.manifest_out, payload)
        print(json.dumps(payload, indent=2))
        return

    esm_emb = _load_pickle(args.esm_matrix_path)
    if not isinstance(esm_emb, torch.Tensor):
        raise TypeError(f"Expected tensor in {args.esm_matrix_path}, got {type(esm_emb)!r}")
    esm_emb = esm_emb.to(device=device, dtype=dtype)
    active_dist_map = _load_pickle(active_dist_map_path)
    train_loader = _get_dataloader(active_dist_map, id_ec, ec_id, args.n_pos, args.n_neg)

    last_completed_epoch = start_epoch - 1
    last_loss = None
    for epoch in range(start_epoch, target_end_epoch + 1):
        if epoch % args.adaptive_rate == 0:
            boundary_weights_path = args.checkpoint_dir / f"{args.model_name}_weights_epoch_{epoch}.pth"
            boundary_dist_map_path = args.checkpoint_dir / f"{args.model_name}_dist_map_epoch_{epoch}.pkl"
            boundary_weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), boundary_weights_path)
            active_dist_map = get_dist_map(ec_id_dict_raw, esm_emb, device, dtype, model=model)
            _save_pickle(boundary_dist_map_path, active_dist_map)
            active_dist_map_path = boundary_dist_map_path
            train_loader = _get_dataloader(active_dist_map, id_ec, ec_id, args.n_pos, args.n_neg)

        epoch_started = time.time()
        train_loss = _train_epoch(
            model=model,
            learning_rate=float(args.learning_rate),
            temp=float(args.temp),
            n_pos=int(args.n_pos),
            epoch=epoch,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            dtype=dtype,
            verbose=bool(args.verbose),
        )
        elapsed = time.time() - epoch_started
        cumulative_train_runtime_s += elapsed
        last_completed_epoch = epoch
        last_loss = train_loss

        if train_loss < best_loss and epoch > 0.8 * int(args.epoch):
            args.final_model_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.final_model_out)
            best_loss = train_loss
            best_epoch = epoch
            print(f"Best checkpoint updated at epoch {epoch}: loss={train_loss:.6f}")

        _save_state(
            state_out=args.state_out,
            epoch=epoch,
            best_loss=best_loss,
            best_epoch=best_epoch,
            cumulative_train_runtime_s=cumulative_train_runtime_s,
            model=model,
            optimizer=optimizer,
            dist_map_path=active_dist_map_path,
            config={
                "training_data": args.training_data,
                "target_epoch": int(args.epoch),
                "n_pos": int(args.n_pos),
                "n_neg": int(args.n_neg),
                "temp": float(args.temp),
                "adaptive_rate": int(args.adaptive_rate),
            },
        )
        print("-" * 75)
        print(f"| end of epoch {epoch:4d} | time: {elapsed:7.2f}s | training loss {train_loss:8.6f}")
        print("-" * 75)

    status = "completed" if last_completed_epoch >= int(args.epoch) else "partial"

    if status == "completed" and not args.final_model_out.exists():
        args.final_model_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.final_model_out)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "model_name": args.model_name,
        "training_data": args.training_data,
        "start_epoch": start_epoch,
        "target_epoch": int(args.epoch),
        "end_epoch_this_run": last_completed_epoch,
        "epochs_completed_this_run": max(0, last_completed_epoch - start_epoch + 1),
        "cumulative_train_runtime_s": cumulative_train_runtime_s,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "last_loss": last_loss,
        "final_model_out": str(args.final_model_out),
        "final_model_size_mib": file_size_mib(args.final_model_out),
        "state_out": str(args.state_out),
        "active_dist_map_path": str(active_dist_map_path),
        "distance_map_path": str(args.distance_map_path),
        "esm_matrix_path": str(args.esm_matrix_path),
        "runtime_context": runtime_context(),
    }
    write_json(args.manifest_out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
