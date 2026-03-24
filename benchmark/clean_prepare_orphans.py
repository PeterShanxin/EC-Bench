from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from benchmark._common import write_json


def mutate(seq: str, position: int) -> str:
    return seq[:position] + "*" + seq[position + 1 :]


def _load_train_rows(csv_path: Path) -> tuple[Dict[str, str], Dict[str, List[str]], Dict[str, Set[str]]]:
    seq_by_id: Dict[str, str] = {}
    id_ec: Dict[str, List[str]] = {}
    ec_id: Dict[str, Set[str]] = defaultdict(set)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            entry = row[0].strip()
            seq = row[1].strip()
            ecs = [item.strip() for item in row[2].split(",") if item.strip()]
            seq_by_id[entry] = seq
            id_ec[entry] = ecs
            for ec in ecs:
                ec_id[ec].add(entry)
    return seq_by_id, id_ec, ec_id


def _single_sequence_ids(id_ec: Dict[str, List[str]], ec_id: Dict[str, Set[str]]) -> tuple[int, List[str]]:
    single_ec = {ec for ec, ids in ec_id.items() if len(ids) == 1}
    single_ids = sorted({entry for entry, ecs in id_ec.items() for ec in ecs if ec in single_ec})
    return len(single_ec), single_ids


def _write_mutated_fasta(seq_by_id: Dict[str, str], single_ids: List[str], output_fasta: Path, copies_per_sequence: int) -> int:
    written = 0
    with output_fasta.open("w", encoding="utf-8") as handle:
        for entry in single_ids:
            seq = seq_by_id[entry]
            for copy_idx in range(copies_per_sequence):
                mutated = seq
                mut_rate = float(np.random.normal(0.10, 0.02, 1)[0])
                times = math.ceil(len(mutated) * mut_rate)
                for _ in range(max(0, times)):
                    position = random.randint(1, len(mutated) - 1)
                    mutated = mutate(mutated, position)
                mutated = mutated.replace("*", "<mask>")
                handle.write(f">{entry}_{copy_idx}\n")
                handle.write(mutated + "\n")
                written += 1
    return written


def _count_fasta_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare CLEAN orphan-sequence mutation FASTA for SupCon-Hard training.")
    ap.add_argument("--ecbench-root", type=Path, required=True)
    ap.add_argument("--training-data", default="train_ec_100")
    ap.add_argument("--output-fasta", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    ap.add_argument("--copies-per-sequence", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    csv_path = args.ecbench_root / "data" / f"{args.training_data}.csv"
    seq_by_id, id_ec, ec_id = _load_train_rows(csv_path)
    single_ec_count, single_ids = _single_sequence_ids(id_ec, ec_id)
    expected_records = len(single_ids) * int(args.copies_per_sequence)

    status = "written"
    actual_records = 0
    if args.resume and args.output_fasta.exists():
        actual_records = _count_fasta_records(args.output_fasta)
        if actual_records == expected_records:
            status = "skipped_resume"
        else:
            args.output_fasta.unlink()

    if status != "skipped_resume":
        args.output_fasta.parent.mkdir(parents=True, exist_ok=True)
        actual_records = _write_mutated_fasta(seq_by_id, single_ids, args.output_fasta, int(args.copies_per_sequence))

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "ecbench_root": str(args.ecbench_root),
        "training_data": args.training_data,
        "train_csv": str(csv_path),
        "output_fasta": str(args.output_fasta),
        "copies_per_sequence": int(args.copies_per_sequence),
        "single_sequence_ec_count": single_ec_count,
        "single_sequence_id_count": len(single_ids),
        "expected_records": expected_records,
        "actual_records": actual_records,
        "seed": int(args.seed),
    }
    write_json(args.manifest_out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
