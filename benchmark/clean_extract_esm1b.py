from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Sequence

import esm
import torch


MAX_MODEL_TOKENS = 1022


@dataclass(frozen=True)
class SeqRecord:
    entry: str
    sequence: str


def _read_fasta(path: Path) -> List[SeqRecord]:
    records: List[SeqRecord] = []
    current_id = ""
    chunks: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id:
                    records.append(SeqRecord(current_id, "".join(chunks)))
                current_id = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
    if current_id:
        records.append(SeqRecord(current_id, "".join(chunks)))
    return records


def _truncate_sequence(seq: str, max_len: int) -> str:
    return seq[:max_len] if len(seq) > max_len else seq


def _iter_batches(records: Sequence[SeqRecord], max_tokens: int) -> Iterator[List[SeqRecord]]:
    batch: List[SeqRecord] = []
    current_tokens = 0
    for record in records:
        seq_tokens = len(record.sequence) + 2
        if batch and current_tokens + seq_tokens > max_tokens:
            yield batch
            batch = []
            current_tokens = 0
        batch.append(record)
        current_tokens += seq_tokens
    if batch:
        yield batch


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract ESM1b mean embeddings for CLEAN-compatible per-sequence .pt files.")
    ap.add_argument("--input-fasta", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-tokens", type=int, default=12000)
    ap.add_argument("--manifest-out", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    started = time.time()
    raw_records = _read_fasta(args.input_fasta)
    if args.limit and args.limit > 0:
        raw_records = raw_records[: args.limit]

    truncated = 0
    records: List[SeqRecord] = []
    for item in raw_records:
        clipped = _truncate_sequence(item.sequence, MAX_MODEL_TOKENS)
        if len(clipped) != len(item.sequence):
            truncated += 1
        records.append(SeqRecord(item.entry, clipped))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model.eval()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped_existing = 0
    for batch in _iter_batches(records, max(128, int(args.batch_tokens))):
        todo = [item for item in batch if args.overwrite or not (args.output_dir / f"{item.entry}.pt").exists()]
        skipped_existing += len(batch) - len(todo)
        if not todo:
            continue
        labels_and_strs = [(item.entry, item.sequence) for item in todo]
        _, _, tokens = batch_converter(labels_and_strs)
        tokens = tokens.to(device)
        with torch.no_grad():
            result = model(tokens, repr_layers=[33], return_contacts=False)
        reps = result["representations"][33].detach().cpu()
        for idx, item in enumerate(todo):
            seq_len = len(item.sequence)
            mean_repr = reps[idx, 1 : seq_len + 1].mean(0)
            torch.save({"mean_representations": {33: mean_repr}}, args.output_dir / f"{item.entry}.pt")
            processed += 1

    duration_s = time.time() - started
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_fasta": str(args.input_fasta),
        "output_dir": str(args.output_dir),
        "requested_limit": int(args.limit),
        "records_seen": len(raw_records),
        "processed": processed,
        "skipped_existing": skipped_existing,
        "truncated_to_1022": truncated,
        "device": str(device),
        "batch_tokens": int(args.batch_tokens),
        "duration_s": duration_s,
    }
    manifest_out = args.manifest_out or (args.output_dir / "manifest.json")
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
