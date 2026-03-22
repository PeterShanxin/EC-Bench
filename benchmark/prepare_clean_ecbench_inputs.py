from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


HEADER = ["Entry", "Sequence", "EC number"]


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_seq(row: Dict[str, str]) -> str:
    if row.get("seq"):
        return str(row["seq"])
    if row.get("sequence"):
        return str(row["sequence"])
    raise KeyError("Missing sequence column; expected `seq` or `sequence`")


def _normalize_ec(row: Dict[str, str]) -> str:
    if row.get("ec_number"):
        return str(row["ec_number"])
    if row.get("EC number"):
        return str(row["EC number"])
    raise KeyError("Missing EC column; expected `ec_number` or `EC number`")


def _write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for row in rows:
            writer.writerow([row["Entry"], row["Sequence"], row["EC number"]])
            count += 1
    return count


def _write_fasta(path: Path, rows: Iterable[Dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f">{row['Entry']}\n{row['Sequence']}\n")
            count += 1
    return count


def _convert_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "Entry": str(row["id"]),
                "Sequence": _normalize_seq(row),
                "EC number": _normalize_ec(row),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare EC-Bench cluster-100 CSV/FASTA inputs for CLEAN.")
    ap.add_argument("--ecbench-root", type=Path, default=Path.cwd())
    ap.add_argument("--threshold", type=int, default=100)
    ap.add_argument("--manifest-out", type=Path, default=None)
    args = ap.parse_args()

    ecbench_root = args.ecbench_root.resolve()
    data_root = ecbench_root / "data"
    clean_root = ecbench_root / "CLEAN" / "data"
    cluster_root = data_root / f"cluster-{args.threshold}"

    train_rows = _convert_rows(_read_rows(cluster_root / "train_ec.csv"))
    test_rows = _convert_rows(_read_rows(cluster_root / "test_ec.csv"))

    train_name = f"train_ec_{args.threshold}"
    test_name = f"test_ec_{args.threshold}"

    outputs = {
        "train_csv_data_root": data_root / f"{train_name}.csv",
        "test_csv_data_root": data_root / f"{test_name}.csv",
        "train_csv_clean_root": clean_root / f"{train_name}.csv",
        "test_csv_clean_root": clean_root / f"{test_name}.csv",
        "train_fasta_clean_root": clean_root / f"{train_name}.fasta",
        "test_fasta_clean_root": clean_root / f"{test_name}.fasta",
    }

    counts = {
        "train_csv_data_root": _write_csv(outputs["train_csv_data_root"], train_rows),
        "test_csv_data_root": _write_csv(outputs["test_csv_data_root"], test_rows),
        "train_csv_clean_root": _write_csv(outputs["train_csv_clean_root"], train_rows),
        "test_csv_clean_root": _write_csv(outputs["test_csv_clean_root"], test_rows),
        "train_fasta_clean_root": _write_fasta(outputs["train_fasta_clean_root"], train_rows),
        "test_fasta_clean_root": _write_fasta(outputs["test_fasta_clean_root"], test_rows),
    }

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ecbench_root": str(ecbench_root),
        "threshold": int(args.threshold),
        "train_name": train_name,
        "test_name": test_name,
        "counts": counts,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    manifest_out = args.manifest_out or (ecbench_root / "CLEAN" / "data" / f"{train_name}_manifest.json")
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
