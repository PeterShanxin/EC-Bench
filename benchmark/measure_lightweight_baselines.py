#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from benchmark._common import ensure_dir


DEFAULT_BASELINES = ("blastp", "catfam", "priam", "ecpred")
BENCHMARK_CODE_ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{BENCHMARK_CODE_ROOT}:{existing}" if existing else str(BENCHMARK_CODE_ROOT)
    )
    return env


def _render(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _run(cmd: Sequence[str], *, cwd: Path, env: Dict[str, str] | None = None) -> None:
    print(f"[measure_lightweight_baselines] running: {_render(cmd)}", flush=True)
    subprocess.run(list(cmd), check=True, cwd=str(cwd), env=env or _subprocess_env())


def _count_fasta_records(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                total += 1
    return total


def _normalize(items: str | Iterable[str]) -> List[str]:
    raw = items.split(",") if isinstance(items, str) else list(items)
    out: List[str] = []
    for item in raw:
        value = str(item).strip().lower()
        if value and value not in out:
            out.append(value)
    return out


def _measurement_command(
    *,
    ecbench_root: Path,
    model_id: str,
    phase: str,
    protocol: str,
    json_out: Path,
    log_file: Path,
    query_count: int,
    shell_command: str,
    meta: Dict[str, object],
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "benchmark.measure_command",
        "--model-id",
        model_id,
        "--phase",
        phase,
        "--protocol",
        protocol,
        "--cwd",
        str(ecbench_root),
        "--json-out",
        str(json_out),
        "--log-file",
        str(log_file),
        "--query-count",
        str(query_count),
    ]
    for key, value in meta.items():
        cmd.extend(["--meta", f"{key}={value}"])
    cmd.extend(["--", "bash", "-lc", shell_command])
    return cmd


def _common_shell_prefix() -> str:
    return (
        "set -euo pipefail && "
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
    )


def _module_prefix(*modules: str) -> str:
    joined = " ".join(shlex.quote(module) for module in modules if module)
    if not joined:
        return _common_shell_prefix()
    return (
        "set -euo pipefail && "
        "source /app1/ebapps/ebenv_hopper.sh >/dev/null 2>&1 && "
        f"module load {joined} >/dev/null 2>&1 && "
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
    )


def _measure_shell(
    *,
    ecbench_root: Path,
    measurement_dir: Path,
    logs_dir: Path,
    model_id: str,
    phase: str,
    protocol: str,
    query_count: int,
    shell_command: str,
    meta: Dict[str, object],
) -> None:
    ensure_dir(measurement_dir)
    ensure_dir(logs_dir)
    json_out = measurement_dir / f"{model_id}_{protocol}_{phase}.json"
    log_file = logs_dir / f"{model_id}_{protocol}_{phase}.log"
    _run(
        _measurement_command(
            ecbench_root=ecbench_root,
            model_id=model_id,
            phase=phase,
            protocol=protocol,
            json_out=json_out,
            log_file=log_file,
            query_count=query_count if phase == "test" else 0,
            shell_command=shell_command,
            meta=meta,
        ),
        cwd=BENCHMARK_CODE_ROOT,
    )


def _measure_blastp(
    *,
    ecbench_root: Path,
    measurement_dir: Path,
    logs_dir: Path,
    baseline_root: Path,
    train_fasta: Path,
    test_fasta: Path,
    protocol: str,
    query_count: int,
    meta: Dict[str, object],
    diamond_module: str,
) -> None:
    db_base = baseline_root / "db" / train_fasta.stem
    results_path = baseline_root / "results" / f"{test_fasta.stem}.diamond.tsv"
    setup_shell = (
        f"{_module_prefix(diamond_module)} && "
        f"mkdir -p {shlex.quote(str(db_base.parent))} {shlex.quote(str(results_path.parent))} && "
        f"diamond makedb --threads 1 --in {shlex.quote(str(train_fasta))} --db {shlex.quote(str(db_base))}"
    )
    _measure_shell(
        ecbench_root=ecbench_root,
        measurement_dir=measurement_dir,
        logs_dir=logs_dir,
        model_id="blastp",
        phase="setup",
        protocol=protocol,
        query_count=0,
        shell_command=setup_shell,
        meta={**meta, "notes": "Diamond makedb against the ID30 train FASTA."},
    )
    test_shell = (
        f"{_module_prefix(diamond_module)} && "
        f"mkdir -p {shlex.quote(str(results_path.parent))} && "
        f"diamond blastp --threads 1 --db {shlex.quote(str(db_base))} --query {shlex.quote(str(test_fasta))} "
        f"--out {shlex.quote(str(results_path))} --outfmt 6 qseqid sseqid bitscore evalue pident length qlen slen "
        "--max-target-seqs 1"
    )
    _measure_shell(
        ecbench_root=ecbench_root,
        measurement_dir=measurement_dir,
        logs_dir=logs_dir,
        model_id="blastp",
        phase="test",
        protocol=protocol,
        query_count=query_count,
        shell_command=test_shell,
        meta={**meta, "notes": "Single-thread Diamond-BLASTp query pass on the ID30 test FASTA."},
    )


def _measure_catfam(
    *,
    ecbench_root: Path,
    measurement_dir: Path,
    logs_dir: Path,
    baseline_root: Path,
    test_fasta: Path,
    protocol: str,
    query_count: int,
    meta: Dict[str, object],
    blastplus_module: str,
) -> None:
    archive_path = baseline_root / "catfam.tar.gz"
    install_root = baseline_root / "install"
    result_path = baseline_root / "results" / f"{test_fasta.stem}.catfam.output"
    setup_shell = (
        f"{_common_shell_prefix()} && "
        f"mkdir -p {shlex.quote(str(baseline_root))} {shlex.quote(str(install_root))} && "
        f"cd {shlex.quote(str(baseline_root))} && "
        f"wget -nv -O {shlex.quote(str(archive_path))} https://bhsai.org/downloads/catfam.tar.gz && "
        f"tar -xzf {shlex.quote(str(archive_path))} -C {shlex.quote(str(install_root))}"
    )
    _measure_shell(
        ecbench_root=ecbench_root,
        measurement_dir=measurement_dir,
        logs_dir=logs_dir,
        model_id="catfam",
        phase="setup",
        protocol=protocol,
        query_count=0,
        shell_command=setup_shell,
        meta={**meta, "notes": "Fresh CatFam archive download and extraction."},
    )
    test_shell = (
        f"{_module_prefix(blastplus_module)} && "
        f"mkdir -p {shlex.quote(str(result_path.parent))} && "
        f"perl {shlex.quote(str(install_root / 'source' / 'catsearch.pl'))} "
        f"-d {shlex.quote(str(install_root / 'CatFamDB' / 'CatFam_v2.0' / 'CatFam4D99R'))} "
        f"-i {shlex.quote(str(test_fasta))} "
        f"-o {shlex.quote(str(result_path))}"
    )
    _measure_shell(
        ecbench_root=ecbench_root,
        measurement_dir=measurement_dir,
        logs_dir=logs_dir,
        model_id="catfam",
        phase="test",
        protocol=protocol,
        query_count=query_count,
        shell_command=test_shell,
        meta={**meta, "notes": "Single-thread CatFam search on the ID30 test FASTA."},
    )


def _blast_legacy_prefix(blast_module: str) -> str:
    return (
        f"{_module_prefix(blast_module)} && "
        'BLAST_ROOT="$(dirname "$(dirname "$(command -v blastall)")")" && '
        'export BLAST_ROOT'
    )


def _measure_priam(
    *,
    ecbench_root: Path,
    measurement_dir: Path,
    logs_dir: Path,
    baseline_root: Path,
    price_fasta: Path,
    test_fasta: Path,
    protocol: str,
    query_count: int,
    meta: Dict[str, object],
    blast_module: str,
) -> None:
    archive_path = baseline_root / "Distribution.zip"
    install_root = baseline_root / "install"
    results_root = baseline_root / "results"
    warmup_root = baseline_root / "warmup"
    setup_shell = (
        f"{_common_shell_prefix()} && "
        f"mkdir -p {shlex.quote(str(baseline_root))} {shlex.quote(str(install_root))} && "
        f"cd {shlex.quote(str(baseline_root))} && "
        f"wget -nv -O {shlex.quote(str(archive_path))} "
        "'https://web.archive.org/web/20160806022844/http://priam.prabi.fr/REL_MAR13/Distribution.zip' && "
        f"unzip -q {shlex.quote(str(archive_path))} -d {shlex.quote(str(install_root))} || test $? -le 1"
    )
    _measure_shell(
        ecbench_root=ecbench_root,
        measurement_dir=measurement_dir,
        logs_dir=logs_dir,
        model_id="priam",
        phase="setup",
        protocol=protocol,
        query_count=0,
        shell_command=setup_shell,
        meta={**meta, "notes": "Fresh PRIAM archive download and extraction."},
    )
    priam_jar = install_root / "PRIAM_search.jar"
    if not priam_jar.exists():
        raise SystemExit(
            "PRIAM setup finished extracting the official profiles archive, but PRIAM_search.jar is not present. "
            "The public Distribution.zip appears to bundle profiles only, so the runnable PRIAM search executable "
            "must be supplied from another source before test measurement can continue."
        )
    warmup_shell = (
        f"{_blast_legacy_prefix(blast_module)} && "
        f"mkdir -p {shlex.quote(str(warmup_root))} && "
        f"cd {shlex.quote(str(install_root))} && "
        f"java -jar PRIAM_search.jar -n warmup_price149 -i {shlex.quote(str(price_fasta))} -p PRIAM_MAR13 "
        f"-o {shlex.quote(str(warmup_root))} --pt 0 --mp 60 --cc T --bd \"$BLAST_ROOT/bin\" --np 1"
    )
    _run(["bash", "-lc", warmup_shell], cwd=ecbench_root)
    test_shell = (
        f"{_blast_legacy_prefix(blast_module)} && "
        f"mkdir -p {shlex.quote(str(results_root))} && "
        f"cd {shlex.quote(str(install_root))} && "
        f"java -jar PRIAM_search.jar -n {shlex.quote(test_fasta.stem)} -i {shlex.quote(str(test_fasta))} -p PRIAM_MAR13 "
        f"-o {shlex.quote(str(results_root))} --pt 0 --mp 60 --cc T --bd \"$BLAST_ROOT/bin\" --np 1"
    )
    _measure_shell(
        ecbench_root=ecbench_root,
        measurement_dir=measurement_dir,
        logs_dir=logs_dir,
        model_id="priam",
        phase="test",
        protocol=protocol,
        query_count=query_count,
        shell_command=test_shell,
        meta={
            **meta,
            "notes": "PRIAM test pass after an unmeasured price-149 warmup that builds the profile library.",
        },
    )


def _measure_ecpred(
    *,
    ecbench_root: Path,
    measurement_dir: Path,
    logs_dir: Path,
    baseline_root: Path,
    test_fasta: Path,
    protocol: str,
    query_count: int,
    meta: Dict[str, object],
) -> None:
    archive_path = baseline_root / "ECPred.tar.gz"
    install_root = baseline_root / "install"
    temp_root = baseline_root / "temp"
    results_root = baseline_root / "results"
    setup_shell = (
        f"{_common_shell_prefix()} && "
        f"mkdir -p {shlex.quote(str(baseline_root))} {shlex.quote(str(install_root))} && "
        f"cd {shlex.quote(str(baseline_root))} && "
        f"wget -nv -O {shlex.quote(str(archive_path))} https://goo.gl/g2tMJ4 && "
        f"tar -xf {shlex.quote(str(archive_path))} -C {shlex.quote(str(install_root))} && "
        'INSTALL_DIR="$(find '
        + shlex.quote(str(install_root))
        + ' -maxdepth 3 -type f -name ECPred.jar -printf \'%h\\n\' | head -n 1)" && '
        'test -n "$INSTALL_DIR" && '
        'cd "$INSTALL_DIR" && '
        'if [ -f runLinux.sh ]; then bash runLinux.sh; fi'
    )
    _measure_shell(
        ecbench_root=ecbench_root,
        measurement_dir=measurement_dir,
        logs_dir=logs_dir,
        model_id="ecpred",
        phase="setup",
        protocol=protocol,
        query_count=0,
        shell_command=setup_shell,
        meta={**meta, "notes": "Fresh ECPred bundle download, extraction, and setup script run."},
    )
    test_shell = (
        f"{_common_shell_prefix()} && "
        f"mkdir -p {shlex.quote(str(temp_root))} {shlex.quote(str(results_root))} && "
        'INSTALL_DIR="$(find '
        + shlex.quote(str(install_root))
        + ' -maxdepth 3 -type f -name ECPred.jar -printf \'%h\\n\' | head -n 1)" && '
        'test -n "$INSTALL_DIR" && '
        'cd "$INSTALL_DIR" && '
        f"java -jar ECPred.jar weighted {shlex.quote(str(test_fasta))} \"$INSTALL_DIR/\" {shlex.quote(str(temp_root))} {shlex.quote(str(results_root / (test_fasta.stem + '.tsv')))}"
    )
    _measure_shell(
        ecbench_root=ecbench_root,
        measurement_dir=measurement_dir,
        logs_dir=logs_dir,
        model_id="ecpred",
        phase="test",
        protocol=protocol,
        query_count=query_count,
        shell_command=test_shell,
        meta={**meta, "notes": "Weighted ECPred inference on the ID30 test FASTA."},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure lightweight operational baselines for EC-Bench Task 1.")
    ap.add_argument("--ecbench-root", type=Path, required=True)
    ap.add_argument("--scratch-root", type=Path, required=True)
    ap.add_argument("--threshold", type=int, default=30)
    ap.add_argument("--baselines", default="blastp,catfam,priam,ecpred")
    ap.add_argument("--run-tag", default="20260330_id30_single_thread")
    ap.add_argument("--diamond-module", default="DIAMOND/2.1.11-GCC-13.3.0")
    ap.add_argument("--blast-module", default="BLAST/2.2.26-Linux_x86_64")
    ap.add_argument("--blastplus-module", default="BLAST+/2.15.0-gompi-2023a")
    ap.add_argument("--hardware-class", default="cpu_login_node")
    args = ap.parse_args()

    ecbench_root = args.ecbench_root.resolve()
    scratch_root = args.scratch_root.resolve()
    baselines = _normalize(args.baselines)
    unknown = [item for item in baselines if item not in DEFAULT_BASELINES]
    if unknown:
        raise SystemExit(f"Unsupported baselines requested: {', '.join(unknown)}")

    threshold = int(args.threshold)
    train_fasta = ecbench_root / "CLEAN" / "data" / f"train_ec_{threshold}.fasta"
    test_fasta = ecbench_root / "CLEAN" / "data" / f"test_ec_{threshold}.fasta"
    price_fasta = ecbench_root / "data" / "price-149.fasta"
    for path in (train_fasta, test_fasta, price_fasta):
        if not path.exists():
            raise SystemExit(f"Required FASTA not found: {path}")

    query_count = _count_fasta_records(test_fasta)
    measurement_dir = scratch_root / "measurements"
    logs_dir = scratch_root / "logs"
    run_root = scratch_root / "lightweight_baselines" / args.run_tag
    protocol = f"id{threshold}_single_thread_cpu"
    base_meta = {
        "split_threshold": threshold,
        "runtime_scope": f"id{threshold}_single_thread_cpu_baseline",
        "threads_requested": 1,
        "hardware_class": args.hardware_class,
    }

    for model_id in baselines:
        baseline_root = run_root / model_id
        ensure_dir(baseline_root)
        if model_id == "blastp":
            _measure_blastp(
                ecbench_root=ecbench_root,
                measurement_dir=measurement_dir,
                logs_dir=logs_dir,
                baseline_root=baseline_root,
                train_fasta=train_fasta,
                test_fasta=test_fasta,
                protocol=protocol,
                query_count=query_count,
                meta=base_meta,
                diamond_module=args.diamond_module,
            )
        elif model_id == "catfam":
            _measure_catfam(
                ecbench_root=ecbench_root,
                measurement_dir=measurement_dir,
                logs_dir=logs_dir,
                baseline_root=baseline_root,
                test_fasta=test_fasta,
                protocol=protocol,
                query_count=query_count,
                meta=base_meta,
                blastplus_module=args.blastplus_module,
            )
        elif model_id == "priam":
            _measure_priam(
                ecbench_root=ecbench_root,
                measurement_dir=measurement_dir,
                logs_dir=logs_dir,
                baseline_root=baseline_root,
                price_fasta=price_fasta,
                test_fasta=test_fasta,
                protocol=protocol,
                query_count=query_count,
                meta=base_meta,
                blast_module=args.blast_module,
            )
        elif model_id == "ecpred":
            _measure_ecpred(
                ecbench_root=ecbench_root,
                measurement_dir=measurement_dir,
                logs_dir=logs_dir,
                baseline_root=baseline_root,
                test_fasta=test_fasta,
                protocol=protocol,
                query_count=query_count,
                meta=base_meta,
            )


if __name__ == "__main__":
    main()
