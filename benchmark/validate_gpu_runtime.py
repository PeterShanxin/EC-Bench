from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.clean_runtime_common import gpu_model_matches, runtime_context


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate in-container CUDA/GPU visibility for CLEAN runtime jobs.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--require-gpu-model", default="")
    args = ap.parse_args()

    payload = runtime_context()
    rendered = json.dumps(payload, indent=2, sort_keys=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)

    if args.require_cuda and not payload.get("torch_cuda_available", False):
        raise SystemExit(2)

    required_gpu_model = str(args.require_gpu_model or "").strip()
    if required_gpu_model and not gpu_model_matches(required_gpu_model, list(payload.get("gpu_probe") or [])):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
