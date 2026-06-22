#!/usr/bin/env python3
"""Validate the shipped zkgemm_cuda extension against this Python/Torch ABI."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = ROOT / "zkgemm" / "cuda"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda-smoke",
        action="store_true",
        help="also run a tiny CUDA call when a GPU is available",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(CUDA_DIR))

    try:
        import torch
    except Exception as exc:
        print(f"torch import failed: {exc}", file=sys.stderr)
        return 1

    try:
        ext = importlib.import_module("zkgemm_cuda")
    except Exception as exc:
        print(f"zkgemm_cuda import failed: {exc}", file=sys.stderr)
        return 1

    try:
        rows = ext.native_generate_rows(12345, 0, 0, 2, 4)
        if len(rows) != 8:
            raise RuntimeError(f"native_generate_rows returned {len(rows)} values")
        rows_np = ext.native_generate_rows_np(12345, 0, 0, 2, 4)
        if len(rows_np) != 8:
            raise RuntimeError(f"native_generate_rows_np returned {len(rows_np)} values")
        cols_np = ext.native_generate_cols_np(12345, 1, 0, 2, 4)
        if len(cols_np) != 8:
            raise RuntimeError(f"native_generate_cols_np returned {len(cols_np)} values")
    except Exception as exc:
        print(f"zkgemm_cuda native smoke failed: {exc}", file=sys.stderr)
        return 1

    if args.cuda_smoke and torch.cuda.is_available():
        try:
            values = ext.philox_generate_field(12345, 8, 0, 0)
            if int(values.numel()) != 64:
                raise RuntimeError(f"philox_generate_field returned {values.numel()} values")
            torch.cuda.synchronize(0)
        except Exception as exc:
            print(f"zkgemm_cuda CUDA smoke failed: {exc}", file=sys.stderr)
            return 1

    print(
        "zkgemm_cuda ok "
        f"(python={sys.version.split()[0]} torch={torch.__version__} "
        f"torch_cuda={torch.version.cuda})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
