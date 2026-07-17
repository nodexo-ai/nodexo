"""
CPU-side Philox 4x32-10 PRF for deterministic matrix generation.

Matches the CUDA implementation in fused_pipeline.cu exactly, so the
verifier can regenerate sub-matrices from a seed without a GPU.

Philox 4x32-10: a counter-based RNG from Salmon et al. (2011).
Each (key, counter) pair deterministically produces 4 × 32-bit outputs.
"""

from __future__ import annotations

import os

from zkgemm.field import P
from zkgemm.native_import import import_zkgemm_cuda

try:
    _zk = import_zkgemm_cuda()
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False

# 32-bit mask
_U32 = 0xFFFFFFFF
_PYTHON_FALLBACK_MAX_VALUES = int(os.environ.get(
    "NODEXO_PYTHON_PRF_FALLBACK_MAX_VALUES",
    "65536",
))


def native_prf_available() -> bool:
    return bool(_HAS_NATIVE)


def _require_native_prf_np(name: str):
    fn = getattr(_zk, name, None)
    if fn is None:
        raise RuntimeError(
            "zkgemm_cuda artifact is missing optimized native PRF slice support. "
            "Rebuild or reinstall the current release artifact before running proofs."
        )
    return fn


def _allow_python_prf_fallback(num_values: int) -> bool:
    if os.environ.get("NODEXO_ALLOW_PYTHON_PRF_FALLBACK") == "1":
        return True
    if os.environ.get("NODEXO_REQUIRE_NATIVE_ZKGEMM") == "1":
        return False
    return int(num_values) <= _PYTHON_FALLBACK_MAX_VALUES


def _require_prf_fallback_budget(num_values: int) -> None:
    if _allow_python_prf_fallback(num_values):
        return
    raise RuntimeError(
        "zkgemm_cuda native PRF slice support is required for production-sized "
        "proof verification. Run scripts/setup_validator.sh or install the "
        "current zkgemm_cuda artifact before verifying proofs."
    )


def _philox_round(c0: int, c1: int, c2: int, c3: int,
                  k0: int, k1: int) -> tuple[int, int, int, int, int, int]:
    """One Philox 4x32 round. Returns (new_c0..c3, new_k0, new_k1)."""
    M0 = 0xD256D193
    M1 = 0x9E3779B9

    p0 = M0 * c0  # 64-bit product
    p1 = M1 * c2

    hi0 = (p0 >> 32) & _U32
    lo0 = p0 & _U32
    hi1 = (p1 >> 32) & _U32
    lo1 = p1 & _U32

    new_c0 = (hi1 ^ c1 ^ k0) & _U32
    new_c1 = lo1 & _U32
    new_c2 = (hi0 ^ c3 ^ k1) & _U32
    new_c3 = lo0 & _U32

    new_k0 = (k0 + 0x9E3779B9) & _U32
    new_k1 = (k1 + 0xBB67AE85) & _U32

    return new_c0, new_c1, new_c2, new_c3, new_k0, new_k1


def philox_10(c0: int, c1: int, c2: int, c3: int,
              k0: int, k1: int) -> tuple[int, int, int, int]:
    """Philox 4x32-10: 10 rounds. Returns (c0, c1, c2, c3)."""
    for _ in range(10):
        c0, c1, c2, c3, k0, k1 = _philox_round(c0, c1, c2, c3, k0, k1)
    return c0, c1, c2, c3


def philox_to_field(seed: int, matrix_id: int, row: int, col: int) -> int:
    """Generate one Mersenne field element from Philox.

    Matches the CUDA philox_to_field exactly:
      key = (seed_lo ^ (0x41414141 + matrix_id),
             seed_hi ^ (0xA1A1A1A1 + matrix_id))
      ctr = (row, col, matrix_id, 0)
      output: ((c1 << 32) | c0) & (2^61 - 1), with reduction if == P.
    """
    seed_lo = seed & _U32
    seed_hi = (seed >> 32) & _U32

    k0 = (seed_lo ^ ((0x41414141 + matrix_id) & _U32)) & _U32
    k1 = (seed_hi ^ ((0xA1A1A1A1 + matrix_id) & _U32)) & _U32

    c0_in = row & _U32
    c1_in = col & _U32
    c2_in = matrix_id & _U32
    c3_in = 0

    c0, c1, c2, c3 = philox_10(c0_in, c1_in, c2_in, c3_in, k0, k1)

    val = ((c1 << 32) | c0) & P  # mask to 61 bits
    if val == P:
        val = 0
    return val


def generate_matrix(seed: int, n: int, matrix_id: int = 0) -> list[int]:
    """Generate an n×n matrix of field elements, returned as flat row-major list.

    Matches zkgemm_cuda.philox_generate_field(seed, n, matrix_id).
    """
    flat = [0] * (n * n)
    for i in range(n):
        for j in range(n):
            flat[i * n + j] = philox_to_field(seed, matrix_id, i, j)
    return flat


def generate_rows(seed: int, matrix_id: int,
                  row_start: int, num_rows: int, n_cols: int) -> list[int]:
    """Generate a sub-matrix: rows [row_start, row_start+num_rows) of an n_cols-wide matrix.

    Returns flat row-major list of length num_rows * n_cols.
    Used by the verifier to regenerate only A_sub (bs rows × n cols).
    """
    if _HAS_NATIVE:
        return _require_native_prf_np("native_generate_rows_np")(
            seed, matrix_id, row_start, num_rows, n_cols
        )
    _require_prf_fallback_budget(num_rows * n_cols)
    flat = [0] * (num_rows * n_cols)
    for i in range(num_rows):
        for j in range(n_cols):
            flat[i * n_cols + j] = philox_to_field(seed, matrix_id, row_start + i, j)
    return flat


def generate_cols(seed: int, matrix_id: int,
                  col_start: int, num_cols: int, n_rows: int) -> list[int]:
    """Generate a sub-matrix: cols [col_start, col_start+num_cols) of an n_rows-tall matrix.

    Returns flat row-major list of length n_rows * num_cols.
    Used by the verifier to regenerate only B_sub (n rows × bs cols).
    """
    if _HAS_NATIVE:
        return _require_native_prf_np("native_generate_cols_np")(
            seed, matrix_id, col_start, num_cols, n_rows
        )
    _require_prf_fallback_budget(n_rows * num_cols)
    flat = [0] * (n_rows * num_cols)
    for i in range(n_rows):
        for j in range(num_cols):
            flat[i * num_cols + j] = philox_to_field(seed, matrix_id, i, col_start + j)
    return flat
