"""
Verifiable GEMM protocol using GKR / Sumcheck.

Prover: given A, B, computes C = A * B and generates a proof.
Verifier: given A, B and a proof (containing C), checks correctness.

The verifier evaluates the multilinear extensions directly for local
proof checks.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field

from zkgemm.field import P, add, mul, from_int
from zkgemm.mle import (
    build_sumcheck_tables, matrix_mle_eval,
    build_block_sumcheck_tables, rect_matrix_mle_eval,
)
from zkgemm.sumcheck import SumcheckProof, sumcheck_prove, sumcheck_verify
from zkgemm.transcript import Transcript
from zkgemm.native_import import import_zkgemm_cuda

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    _zk = import_zkgemm_cuda()
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False


def _is_ndarray(x):
    return _HAS_NUMPY and isinstance(x, np.ndarray)


def _proof_profile_enabled() -> bool:
    return os.environ.get("ZKGEMM_PROOF_PROFILE", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _proof_profile(msg: str) -> None:
    sys.stderr.write(f"[zkgemm-proof] {msg}\n")
    sys.stderr.flush()


@dataclass
class GemmProof:
    """Proof that C = A * B for n x n matrices over F_p."""

    n: int
    C_flat: list[int]
    sumcheck_proof: SumcheckProof
    final_A_eval: int  # A_tilde(r_i, r_k)
    final_B_eval: int  # B_tilde(r_k, r_j)


def _flatten(M: list[list[int]]) -> list[int]:
    """Flatten a 2D matrix to row-major 1D list."""
    return [v for row in M for v in row]


def matmul_field(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """Standard O(n^3) matrix multiplication over F_p."""
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            a_ik = A[i][k]
            if a_ik == 0:
                continue
            for j in range(n):
                C[i][j] = add(C[i][j], mul(a_ik, B[k][j]))
    return C


def _init_transcript(A_flat: list[int], B_flat: list[int], C_flat: list[int]) -> Transcript:
    """Create and seed a transcript with the three matrices."""
    t = Transcript(b"zkgemm-v1")
    t.absorb_many(b"A", A_flat)
    t.absorb_many(b"B", B_flat)
    t.absorb_many(b"C", C_flat)
    return t


class GemmProver:
    """Generates a proof that C = A * B."""

    @staticmethod
    def prove(A: list[list[int]], B: list[list[int]]) -> GemmProof:
        """Produce a verifiable GEMM proof.

        Args:
            A: n x n matrix of field elements.
            B: n x n matrix of field elements.

        Returns:
            GemmProof containing C and the sumcheck transcript.
        """
        n = len(A)
        m = int(math.log2(n))
        assert n == (1 << m), "Matrix dimension must be a power of 2"
        assert all(len(row) == n for row in A)
        assert len(B) == n and all(len(row) == n for row in B)

        # 1. Compute C = A * B
        C = matmul_field(A, B)

        # 2. Flatten
        A_flat = _flatten(A)
        B_flat = _flatten(B)
        C_flat = _flatten(C)

        # 3. Initialise Fiat-Shamir transcript
        transcript = _init_transcript(A_flat, B_flat, C_flat)

        # 4. Squeeze random evaluation point (r_i, r_j)
        r_i = transcript.squeeze_n(b"r_i", m)
        r_j = transcript.squeeze_n(b"r_j", m)

        # 5. Evaluate C_tilde at the random point
        claimed_sum = matrix_mle_eval(C_flat, n, r_i, r_j)

        # 6. Build sumcheck tables
        T_A, T_B = build_sumcheck_tables(A_flat, B_flat, n, r_i, r_j)

        # 7. Run sumcheck prover
        sc_proof, challenges = sumcheck_prove(T_A, T_B, claimed_sum, transcript)

        # 8. Final evaluations (r_k = challenges from sumcheck)
        r_k = challenges
        final_A = matrix_mle_eval(A_flat, n, r_i, r_k)
        final_B = matrix_mle_eval(B_flat, n, r_k, r_j)

        return GemmProof(
            n=n,
            C_flat=C_flat,
            sumcheck_proof=sc_proof,
            final_A_eval=final_A,
            final_B_eval=final_B,
        )


class GemmVerifier:
    """Verifies a proof that C = A * B."""

    @staticmethod
    def verify(A: list[list[int]], B: list[list[int]], proof: GemmProof) -> bool:
        """Verify a GEMM proof.

        The verifier evaluates the multilinear extensions directly.

        Args:
            A: n x n matrix of field elements.
            B: n x n matrix of field elements.
            proof: GemmProof produced by the prover.

        Returns:
            True if the proof is valid, False otherwise.
        """
        n = proof.n
        m = int(math.log2(n))

        A_flat = _flatten(A)
        B_flat = _flatten(B)
        C_flat = proof.C_flat

        # 1. Rebuild transcript identically
        transcript = _init_transcript(A_flat, B_flat, C_flat)

        # 2. Squeeze the same random point
        r_i = transcript.squeeze_n(b"r_i", m)
        r_j = transcript.squeeze_n(b"r_j", m)

        # 3. Compute expected C_tilde(r_i, r_j) from the claimed C
        expected_sum = matrix_mle_eval(C_flat, n, r_i, r_j)

        # 4. Check claimed sum matches
        if expected_sum != proof.sumcheck_proof.claimed_sum:
            return False

        # 5. Run sumcheck verifier
        ok, final_claim, challenges = sumcheck_verify(proof.sumcheck_proof, transcript)
        if not ok:
            return False

        # 6. Compute the MLE evaluations at the final point
        r_k = challenges
        a_val = matrix_mle_eval(A_flat, n, r_i, r_k)
        b_val = matrix_mle_eval(B_flat, n, r_k, r_j)

        # 7. Final check: the sumcheck's final claim must equal A_tilde * B_tilde
        if final_claim != mul(a_val, b_val):
            return False

        return True


# =====================================================================
#  Block ZkGEMM — proves a single bs×bs block of C = A × B
# =====================================================================

@dataclass
class BlockGemmProof:
    """Proof that C_block = A_sub × B_sub for one block.

    A_sub is bs × n_inner, B_sub is n_inner × bs, C_block is bs × bs.
    The sumcheck runs over the inner dimension k ∈ {0,1}^m_inner.

    C_block is NOT in the proof — the Merkle tree commits to its hash.
    The proof binds to the committed block via leaf_hash in the transcript.

    spot_A/spot_B: Optional PRF spot-check entries. Each entry is
    (local_row, local_col, value) sampled at Fiat-Shamir-derived positions.
    The verifier regenerates these from the PRF seed and checks equality,
    probabilistically confirming the prover used the correct A/B matrices.
    """

    bs: int
    n_inner: int
    sumcheck_proof: SumcheckProof
    final_A_eval: int   # A_tilde(r_a, r_k)
    final_B_eval: int   # B_tilde(r_k, r_b)
    spot_A: list[tuple[int, int, int]] = field(default_factory=list)
    spot_B: list[tuple[int, int, int]] = field(default_factory=list)


def _init_block_transcript(
    leaf_hash: bytes,
    seed: int, n: int, bi: int, bj: int,
) -> Transcript:
    """Create and seed a transcript for a block proof.

    Binds to (seed, n, block position, committed leaf hash).
    C_block is NOT absorbed — only its 32-byte leaf hash.
    """
    t = Transcript(b"zkgemm-block-v3")
    t.absorb(b"seed", seed)
    t.absorb(b"n", n)
    t.absorb(b"bi", bi)
    t.absorb(b"bj", bj)
    t.absorb_bytes(b"leaf_hash", leaf_hash)
    return t


def _native_or_py_matrix_mle_eval(flat, n, r_row, r_col):
    if _HAS_NATIVE:
        if _is_ndarray(flat):
            return _zk.native_matrix_mle_eval_np(flat, n, r_row, r_col)
        return _zk.native_matrix_mle_eval(flat, n, r_row, r_col)
    return matrix_mle_eval(flat, n, r_row, r_col)


def _native_or_py_rect_mle_eval(flat, nr, nc, r_row, r_col):
    if _HAS_NATIVE:
        if _is_ndarray(flat):
            return _zk.native_rect_matrix_mle_eval_np(flat, nr, nc, r_row, r_col)
        return _zk.native_rect_matrix_mle_eval(flat, nr, nc, r_row, r_col)
    return rect_matrix_mle_eval(flat, nr, nc, r_row, r_col)


def _native_or_py_block_tables(A, B, bs, ni, ra, rb):
    if _HAS_NATIVE:
        if _is_ndarray(A):
            return _zk.native_build_block_sumcheck_tables_np(A, B, bs, ni, ra, rb)
        return _zk.native_build_block_sumcheck_tables(A, B, bs, ni, ra, rb)
    return build_block_sumcheck_tables(A, B, bs, ni, ra, rb)


def _next_pow2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


class BlockGemmProver:
    """Proves C_block = A_sub × B_sub for one block."""

    @staticmethod
    def prove(
        A_sub_flat: list[int],
        B_sub_flat: list[int],
        C_block_flat: list[int],
        bs: int,
        n_inner: int,
        seed: int = 0,
        bi: int = 0,
        bj: int = 0,
        leaf_hash: bytes = b"",
        num_spot_checks: int = 0,
    ) -> BlockGemmProof:
        """Produce a block ZkGEMM proof.

        Args:
            A_sub_flat: bs × n_inner matrix (row-major).
            B_sub_flat: n_inner × bs matrix (row-major).
            C_block_flat: bs × bs matrix (row-major, prover-side only).
            bs: block size (power of 2).
            n_inner: inner dimension (any positive multiple of bs).
            seed: PRF seed (for transcript binding).
            bi, bj: block indices (for transcript domain separation).
            leaf_hash: SHA-256 leaf hash of C_block (32 bytes).
        """
        m_bs = int(math.log2(bs))
        assert bs == (1 << m_bs)
        assert n_inner > 0
        assert len(A_sub_flat) == bs * n_inner
        assert len(B_sub_flat) == n_inner * bs
        assert len(C_block_flat) == bs * bs

        # Pad inner dimension to next power of 2 for sumcheck
        profile = _proof_profile_enabled()
        total_t0 = time.perf_counter() if profile else 0.0
        n_padded = _next_pow2(n_inner)
        m_inner = n_padded.bit_length() - 1

        # Compute leaf_hash if not provided
        hash_t0 = time.perf_counter() if profile else 0.0
        if not leaf_hash:
            from zkgemm.merkle import hash_block
            leaf_hash = hash_block(C_block_flat, bs)
        hash_dt = time.perf_counter() - hash_t0 if profile else 0.0

        # 1. Transcript binds to leaf_hash (not full C_block)
        transcript_t0 = time.perf_counter() if profile else 0.0
        transcript = _init_block_transcript(
            leaf_hash, seed, n_inner, bi, bj
        )

        # 2. Squeeze random evaluation point (r_a, r_b) over block indices
        r_a = transcript.squeeze_n(b"r_a", m_bs)
        r_b = transcript.squeeze_n(b"r_b", m_bs)
        transcript_dt = time.perf_counter() - transcript_t0 if profile else 0.0

        # 3. Evaluate C_block_tilde at (r_a, r_b) — prover has C_block
        claimed_t0 = time.perf_counter() if profile else 0.0
        claimed_sum = _native_or_py_matrix_mle_eval(C_block_flat, bs, r_a, r_b)
        claimed_dt = time.perf_counter() - claimed_t0 if profile else 0.0

        # 4. Build sumcheck tables over inner dimension
        #    Tables have n_inner entries from actual data, zero-padded to n_padded
        tables_t0 = time.perf_counter() if profile else 0.0
        T_A, T_B = _native_or_py_block_tables(
            A_sub_flat, B_sub_flat, bs, n_inner, r_a, r_b
        )
        if n_padded > n_inner:
            pad = [0] * (n_padded - n_inner)
            T_A = list(T_A) + pad
            T_B = list(T_B) + pad
        tables_dt = time.perf_counter() - tables_t0 if profile else 0.0

        # 5. Run sumcheck prover (m_inner rounds over padded tables)
        sumcheck_t0 = time.perf_counter() if profile else 0.0
        sc_proof, challenges = sumcheck_prove(T_A, T_B, claimed_sum, transcript)
        sumcheck_dt = time.perf_counter() - sumcheck_t0 if profile else 0.0

        # 6. Final evaluations — r_k has m_inner challenges (log2(n_padded))
        #    MLE eval iterates over actual n_inner columns; eq_vector(r_k) has
        #    n_padded entries but only the first n_inner are accessed.
        final_t0 = time.perf_counter() if profile else 0.0
        r_k = challenges
        final_A = _native_or_py_rect_mle_eval(A_sub_flat, bs, n_inner, r_a, r_k)
        final_B = _native_or_py_rect_mle_eval(B_sub_flat, n_inner, bs, r_k, r_b)
        final_dt = time.perf_counter() - final_t0 if profile else 0.0

        # 7. PRF spot checks — sample random entries from A_sub/B_sub
        #    Positions derived from Fiat-Shamir transcript (prover can't predict).
        #    Verifier regenerates these entries from PRF and checks equality.
        #    A and B checks are INTERLEAVED in the Fiat-Shamir sequence so that
        #    the first N checks form a valid prefix (for two-tier verification).
        spot_A: list[tuple[int, int, int]] = []
        spot_B: list[tuple[int, int, int]] = []
        spots_t0 = time.perf_counter() if profile else 0.0
        if num_spot_checks > 0:
            for _ in range(num_spot_checks):
                a = transcript.squeeze(b"spot_a") % bs
                k_a = transcript.squeeze(b"spot_k_a") % n_inner
                spot_A.append((a, k_a, int(A_sub_flat[a * n_inner + k_a])))
                k_b = transcript.squeeze(b"spot_k_b") % n_inner
                b = transcript.squeeze(b"spot_b") % bs
                spot_B.append((k_b, b, int(B_sub_flat[k_b * bs + b])))
        spots_dt = time.perf_counter() - spots_t0 if profile else 0.0

        if profile:
            _proof_profile(
                f"block_prove block=({bi},{bj}) hash={hash_dt:.4f}s "
                f"transcript={transcript_dt:.4f}s claimed={claimed_dt:.4f}s "
                f"tables={tables_dt:.4f}s sumcheck={sumcheck_dt:.4f}s "
                f"final={final_dt:.4f}s spots={spots_dt:.4f}s "
                f"total={time.perf_counter() - total_t0:.4f}s"
            )

        return BlockGemmProof(
            bs=bs,
            n_inner=n_inner,
            sumcheck_proof=sc_proof,
            final_A_eval=final_A,
            final_B_eval=final_B,
            spot_A=spot_A,
            spot_B=spot_B,
        )


class BlockGemmVerifier:
    """Verifies a block ZkGEMM proof."""

    @staticmethod
    def verify_light(
        proof: BlockGemmProof,
        leaf_hash: bytes = b"",
        seed: int = 0,
        bi: int = 0,
        bj: int = 0,
    ) -> bool:
        """Verify sumcheck structure only (no MLE recomputation).

        Checks: Fiat-Shamir consistency, round equations, and that
        final_claim == final_A_eval * final_B_eval.

        Does NOT verify that final_A_eval/final_B_eval are correct —
        the caller must check these separately (e.g. via fraud proof
        or by calling verify_full).
        """
        bs = proof.bs
        n_inner = proof.n_inner
        m_bs = int(math.log2(bs))

        # 1. Rebuild transcript from leaf_hash
        transcript = _init_block_transcript(
            leaf_hash, seed, n_inner, bi, bj
        )

        # 2. Squeeze the same random point
        r_a = transcript.squeeze_n(b"r_a", m_bs)
        r_b = transcript.squeeze_n(b"r_b", m_bs)

        # 3. Verify sumcheck rounds (no C_block needed)
        ok, final_claim, challenges = sumcheck_verify(
            proof.sumcheck_proof, transcript
        )
        if not ok:
            return False

        # 4. Check final_claim == final_A_eval * final_B_eval
        if final_claim != mul(proof.final_A_eval, proof.final_B_eval):
            return False

        return True

    @staticmethod
    def verify_spot(
        proof: BlockGemmProof,
        seed: int,
        bi: int = 0,
        bj: int = 0,
        leaf_hash: bytes = b"",
        max_checks: int | None = None,
    ) -> bool:
        """Spot-check verification: light check + PRF entry sampling.

        Stronger than verify_light (probabilistically verifies the prover
        used the correct PRF-generated A/B), much cheaper than verify_full
        (O(s) Philox calls instead of O(bs × n) regeneration).

        Args:
            max_checks: If set, only verify the first max_checks entries
                from each of spot_A and spot_B. Allows two-tier verification:
                - Validator (off-chain): verify_spot(proof) — all entries
                - On-chain contract: verify_spot(proof, max_checks=20) — subset
                The prover generates one proof; the first max_checks positions
                are a deterministic prefix of the full Fiat-Shamir sequence.
        """
        if not proof.spot_A and not proof.spot_B:
            return BlockGemmVerifier.verify_light(
                proof, leaf_hash=leaf_hash, seed=seed, bi=bi, bj=bj,
            )

        bs = proof.bs
        n_inner = proof.n_inner
        m_bs = int(math.log2(bs))

        # 1. Rebuild transcript
        transcript = _init_block_transcript(leaf_hash, seed, n_inner, bi, bj)
        r_a = transcript.squeeze_n(b"r_a", m_bs)
        r_b = transcript.squeeze_n(b"r_b", m_bs)

        # 2. Verify sumcheck rounds (advances transcript to post-sumcheck state)
        ok, final_claim, challenges = sumcheck_verify(
            proof.sumcheck_proof, transcript
        )
        if not ok:
            return False

        # 3. Check final_claim == final_A_eval * final_B_eval
        if final_claim != mul(proof.final_A_eval, proof.final_B_eval):
            return False

        # 4. Derive spot positions from transcript and verify against PRF
        #    A and B checks are INTERLEAVED (matching the prover):
        #    each round squeezes spot_a, spot_k_a, spot_k_b, spot_b.
        #    This means the first N rounds form a valid prefix for two-tier
        #    verification (on-chain checks fewer rounds than validator).
        from zkgemm.prf import philox_to_field

        n_checks = len(proof.spot_A) if max_checks is None else min(max_checks, len(proof.spot_A))

        for i in range(n_checks):
            # A entry
            a, k, claimed_val = proof.spot_A[i]
            expected_a = transcript.squeeze(b"spot_a") % bs
            expected_k = transcript.squeeze(b"spot_k_a") % n_inner
            if a != expected_a or k != expected_k:
                return False
            actual_val = philox_to_field(seed, 0, bi * bs + a, k)
            if actual_val != claimed_val:
                return False

            # B entry (same round, interleaved)
            k_b, b, claimed_val_b = proof.spot_B[i]
            expected_kb = transcript.squeeze(b"spot_k_b") % n_inner
            expected_b = transcript.squeeze(b"spot_b") % bs
            if k_b != expected_kb or b != expected_b:
                return False
            actual_val_b = philox_to_field(seed, 1, k_b, bj * bs + b)
            if actual_val_b != claimed_val_b:
                return False

        return True

    @staticmethod
    def verify_full(
        A_sub_flat: list[int],
        B_sub_flat: list[int],
        proof: BlockGemmProof,
        leaf_hash: bytes = b"",
        seed: int = 0,
        bi: int = 0,
        bj: int = 0,
    ) -> bool:
        """Full verification: recompute MLE evaluations from A_sub/B_sub.

        Independently verifies that the prover's final_A_eval and
        final_B_eval match the actual MLE evaluations of A_sub and B_sub.
        """
        bs = proof.bs
        n_inner = proof.n_inner
        m_bs = int(math.log2(bs))

        # 1. Rebuild transcript from leaf_hash
        transcript = _init_block_transcript(
            leaf_hash, seed, n_inner, bi, bj
        )

        # 2. Squeeze the same random point
        r_a = transcript.squeeze_n(b"r_a", m_bs)
        r_b = transcript.squeeze_n(b"r_b", m_bs)

        # 3. Verify sumcheck rounds
        ok, final_claim, challenges = sumcheck_verify(
            proof.sumcheck_proof, transcript
        )
        if not ok:
            return False

        # 4. Independently evaluate MLEs at the final point
        r_k = challenges
        a_val = _native_or_py_rect_mle_eval(A_sub_flat, bs, n_inner, r_a, r_k)
        b_val = _native_or_py_rect_mle_eval(B_sub_flat, n_inner, bs, r_k, r_b)

        # 5. Final check: recomputed values must match the claimed product
        if final_claim != mul(a_val, b_val):
            return False

        return True
