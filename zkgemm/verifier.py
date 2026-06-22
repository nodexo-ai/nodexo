"""
Block-committed ZkGEMM verifier.

Three verification modes:

  verify() — Full verification (sound):
    1. Merkle path: leaf_hash is in the committed tree
    2. Sumcheck rounds: algebraic consistency
    3. Final MLE check: recompute A_tilde(r_a, r_k) and B_tilde(r_k, r_b)
       from the PRF seed and verify final_claim == A_tilde * B_tilde

  verify_spot() — Spot-check verification (probabilistic):
    1. Merkle path: leaf_hash is in the committed tree
    2. Sumcheck rounds: algebraic consistency
    3. PRF spot checks: sample random entries from A/B and verify via PRF
    Supports max_checks for two-tier verification:
      - Validator (off-chain): check all entries (high detection)
      - On-chain: check first N entries only (low gas)

  verify_light() — Optimistic verification (for on-chain / fraud-proof):
    1. Merkle path: leaf_hash is in the committed tree
    2. Sumcheck rounds: algebraic consistency
    3. Trusts the prover's final_A_eval, final_B_eval

    O(log n) per block, microseconds. Sound only with a fraud-proof
    mechanism where anyone can dispute the final evaluations.
"""

from __future__ import annotations

from zkgemm.field import mul
from zkgemm.gemm import BlockGemmVerifier
from zkgemm.merkle import verify_merkle_path
from zkgemm.prf import generate_rows, generate_cols
from zkgemm.prover import BlockProof


class ZkGemmBlockVerifier:
    """Verifies block-committed ZkGEMM proofs. CPU-only."""

    def __init__(self, seed: int, n: int, block_size: int = 256):
        assert n > 0 and n % block_size == 0
        self.seed = seed
        self.n = n
        self.bs = block_size
        self.blocks_per_row = n // block_size

    def verify(self, root: bytes, proofs: list[BlockProof]) -> bool:
        """Full verification — sound, recomputes MLE from PRF.

        Returns True only if all proofs are valid.
        """
        for proof in proofs:
            if not self._verify_one(root, proof):
                return False
        return True

    def verify_light(self, root: bytes, proofs: list[BlockProof]) -> bool:
        """Optimistic verification — trusts prover's final evaluations.

        Fast (O(log n) per block) but NOT sound on its own.
        Use with a fraud-proof mechanism for on-chain deployment.
        """
        for proof in proofs:
            if not self._verify_one_light(root, proof):
                return False
        return True

    def verify_spot(
        self, root: bytes, proofs: list[BlockProof],
        max_checks: int | None = None,
    ) -> bool:
        """Spot-check verification — light check + PRF entry sampling.

        Args:
            max_checks: If set, only verify the first max_checks entries
                per matrix per block. Use None for validator (all entries),
                or a small number (e.g. 20) for on-chain gas savings.
        """
        for proof in proofs:
            if not self._verify_one_spot(root, proof, max_checks):
                return False
        return True

    def _verify_one(self, root: bytes, proof: BlockProof) -> bool:
        """Full single-block verification."""
        bi, bj = proof.bi, proof.bj
        bs = self.bs
        n = self.n

        # 1. Verify Merkle inclusion (leaf_hash is in the committed tree)
        if not verify_merkle_path(root, proof.leaf_hash, proof.merkle_path):
            return False

        # 2. Verify sumcheck structure (light check)
        if not BlockGemmVerifier.verify_light(
            proof.zkgemm_proof, leaf_hash=proof.leaf_hash,
            seed=self.seed, bi=bi, bj=bj,
        ):
            return False

        # 3. Recompute the final MLE evaluations from PRF
        #    Uses native C++ when available (~ms vs ~seconds in Python)
        A_sub_flat = generate_rows(self.seed, 0, bi * bs, bs, n)
        B_sub_flat = generate_cols(self.seed, 1, bj * bs, bs, n)

        if not BlockGemmVerifier.verify_full(
            A_sub_flat, B_sub_flat, proof.zkgemm_proof,
            leaf_hash=proof.leaf_hash,
            seed=self.seed, bi=bi, bj=bj,
        ):
            return False

        return True

    def _verify_one_spot(
        self, root: bytes, proof: BlockProof,
        max_checks: int | None = None,
    ) -> bool:
        """Spot-check single-block verification (light + PRF sampling)."""
        bi, bj = proof.bi, proof.bj

        # 1. Verify Merkle inclusion
        if not verify_merkle_path(root, proof.leaf_hash, proof.merkle_path):
            return False

        # 2. Verify sumcheck + spot-check A/B entries against PRF
        if not BlockGemmVerifier.verify_spot(
            proof.zkgemm_proof, seed=self.seed, bi=bi, bj=bj,
            leaf_hash=proof.leaf_hash, max_checks=max_checks,
        ):
            return False

        return True

    def _verify_one_light(self, root: bytes, proof: BlockProof) -> bool:
        """Optimistic single-block verification (no MLE recomputation)."""
        bi, bj = proof.bi, proof.bj

        # 1. Verify Merkle inclusion
        if not verify_merkle_path(root, proof.leaf_hash, proof.merkle_path):
            return False

        # 2. Verify sumcheck + trust final evaluations
        if not BlockGemmVerifier.verify_light(
            proof.zkgemm_proof, leaf_hash=proof.leaf_hash,
            seed=self.seed, bi=bi, bj=bj,
        ):
            return False

        return True
