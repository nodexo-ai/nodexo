"""
Two-phase block-committed ZkGEMM prover.

Phase 1 (commit): Generate A, B from PRF seed, compute C = A × B on GPU,
    hash 256×256 blocks, build Merkle tree, return root.
Phase 2 (prove): Given challenged block indices, produce ZkGEMM proofs
    + Merkle authentication paths for each block.

The prover uses GPU acceleration for GEMM, hashing, and Merkle tree
construction via the zkgemm_cuda extension.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from zkgemm.field import add, mul
from zkgemm.gemm import BlockGemmProver, BlockGemmProof
from zkgemm.merkle import extract_merkle_path, block_index, hash_block


def _proof_profile_enabled() -> bool:
    return os.environ.get("ZKGEMM_PROOF_PROFILE", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _proof_profile(msg: str) -> None:
    sys.stderr.write(f"[zkgemm-proof] {msg}\n")
    sys.stderr.flush()


@dataclass
class BlockProof:
    """Proof for a single challenged block.

    Contains only the cryptographic proof data:
    - leaf_hash: SHA-256 hash of the block (32 bytes, binds to Merkle tree)
    - merkle_path: authentication path to the committed root
    - zkgemm_proof: sumcheck proof + final MLE evaluations

    Neither C_block nor A_sub/B_sub are included:
    - C_block is committed via leaf_hash in the Merkle tree
    - A_sub/B_sub are PRF-determined by the public seed
    """

    bi: int
    bj: int
    leaf_hash: bytes
    merkle_path: list[tuple[bytes, bool]]
    zkgemm_proof: BlockGemmProof


class ZkGemmBlockProver:
    """Two-phase block-committed ZkGEMM prover."""

    def __init__(
        self,
        seed: int,
        n: int,
        block_size: int = 256,
        device: int = 0,
        num_spot_checks: int = 50,
        use_8bit: bool = True,
    ):
        assert n > 0 and n % block_size == 0, "n must be a positive multiple of block_size"
        m_bs = int(math.log2(block_size))
        assert block_size == (1 << m_bs), "block_size must be a power of 2"

        self.seed = seed
        self.n = n
        self.bs = block_size
        self.device = device
        self.blocks_per_row = n // block_size
        self.num_spot_checks = num_spot_checks
        self.use_8bit = use_8bit

        # Populated by phase1
        self.A: torch.Tensor | None = None
        self.B: torch.Tensor | None = None
        self.C: torch.Tensor | None = None
        self.root: bytes | None = None
        self.flat_tree: bytes | None = None

    def phase1_commit(self) -> bytes:
        """Phase 1: Generate matrices, compute C, build Merkle tree.

        Returns:
            Merkle root (32 bytes).
        """
        import zkgemm_cuda as zk

        # Generate A, B from PRF seed
        self.A = zk.philox_generate_field(self.seed, self.n, 0, self.device)
        self.B = zk.philox_generate_field(self.seed, self.n, 1, self.device)

        # Compute C = A × B (INT8 tensor-core decomposition)
        if self.use_8bit:
            self.C = zk.field_gemm_8bit(self.A, self.B, self.device)
        else:
            self.C = zk.field_gemm(self.A, self.B, self.device)

        # Build Merkle commitment over bs×bs blocks of C
        root_tensor, tree_tensor = zk.block_merkle_commit(
            self.C, self.bs, self.device
        )

        self.root = bytes(root_tensor.tolist())
        self.flat_tree = bytes(tree_tensor.tolist())

        return self.root

    def phase2_prove(
        self, challenged_blocks: list[tuple[int, int]]
    ) -> list[BlockProof]:
        """Phase 2: Generate ZkGEMM proofs for challenged blocks.

        Args:
            challenged_blocks: List of (bi, bj) block coordinates.

        Returns:
            List of BlockProof, one per challenged block.
        """
        assert self.C is not None, "Must call phase1_commit first"

        n = self.n
        bs = self.bs
        bpr = self.blocks_per_row
        num_leaves = bpr * bpr

        proofs = []
        use_prf_slices = self.A is None or self.B is None
        if use_prf_slices:
            from zkgemm.prf import generate_cols, generate_rows

        profile = _proof_profile_enabled()
        phase_t0 = time.perf_counter() if profile else 0.0
        for bi, bj in challenged_blocks:
            block_t0 = time.perf_counter() if profile else 0.0
            if use_prf_slices:
                # Pass-0 optimization path: after commitment we only keep C.
                # A and B are public PRF matrices, so challenged slices can be
                # regenerated exactly when root_1 reveals the challenge set.
                c_t0 = time.perf_counter() if profile else 0.0
                C_block = self.C[bi*bs:(bi+1)*bs, bj*bs:(bj+1)*bs].contiguous()
                C_block_flat = C_block.cpu().numpy().flatten().tolist()
                c_dt = time.perf_counter() - c_t0 if profile else 0.0
                a_t0 = time.perf_counter() if profile else 0.0
                A_sub_np = np.asarray(
                    generate_rows(self.seed, 0, bi * bs, bs, n),
                    dtype=np.int64,
                )
                a_dt = time.perf_counter() - a_t0 if profile else 0.0
                b_t0 = time.perf_counter() if profile else 0.0
                B_sub_np = np.asarray(
                    generate_cols(self.seed, 1, bj * bs, bs, n),
                    dtype=np.int64,
                )
                b_dt = time.perf_counter() - b_t0 if profile else 0.0
            else:
                # Extract sub-matrices on GPU first, then transfer only the
                # small slices to CPU. This avoids transferring the full n×n
                # matrices for the normal resident-GPU proof path.
                c_t0 = time.perf_counter() if profile else 0.0
                C_block_gpu = self.C[bi*bs:(bi+1)*bs, bj*bs:(bj+1)*bs].contiguous()
                A_sub_gpu = self.A[bi*bs:(bi+1)*bs, :]
                B_sub_gpu = self.B[:, bj*bs:(bj+1)*bs].contiguous()

                C_block_flat = C_block_gpu.cpu().numpy().flatten().tolist()
                c_dt = time.perf_counter() - c_t0 if profile else 0.0
                a_t0 = time.perf_counter() if profile else 0.0
                A_sub_np = A_sub_gpu.cpu().numpy().flatten()
                a_dt = time.perf_counter() - a_t0 if profile else 0.0
                b_t0 = time.perf_counter() if profile else 0.0
                B_sub_np = B_sub_gpu.cpu().numpy().flatten()
                b_dt = time.perf_counter() - b_t0 if profile else 0.0

            # Compute leaf hash (32 bytes — this goes into the proof, NOT C_block)
            hash_t0 = time.perf_counter() if profile else 0.0
            leaf_hash = hash_block(C_block_flat, bs)
            hash_dt = time.perf_counter() - hash_t0 if profile else 0.0

            # Merkle path
            merkle_t0 = time.perf_counter() if profile else 0.0
            leaf_idx = block_index(bi, bj, bpr)
            merkle_path = extract_merkle_path(
                self.flat_tree, num_leaves, leaf_idx
            )
            merkle_dt = time.perf_counter() - merkle_t0 if profile else 0.0

            # ZkGEMM block proof — numpy arrays passed directly to C++ (no .tolist())
            proof_t0 = time.perf_counter() if profile else 0.0
            zkgemm_proof = BlockGemmProver.prove(
                A_sub_np, B_sub_np, C_block_flat,
                bs, n, seed=self.seed, bi=bi, bj=bj,
                leaf_hash=leaf_hash,
                num_spot_checks=self.num_spot_checks,
            )
            proof_dt = time.perf_counter() - proof_t0 if profile else 0.0

            proofs.append(BlockProof(
                bi=bi, bj=bj,
                leaf_hash=leaf_hash,
                merkle_path=merkle_path,
                zkgemm_proof=zkgemm_proof,
            ))

            if profile:
                total_dt = time.perf_counter() - block_t0
                mode = "prf" if use_prf_slices else "resident"
                _proof_profile(
                    f"phase2 block=({bi},{bj}) mode={mode} "
                    f"c={c_dt:.4f}s a={a_dt:.4f}s b={b_dt:.4f}s "
                    f"hash={hash_dt:.4f}s merkle={merkle_dt:.4f}s "
                    f"prove={proof_dt:.4f}s total={total_dt:.4f}s"
                )

        if profile:
            _proof_profile(
                f"phase2 blocks={len(challenged_blocks)} "
                f"mode={'prf' if use_prf_slices else 'resident'} "
                f"total={time.perf_counter() - phase_t0:.4f}s"
            )

        return proofs
