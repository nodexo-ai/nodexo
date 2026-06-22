"""ZkGEMM — Zero-Knowledge Verifiable Matrix Multiplication."""

from zkgemm.prover import ZkGemmBlockProver, BlockProof
from zkgemm.verifier import ZkGemmBlockVerifier

__all__ = [
    "ZkGemmBlockProver",
    "ZkGemmBlockVerifier",
    "BlockProof",
]
