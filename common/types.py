"""
Nodexo — shared types for proof recipes, commitments, and hardware attestation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HardwareAttestation:
    """Self-reported hardware specs, included in proof commitments.
    Signed by miner hotkey. Cross-checked during deep verification (SSH).
    """
    gpu_model: str
    gpu_uuids: list[str]
    gpu_count: int
    vram_mb: int          # Per GPU
    driver_version: str
    nvml_digest: str       # MD5:SHA256 of libnvidia-ml.so
    sysbox_detected: bool
    cpu_model: str
    ram_gb: int
    software_version: str  # Nodexo version string
    tee_platform: str = ""  # "" = none, "tdx", "sev-snp"
    tee_quote: str = ""     # Base64-encoded attestation quote


@dataclass
class BlockProof:
    """Proof for a single challenged block within one pass."""
    bi: int                          # Block row index
    bj: int                          # Block column index
    leaf_hash: bytes                 # SHA256 of C_block (32 bytes)
    merkle_path: list[bytes]         # Authentication path to root
    sumcheck_rounds: list[tuple[int, int, int]]  # (e0, e1, e2) per round
    claimed_sum: int                 # C̃(r_a, r_b)
    final_a_eval: int               # Ã(r_a, r_k)
    final_b_eval: int               # B̃(r_k, r_b)
    spot_checks_a: list[tuple[int, int, int]]  # (row, col, value)
    spot_checks_b: list[tuple[int, int, int]]  # (row, col, value)


@dataclass
class PassData:
    """Data for one pass (pass_0 or pass_1) of the chained proof."""
    merkle_root: bytes               # 32-byte Merkle root
    block_proofs: list[BlockProof]   # One per challenged block


@dataclass
class ProofCommitment:
    """Phase A message: commitment after pass_0 completes.
    Sent to all validators immediately after pass_0 GEMM finishes.
    """
    epoch_id: int
    executor_id: str                 # Hex-encoded bytes32
    seed: bytes                      # 8-byte seed derived from beacon
    matrix_dim: int                  # n (VRAM-scaled)
    pass_0_root: bytes               # root_0 (32 bytes)
    hw_attestation: HardwareAttestation
    timestamp_pass0: float           # Unix timestamp when pass_0 completed
    signature: bytes = b""           # SR25519 signature over the commitment


@dataclass
class ProofRecipe:
    """Phase B message: full recipe after pass_1 + proofs complete.
    Sent to all validators after both passes and proof generation.
    """
    epoch_id: int
    executor_id: str
    seed: bytes
    matrix_dim: int
    pass_0: PassData
    pass_1: PassData
    timestamp_pass0: float           # Same as commitment (consistency check)
    timestamp_pass1: float           # When pass_1 completed
    compute_time: float              # Total GPU time (seconds)
    signature: bytes = b""           # SR25519 signature over the recipe


@dataclass
class ExecutorSpec:
    """Executor specification as read from ComputeRegistry on-chain."""
    executor_id: str
    miner_address: str
    endpoint: str
    gpu_model_hash: str
    gpu_count: int
    vram_mb: int
    price_per_gpu_hour: int  # RAO
    expires_at: int          # Unix timestamp
    is_active: bool
    is_rented: bool
    miner_uid: Optional[int] = None
    miner_registered: Optional[bool] = None
    uid_owner_match: Optional[bool] = None


@dataclass
class ValidatorEndpoint:
    """Validator info as read from ValidatorRegistry on-chain."""
    address: str
    proxy_endpoint: str
    uid: int
    is_active: bool


class ValidatorDiscoveryResult(list):
    """Validator discovery list with source health metadata."""

    def __init__(
        self,
        validators=None,
        *,
        registry_failed: bool = False,
        partial: bool = False,
        inconclusive_native_count: int = 0,
    ):
        super().__init__(validators or [])
        self.registry_failed = bool(registry_failed)
        self.partial = bool(partial)
        self.inconclusive_native_count = max(0, int(inconclusive_native_count or 0))


@dataclass
class ProofScore:
    """Per-executor proof score computed by the validator analyzer."""
    executor_id: str
    pass_rate_1h: float     # 0.0 - 1.0
    pass_rate_24h: float
    pass_rate_all: float
    avg_delta: float         # Seconds (timing between pass_0 and pass_1)
    avg_compute_time: float  # Seconds
    status: str              # pass, pending, fail, stale, offline
    last_verified_epoch: int
    trust_level: str         # light, spot, full (verification tier)
