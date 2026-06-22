"""
ZkGEMM proof verifier — per-GPU, three-tier verification.

Each GPU in an executor generates its own proof with a unique seed
(gpu_index baked into seed derivation). The verifier checks ALL GPUs
independently. Missing or failed GPU proof = executor fails.

Security checks per GPU:
  1. Seed verification — re-derive from beacon + executor_id + gpu_index
  2. VRAM consistency — matrix_dim matches registered VRAM
  3. Chain integrity — seed_1 from seed + root_0
  4. Challenge re-derivation — independently compute challenged blocks
  5. Cryptographic verification — Merkle + sumcheck (light/spot/full)

Global checks:
  6. Epoch freshness — reject old/future epochs
  7. GPU count — must match registered gpu_count
  8. Wall-clock timing — proof within deadline of beacon block

Per-GPU verification loop (prior-art context: protocol design notes).
"""
from __future__ import annotations

import hashlib
import logging
import math
import struct
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from common.proof_timing import (
    DEFAULT_WALL_CLOCK_GRACE_SEC,
    DEFAULT_RECEIPT_SPREAD_GRACE_SEC,
    GPU_TIMING_MODELS,
    GpuTimingModel,
    compute_delta_deadline,
    compute_delta_floor,
    compute_gpu_count_scaling,
    compute_receipt_spread_deadline,
    compute_wall_clock_deadline,
)

# Domain separation (must match proof_service.py)
SEED_PREFIX = b"NODEXO_SEED_V1"
CHAIN_PREFIX = b"NODEXO_CHAIN_V1"
CHALLENGE_PREFIX = b"NODEXO_CHALLENGE_V1"

# Timing
WALL_CLOCK_MAX_SEC = 300.0
MAX_EPOCH_DRIFT = 2

# T_commit domain separation (must match gpu_worker.py)
COMMIT_PREFIX = b"NODEXO_COMMIT_V1"


def compute_t_commit(seed: bytes, root: bytes, u: int, n: int, epoch_id: int) -> bytes:
    """Re-derive T_commit for verification (must match gpu_worker.py)."""
    payload = COMMIT_PREFIX + seed + root + struct.pack("<I", u) + struct.pack("<I", n) + struct.pack("<I", epoch_id)
    return hashlib.sha256(payload).digest()


@dataclass
class VerificationResult:
    valid: bool
    tier: str
    reason: str = ""
    verification_time_ms: float = 0
    gpu_results: list = field(default_factory=list)


@dataclass
class VerificationContext:
    """All data needed to verify a recipe."""
    beacon: bytes = b""
    executor_id: str = ""
    expected_vram_mb: int = 0
    expected_gpu_count: int = 1
    gpu_model_name: str = ""    # For timing model lookup
    is_rented: bool = False
    current_epoch: int = 0
    epoch_interval: int = 15
    num_challenges: int = 4
    recv_time: float = 0
    beacon_timestamp: float = 0
    beacon_block: int = 0
    require_beacon: bool = False
    require_registry: bool = False


def derive_seed_with_gpu(beacon: bytes, executor_id: str, gpu_index: int) -> bytes:
    """Re-derive per-GPU seed (must match proof_service.derive_seed)."""
    payload = SEED_PREFIX + beacon + bytes.fromhex(executor_id) + struct.pack("<I", gpu_index)
    return hashlib.sha256(payload).digest()[:8]


def _derive_challenge_indices(seed: bytes, root_1: bytes, num_challenges: int, total_blocks: int) -> list[int]:
    """Re-derive challenge indices (must match proof_service exactly)."""
    indices = []
    seen = set()
    counter = 0
    while len(indices) < num_challenges:
        payload = CHALLENGE_PREFIX + seed + root_1 + struct.pack("<I", counter)
        h = hashlib.sha256(payload).digest()
        idx = struct.unpack("<I", h[:4])[0] % total_blocks
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
        counter += 1
    return indices


def _deserialize_block_proofs(pass_data: dict, seed_int: int, n: int, bs: int):
    """Deserialize JSON block proofs into zkgemm.prover.BlockProof objects."""
    from zkgemm.prover import BlockProof
    from zkgemm.gemm import BlockGemmProof
    from zkgemm.sumcheck import SumcheckProof, SumcheckRound

    proofs = []
    for bp_data in pass_data.get("block_proofs", []):
        merkle_path = [(bytes.fromhex(e[0]), e[1]) for e in bp_data["merkle_path"]]
        rounds = [SumcheckRound(evals=tuple(r)) for r in bp_data.get("sumcheck_rounds", [])]
        sumcheck_proof = SumcheckProof(claimed_sum=bp_data.get("claimed_sum", 0), rounds=rounds)
        zkgemm_proof = BlockGemmProof(
            bs=bp_data.get("bs", bs), n_inner=bp_data.get("n_inner", n),
            sumcheck_proof=sumcheck_proof,
            final_A_eval=bp_data.get("final_A_eval", 0),
            final_B_eval=bp_data.get("final_B_eval", 0),
            spot_A=[tuple(t) for t in bp_data.get("spot_A", [])],
            spot_B=[tuple(t) for t in bp_data.get("spot_B", [])],
        )
        proofs.append(BlockProof(
            bi=bp_data["bi"], bj=bp_data["bj"],
            leaf_hash=bytes.fromhex(bp_data["leaf_hash"]),
            merkle_path=merkle_path, zkgemm_proof=zkgemm_proof,
        ))
    return proofs


def _verify_single_gpu(
    gp: dict, executor_id: str, n: int, bs: int, tier: str,
    beacon: bytes, num_challenges: int, epoch_id: int = 0,
) -> tuple[bool, str]:
    """Verify one GPU's proof (both passes). Returns (ok, reason)."""
    from zkgemm.verifier import ZkGemmBlockVerifier

    gpu_idx = gp["gpu_index"]
    seed_hex = gp["seed"]
    seed_bytes = bytes.fromhex(seed_hex)
    seed_int = struct.unpack("<Q", seed_bytes)[0]

    # ── Per-GPU seed verification ──────────────────────────
    if beacon:
        expected_seed = derive_seed_with_gpu(beacon, executor_id, gpu_idx)
        if expected_seed != seed_bytes:
            return False, f"GPU[{gpu_idx}]: seed mismatch (fabricated or wrong gpu_index)"

    # ── T_commit verification (binds proof to seed/root/epoch) ──
    root_0 = bytes.fromhex(gp["pass_0"]["merkle_root"])
    claimed_tc0 = gp.get("T_commit_0", "")
    if claimed_tc0:
        expected_tc0 = compute_t_commit(seed_bytes, root_0, 0, n, epoch_id)
        if bytes.fromhex(claimed_tc0) != expected_tc0:
            return False, f"GPU[{gpu_idx}]: T_commit_0 mismatch (proof not bound to this epoch)"

    root_1 = bytes.fromhex(gp["pass_1"]["merkle_root"])
    seed_1 = hashlib.sha256(CHAIN_PREFIX + seed_bytes + root_0).digest()[:8]
    claimed_tc1 = gp.get("T_commit_1", "")
    if claimed_tc1:
        expected_tc1 = compute_t_commit(seed_1, root_1, 1, n, epoch_id)
        if bytes.fromhex(claimed_tc1) != expected_tc1:
            return False, f"GPU[{gpu_idx}]: T_commit_1 mismatch"

    # ── Chain integrity (seed_1 already derived above in T_commit section) ──
    seed_1_int = struct.unpack("<Q", seed_1)[0]

    # ── Challenge re-derivation for this GPU ───────────────
    blocks_per_row = n // bs
    total_blocks = blocks_per_row * blocks_per_row

    if num_challenges > 0:
        expected_indices = _derive_challenge_indices(seed_bytes, root_1, num_challenges, total_blocks)
        expected_blocks = set()
        for idx in expected_indices:
            expected_blocks.add((idx // blocks_per_row, idx % blocks_per_row))

        proven_0 = {(bp["bi"], bp["bj"]) for bp in gp["pass_0"]["block_proofs"]}
        proven_1 = {(bp["bi"], bp["bj"]) for bp in gp["pass_1"]["block_proofs"]}

        if proven_0 != expected_blocks:
            return False, f"GPU[{gpu_idx}]: challenge mismatch pass_0"
        if proven_1 != expected_blocks:
            return False, f"GPU[{gpu_idx}]: challenge mismatch pass_1"

    # ── Deserialize and verify cryptographic proofs ────────
    proofs_0 = _deserialize_block_proofs(gp["pass_0"], seed_int, n, bs)
    proofs_1 = _deserialize_block_proofs(gp["pass_1"], seed_1_int, n, bs)

    if not proofs_0 or not proofs_1:
        return False, f"GPU[{gpu_idx}]: missing block proofs"

    # Cap PRF spot samples per block in spot mode. None = check ALL 4M entries,
    # which makes spot as expensive as full. 50 random samples gives overwhelming
    # detection probability against any non-trivial cheat.
    SPOT_MAX_CHECKS = 50

    # Pass 0
    v0 = ZkGemmBlockVerifier(seed=seed_int, n=n, block_size=bs)
    if tier == "light":
        ok_0 = v0.verify_light(root_0, proofs_0)
    elif tier == "spot":
        ok_0 = v0.verify_spot(root_0, proofs_0, max_checks=SPOT_MAX_CHECKS)
    else:
        ok_0 = v0.verify(root_0, proofs_0)
    if not ok_0:
        return False, f"GPU[{gpu_idx}]: pass_0 cryptographic verification failed"

    # Pass 1
    v1 = ZkGemmBlockVerifier(seed=seed_1_int, n=n, block_size=bs)
    if tier == "light":
        ok_1 = v1.verify_light(root_1, proofs_1)
    elif tier == "spot":
        ok_1 = v1.verify_spot(root_1, proofs_1, max_checks=SPOT_MAX_CHECKS)
    else:
        ok_1 = v1.verify(root_1, proofs_1)
    if not ok_1:
        return False, f"GPU[{gpu_idx}]: pass_1 cryptographic verification failed"

    return True, ""


def verify_recipe(
    recipe_data: dict,
    tier: str = "spot",
    ctx: VerificationContext = None,
) -> VerificationResult:
    """Verify a proof recipe with per-GPU verification.

    ALL GPUs must pass independently. Missing GPU proofs = fail.
    """
    if ctx is None:
        ctx = VerificationContext()
    t0 = time.time()

    def _fail(reason):
        return VerificationResult(False, tier, reason, (time.time() - t0) * 1000)

    try:
        from common.config import compute_matrix_dim

        n = recipe_data["matrix_dim"]
        bs = 256
        epoch_id = recipe_data.get("epoch_id", 0)
        executor_id = recipe_data.get("executor_id", "")
        gpu_proofs = recipe_data.get("gpu_proofs", [])
        run_id = recipe_data.get("run_id", "")
        beacon_meta = recipe_data.get("beacon_meta", {})
        allocation_state = recipe_data.get("allocation_state", "")

        # ── GLOBAL CHECK 0: run_id presence (replay prevention) ──
        if not run_id:
            return _fail("Missing run_id (required for replay prevention)")

        # Live validator verification must have both the block beacon and
        # registry context. Without the beacon, the miner-controlled seed would
        # not be bound to the current chain cycle. Without registry context, a
        # weaker executor could bypass GPU-count/VRAM workload checks.
        if ctx.require_beacon and not ctx.beacon:
            return _fail("Missing chain beacon for seed verification")
        if ctx.require_registry:
            if ctx.expected_vram_mb <= 0 or ctx.expected_gpu_count <= 0:
                return _fail("Missing registry context for proof verification")

        # ── GLOBAL CHECK 0b: beacon_meta (auditability) ──────────
        if beacon_meta:
            claimed_b_num = beacon_meta.get("b_num")
            if ctx.require_beacon and ctx.beacon_block > 0:
                try:
                    if int(claimed_b_num) != int(ctx.beacon_block):
                        return _fail("beacon_meta.b_num doesn't match scheduled beacon block")
                except Exception:
                    return _fail("beacon_meta.b_num missing or invalid")
            claimed_b_hash = beacon_meta.get("b_hash", "")
            if ctx.beacon and claimed_b_hash:
                if bytes.fromhex(claimed_b_hash) != ctx.beacon:
                    return _fail("beacon_meta.b_hash doesn't match actual chain beacon")
        elif ctx.require_beacon and ctx.beacon_block > 0:
            return _fail("Missing beacon_meta for scheduled beacon verification")

        # `allocation_state` is advisory. Rental state can change inside a
        # 15-block proof cycle, while the recipe is anchored to the cycle
        # start block. Enforcing exact mode here false-rejects honest proofs
        # around rent/end transitions. The matrix-dimension check below still
        # accepts only the heavy or micro workload, and the cryptographic proof
        # is verified either way.

        # ── GLOBAL CHECK 1: Epoch freshness ────────────────────
        if ctx.current_epoch > 0:
            if abs(epoch_id - ctx.current_epoch) > MAX_EPOCH_DRIFT:
                return _fail(f"Epoch drift: recipe={epoch_id}, current={ctx.current_epoch}")

        # ── GLOBAL CHECK 2: GPU count matches registration ─────
        if ctx.expected_gpu_count > 0:
            if len(gpu_proofs) != ctx.expected_gpu_count:
                return _fail(
                    f"GPU count mismatch: got {len(gpu_proofs)} proofs, "
                    f"expected {ctx.expected_gpu_count} (registered)"
                )

        # ── GLOBAL CHECK 3: VRAM consistency ───────────────────
        # Free executors must prove the VRAM-scaled heavy workload. This is
        # the advertised-hardware signal: accepting the fixed micro workload
        # while free would let a weaker GPU masquerade as a larger one.
        #
        # Rented executors may submit micro proofs so renter workloads can use
        # most VRAM. We also accept heavy while rented because it is strictly
        # stronger and covers rent/start transition races where the miner began
        # a free-cycle proof before the rental flag reached the validator.
        if ctx.expected_vram_mb > 0:
            from common.config import BUFFER_HEAVY, compute_micro_matrix_dim
            vram_gb = ctx.expected_vram_mb / 1024
            heavy_n = compute_matrix_dim(vram_gb, buffer=BUFFER_HEAVY, block_size=bs)
            micro_n = compute_micro_matrix_dim(block_size=bs)
            accepted = {heavy_n, micro_n} if ctx.is_rented else {heavy_n}
            if not any(abs(n - e) <= bs for e in accepted):
                expected = (
                    f"heavy={heavy_n} or micro={micro_n}"
                    if ctx.is_rented
                    else f"heavy={heavy_n}"
                )
                return _fail(
                    f"Matrix dim {n} invalid for "
                    f"{'rented' if ctx.is_rented else 'free'} executor "
                    f"(expected {expected})"
                )
            # Decide which mode this was so timing/security checks downstream
            # can use the right expected baseline.
            ctx._actual_mode = "micro" if abs(n - micro_n) <= bs else "heavy"

        # ── GLOBAL CHECK 4: Wall-clock timing ──────────────────
        if ctx.recv_time > 0 and ctx.beacon_timestamp > 0:
            wall = ctx.recv_time - ctx.beacon_timestamp
            if wall < -DEFAULT_WALL_CLOCK_GRACE_SEC:
                return _fail(f"Causality violation: proof {-wall:.1f}s before beacon")
            if wall > WALL_CLOCK_MAX_SEC:
                return _fail(f"Stale proof: {wall:.1f}s after beacon (max {WALL_CLOCK_MAX_SEC}s)")

        # ── GLOBAL CHECK 5: Must have at least 1 GPU proof ─────
        if not gpu_proofs:
            return _fail("No GPU proofs in recipe")

        # ── PER-GPU VERIFICATION ───────────────────────────────
        gpu_results = []
        for gp in gpu_proofs:
            ok, reason = _verify_single_gpu(
                gp, executor_id, n, bs, tier,
                ctx.beacon, ctx.num_challenges, epoch_id,
            )
            gpu_results.append({"gpu_index": gp["gpu_index"], "valid": ok, "reason": reason})
            if not ok:
                return VerificationResult(
                    False, tier, reason, (time.time() - t0) * 1000, gpu_results,
                )

        return VerificationResult(
            True, tier, "", (time.time() - t0) * 1000, gpu_results,
        )

    except Exception as e:
        return VerificationResult(False, tier, f"Exception: {e}", (time.time() - t0) * 1000)


def select_verification_tier(trust_score: float, is_new: bool) -> str:
    if is_new:
        return "full"
    if trust_score < 0.3:
        return "full"
    if trust_score < 0.7:
        return "spot"
    return "light"
