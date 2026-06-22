"""
Proof ingestion API — receives commitments and recipes from executors.

Two-phase protocol:
  POST /proofs/commit  — Phase A (after pass_0)
  POST /proofs/recipe  — Phase B (after pass_1 + proofs)

Security layers:
  1. Registered-executor check — executor_id must exist in ComputeRegistry
  2. Rate limiting — max 1 commit + 1 recipe per executor per epoch
  3. Payload size limit — reject oversized bodies
  4. Signature verification — SR25519 signed by the executor's miner hotkey

Reference: see protocol design notes
"""
from __future__ import annotations

import time
import asyncio
import hashlib
import heapq
import json
import os
import string
from functools import partial
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import bittensor as bt
from fastapi import APIRouter, HTTPException, Request

if TYPE_CHECKING:
    from neurons.validator.proof.analyzer import ProofAnalyzer
router = APIRouter(prefix="/proofs", tags=["proofs"])

# Set by the main validator app at startup
analyzer: ProofAnalyzer = None
verification_queue = None  # asyncio.Queue, set by validator daemon
follower_mode = False      # True for validators that import upstream scores
verification_priority_salt = "validator"
registry_client = None     # ComputeRegistryClient, for executor existence check
rpc_client = None          # SubtensorRPC, for hotkey→UID checks
chain_snapshot = None      # ChainSnapshot, set by validator daemon

# Rate limiting: track last submission per executor
_last_commit: dict[str, float] = {}
_last_recipe: dict[str, float] = {}
_MIN_INTERVAL = 30  # Minimum seconds between submissions from same executor

# Duplicate epoch rejection: track (executor_id, epoch_id) pairs already received
_received_recipes: set[tuple[str, int]] = set()

# Per-pass receipt arrival times. Keyed by (executor_id, epoch_id,
# gpu_index) → {0: ts_pass0_recv, 1: ts_pass1_recv}. Used to derive
# external_pass_delta = ts_pass1 - ts_pass0 (network jitter cancels).
# Bounded by _PASS_RECV_MAX entries; LRU-pruned.
_pass_recv: dict[tuple[str, int, int], dict[int, float]] = {}
_pass_recv_meta: dict[tuple[str, int, int, int], dict[str, object]] = {}
_PASS_RECV_MAX = 5000


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


_MAX_RECEIPT_BODY_BYTES = max(512, _env_int("NODEXO_RECEIPT_MAX_BODY_BYTES", 4096))
_RECEIPT_CRYPTO_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, min(128, _env_int("NODEXO_RECEIPT_CRYPTO_WORKERS", 16))),
    thread_name_prefix="receipt-crypto",
)
_RECEIPT_STORE_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, min(32, _env_int("NODEXO_RECEIPT_STORE_WORKERS", 4))),
    thread_name_prefix="receipt-store",
)


async def _read_limited_receipt_body(
    request: Request,
    max_bytes: int = _MAX_RECEIPT_BODY_BYTES,
) -> bytes:
    try:
        content_length = int(request.headers.get("content-length", "0") or "0")
    except ValueError:
        raise HTTPException(400, "Invalid content-length")
    if content_length > max_bytes:
        raise HTTPException(413, "Receipt too large")

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(413, "Receipt too large")
        chunks.append(chunk)
    return b"".join(chunks)


def _clean_receipt_hex_field(value: object, *, expected_len: int = 64) -> str:
    out = str(value or "").strip().lower().removeprefix("0x")
    if len(out) != expected_len or any(c not in string.hexdigits for c in out):
        return ""
    return out

# Cache expensive proof-identity checks. The proof HTTP hot path verifies every
# SR25519 request signature, nonce, active lease, and cached hotkey ownership.
# It must not perform chain RPC, metagraph refreshes, or endpoint challenges.
_ownership_cache: dict[tuple[str, str], tuple[float, str, int]] = {}
_ownership_locks: dict[tuple[str, str], asyncio.Lock] = {}
_uid_hotkey_cache: dict[int, tuple[float, bytes]] = {}
_endpoint_identity_cache: dict[tuple[str, str, str], float] = {}
_executor_refresh_tasks: dict[str, asyncio.Task] = {}
_metagraph_refresh_task: asyncio.Task | None = None
_OWNERSHIP_CACHE_TTL = 180.0
_UID_HOTKEY_CACHE_TTL = 300.0
_IDENTITY_CACHE_TTL = 600.0
_OWNERSHIP_CACHE_CAP = 50000
_IDENTITY_CACHE_CAP = 50000
try:
    _PROOF_REGISTRY_CACHE_MAX_AGE = max(
        30.0,
        float(os.environ.get("NODEXO_PROOF_REGISTRY_CACHE_MAX_AGE_S", "900.0")),
    )
except Exception:
    _PROOF_REGISTRY_CACHE_MAX_AGE = 900.0
try:
    _CHAIN_OWNERSHIP_TIMEOUT = max(
        1.0,
        float(os.environ.get("NODEXO_PROOF_CHAIN_OWNERSHIP_TIMEOUT_S", "24.0")),
    )
except Exception:
    _CHAIN_OWNERSHIP_TIMEOUT = 24.0


def _hotkey_account_id_bytes(hotkey_ss58: str) -> bytes:
    from substrateinterface.utils.ss58 import ss58_decode

    decoded = ss58_decode(hotkey_ss58)
    return bytes.fromhex(decoded.removeprefix("0x"))


def _cached_subtensor_uid_for_hotkey(hotkey_ss58: str) -> tuple[bool, int | None]:
    """Return cached Subtensor hotkey ownership without refreshing metagraph."""
    if rpc_client is None:
        return False, None
    cached_lookup = getattr(rpc_client, "get_cached_uid_for_hotkey", None)
    if not callable(cached_lookup):
        return False, None
    try:
        hit, uid = cached_lookup(hotkey_ss58, max_age_s=_UID_HOTKEY_CACHE_TTL)
        return bool(hit), None if uid is None else int(uid)
    except Exception:
        return False, None


def _prune_ownership_cache(now: float | None = None) -> None:
    if len(_ownership_cache) <= _OWNERSHIP_CACHE_CAP:
        return
    if now is None:
        now = time.time()
    cutoff = now - _OWNERSHIP_CACHE_TTL
    for k, entry in list(_ownership_cache.items()):
        if entry[0] < cutoff:
            _ownership_cache.pop(k, None)
    if len(_ownership_cache) > _OWNERSHIP_CACHE_CAP:
        _ownership_cache.clear()


def _schedule_metagraph_cache_refresh() -> None:
    """Kick metagraph refresh in the background; never await from proof ingress."""
    global _metagraph_refresh_task
    if rpc_client is None:
        return
    get_metagraph = getattr(rpc_client, "get_metagraph", None)
    if not callable(get_metagraph):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if _metagraph_refresh_task is not None and not _metagraph_refresh_task.done():
        return

    async def _refresh() -> None:
        try:
            await asyncio.to_thread(get_metagraph, False)
            bt.logging.debug("Proof hotkey ownership cache refreshed")
        except Exception as e:
            bt.logging.debug(f"Proof hotkey ownership cache refresh failed: {e}")

    _metagraph_refresh_task = loop.create_task(_refresh())


def _schedule_executor_info_refresh(executor_id_hex: str) -> None:
    """Refresh one executor in the background; never await from proof ingress."""
    if registry_client is None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = _executor_refresh_tasks.get(executor_id_hex)
    if task is not None and not task.done():
        return

    async def _refresh() -> None:
        try:
            await _get_executor_info_cached(executor_id_hex)
        except Exception as e:
            bt.logging.debug(
                f"Proof executor cache refresh failed for {executor_id_hex[:16]}: {e}"
            )
        finally:
            _executor_refresh_tasks.pop(executor_id_hex, None)

    _executor_refresh_tasks[executor_id_hex] = loop.create_task(_refresh())


def _db_registry_executor_spec(executor_id_hex: str):
    """Load executor state from the validator-local DB index only."""
    if db_instance is None:
        return None
    try:
        from datetime import datetime, timezone

        from common.db import RegistryExecutorState
        from common.types import ExecutorSpec

        with db_instance.session() as s:
            row = s.query(RegistryExecutorState).filter_by(
                executor_id=executor_id_hex,
            ).first()
            if row is None:
                return None
            updated_at = row.updated_at
            if updated_at is not None:
                if updated_at.tzinfo is None:
                    age_s = (datetime.utcnow() - updated_at).total_seconds()
                else:
                    age_s = (datetime.now(timezone.utc) - updated_at).total_seconds()
                if age_s > _PROOF_REGISTRY_CACHE_MAX_AGE:
                    return None
            return ExecutorSpec(
                executor_id=row.executor_id,
                miner_address=row.miner_address or "",
                endpoint=row.endpoint or "",
                gpu_model_hash=row.gpu_model_hash or "",
                gpu_count=int(row.gpu_count or 0),
                vram_mb=int(row.vram_mb or 0),
                price_per_gpu_hour=int(row.price_per_gpu_hour or 0),
                expires_at=int(row.expires_at or 0),
                is_active=bool(row.is_active),
                is_rented=bool(row.is_rented),
                miner_uid=int(row.miner_uid or 0),
                miner_registered=bool(row.miner_registered),
                uid_owner_match=bool(row.uid_owner_match),
            )
    except Exception as e:
        bt.logging.debug(f"DB registry executor cache unavailable: {e}")
        return None


def _get_executor_info_local(executor_id_hex: str):
    """Return executor state from local caches only.

    This function is safe for /proofs/* ingress: it never calls chain RPC.
    """
    now = time.time()
    cached = _executor_info_cache.get(executor_id_hex)
    if cached:
        ts, spec = cached
        if (
            spec is not None
            and bool(getattr(spec, "is_active", False))
            and now - ts <= _PROOF_REGISTRY_CACHE_MAX_AGE
            and int(getattr(spec, "expires_at", 0) or 0) > int(now)
        ):
            return spec

    if chain_snapshot is not None:
        try:
            snap = chain_snapshot.get(executor_id_hex)
            age_fn = getattr(chain_snapshot, "age_seconds", None)
            age_s = age_fn() if callable(age_fn) else None
            if (
                snap is not None
                and (age_s is None or float(age_s) <= _PROOF_REGISTRY_CACHE_MAX_AGE)
            ):
                _executor_info_cache[executor_id_hex] = (now, snap)
                return snap
        except Exception:
            pass

    spec = _db_registry_executor_spec(executor_id_hex)
    if spec is not None:
        _executor_info_cache[executor_id_hex] = (now, spec)
        return spec
    return None


def _verify_cached_ownership(
    executor_id: str,
    hotkey_ss58: str,
    spec,
    existing_hotkey: str | None,
) -> None:
    """Verify hotkey ownership using only local cache state."""
    owner = str(getattr(spec, "miner_address", "") or "")
    if not owner or owner.lower() == "0x0000000000000000000000000000000000000000":
        raise HTTPException(403, "Executor owner missing")
    if getattr(spec, "miner_registered", None) is not True:
        raise HTTPException(403, "Executor owner EVM is not registered")
    if getattr(spec, "uid_owner_match", None) is not True:
        raise HTTPException(403, "On-chain UID→EVM mismatch for executor")
    uid_raw = getattr(spec, "miner_uid", None)
    if uid_raw is None:
        raise HTTPException(503, "Executor UID cache unavailable")
    uid = int(uid_raw)

    now = time.time()
    cache_key = (executor_id, hotkey_ss58)
    cached = _ownership_cache.get(cache_key)
    if cached and now - cached[0] < _OWNERSHIP_CACHE_TTL:
        cached_owner, cached_uid = cached[1], int(cached[2])
        if cached_owner.lower() == owner.lower() and cached_uid == uid:
            return

    cached_metagraph_checked, cached_uid = _cached_subtensor_uid_for_hotkey(
        hotkey_ss58
    )
    if not cached_metagraph_checked:
        _schedule_metagraph_cache_refresh()
        raise HTTPException(503, "Hotkey ownership cache unavailable")
    if cached_uid is None:
        raise HTTPException(403, "Proof hotkey not registered on subnet")
    if uid != int(cached_uid):
        if existing_hotkey is not None and existing_hotkey != hotkey_ss58:
            bt.logging.warning(
                f"Hotkey rotation rejected: executor={executor_id[:16]} "
                f"bound={existing_hotkey[:16]} request={hotkey_ss58[:16]}"
            )
        raise HTTPException(403, "Proof hotkey does not own executor UID")

    _ownership_cache[cache_key] = (now, owner, uid)
    _prune_ownership_cache(now)


async def _get_evm_metagraph_hotkey_for_uid(uid: int, get_hotkey_for_uid) -> bytes:
    """Return UID hotkey bytes, cached across first-bind proof bursts."""
    import asyncio as _asyncio

    now = time.time()
    cached = _uid_hotkey_cache.get(int(uid))
    if cached and now - cached[0] < _UID_HOTKEY_CACHE_TTL:
        return cached[1]
    value = await _asyncio.to_thread(get_hotkey_for_uid, int(uid))
    value_bytes = bytes(value)
    _uid_hotkey_cache[int(uid)] = (time.time(), value_bytes)
    if len(_uid_hotkey_cache) > 10000:
        cutoff = time.time() - _UID_HOTKEY_CACHE_TTL
        for k, entry in list(_uid_hotkey_cache.items()):
            if entry[0] < cutoff:
                _uid_hotkey_cache.pop(k, None)
        if len(_uid_hotkey_cache) > 10000:
            _uid_hotkey_cache.clear()
    return value_bytes


def _record_pass_recv(executor_id: str, epoch_id: int, gpu_index: int,
                       u: int, recv_time: float, root: str = "",
                       t_commit: str = "", hotkey_ss58: str = "") -> None:
    key = (executor_id, epoch_id, gpu_index)
    if key not in _pass_recv:
        if len(_pass_recv) >= _PASS_RECV_MAX:
            # Crude eviction: drop oldest 10%
            for k in list(_pass_recv.keys())[:_PASS_RECV_MAX // 10]:
                _pass_recv.pop(k, None)
                for u in (0, 1):
                    _pass_recv_meta.pop((k[0], k[1], k[2], u), None)
        _pass_recv[key] = {}
    _pass_recv[key][u] = recv_time
    _pass_recv_meta[(executor_id, epoch_id, gpu_index, u)] = {
        "recv_time": recv_time,
        "root": (root or "").lower().removeprefix("0x"),
        "t_commit": (t_commit or "").lower().removeprefix("0x"),
        "hotkey_ss58": hotkey_ss58 or "",
    }


def get_external_pass_delta(executor_id: str, epoch_id: int) -> float:
    """Return the external pass_1 - pass_0 delta for an (executor,
    epoch). Averages over per-GPU deltas if multiple GPUs reported.
    Returns 0 if either pass receipt is missing.
    """
    deltas = []
    for (eid, ep, _gi), passes in _pass_recv.items():
        if eid != executor_id or ep != epoch_id:
            continue
        t0 = passes.get(0)
        t1 = passes.get(1)
        if t0 and t1 and t1 > t0:
            deltas.append(t1 - t0)
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)


def get_bound_pass_delta(recipe_data: dict) -> float:
    """Return mean receipt delta only when receipts match final recipe roots.

    This prevents timing receipts from being spoofed independently of the
    recipe that was ultimately verified.
    """
    stats = get_bound_pass_timing_stats(recipe_data)
    return float(stats.get("mean_delta") or 0.0)


def get_bound_pass_timing_stats(recipe_data: dict) -> dict[str, float]:
    """Return bound receipt timing stats for recipe-matching GPU receipts.

    The validator uses max_delta for security decisions. mean_delta is retained
    for telemetry. Missing or mismatched receipts are ignored and reflected in
    `count`, letting the caller decide whether timing evidence is mandatory.
    """
    if db_instance is not None:
        try:
            from common.db import get_bound_proof_receipt_timing_stats
            db_stats = get_bound_proof_receipt_timing_stats(db_instance, recipe_data)
            if float(db_stats.get("count") or 0.0) > 0:
                return db_stats
        except Exception as e:
            bt.logging.debug(f"DB receipt timing lookup failed: {e}")

    executor_id = str(recipe_data.get("executor_id") or "")
    epoch_id = int(recipe_data.get("epoch_id") or 0)
    allowed_hotkey = str(recipe_data.get("_nodexo_hotkey_ss58") or "")
    deltas = []
    pass0_times = []
    pass1_times = []
    for gp in recipe_data.get("gpu_proofs") or []:
        try:
            gpu_index = int(gp.get("gpu_index"))
        except Exception:
            continue
        p0 = gp.get("pass_0") or {}
        p1 = gp.get("pass_1") or {}
        root0 = str(p0.get("merkle_root") or "").lower().removeprefix("0x")
        root1 = str(p1.get("merkle_root") or "").lower().removeprefix("0x")
        tc0 = str(gp.get("T_commit_0") or "").lower().removeprefix("0x")
        tc1 = str(gp.get("T_commit_1") or "").lower().removeprefix("0x")
        r0 = _pass_recv_meta.get((executor_id, epoch_id, gpu_index, 0))
        r1 = _pass_recv_meta.get((executor_id, epoch_id, gpu_index, 1))
        if not r0 or not r1:
            continue
        if allowed_hotkey:
            if r0.get("hotkey_ss58") and r0.get("hotkey_ss58") != allowed_hotkey:
                continue
            if r1.get("hotkey_ss58") and r1.get("hotkey_ss58") != allowed_hotkey:
                continue
        if r0.get("root") != root0 or r1.get("root") != root1:
            continue
        if r0.get("t_commit") != tc0 or r1.get("t_commit") != tc1:
            continue
        t0 = float(r0.get("recv_time") or 0)
        t1 = float(r1.get("recv_time") or 0)
        if t0 > 0 and t1 > t0:
            deltas.append(t1 - t0)
            pass0_times.append(t0)
            pass1_times.append(t1)
    if not deltas:
        return {
            "count": 0.0,
            "mean_delta": 0.0,
            "min_delta": 0.0,
            "max_delta": 0.0,
            "pass0_spread": 0.0,
            "pass1_spread": 0.0,
            "min_pass0_recv": 0.0,
            "max_pass0_recv": 0.0,
            "min_pass1_recv": 0.0,
            "max_pass1_recv": 0.0,
        }
    return {
        "count": float(len(deltas)),
        "mean_delta": sum(deltas) / len(deltas),
        "min_delta": min(deltas),
        "max_delta": max(deltas),
        "pass0_spread": max(pass0_times) - min(pass0_times) if len(pass0_times) > 1 else 0.0,
        "pass1_spread": max(pass1_times) - min(pass1_times) if len(pass1_times) > 1 else 0.0,
        "min_pass0_recv": min(pass0_times),
        "max_pass0_recv": max(pass0_times),
        "min_pass1_recv": min(pass1_times),
        "max_pass1_recv": max(pass1_times),
    }

# Hotkey→executor ownership binding (first-claim + chain validation).
# Maps executor_id → authorized hotkey_ss58. DB-backed (audit C-7 fix):
# without persistence, a validator restart wiped this dict and any
# hotkey could re-claim ownership of any executor in the first proof
# of the next cycle. _load_executor_bindings_from_db is called from
# the validator's lifespan after the DB is ready.
_executor_hotkey_binding: dict[str, str] = {}

# Set by validator.lifespan once db is initialized — caching the db
# reference lets the write path persist new bindings without an
# import dance.
db_instance = None


def _load_executor_bindings_from_db() -> None:
    """Hydrate _executor_hotkey_binding from the DB. Called once at
    validator startup so existing bindings survive restart."""
    global _executor_hotkey_binding
    if db_instance is None:
        return
    try:
        from common.db import get_all_executor_hotkey_bindings
        rows = get_all_executor_hotkey_bindings(db_instance)
        _executor_hotkey_binding.update(rows)
        bt.logging.info(
            f"Loaded {len(rows)} executor hotkey bindings from DB"
        )
    except Exception as e:
        bt.logging.warning(f"_load_executor_bindings_from_db: {e}")


def _persist_executor_binding(executor_id: str, hotkey_ss58: str) -> None:
    """Write-through to DB on every new/changed binding."""
    if db_instance is None:
        return
    try:
        from common.db import set_executor_hotkey_binding
        set_executor_hotkey_binding(db_instance, executor_id, hotkey_ss58)
    except Exception as e:
        bt.logging.warning(f"_persist_executor_binding: {e}")

# Max body size (32KB should be enough for a recipe with 4 challenged blocks × 500 spot checks)
_MAX_BODY_SIZE = 2 * 1024 * 1024  # 2MB (production recipes with spot checks are ~500KB-1MB)

# In-memory cache for ComputeRegistry.get_executor_info, keyed by executor_id_hex.
# At a cycle-boundary spike, every recipe POST hits the chain twice for the same
# executor (once in _check_registered_executor, once in _verify_request_signature).
# Cache reduces 2N RPCs/spike to ~2 (one fill per executor that hasn't been seen
# in 60s), eliminating the cycle-boundary 429 storms on testnet.
_executor_info_cache: dict[str, tuple[float, "object | None"]] = {}
_EXECUTOR_INFO_TTL = 60.0  # seconds

VERIFY_PRIORITY_PREFIX = b"NODEXO_VERIFY_PRIORITY_V1"


def _verification_priority(epoch_id: int, executor_id: str) -> int:
    """Per-cycle fair ordering for overload protection.

    When a validator cannot process every submitted proof, this priority makes
    the verified subset rotate by epoch and validator identity instead of
    consistently favoring the fastest network paths.
    """
    try:
        executor_bytes = bytes.fromhex(executor_id)
    except ValueError:
        executor_bytes = executor_id.encode("utf-8")
    salt = str(verification_priority_salt or "validator").encode("utf-8")
    payload = (
        VERIFY_PRIORITY_PREFIX
        + salt
        + int(epoch_id).to_bytes(8, "big", signed=False)
        + executor_bytes
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _enqueue_verification_or_skip(body: dict, recv_time: float) -> bool:
    """Queue a recipe, replacing a lower-priority queued recipe if needed.

    Returns True when the recipe will be verified. Returns False when it was
    accepted but skipped due to local validator capacity.
    """
    if verification_queue is None:
        return True

    epoch_id = int(body.get("epoch_id") or 0)
    executor_id = str(body.get("executor_id") or "")
    priority = _verification_priority(epoch_id, executor_id)
    entry = (priority, recv_time, {"recipe": body, "recv_time": recv_time})

    try:
        verification_queue.put_nowait(entry)
        return True
    except asyncio.QueueFull:
        # PriorityQueue keeps a min-heap. If a better-priority proof arrives
        # after the queue filled, replace the current worst queued proof so
        # overload sampling is priority-based, not arrival-order based.
        queued = getattr(verification_queue, "_queue", None)
        if queued:
            worst_idx = max(range(len(queued)), key=lambda i: queued[i][0])
            worst = queued[worst_idx]
            if priority < worst[0]:
                queued[worst_idx] = entry
                heapq.heapify(queued)
                evicted = worst[2]
                _record_validator_capacity_skip(evicted["recipe"], evicted["recv_time"])
                return True

        _record_validator_capacity_skip(body, recv_time)
        return False


def _record_validator_capacity_skip(
    recipe: dict,
    recv_time: float,
    reason: str = "validator_capacity_skip",
) -> None:
    """Record a neutral skip when this validator cannot verify.

    This is explicitly not a cryptographic verification. It is persisted for
    auditability and bypasses ProofAnalyzer so it does not change pass-rate or
    timing baselines. For already-known-good executors we store it as valid so
    validator-side outages do not age out the recent-proof gate; for never
    verified executors it remains non-valid and cannot bootstrap eligibility.
    """
    try:
        if db_instance is None:
            return
        from common.db import ProofResult, store_proof_result
        executor_id = str(recipe.get("executor_id") or "")
        has_real_valid_proof = False
        with db_instance.session() as s:
            has_real_valid_proof = s.query(ProofResult).filter(
                ProofResult.executor_id == executor_id,
                ProofResult.valid == True,  # noqa: E712
                ProofResult.tier != "skipped",
            ).first() is not None
        store_proof_result(
            db_instance,
            executor_id,
            int(recipe.get("epoch_id") or 0),
            valid=has_real_valid_proof,
            tier="skipped",
            reason=reason,
            compute_time=float(recipe.get("compute_time") or 0),
            recv_time=float(recv_time or 0),
        )
    except Exception as e:
        bt.logging.debug(f"capacity skip record failed: {e}")


async def _get_executor_info_cached(executor_id_hex: str):
    """Fetch and cache ExecutorSpec for an executor_id. None on RPC failure.

    Background refresh helper only. Do not call this from proof HTTP ingress.
    Prefer the validator-wide ChainSnapshot when it is fresh. That keeps the
    proof hot path off the EVM RPC even when thousands of executors broadcast
    around the same cycle boundary. If the snapshot is stale and direct RPC is
    rate-limited, fall back to the stale snapshot for unexpired executors. That
    prevents public-RPC outages from turning honest proof submissions into hard
    proof failures. Rental availability still uses the ChainSnapshot/read gates;
    this fallback only supplies hardware/owner context for proof verification.
    """
    now = time.time()
    stale_snapshot_spec = None
    if chain_snapshot is not None:
        try:
            snap = chain_snapshot.get(executor_id_hex)
            if snap is not None and chain_snapshot.is_fresh():
                _executor_info_cache[executor_id_hex] = (now, snap)
                return snap
            if snap is not None and getattr(snap, "expires_at", 0) > int(now):
                stale_snapshot_spec = snap
        except Exception:
            pass
    if registry_client is None:
        return None
    import asyncio as _asyncio
    cached = _executor_info_cache.get(executor_id_hex)
    if cached and now - cached[0] < _EXECUTOR_INFO_TTL:
        return cached[1]
    try:
        spec = await _asyncio.wait_for(
            _asyncio.to_thread(
                registry_client.get_executor_info, bytes.fromhex(executor_id_hex)
            ),
            timeout=3.0,
        )
    except _asyncio.TimeoutError:
        bt.logging.debug(f"get_executor_info timeout for {executor_id_hex[:16]}")
        if stale_snapshot_spec is not None:
            _executor_info_cache[executor_id_hex] = (now, stale_snapshot_spec)
            return stale_snapshot_spec
        return None
    except Exception as e:
        bt.logging.debug(f"get_executor_info failed for {executor_id_hex[:16]}: {e}")
        if stale_snapshot_spec is not None:
            _executor_info_cache[executor_id_hex] = (now, stale_snapshot_spec)
            return stale_snapshot_spec
        return None
    if spec is None and stale_snapshot_spec is not None:
        _executor_info_cache[executor_id_hex] = (now, stale_snapshot_spec)
        return stale_snapshot_spec
    _executor_info_cache[executor_id_hex] = (now, spec)
    # Cap cache size to avoid unbounded growth in pathological cases.
    if len(_executor_info_cache) > 10000:
        _executor_info_cache.clear()
    return spec


async def _verify_endpoint_identity_cached(executor_id: str, spec) -> None:
    """Verify registered endpoint controls the executor owner's EVM key."""
    import asyncio as _asyncio
    import secrets
    import aiohttp

    from common.identity import (
        normalize_evm_address, recover_identity_challenge,
    )

    endpoint = (getattr(spec, "endpoint", "") or "").strip()
    owner = normalize_evm_address(getattr(spec, "miner_address", "") or "")
    if not endpoint:
        raise HTTPException(403, "Executor endpoint missing")
    cache_key = (executor_id, endpoint, owner)
    now = time.time()
    cached = _endpoint_identity_cache.get(cache_key)
    if cached and now - cached < _IDENTITY_CACHE_TTL:
        return

    nonce = secrets.token_bytes(32)
    url = endpoint.rstrip("/") + "/identity/challenge"
    timeout = aiohttp.ClientTimeout(total=5, connect=2)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url,
                json={"nonce": nonce.hex(), "executor_id": executor_id},
            ) as resp:
                if resp.status != 200:
                    detail = await resp.text()
                    raise HTTPException(
                        403,
                        f"Endpoint identity challenge failed ({resp.status}): {detail[:160]}",
                    )
                data = await resp.json()
    except HTTPException:
        raise
    except (_asyncio.TimeoutError, Exception) as e:
        raise HTTPException(403, f"Endpoint identity challenge failed: {e}")

    claimed = normalize_evm_address(str(data.get("address") or ""))
    if claimed != owner:
        raise HTTPException(403, "Endpoint identity address mismatch")
    recovered = recover_identity_challenge(
        nonce, owner, executor_id, str(data.get("signature") or ""),
    )
    if recovered != owner:
        raise HTTPException(403, "Endpoint identity signature mismatch")
    _endpoint_identity_cache[cache_key] = now
    if len(_endpoint_identity_cache) > _IDENTITY_CACHE_CAP:
        cutoff = now - _IDENTITY_CACHE_TTL
        for k, ts in list(_endpoint_identity_cache.items()):
            if ts < cutoff:
                _endpoint_identity_cache.pop(k, None)
        if len(_endpoint_identity_cache) > _IDENTITY_CACHE_CAP:
            _endpoint_identity_cache.clear()


def _check_rate_limit(executor_id: str, tracker: dict, label: str):
    """Rate limit per executor. Raises 429 if too frequent."""
    now = time.time()
    last = tracker.get(executor_id, 0)
    if now - last < _MIN_INTERVAL:
        raise HTTPException(429, f"Rate limited: {label} from {executor_id[:16]} (wait {_MIN_INTERVAL}s)")
    tracker[executor_id] = now


async def _check_registered_executor(executor_id: str):
    """Verify executor_id is live using validator-local registry cache only.

    `isActive` alone is sticky (only flipped by reportOffline quorum or
    self-deregister), so without the `expires_at > now` clause an executor that
    stops renewing keeps getting its proofs scored forever.
    """
    import time as _time
    if registry_client is None:
        return  # No registry = accept all (dev mode)
    spec = _get_executor_info_local(executor_id)
    if spec is None:
        _schedule_executor_info_refresh(executor_id)
        raise HTTPException(503, f"Executor registry cache unavailable: {executor_id[:16]}")
    if not spec.is_active:
        raise HTTPException(403, f"Inactive executor: {executor_id[:16]}")
    if spec.expires_at <= int(_time.time()):
        raise HTTPException(403, f"Lease expired: {executor_id[:16]}")


async def _verify_request_signature(request: Request, body_bytes: bytes, executor_id: str):
    """Verify SR25519 signature on the request.

    Two-layer verification:
      1. Cryptographic: SR25519 signature is valid and fresh.
      2. Ownership: The signing hotkey is bound to this executor_id.

    Ownership uses first-claim binding: the first hotkey to submit for an
    executor_id is cached as the authorized signer. A different hotkey
    claiming the same executor is rejected.

    When the chain registry is available, the binding is validated from
    validator-local registry/metagraph caches. The proof ingress path must not
    perform chain RPC, metagraph refreshes, or endpoint identity probes.
    """
    from common.crypto import extract_signature_headers, verify_signature

    headers = dict(request.headers)
    sig_parts = extract_signature_headers(headers)

    if sig_parts is None:
        raise HTTPException(401, "Missing signature headers (X-Nodexo-Signature, X-Nodexo-Hotkey, X-Nodexo-Timestamp, X-Nodexo-Nonce)")

    sig_hex, hotkey_ss58, timestamp, nonce = sig_parts

    # Layer 1: Verify the cryptographic signature
    if not await asyncio.to_thread(
        verify_signature,
        body_bytes,
        sig_hex,
        hotkey_ss58,
        timestamp,
        nonce,
    ):
        raise HTTPException(401, "Invalid signature or expired request")

    # Layer 2: Verify hotkey→executor ownership binding (DB-backed).
    # A stored mismatch is not rejected before the local ownership-cache check:
    # honest hotkey rotation is allowed once the cached UID→hotkey mapping
    # proves it.
    existing_hotkey = _executor_hotkey_binding.get(executor_id)
    if existing_hotkey is not None and existing_hotkey != hotkey_ss58 and registry_client is None:
        bt.logging.warning(
            f"Hotkey mismatch: executor={executor_id[:16]} bound to {existing_hotkey[:16]}, request from {hotkey_ss58[:16]}"
        )
        raise HTTPException(403, "Hotkey does not match executor binding")

    # Layer 2b: Local-cache ownership validation. Cache miss is a validator
    # readiness problem; do not shift chain/RPC latency into miner proof timing.
    if registry_client is not None:
        spec = _get_executor_info_local(executor_id)
        if spec is None:
            _schedule_executor_info_refresh(executor_id)
            raise HTTPException(503, "Executor registry cache unavailable")
        _verify_cached_ownership(
            executor_id,
            hotkey_ss58,
            spec,
            existing_hotkey,
        )

    # Replay protection — even a valid signature cannot be reused inside
    # the freshness window. Record the nonce only after expensive ownership
    # checks complete; otherwise a client-side timeout during local cache/auth
    # checks can poison the retry with "Replay detected" before the proof is
    # accepted.
    from common.crypto import check_and_record_nonce
    if not await asyncio.to_thread(
        check_and_record_nonce,
        "validator-ingress",
        hotkey_ss58,
        nonce,
        body_bytes,
    ):
        raise HTTPException(401, "Replay detected")

    if existing_hotkey != hotkey_ss58:
        for k in list(_ownership_cache):
            if k[0] == executor_id and k[1] != hotkey_ss58:
                _ownership_cache.pop(k, None)
        _executor_hotkey_binding[executor_id] = hotkey_ss58
        _persist_executor_binding(executor_id, hotkey_ss58)
        action = "rotated" if existing_hotkey else "bound"
        bt.logging.info(
            f"Hotkey binding {action}: executor={executor_id[:16]} → hotkey={hotkey_ss58[:16]}"
        )
    return hotkey_ss58


async def _verify_request_signature_crypto(request: Request, body_bytes: bytes) -> str:
    """Verify request signature freshness/replay without ownership checks.

    Timing-critical receipts use this fast path. The final recipe still goes
    through _verify_request_signature(), which enforces local registry and
    hotkey ownership before any proof can score.
    """
    from common.crypto import extract_signature_headers

    sig_parts = extract_signature_headers(dict(request.headers))
    if sig_parts is None:
        raise HTTPException(401, "Missing signature headers (X-Nodexo-Signature, X-Nodexo-Hotkey, X-Nodexo-Timestamp, X-Nodexo-Nonce)")

    sig_hex, hotkey_ss58, timestamp, nonce = sig_parts
    loop = asyncio.get_running_loop()
    ok, replay = await loop.run_in_executor(
        _RECEIPT_CRYPTO_EXECUTOR,
        _verify_and_record_receipt_signature,
        body_bytes,
        sig_hex,
        hotkey_ss58,
        timestamp,
        nonce,
    )
    if replay:
        raise HTTPException(401, "Replay detected")
    if not ok:
        raise HTTPException(401, "Invalid signature or expired request")

    return hotkey_ss58


def _verify_and_record_receipt_signature(
    body_bytes: bytes,
    sig_hex: str,
    hotkey_ss58: str,
    timestamp: str,
    nonce: str,
) -> tuple[bool, bool]:
    """Return (signature_ok, replay_seen) for timing-critical receipts."""
    from common.crypto import check_and_record_nonce, verify_signature

    if not verify_signature(body_bytes, sig_hex, hotkey_ss58, timestamp, nonce):
        return False, False
    if not check_and_record_nonce(
        "validator-ingress",
        hotkey_ss58,
        nonce,
        body_bytes,
    ):
        return False, True
    return True, False


@router.post("/commit")
async def receive_commitment(request: Request):
    """Phase A: Receive pass_0 commitment from an executor."""
    # Timing analysis uses validator-side ingress time. Capture it before body
    # parsing, local registry checks, signature checks, or rate limiting.
    recv_time = time.time()
    # Size check
    content_length = request.headers.get("content-length", "0")
    if int(content_length) > _MAX_BODY_SIZE:
        raise HTTPException(413, "Payload too large")

    body_bytes = await request.body()
    body = json.loads(body_bytes)
    executor_id = body.get("executor_id", "")
    epoch_id = body.get("epoch_id", 0)

    if not executor_id or not epoch_id:
        raise HTTPException(400, "executor_id and epoch_id required")

    await _check_registered_executor(executor_id)
    await _verify_request_signature(request, body_bytes, executor_id)
    _check_rate_limit(executor_id, _last_commit, "commit")

    # Record commitment arrival time
    if analyzer:
        analyzer.record_commitment(executor_id, epoch_id, recv_time)

    bt.logging.debug(f"Commitment received: executor={executor_id[:16]} epoch={epoch_id}")
    return {"status": "accepted", "epoch_id": epoch_id}


@router.post("/receipt")
async def receive_receipt(request: Request):
    """Phase A (per-GPU): Receive per-GPU per-pass receipt.

    Sent immediately after each GPU completes pass_0 or pass_1.
    The validator records its own recv_time for delta measurement.
    """
    # Stamp ingress before body parsing, signature checks, registry cache reads,
    # or any other validator-local work. This timestamp is the timing signal.
    recv_time = time.time()

    body_bytes = await _read_limited_receipt_body(request)
    # Receipts are deliberately tiny. Keep this parse on the event loop instead
    # of the shared default executor; public-RPC chain calls can occupy that
    # executor, and receipt ingress must stamp and record without queueing behind
    # unrelated validator work.
    try:
        body = json.loads(body_bytes)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Receipt JSON must be an object")
    executor_id = _clean_receipt_hex_field(body.get("executor_id"))
    root = _clean_receipt_hex_field(body.get("root"))
    t_commit = _clean_receipt_hex_field(body.get("T_commit") or body.get("t_commit"))
    try:
        epoch_id = int(body.get("epoch_id") or 0)
        gpu_index = int(body.get("gpu_index") if body.get("gpu_index") is not None else -1)
        u = int(body.get("u") if body.get("u") is not None else -1)
    except (TypeError, ValueError):
        raise HTTPException(400, "Invalid receipt numeric field")

    if (
        not executor_id
        or epoch_id <= 0
        or gpu_index < 0
        or gpu_index > 1024
        or u not in {0, 1}
        or not root
        or not t_commit
    ):
        raise HTTPException(
            400,
            "valid executor_id, epoch_id, gpu_index, u, root, and T_commit required",
        )

    hotkey_ss58 = await _verify_request_signature_crypto(request, body_bytes)
    existing_hotkey = _executor_hotkey_binding.get(executor_id)
    if existing_hotkey is not None and existing_hotkey != hotkey_ss58:
        raise HTTPException(403, "Hotkey does not match executor binding")

    # Record validator-side HTTP arrival time (this is the unfakeable
    # timestamp). Receipt ingress is intentionally independent from local
    # registry/metagraph cache misses; final recipe verification enforces
    # ownership before any proof can score.
    if analyzer:
        analyzer.record_receipt(executor_id, epoch_id, gpu_index, u, recv_time)

    if db_instance is not None:
        loop = asyncio.get_running_loop()
        try:
            from common.db import store_proof_receipt
            await loop.run_in_executor(
                _RECEIPT_STORE_EXECUTOR,
                partial(
                    store_proof_receipt,
                    db_instance,
                    executor_id=executor_id,
                    epoch_id=int(epoch_id),
                    gpu_index=int(gpu_index),
                    u=int(u),
                    recv_time=float(recv_time),
                    root=root,
                    t_commit=t_commit,
                    hotkey_ss58=hotkey_ss58,
                ),
            )
        except Exception as e:
            bt.logging.warning(
                f"Receipt store failed: executor={executor_id[:16]} "
                f"epoch={epoch_id} gpu={gpu_index} u={u}: {e}"
            )
            raise HTTPException(503, "receipt store unavailable")

    # ALSO record in a separate map for the external_pass_delta signal.
    # Network jitter cancels when we take the delta between pass_0 and
    # pass_1 receipts (same path, similar RTT). Stored keyed by
    # (executor_id, epoch_id, gpu_index) so we can compute the delta
    # when both passes have arrived.
    _record_pass_recv(
        executor_id, epoch_id, gpu_index, int(u), recv_time,
        root=root, t_commit=t_commit, hotkey_ss58=hotkey_ss58,
    )

    bt.logging.debug(
        f"Receipt: executor={executor_id[:16]} epoch={epoch_id} gpu={gpu_index} u={u}"
    )
    return {"status": "accepted"}


@router.post("/recipe")
async def receive_recipe(request: Request):
    """Phase B: Receive full recipe from an executor."""
    # Queue timing is based on HTTP ingress, not on validator-local auth,
    # duplicate, or rate-limit work.
    recv_time = time.time()
    content_length = request.headers.get("content-length", "0")
    if int(content_length) > _MAX_BODY_SIZE:
        raise HTTPException(413, "Payload too large")

    body_bytes = await request.body()
    body = await asyncio.to_thread(json.loads, body_bytes)
    executor_id = body.get("executor_id", "")
    epoch_id = body.get("epoch_id", 0)

    if not executor_id or not epoch_id:
        raise HTTPException(400, "executor_id and epoch_id required")

    # Security: reject unknown executors
    await _check_registered_executor(executor_id)

    # Security: verify SR25519 signature
    recipe_hotkey_ss58 = await _verify_request_signature(request, body_bytes, executor_id)
    body["_nodexo_hotkey_ss58"] = recipe_hotkey_ss58

    # Rate limit
    _check_rate_limit(executor_id, _last_recipe, "recipe")

    # Duplicate epoch rejection — same (executor, epoch) already received
    dedup_key = (executor_id, epoch_id)
    if dedup_key in _received_recipes:
        raise HTTPException(409, f"Duplicate: already received recipe for epoch {epoch_id}")
    _received_recipes.add(dedup_key)
    # Prune old entries
    if len(_received_recipes) > 10000:
        _received_recipes.clear()

    # Record recipe arrival time
    if analyzer:
        analyzer.record_recipe_time(executor_id, epoch_id, recv_time)

    if follower_mode:
        bt.logging.debug(
            f"Recipe accepted in follower mode: executor={executor_id[:16]} epoch={epoch_id}"
        )
        return {
            "status": "accepted",
            "epoch_id": epoch_id,
            "verification": "follower_mode",
        }
    if verification_queue is None:
        bt.logging.error(
            f"Verification queue unavailable; rejecting proof "
            f"executor={executor_id[:16]} epoch={epoch_id}"
        )
        raise HTTPException(503, "verification queue unavailable")

    # Queue for verification. If this validator is at capacity, accept the
    # proof as unverified instead of blocking the HTTP handler until the miner
    # times out. The queue is priority-ordered per cycle so overload sampling
    # rotates across executors instead of always favoring the earliest arrivals.
    if not _enqueue_verification_or_skip(body, recv_time):
        bt.logging.warning(
            f"Verification queue full; accepted unverified proof "
            f"executor={executor_id[:16]} epoch={epoch_id}"
        )
        return {
            "status": "accepted",
            "epoch_id": epoch_id,
            "verification": "skipped_capacity",
        }

    try:
        qsize = verification_queue.qsize() if verification_queue is not None else -1
    except Exception:
        qsize = -1
    bt.logging.info(
        f"Recipe queued: executor={executor_id[:16]} epoch={epoch_id} qsize={qsize}"
    )
    return {"status": "accepted", "epoch_id": epoch_id}
