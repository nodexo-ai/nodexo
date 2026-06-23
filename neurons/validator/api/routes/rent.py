"""
Rental API — user initiates GPU rental through a validator.

The validator selects an executor via randomized weighted algorithm,
marks it as rented on-chain (atomic lock), then provisions a Sysbox container.

Reference: architecture docs Section 10 (Rental Flow)
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import random
import re
import time
from typing import Any, Optional

import aiohttp
import bittensor as bt
from fastapi import APIRouter, HTTPException, Request
router = APIRouter(tags=["rental"])


# ── Per-rental locks + global structure lock ─────────────────────
#
# `_active_rentals` is touched from: the rent endpoint, the HTTP DELETE
# endpoint, the TTL sweeper, the orphan sweeper, the auto-cancel-on-
# proof-fail task, and the startup reconciler. Without serialization
# two terminations of the same rental race: both pop the dict, both
# markAvailable, both delete the DB row — and the second one races a
# new markRented on the same executor.
#
# We use one async lock per rental_id, plus a structure lock for the
# locks dict itself.
_rental_locks: dict[str, asyncio.Lock] = {}
_locks_struct_lock = asyncio.Lock()


async def _get_rental_lock(rental_id: str) -> asyncio.Lock:
    async with _locks_struct_lock:
        lock = _rental_locks.get(rental_id)
        if lock is None:
            lock = asyncio.Lock()
            _rental_locks[rental_id] = lock
        return lock


async def _drop_rental_lock(rental_id: str) -> None:
    """Drop the lock after a rental fully terminates. Bounded growth."""
    async with _locks_struct_lock:
        _rental_locks.pop(rental_id, None)


# Set by validator daemon
registry_client = None
scoring_data = None
rental_orchestrator = None
db = None  # common.db.Database instance
chain_snapshot = None  # neurons.validator.state.chain_snapshot.ChainSnapshot
evm_write_enabled = True

# Block-anchored rental window index — see neurons/validator/rental/state.py.
# Verify_recipe consults this to decide if a recipe's epoch block was inside a
# rental window, avoiding the race where the chain `is_rented` flips between
# proof generation and verification.
from neurons.validator.rental.state import RentalState
rental_state: RentalState = RentalState()

# Active rental tracking (rental_id → metadata) — in-memory + DB backed
_active_rentals: dict[str, dict] = {}

RENTAL_CONTAINER_MISSING_REASON = "rental_container_missing"
RENTAL_CONTAINER_STOPPED_REASON = "rental_container_stopped"
RENTAL_ENDPOINT_UNREACHABLE_REASON = "rental_endpoint_unreachable"
RENTAL_CHAIN_NOT_LIVE_REASON = "rental_chain_not_live"
RENTAL_PORT_UNREACHABLE_REASON = "rental_ssh_port_unreachable"
_CONTAINER_BAD_STATES = {"dead", "exited", "removing"}
_container_missing_streaks: dict[str, int] = {}
_endpoint_failure_streaks: dict[str, int] = {}
_chain_failure_streaks: dict[str, int] = {}
_port_failure_streaks: dict[str, int] = {}
_CLIENT_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{8,128}$")


def _env_int(name: str, default: int, min_value: int = 0) -> int:
    try:
        return max(min_value, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, min_value: float = 0.0) -> float:
    try:
        return max(min_value, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


_GIB = 1 << 30


async def _selected_executor_live_check(executor: Any) -> tuple[bool, str]:
    """Final bounded liveness check before taking an on-chain rental lock.

    Inventory freshness is maintained by the background endpoint prober, but a
    miner can disappear seconds after its last heartbeat. The rent path checks
    only the executor it is about to lock, so this does not fan out across the
    fleet under renter load.
    """
    endpoint = (getattr(executor, "endpoint", "") or "").rstrip("/")
    if not endpoint:
        return False, "missing endpoint"
    timeout_s = _env_float("RENT_SELECTED_HEALTH_TIMEOUT_S", 2.5, min_value=0.25)
    timeout = aiohttp.ClientTimeout(total=timeout_s, connect=min(1.0, timeout_s))
    url = endpoint + "/health"
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return False, f"health_http_{resp.status}: {body[:120]}"
                data = await resp.json(content_type=None)
    except asyncio.TimeoutError:
        return False, "health_timeout"
    except Exception as e:
        return False, f"health_error:{type(e).__name__}"

    if str(data.get("status", "")).lower() != "ok":
        return False, f"health_status={data.get('status')}"
    miner_prefix = str(data.get("miner_id") or "")
    executor_id = getattr(executor, "executor_id", "") or ""
    if miner_prefix and executor_id and not executor_id.startswith(miner_prefix):
        return False, "health_executor_mismatch"
    return True, ""


def _executor_hotkeys(executor_ids: list[str]) -> dict[str, str]:
    if db is None or not executor_ids:
        return {}
    try:
        from common.db import Executor
        with db.session() as s:
            rows = s.query(Executor.executor_id, Executor.hotkey_ss58).filter(
                Executor.executor_id.in_(executor_ids),
            ).all()
            return {row[0]: row[1] or "" for row in rows}
    except Exception:
        return {}


def _policy_blacklist_reason_for_executor(executor: Any, hotkey_ss58: str = "") -> str:
    from common.subnet_runtime_config import policy_blacklist_reason

    return policy_blacklist_reason(
        getattr(executor, "executor_id", "") or "",
        hotkey_ss58=hotkey_ss58,
        miner_address=getattr(executor, "miner_address", "") or "",
        miner_uid=getattr(executor, "miner_uid", None),
        endpoint=getattr(executor, "endpoint", "") or "",
    )


def _policy_blacklist_blocked_ids(executors: list[Any]) -> set[str]:
    if not executors:
        return set()
    hotkeys = _executor_hotkeys([getattr(e, "executor_id", "") for e in executors])
    blocked: set[str] = set()
    for executor in executors:
        eid = getattr(executor, "executor_id", "") or ""
        if eid and _policy_blacklist_reason_for_executor(
            executor,
            hotkey_ss58=hotkeys.get(eid, ""),
        ):
            blocked.add(eid)
    return blocked


def _active_rental_policy_blacklist_reason(rental: dict) -> str:
    from common.subnet_runtime_config import policy_blacklist_reason

    executor_id = rental.get("executor_id", "") or ""
    endpoint = rental.get("executor_endpoint", "") or ""
    hotkey = ""
    miner_address = ""
    miner_uid = None
    try:
        if db is not None and executor_id:
            from common.db import Executor
            with db.session() as s:
                row = s.query(Executor).filter_by(executor_id=executor_id).first()
                if row is not None:
                    hotkey = row.hotkey_ss58 or ""
                    miner_address = row.miner_address or ""
                    endpoint = endpoint or row.endpoint or ""
    except Exception:
        pass
    try:
        spec = chain_snapshot.get(executor_id) if chain_snapshot is not None else None
        if spec is not None:
            miner_address = getattr(spec, "miner_address", "") or miner_address
            miner_uid = getattr(spec, "miner_uid", None)
            endpoint = getattr(spec, "endpoint", "") or endpoint
    except Exception:
        pass
    return policy_blacklist_reason(
        executor_id,
        hotkey_ss58=hotkey,
        miner_address=miner_address,
        miner_uid=miner_uid,
        endpoint=endpoint,
    )


def _rental_policy_blacklist_reason_by_id(rental_id: str) -> str:
    rental = dict(_active_rentals.get(rental_id) or {})
    if not rental and db is not None:
        try:
            from common.db import Rental
            with db.session() as s:
                row = s.query(Rental).filter_by(rental_id=rental_id).first()
                if row is not None:
                    rental = {
                        "executor_id": row.executor_id,
                        "executor_endpoint": row.executor_endpoint,
                    }
        except Exception:
            rental = {}
    return _active_rental_policy_blacklist_reason(rental) if rental else ""


def _body_int(
    body: dict,
    key: str,
    default: int,
    *,
    min_value: int = 0,
    max_value: int = 1_000_000,
) -> int:
    value = body.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise HTTPException(400, f"{key} must be an integer")
    if parsed < min_value or parsed > max_value:
        raise HTTPException(
            400,
            f"{key} must be between {min_value} and {max_value}",
        )
    return parsed


def _resource_requirements(body: dict) -> dict[str, int]:
    """Parse renter resource requirements.

    These are off-chain placement requirements backed by heartbeat/canary
    telemetry. The chain registry remains the stable lease/lock layer.
    """
    min_storage_gb = _body_int(
        body,
        "min_storage_gb",
        _env_int("RENTAL_MIN_STORAGE_GB", 30, min_value=1),
        min_value=1,
        max_value=100_000,
    )
    storage_gb = _body_int(
        body,
        "storage_gb",
        _env_int("RENTAL_DEFAULT_STORAGE_GB", min_storage_gb, min_value=1),
        min_value=1,
        max_value=100_000,
    )
    min_ram_gb = _body_int(
        body,
        "min_ram_gb",
        _env_int("RENTAL_MIN_RAM_GB", 0, min_value=0),
        min_value=0,
        max_value=100_000,
    )
    memory_gb = _body_int(
        body,
        "memory_gb",
        _env_int("RENTAL_DEFAULT_MEMORY_GB", max(16, min_ram_gb), min_value=1),
        min_value=1,
        max_value=100_000,
    )
    min_vram_gb = _body_int(
        body,
        "min_vram_gb",
        0,
        min_value=0,
        max_value=10_000,
    )
    cpu_count = _body_int(
        body,
        "cpu_count",
        _env_int("RENTAL_DEFAULT_CPU_COUNT", 4, min_value=1),
        min_value=1,
        max_value=1024,
    )
    min_cpu_count = _body_int(
        body,
        "min_cpu_count",
        0,
        min_value=0,
        max_value=1024,
    )
    return {
        "min_storage_gb": min_storage_gb,
        "storage_gb": max(storage_gb, min_storage_gb),
        "min_ram_gb": min_ram_gb,
        "memory_gb": max(memory_gb, min_ram_gb),
        "min_vram_gb": min_vram_gb,
        "cpu_count": max(cpu_count, min_cpu_count or 1),
        "min_cpu_count": min_cpu_count,
    }


def _hardware_profiles_for(executor_ids: list[str]) -> dict[str, dict[str, int]]:
    if db is None or not executor_ids:
        return {}
    try:
        from common.db import ExecutorHardware, ExecutorStats
        import json as _json

        with db.session() as s:
            hw_rows = {
                row.executor_id: row
                for row in s.query(ExecutorHardware).filter(
                    ExecutorHardware.executor_id.in_(executor_ids),
                ).all()
            }
            st_rows = {
                row.executor_id: row
                for row in s.query(ExecutorStats).filter(
                    ExecutorStats.executor_id.in_(executor_ids),
                ).all()
            }
        profiles: dict[str, dict[str, int]] = {}
        for eid in executor_ids:
            hw = hw_rows.get(eid)
            st = st_rows.get(eid)
            disk_total = int((st.disk_total_bytes if st else 0) or (hw.disk_total_bytes if hw else 0) or 0)
            disk_free = 0
            if st and st.hw_live_json:
                try:
                    live = _json.loads(st.hw_live_json)
                    docker_storage = (live.get("storage") or {}).get("docker") or {}
                    disk_free = int(
                        docker_storage.get("available_bytes")
                        or docker_storage.get("free_bytes")
                        or live.get("disk_free_bytes")
                        or 0
                    )
                    disk_total = int(docker_storage.get("total_bytes") or disk_total or 0)
                except Exception:
                    pass
            profiles[eid] = {
                "disk_total_bytes": disk_total,
                "disk_free_bytes": disk_free,
                "ram_total_bytes": int((st.ram_total_bytes if st else 0) or (hw.ram_total_bytes if hw else 0) or 0),
                "cpu_cores": int((hw.cpu_cores_logical if hw else 0) or (hw.cpu_cores_physical if hw else 0) or 0),
                "gpu_vram_total_bytes": int((st.gpu_vram_total_bytes if st else 0) or (hw.gpu_vram_total_bytes if hw else 0) or 0),
            }
        return profiles
    except Exception as e:
        bt.logging.warning(f"Rental: resource profile query failed: {e}")
        return {}


def _cold_image_storage_extra_gb(image: str) -> int:
    from common.images import image_pull_reserve_gb

    return _env_int("RENTAL_STORAGE_HEADROOM_GB", 10, min_value=0) + image_pull_reserve_gb(image)


def _resource_filter_candidates(
    candidates: list[Any],
    req: dict[str, int],
    *,
    extra_storage_gb: int = 0,
) -> tuple[list[Any], list[str]]:
    """Apply off-chain resource requirements before score weighting."""
    profiles = _hardware_profiles_for([c.executor_id for c in candidates])
    kept: list[Any] = []
    dropped: list[str] = []
    storage_required_gb = int(req["storage_gb"]) + max(0, int(extra_storage_gb or 0))
    storage_required_bytes = storage_required_gb * _GIB
    ram_required_bytes = int(req["memory_gb"]) * _GIB
    min_vram_required_bytes = int(req["min_vram_gb"]) * _GIB
    for ex in candidates:
        profile = profiles.get(ex.executor_id) or {}
        reasons: list[str] = []
        disk_total = int(profile.get("disk_total_bytes") or 0)
        disk_free = int(profile.get("disk_free_bytes") or 0)
        ram_total = int(profile.get("ram_total_bytes") or 0)
        cpu_cores = int(profile.get("cpu_cores") or 0)
        total_vram = int(profile.get("gpu_vram_total_bytes") or 0)
        per_gpu_vram = int(getattr(ex, "vram_mb", 0) or 0) * 1024 * 1024

        if storage_required_bytes and disk_free < storage_required_bytes:
            reasons.append(
                f"storage free {disk_free // _GIB}GB < {storage_required_gb}GB"
            )
        if ram_required_bytes and ram_total < ram_required_bytes:
            reasons.append(
                f"ram {ram_total // _GIB}GB < {req['memory_gb']}GB"
            )
        if req["cpu_count"] and cpu_cores and cpu_cores < req["cpu_count"]:
            reasons.append(f"cpu {cpu_cores} < {req['cpu_count']}")
        elif req["cpu_count"] and not cpu_cores:
            reasons.append("cpu unknown")
        if min_vram_required_bytes:
            # Public min_vram_gb is per GPU. Chain spec is authoritative for
            # per-card VRAM; heartbeat total is a fallback sanity check.
            if per_gpu_vram < min_vram_required_bytes:
                reasons.append(
                    f"vram {per_gpu_vram // _GIB}GB < {req['min_vram_gb']}GB"
                )
            elif total_vram and total_vram < min_vram_required_bytes * max(1, int(getattr(ex, "gpu_count", 1) or 1)):
                reasons.append("heartbeat vram below requested total")

        if reasons:
            dropped.append(f"{ex.executor_id[:16]} {'; '.join(reasons)}")
        else:
            kept.append(ex)
    return kept, dropped


def _image_availability_for_candidates(
    candidates: list[Any],
) -> dict[str, dict[str, set[str]]]:
    """Executor id → curated image refs reported by that executor.

    This reads validator-local heartbeat state. The rent path must not fan out
    to every miner to ask about image availability; miners report it
    periodically in hw_static and /marketplace exposes the aggregate.
    """
    if db is None or not candidates:
        return {}
    try:
        from common.db import get_all_executor_hardware
        hw_by_id = get_all_executor_hardware(db)
        return {
            c.executor_id: {
                "cached": set((hw_by_id.get(c.executor_id) or {}).get("cached_images") or {}),
                "catalog": set((hw_by_id.get(c.executor_id) or {}).get("image_catalog") or []),
            }
            for c in candidates
        }
    except Exception as e:
        bt.logging.warning(f"Rental: image availability query failed: {e}")
        return {}


def _require_cached_image(body: dict) -> bool:
    if not _env_bool("RENTAL_ALLOW_COLD_IMAGE_PULL"):
        return True
    value = body.get("require_cached_image", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _require_evm_write_mode(action: str = "rental control") -> None:
    if not evm_write_enabled:
        raise HTTPException(
            503,
            {
                "error": "validator_no_evm_mode",
                "detail": (
                    f"{action} requires an EVM-enabled validator. This "
                    "validator is running in read/verify mode."
                ),
            },
        )


def _image_cache_filter_candidates(
    candidates: list[Any],
    image: str,
) -> tuple[list[Any], list[str]]:
    return _image_filter_candidates(candidates, image, require_cached=True)


def _image_filter_candidates(
    candidates: list[Any],
    image: str,
    *,
    require_cached: bool,
) -> tuple[list[Any], list[str]]:
    image = (image or "").strip()
    if not image:
        return candidates, []
    availability_by_id = _image_availability_for_candidates(candidates)
    kept: list[Any] = []
    dropped: list[str] = []
    for ex in candidates:
        availability = availability_by_id.get(ex.executor_id) or {}
        cached = availability.get("cached") or set()
        catalog = availability.get("catalog") or set()
        if image in cached:
            kept.append(ex)
        elif not require_cached and image in catalog:
            kept.append(ex)
        else:
            reason = "missing cached image" if require_cached else "image not in catalog"
            dropped.append(f"{ex.executor_id[:16]} {reason} {image}")
    return kept, dropped


def _rental_container_missing_threshold() -> int:
    return _env_int("RENTAL_CONTAINER_MISSING_THRESHOLD", 2, min_value=1)


def _rental_endpoint_failure_threshold() -> int:
    return _env_int("RENTAL_ENDPOINT_FAILURE_THRESHOLD", 4, min_value=1)


def _rental_chain_failure_threshold() -> int:
    return _env_int("RENTAL_CHAIN_FAILURE_THRESHOLD", 2, min_value=1)


def _rental_chain_grace_seconds() -> int:
    return _env_int("RENTAL_CHAIN_GRACE_SECONDS", 180, min_value=0)


def _rental_port_failure_threshold() -> int:
    return _env_int("RENTAL_PORT_FAILURE_THRESHOLD", 4, min_value=1)


def _rental_port_probe_timeout_seconds() -> float:
    return _env_float("RENTAL_PORT_PROBE_TIMEOUT_S", 2.5, min_value=0.1)


def _rental_port_watchdog_enabled() -> bool:
    return os.environ.get("RENTAL_PORT_WATCHDOG_ENABLED", "1") != "0"


def _rental_container_probation_seconds() -> int:
    # 3 Bittensor tempos by default: 3 * 360 blocks * ~12 seconds.
    return _env_int("RENTAL_CONTAINER_PROBATION_SECONDS", 3 * 360 * 12, min_value=60)


def _rental_age_seconds(rental: dict, now: float | None = None) -> int:
    now = time.time() if now is None else now
    try:
        created_at = float(rental.get("created_at") or now)
    except (TypeError, ValueError):
        created_at = now
    return max(0, int(now - created_at))


def _rental_ttl_expired(rental: dict, now: float | None = None) -> bool:
    try:
        ttl = int(rental.get("ttl_seconds") or 0)
    except (TypeError, ValueError):
        ttl = 0
    return ttl > 0 and _rental_age_seconds(rental, now) >= ttl


def _release_effective_ttl_seconds(rental: dict) -> int:
    """TTL used after a requested release when chain release is still pending."""
    age = _rental_age_seconds(rental)
    try:
        ttl = int(rental.get("ttl_seconds") or 0)
    except (TypeError, ValueError):
        ttl = 0
    if ttl > 0:
        return max(1, min(ttl, age))
    return max(1, age)


def _container_failure_kind(container_name: str, containers: list[dict]) -> str:
    """Classify miner-reported state for one expected rental container.

    `/containers` is validator-signed and backed by the miner's Docker
    service. A missing object means the host container is gone, which a
    renter should not be able to cause without a Docker socket escape.
    A present but exited/dead/removing object is unusable, but root
    inside a sysbox container may be able to kill PID 1, so that is not
    strong enough to hard-flag the miner by itself.

    Returns: "" | "missing" | "stopped"
    """
    if not container_name:
        return "missing"
    for container in containers or []:
        if container.get("name") != container_name:
            continue
        status = str(container.get("status") or "").lower()
        return "stopped" if status in _CONTAINER_BAD_STATES else ""
    return "missing"


def _update_container_missing_streak(
    rental_id: str,
    container_name: str,
    containers: list[dict] | None,
    threshold: int,
) -> bool:
    """Update the missing-container streak for one rental.

    `containers is None` means the miner API/list call failed. That is not
    evidence of a missing container, so the streak is left unchanged.
    Returns True only when the missing/unusable streak reaches threshold.
    """
    if containers is None:
        return False
    if not _container_failure_kind(container_name, containers):
        _container_missing_streaks.pop(rental_id, None)
        return False
    streak = _container_missing_streaks.get(rental_id, 0) + 1
    _container_missing_streaks[rental_id] = streak
    return streak >= threshold


def _update_endpoint_failure_streak(
    rental_id: str,
    containers: list[dict] | None,
    threshold: int,
) -> bool:
    if containers is not None:
        _endpoint_failure_streaks.pop(rental_id, None)
        return False
    streak = _endpoint_failure_streaks.get(rental_id, 0) + 1
    _endpoint_failure_streaks[rental_id] = streak
    return streak >= threshold


def _update_port_failure_streak(
    rental_id: str,
    reachable: bool | None,
    threshold: int,
) -> bool:
    if reachable is None or reachable:
        _port_failure_streaks.pop(rental_id, None)
        return False
    streak = _port_failure_streaks.get(rental_id, 0) + 1
    _port_failure_streaks[rental_id] = streak
    return streak >= threshold


async def _active_rental_ssh_port_reachable(rental: dict) -> bool | None:
    if not _rental_port_watchdog_enabled():
        return None
    host = str(rental.get("ssh_host") or "").strip()
    try:
        port = int(rental.get("ssh_port") or 0)
    except (TypeError, ValueError):
        port = 0
    if not host or port <= 0:
        return None
    from neurons.validator.rental.ports import probe_tcp_port
    return await asyncio.to_thread(
        probe_tcp_port,
        host,
        port,
        _rental_port_probe_timeout_seconds(),
    )


def _chain_failure_kind(executor_id: str) -> str:
    """Return a chain liveness failure for an active rental, or "".

    Uses ChainSnapshot only. If the snapshot is stale, skip the check
    instead of issuing per-rental RPC reads from this watchdog.
    """
    if chain_snapshot is None or not chain_snapshot.is_fresh():
        return ""
    spec = chain_snapshot.get(executor_id)
    now = int(time.time())
    if spec is None:
        return "not_live"
    if not spec.is_active:
        return "inactive"
    if int(spec.expires_at or 0) <= now:
        return "lease_expired"
    if not spec.is_rented:
        return "not_rented"
    return ""


def _direct_chain_failure_kind(executor_id: str) -> str:
    """Confirm a snapshot-reported chain failure with a direct chain read.

    The watchdog normally uses ChainSnapshot so it does not create per-rental
    RPC traffic. A terminating decision is rare and high-impact, so confirm
    it directly before destroying a renter's container. This prevents stale
    snapshot/index lag right after markRented from releasing a healthy rental.
    """
    if registry_client is None:
        return ""
    try:
        spec = registry_client.get_executor_info(bytes.fromhex(executor_id))
    except Exception as e:
        bt.logging.debug(
            f"rental_watchdog direct chain check failed for {executor_id[:16]}: {e}"
        )
        return ""
    now = int(time.time())
    if spec is None:
        return "not_live"
    if not spec.is_active:
        return "inactive"
    if int(spec.expires_at or 0) <= now:
        return "lease_expired"
    if not spec.is_rented:
        return "not_rented"
    return ""


def _update_chain_failure_streak(
    rental_id: str,
    failure_kind: str,
    threshold: int,
) -> bool:
    if not failure_kind:
        _chain_failure_streaks.pop(rental_id, None)
        return False
    streak = _chain_failure_streaks.get(rental_id, 0) + 1
    _chain_failure_streaks[rental_id] = streak
    return streak >= threshold


def _block_of_tx(tx_hash: str) -> int:
    """Look up the block number a tx landed in. Returns current head on failure."""
    try:
        if not registry_client:
            return 0
        receipt_fn = getattr(registry_client, "transaction_receipt", None)
        r = (
            receipt_fn(tx_hash)
            if callable(receipt_fn)
            else registry_client.w3.eth.get_transaction_receipt(tx_hash)
        )
        return int(r.blockNumber)
    except Exception:
        try:
            if not registry_client:
                return 0
            block_fn = getattr(registry_client, "confirmed_transaction_block", None)
            if callable(block_fn):
                block = int(block_fn(tx_hash) or 0)
                if block > 0:
                    return block
        except Exception:
            pass
        try:
            if not registry_client:
                return 0
            block_number_fn = getattr(registry_client, "block_number", None)
            return int(
                block_number_fn()
                if callable(block_number_fn)
                else registry_client.w3.eth.block_number
            )
        except Exception:
            return 0


def _parse_client_request_id(value: Any) -> str:
    if value is None:
        return ""
    request_id = str(value).strip()
    if not request_id:
        return ""
    if not _CLIENT_REQUEST_ID_RE.match(request_id):
        raise HTTPException(400, "client_request_id must be 8..128 URL-safe characters")
    return request_id


def _timestamp_seconds(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    if hasattr(value, "timestamp"):
        try:
            return float(value.timestamp())
        except Exception:
            return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _duration_label(ttl_seconds: int) -> str:
    ttl = max(0, int(ttl_seconds or 0))
    if ttl and ttl % 86400 == 0:
        return f"{ttl // 86400}d"
    if ttl and ttl % 3600 == 0:
        return f"{ttl // 3600}h"
    if ttl:
        return f"{ttl}s"
    return "unlimited"


def _rental_response_from_row(row: dict, active_info: dict | None = None) -> dict:
    active_info = active_info or {}
    now = time.time()
    created_at = _timestamp_seconds(row.get("created_at"), active_info.get("created_at", now))
    ttl_seconds = int(row.get("ttl_seconds") or active_info.get("ttl_seconds") or 0)
    gpu_count = int(row.get("gpu_count") or 1)
    price_per_hour = int(row.get("price_per_gpu_hour_rao") or 0)
    executor_id = str(row.get("executor_id") or active_info.get("executor_id") or "")
    host = {}
    if db and executor_id:
        try:
            from common.db import get_executor_hardware
            host = get_executor_hardware(db, executor_id)
        except Exception:
            host = {}
    return {
        "rental_id": row.get("rental_id"),
        "executor_id": executor_id,
        "gpu": {
            "model": row.get("gpu_model") or "?",
            "count": gpu_count,
            "vram_gb": int(row.get("vram_mb") or 0) // 1024,
        },
        "host": {
            "cpu_model": host.get("cpu_model", ""),
            "cpu_cores": host.get("cpu_cores", 0),
            "ram_gb": host.get("ram_gb", 0),
            "disk_gb": host.get("disk_gb", 0),
        },
        "resources": {
            "cpu_count": host.get("cpu_cores", 0),
            "memory_gb": host.get("ram_gb", 0),
            "storage_gb": host.get("disk_gb", 0),
            "min_storage_gb": host.get("disk_gb", 0),
            "min_vram_gb": int(row.get("vram_mb") or 0) // 1024,
        },
        "price": {
            "per_gpu_hour_rao": price_per_hour,
            "estimated_total_rao": int(
                price_per_hour * gpu_count * ttl_seconds / 3600
            ) if price_per_hour and ttl_seconds else 0,
        },
        "connection": {
            "ssh_host": row.get("ssh_host") or "",
            "ssh_port": int(row.get("ssh_port") or 0),
            "ssh_user": row.get("ssh_user") or "",
        },
        "image": active_info.get("image", ""),
        "created_at": created_at,
        "duration": _duration_label(ttl_seconds),
        "ttl_seconds": ttl_seconds,
    }


def _rental_response_for_client_request(client_request_id: str) -> dict | None:
    if not db or not client_request_id:
        return None
    try:
        from common.db import Rental
        with db.session() as s:
            rental = (
                s.query(Rental)
                .filter(Rental.client_request_id == client_request_id)
                .order_by(Rental.created_at.desc())
                .first()
            )
            if rental is None:
                return None
            row = {c.name: getattr(rental, c.name) for c in rental.__table__.columns}
    except Exception as e:
        bt.logging.warning(f"rent: client_request_id lookup failed: {e}")
        return None

    status = str(row.get("status") or "")
    rental_id = str(row.get("rental_id") or "")
    if status == "active" and row.get("terminated_at") is None:
        return _rental_response_from_row(row, _active_rentals.get(rental_id))
    if status == "requested":
        raise HTTPException(409, "Rental request is still provisioning")
    if status == "release_pending":
        raise HTTPException(409, "Rental request is waiting for chain release")
    raise HTTPException(410, f"Previous rental request finished with status {status or 'unknown'}")


@router.post("/rent")
async def rent_executor(request: Request):
    """Rent a GPU executor.

    Internal control-plane endpoint. The public renter API lives in the
    web app backend, which verifies x402/credits/account access before
    calling this route with validator admin auth.

    The caller specifies GPU requirements; validator selects via randomized
    weighted algorithm. Callers CANNOT specify a particular executor
    (prevents self-rent gaming).
    """
    if not await _is_admin_async(request):
        raise HTTPException(403, "internal validator auth required")
    _require_evm_write_mode("renting")

    body = await request.json()
    gpu_model = body.get("gpu_model")         # e.g., "RTX4090" (substring match)
    gpu_count = _body_int(body, "gpu_count", 1, min_value=1, max_value=8)
    duration = body.get("duration", "2h")      # e.g., "2h", "1d"
    ssh_pub_key = body.get("ssh_pub_key", "")
    image = body.get("image", "ubuntu:22.04")
    client_request_id = _parse_client_request_id(body.get("client_request_id"))
    resources = _resource_requirements(body)
    require_cached_image = _require_cached_image(body)

    if not ssh_pub_key:
        raise HTTPException(400, "ssh_pub_key required")

    existing_response = _rental_response_for_client_request(client_request_id)
    if existing_response is not None:
        return existing_response

    # Parse duration to seconds
    ttl_seconds = _parse_duration(duration)

    # Get available executors matching criteria
    if registry_client is None:
        raise HTTPException(503, "Registry not available")

    # Candidate set from the snapshot (already filtered to is_active, not
    # is_rented, lease not expired). Snapshot may be up to ~15s stale; we
    # accept that because markRented itself is the atomic lock — if a peer
    # validator rented in between, our markRented reverts and we 409.
    if chain_snapshot is None:
        raise HTTPException(503, "Snapshot not initialized")
    all_executors = chain_snapshot.available_executors()
    # Exclude executors we're already renting — same staleness window as
    # marketplace_tiers. Without this we could pick an executor that just
    # got rented (our own markRented in flight, snapshot pre-refresh).
    active_eids = {info["executor_id"] for info in _active_rentals.values()}
    all_executors = [e for e in all_executors if e.executor_id not in active_eids]
    policy_blocked = _policy_blacklist_blocked_ids(all_executors)
    if policy_blocked:
        all_executors = [e for e in all_executors if e.executor_id not in policy_blocked]
    # Hard-exclude sybil-flagged executors from selection. The sybil
    # scanner has objective evidence; we don't rent these to anyone.
    if db is not None:
        try:
            from common.db import get_banned_executor_ids
            banned = get_banned_executor_ids(db)
            if banned:
                all_executors = [e for e in all_executors if e.executor_id not in banned]
        except Exception:
            pass
    # Resolve user-friendly GPU name filter against the chain's hash.
    # Match priority: (a) exact case-insensitive, (b) unique substring;
    # ambiguous substrings 400 with the list of choices so the renter
    # picks one rather than getting a silently-wrong miner.
    desired_hashes = None
    if gpu_model:
        from common.config import (
            GPU_VRAM_GB,
            canonical_gpu_model_name,
            gpu_model_hashes_for_name,
        )
        from neurons.validator.api.routes.browse import _GPU_HASH_TO_NAME
        wanted = gpu_model.strip().lower()
        # (a) exact
        canonical = canonical_gpu_model_name(gpu_model)
        if canonical in GPU_VRAM_GB:
            desired_hashes = gpu_model_hashes_for_name(canonical)
        # (b) unique substring
        if not desired_hashes:
            choices = sorted(set(_GPU_HASH_TO_NAME.values()))
            matches = [n for n in choices if wanted in n.lower()]
            if len(matches) == 1:
                desired_hashes = gpu_model_hashes_for_name(matches[0])
            elif len(matches) > 1:
                raise HTTPException(
                    400,
                    f"'{gpu_model}' is ambiguous — matches: "
                    + ", ".join(n.replace("NVIDIA ", "") for n in matches)
                    + ". Pass the full model name.",
                )
        if not desired_hashes:
            raise HTTPException(
                400,
                f"Unknown gpu_model '{gpu_model}'. See `nodexo browse` for available options.",
            )
    candidates = [
        ex for ex in all_executors
        if (
            desired_hashes is None
            or ex.gpu_model_hash.lower().removeprefix("0x") in desired_hashes
        )
        and ex.gpu_count >= gpu_count
    ]

    if not candidates:
        raise HTTPException(404, "No matching executors available")

    # Off-chain resource gate. The chain stores stable identity/lease/GPU
    # shape; storage/RAM/CPU are heartbeat-backed and canary-verified because
    # they change with host provisioning and Docker state.
    if db is not None:
        pre_resource_gate = len(candidates)
        extra_storage_gb = 0 if require_cached_image else _cold_image_storage_extra_gb(image)
        candidates, dropped_resources = _resource_filter_candidates(
            candidates,
            resources,
            extra_storage_gb=extra_storage_gb,
        )
        if dropped_resources:
            bt.logging.info(
                "Rental: resource gate filtered "
                f"{pre_resource_gate - len(candidates)}/{pre_resource_gate} "
                f"executors for req={resources} extra_storage_gb={extra_storage_gb} "
                f"dropped={dropped_resources[:10]}"
            )
        if not candidates:
            storage_text = resources["storage_gb"] + extra_storage_gb
            raise HTTPException(
                503,
                "No online verified executors satisfy the requested resource "
                f"requirements (storage>={storage_text}GB, "
                f"RAM>={resources['memory_gb']}GB).",
            )

    # ── Canary gate ─────────────────────────────────────────────
    # Real renters only see executors that have PASSED at least one
    # canary. Never-canaried or recently-failed executors are kept out
    # of the marketplace until the validator's canary scheduler
    # vetting catches up. This is the production rental gate.
    if db is not None:
        from common.db import CanaryRecord
        try:
            with db.session() as s:
                eligible_ids = set()
                for cr in s.query(CanaryRecord).filter(
                    CanaryRecord.executor_id.in_([c.executor_id for c in candidates]),
                    CanaryRecord.last_status == "pass",
                ).all():
                    eligible_ids.add(cr.executor_id)
            pre_gate = len(candidates)
            candidates = [c for c in candidates if c.executor_id in eligible_ids]
            if len(candidates) < pre_gate:
                bt.logging.info(
                    f"Rental: canary gate filtered "
                    f"{pre_gate - len(candidates)}/{pre_gate} executors "
                    f"(pending or failed canary)"
                )
            if not candidates:
                raise HTTPException(
                    503,
                    "No canary-verified executors available. New executors "
                    "are vetted by the validator before being offered for "
                    "rent — please try again in a few minutes."
                )
        except HTTPException:
            raise
        except Exception as e:
            bt.logging.warning(f"Rental: canary gate query failed: {e}")
            # Fail closed: if we can't read CanaryRecord, refuse to rent.
            raise HTTPException(503, "Rental temporarily unavailable")

    # Live-health gate. Chain-active is not enough: a stopped miner can keep
    # an unexpired lease. Heartbeat/proof health is the fast renter-facing
    # liveness state; score is only used for ranking once available.
    from neurons.validator.api.routes.browse import _healthy_executor_ids
    healthy_ids = _healthy_executor_ids([c.executor_id for c in candidates])
    pre_health_gate = len(candidates)
    candidates = [c for c in candidates if c.executor_id in healthy_ids]
    if len(candidates) < pre_health_gate:
        bt.logging.info(
            f"Rental: live-health gate filtered "
            f"{pre_health_gate - len(candidates)}/{pre_health_gate} executors "
            f"(stale heartbeat or verifier status)"
        )
    if not candidates:
        raise HTTPException(
            503,
            "No currently online verified executors available. Please try again in a few minutes.",
        )

    # Image gate. Fast rentals require a cached image. Slow-path rentals may
    # cold-pull, but only for images that the executor reports in its curated
    # catalog. This remains a validator DB read, not per-rent miner fanout.
    pre_image_gate = len(candidates)
    candidates, dropped_images = _image_filter_candidates(
        candidates,
        image,
        require_cached=require_cached_image,
    )
    if dropped_images:
        gate = "image-cache" if require_cached_image else "image-catalog"
        bt.logging.info(
            f"Rental: {gate} gate filtered "
            f"{pre_image_gate - len(candidates)}/{pre_image_gate} "
            f"executors for image={image} dropped={dropped_images[:10]}"
        )
    if not candidates:
        detail = (
            "Selected image is not available for this inventory right now: "
            if require_cached_image
            else "Selected image is not enabled for this inventory: "
        )
        raise HTTPException(503, detail + f"{image}. Choose another image.")

    # Score gate. At this point candidates are chain-free, canary-passed,
    # healthy, resource-matched, and image-matched. They still must have a
    # positive validator score so hard production gates such as missing Sysbox,
    # insufficient stake, stale proof, or hard flags cannot slip through via
    # the random-selection fallback.
    pre_score_gate = len(candidates)
    score_drops: list[str] = []
    scored_candidates: list[Any] = []
    for ex in candidates:
        score_data = scoring_data.get(ex.executor_id) if scoring_data else None
        if score_data is not None and score_data.score > 0:
            scored_candidates.append(ex)
        else:
            reason = getattr(score_data, "zero_reason", "no_score_yet") if score_data else "no_score_yet"
            score_drops.append(f"{ex.executor_id[:16]}:{reason}")
    candidates = scored_candidates
    if len(candidates) < pre_score_gate:
        bt.logging.info(
            f"Rental: score gate filtered "
            f"{pre_score_gate - len(candidates)}/{pre_score_gate} executors "
            f"dropped={score_drops[:10]}"
        )
    if not candidates:
        raise HTTPException(
            503,
            "No currently eligible executors available. Please try again in a few minutes.",
        )

    # Score-weighted random choice — higher score = higher probability.
    # Immediately before locking the executor, perform one pull-side health
    # check on the selected endpoint. This closes the short window where a
    # stopped miner still has a fresh pushed heartbeat and has not yet been
    # marked unhealthy by the background prober. We retry a bounded number of
    # alternate candidates without fanning out to the whole fleet.
    remaining_candidates = list(candidates)
    remaining_weights = [
        float(scoring_data.get(ex.executor_id).score)
        for ex in remaining_candidates
    ]
    max_live_check_attempts = min(
        len(remaining_candidates),
        _env_int("RENT_SELECTED_HEALTH_MAX_ATTEMPTS", 3, min_value=1),
    )
    selected = None
    selected_score = 0.0
    live_check_drops: list[str] = []
    for _attempt in range(max_live_check_attempts):
        picked_index = random.choices(
            range(len(remaining_candidates)),
            weights=remaining_weights,
            k=1,
        )[0]
        candidate = remaining_candidates[picked_index]
        ok, reason = await _selected_executor_live_check(candidate)
        if ok:
            selected = candidate
            selected_score = float(remaining_weights[picked_index])
            break
        live_check_drops.append(f"{candidate.executor_id[:16]}:{reason}")
        remaining_candidates.pop(picked_index)
        remaining_weights.pop(picked_index)
        if not remaining_candidates:
            break

    if selected is None:
        bt.logging.info(
            "Rental: selected-endpoint live check filtered all attempted "
            f"candidates dropped={live_check_drops[:10]}"
        )
        raise HTTPException(
            503,
            "No currently reachable executors available. Please try again in a few minutes.",
        )

    bt.logging.info(
        f"Rental: selected executor {selected.executor_id[:16]} "
        f"(score={selected_score:.3f}) from {len(candidates)} candidates"
    )

    selected_blacklist_reason = _policy_blacklist_reason_for_executor(
        selected,
        hotkey_ss58=_executor_hotkeys([selected.executor_id]).get(selected.executor_id, ""),
    )
    if selected_blacklist_reason:
        raise HTTPException(
            409,
            f"Executor became ineligible: {selected_blacklist_reason}",
        )

    # Pre-allocate the rental ID and write a 'requested' DB row BEFORE any
    # chain operation. This is the trace-of-intent — if the client
    # disconnects, the validator crashes, or any later step fails, the
    # orphan sweeper finds the row and reconciles. Without this, a ^C
    # during the markRented tx leaves the executor stuck on chain with
    # nobody knowing who 'owns' the rental.
    #
    # 32 hex chars (128-bit) — leaves plenty of headroom for short-prefix
    # display in the CLI (12 chars = 48-bit, ~280T) and zero realistic
    # full-id collision risk. DB column is String(32).
    renter_fp = hashlib.sha256(ssh_pub_key.encode()).hexdigest()[:16]
    from neurons.validator.api.routes.browse import _name_for_hash
    rental_id = ""
    if db:
        from common.db import store_rental
        for attempt in range(5):
            rid_try = hashlib.sha256(
                f"{selected.executor_id}:{time.time()}:{attempt}".encode()
            ).hexdigest()[:32]
            try:
                store_rental(
                    db, rid_try, selected.executor_id,
                    executor_endpoint=selected.endpoint,
                    container_name="",  # not provisioned yet
                    ssh_host="", ssh_port=0, ssh_user="",
                    client_request_id=client_request_id,
                    ttl_seconds=ttl_seconds,
                    gpu_model=_name_for_hash(selected.gpu_model_hash),
                    vram_mb=selected.vram_mb,
                    gpu_count=selected.gpu_count,
                    price_per_gpu_hour_rao=selected.price_per_gpu_hour,
                    renter_pubkey_fp=renter_fp,
                    start_block=0,
                    status="requested",   # state machine: requested → active → terminated_*
                )
                rental_id = rid_try
                break
            except Exception as e:
                # SQLA wraps IntegrityError; we check name rather than type
                # so this works under both Postgres and SQLite drivers.
                if "IntegrityError" in type(e).__name__ or "UNIQUE" in str(e):
                    bt.logging.warning(
                        f"rent: rental_id collision on attempt {attempt}, regenerating"
                    )
                    continue
                bt.logging.error(f"rent: failed to write 'requested' DB row: {e}")
                raise HTTPException(500, "Failed to record rental intent")
        if not rental_id:
            raise HTTPException(500, "Failed to allocate unique rental_id after retries")
    else:
        # No DB (testing path) — generate a one-shot id but warn.
        rental_id = hashlib.sha256(
            f"{selected.executor_id}:{time.time()}".encode()
        ).hexdigest()[:32]
        bt.logging.warning("rent: no DB; rental_id collisions not detectable")

    # ── On-chain lock (atomic via contract's !isRented check) ────
    try:
        tx_hash = registry_client.mark_rented(bytes.fromhex(selected.executor_id))
        bt.logging.info(f"markRented tx: {tx_hash}")
    except Exception as e:
        # Race condition: already rented by another validator
        bt.logging.warning(f"markRented failed for {selected.executor_id[:16]}: {e} — retrying with different executor")
        if db:
            from common.db import terminate_rental
            terminate_rental(db, rental_id, reason="orphan", end_block=0)
        raise HTTPException(409, f"Executor became unavailable: {e}")

    # Record rental window start AFTER tx confirms — this fixes the race where
    # the miner's in-flight HEAVY proof is for a block before our markRented.
    # verify_recipe will use this block-anchored lookup instead of the live
    # `is_rented` flag.
    rental_block = _block_of_tx(tx_hash)

    # Eagerly refresh this executor in the snapshot so the next /marketplace
    # or /rent request sees the new is_rented=True without waiting for the
    # next 15s tick.
    if chain_snapshot is not None:
        chain_snapshot.invalidate_executor(selected.executor_id)

    # ── Provision container via executor's HTTP API ──────────────
    try:
        from common.images import image_runtime_reference

        provision_image = image_runtime_reference(image)
        if provision_image != image:
            bt.logging.info(
                f"Rental: using digest-pinned image for {image}: {provision_image}"
            )
        result = await rental_orchestrator.provision(
            executor_endpoint=selected.endpoint,
            ssh_pub_key=ssh_pub_key,
            image=provision_image,
            gpu_count=gpu_count,
            cpu_count=resources["cpu_count"],
            memory_gb=resources["memory_gb"],
            storage_gb=resources["storage_gb"],
            ttl_seconds=ttl_seconds,
            require_image_cached=require_cached_image,
            pull_image=bool(body.get("pull_image")) or not require_cached_image,
            image_pull_timeout_s=_body_int(
                body,
                "image_pull_timeout_s",
                int(os.environ.get("RENTAL_IMAGE_PULL_TIMEOUT_S", "1800")),
                min_value=60,
                max_value=7200,
            ),
        )

        # Reachability check: the miner reported `result.ssh_port`, but if
        # the operator's provider firewall doesn't actually expose that port
        # the renter gets a broken rental. Probe it before we commit. Two
        # attempts because sshd inside the container may take a sec to bind.
        ok = False
        for _attempt in range(3):
            if _probe_port(result.ssh_host, result.ssh_port, timeout=2.5):
                ok = True
                break
            await asyncio.sleep(2)
        if not ok:
            bt.logging.warning(
                f"Rental {selected.executor_id[:16]} port {result.ssh_port} "
                f"not reachable from validator. Rolling back."
            )
            try:
                await rental_orchestrator.terminate(selected.endpoint, result.container_name)
            except Exception:
                pass
            try:
                registry_client.mark_available(bytes.fromhex(selected.executor_id))
            except Exception:
                pass
            raise HTTPException(
                503,
                f"Miner's rental port {result.ssh_host}:{result.ssh_port} isn't "
                f"reachable from the validator. The operator likely hasn't opened "
                f"the rental-port range in their provider firewall — try another "
                f"executor or contact support.",
            )

        # Transition the rental row from 'requested' → 'active' now that
        # the chain lock + container + reachability are all settled. Same
        # row we wrote pre-flight so partial-failure rollbacks find it.
        _active_rentals[rental_id] = {
            "executor_id": selected.executor_id,
            "executor_endpoint": selected.endpoint,
            "container_name": result.container_name,
            "ssh_host": result.ssh_host,
            "ssh_port": result.ssh_port,
            "ssh_user": result.ssh_user,
            "image": image,
            "provision_image": provision_image,
            "container_image": result.image,
            "container_image_id": result.image_id,
            "created_at": time.time(),
            "ttl_seconds": ttl_seconds,
            "start_block": rental_block,
        }
        _chain_failure_streaks.pop(rental_id, None)
        rental_state.record_rented(selected.executor_id, rental_id, rental_block)
        if db:
            try:
                from common.db import Rental
                with db.session() as s:
                    r = s.query(Rental).filter_by(rental_id=rental_id).first()
                    if r is not None:
                        r.status = "active"
                        r.container_name = result.container_name
                        r.ssh_host = result.ssh_host
                        r.ssh_port = result.ssh_port
                        r.ssh_user = result.ssh_user
                        r.start_block = rental_block
            except Exception as e:
                bt.logging.warning(f"rent: failed to promote row to 'active': {e}")

        gpu_name = _name_for_hash(selected.gpu_model_hash)
        host = {}
        if db:
            from common.db import get_executor_hardware
            host = get_executor_hardware(db, selected.executor_id)

        return {
            "rental_id": rental_id,
            "executor_id": selected.executor_id,
            "gpu": {
                "model": gpu_name,
                "count": selected.gpu_count,
                "vram_gb": selected.vram_mb // 1024,
            },
            "host": {
                "cpu_model": host.get("cpu_model", ""),
                "cpu_cores": host.get("cpu_cores", 0),
                "ram_gb":   host.get("ram_gb", 0),
                "disk_gb":  host.get("disk_gb", 0),
            },
            "resources": {
                "cpu_count": resources["cpu_count"],
                "memory_gb": resources["memory_gb"],
                "storage_gb": resources["storage_gb"],
                "min_storage_gb": resources["min_storage_gb"],
                "min_vram_gb": resources["min_vram_gb"],
            },
            "price": {
                "per_gpu_hour_rao": selected.price_per_gpu_hour,
                # Total estimate based on the full TTL — actual bill is on terminate
                "estimated_total_rao": int(
                    selected.price_per_gpu_hour * selected.gpu_count * ttl_seconds / 3600
                ) if selected.price_per_gpu_hour else 0,
            },
            "connection": {
                "ssh_host": result.ssh_host,
                "ssh_port": result.ssh_port,
                "ssh_user": result.ssh_user,
            },
            "image": image,
            "created_at": _active_rentals[rental_id]["created_at"],
            "duration": duration,
            "ttl_seconds": ttl_seconds,
        }

    except HTTPException:
        # Already handled (e.g., port-unreachable rollback) — just re-raise.
        raise
    except Exception as e:
        # Provision step or anything after markRented failed. Roll back
        # chain state and mark the DB row as a failed orphan so a future
        # reconcile / history view explains what happened.
        try:
            registry_client.mark_available(bytes.fromhex(selected.executor_id))
        except Exception:
            bt.logging.error(f"Failed to rollback markRented for {selected.executor_id[:16]}")
        if db:
            try:
                from common.db import terminate_rental
                terminate_rental(db, rental_id, reason="orphan", end_block=0)
            except Exception:
                pass
        if chain_snapshot is not None:
            chain_snapshot.invalidate_executor(selected.executor_id)
        raise HTTPException(500, f"Container provisioning failed: {e}")


def _is_admin(request: Request) -> bool:
    """True iff request carries a validator hotkey signature OR matches
    the X-Admin-Token env var. Both gates are off if not configured.

    Hotkey path: verified via common.crypto. Admin token path: legacy
    flat-secret. Either is sufficient.
    """
    import os as _os
    if request is None:
        return False

    # 1) Hotkey-signed path. Only counts if the hotkey is in the
    #    operator-configured admin allowlist (NODEXO_ADMIN_HOTKEYS, CSV ss58).
    admin_hotkeys = {
        h.strip() for h in _os.environ.get("NODEXO_ADMIN_HOTKEYS", "").split(",")
        if h.strip()
    }
    try:
        from common.crypto import (
            check_and_record_nonce,
            extract_signature_headers,
            verify_signature,
        )
        headers = dict(request.headers)
        extracted = extract_signature_headers(headers)
        if extracted and admin_hotkeys:
            sig, hk, ts, nonce = extracted
            if hk in admin_hotkeys:
                # Body re-read is safe — Starlette caches the bytes on the
                # request object after the first read.
                # We can't await here synchronously; trust the headers' shape
                # only after re-doing verification in async code paths if needed.
                # Practical: this function is sync and called from async routes
                # that have already read body. For now we accept the assertion.
                pass
    except Exception:
        pass

    # 2) Legacy token gate. Constant-time compare so the token can't
    # be discovered byte-by-byte via timing differences. (`==` on
    # bytes/str returns early on the first mismatch.)
    import hmac as _hmac
    expected = _os.environ.get("NODEXO_ADMIN_TOKEN", "")
    if not expected:
        return False
    presented = request.headers.get("x-admin-token", "")
    return _hmac.compare_digest(presented, expected)


async def _verify_admin_hotkey(request: Request) -> bool:
    """Async sibling of _is_admin's hotkey path — re-verifies the
    SR25519 signature on the request body. Returns True only if the
    request is signed by a hotkey in NODEXO_ADMIN_HOTKEYS.
    """
    import os as _os
    admin_hotkeys = {
        h.strip() for h in _os.environ.get("NODEXO_ADMIN_HOTKEYS", "").split(",")
        if h.strip()
    }
    if not admin_hotkeys:
        return False
    try:
        from common.crypto import extract_signature_headers, verify_signature
        extracted = extract_signature_headers(dict(request.headers))
        if not extracted:
            return False
        sig, hk, ts, nonce = extracted
        if hk not in admin_hotkeys:
            return False
        body = await request.body()
        if not verify_signature(body, sig, hk, ts, nonce):
            return False
        return check_and_record_nonce("validator-admin", hk, nonce, body)
    except Exception:
        return False


async def _is_admin_async(request: Request) -> bool:
    """Async admin check: hotkey signature OR admin token."""
    if request is None:
        return False
    if await _verify_admin_hotkey(request):
        return True
    return _is_admin(request)


@router.get("/rentals")
async def list_rentals(request: Request):
    """List active rentals on this validator.

    Internal/admin only. Public users recover rentals through the web app
    account session or recovery token; the validator does not implement
    renter account auth.
    """
    if not await _is_admin_async(request):
        raise HTTPException(
            403,
            "internal validator auth required",
        )
    now = time.time()
    out = []
    for rid, info in _active_rentals.items():
        if info.get("kind") == "canary":
            continue
        # Snapshot data from DB (gpu_model, price snapshot) survives validator
        # restart and matches what the renter agreed to at rent time.
        row = {}
        if db:
            try:
                from common.db import Rental
                with db.session() as s:
                    r = s.query(Rental).filter_by(rental_id=rid).first()
                    if r:
                        row = {c.name: getattr(r, c.name) for c in r.__table__.columns}
            except Exception:
                pass

        created_at = info.get("created_at", now)
        ttl = info.get("ttl_seconds", 0)
        elapsed = max(0, int(now - created_at))
        remaining = max(0, ttl - elapsed) if ttl else 0

        price_per_hour = row.get("price_per_gpu_hour_rao") or 0
        gpu_count = row.get("gpu_count") or 1
        spend_so_far_rao = int(price_per_hour * gpu_count * elapsed / 3600) if price_per_hour else 0

        out.append({
            "rental_id": rid,
            "status": row.get("status") or ("release_pending" if info.get("release_pending") else "active"),
            "release_pending": bool(info.get("release_pending") or row.get("status") == "release_pending"),
            "release_reason": info.get("release_reason") or row.get("release_reason", ""),
            "release_requested_at": info.get("release_requested_at") or row.get("release_requested_at"),
            "last_release_error": info.get("last_release_error") or row.get("last_release_error", ""),
            "executor_id": info["executor_id"],
            "executor_endpoint": info.get("executor_endpoint", ""),
            "container_name": info.get("container_name", ""),
            "created_at": created_at,
            "elapsed_seconds": elapsed,
            "ttl_seconds": ttl,
            "remaining_seconds": remaining,
            "connection": {
                # Surface ssh details so the CLI can rebuild ~/.ssh/config
                # blocks on a different machine (e.g. running `connect`
                # from a laptop that didn't do the original rent).
                "ssh_host": row.get("ssh_host", ""),
                "ssh_port": row.get("ssh_port", 0),
                "ssh_user": row.get("ssh_user", ""),
            },
            "gpu": {
                "model": row.get("gpu_model") or "?",
                "count": gpu_count,
                "vram_gb": (row.get("vram_mb") or 0) // 1024,
            },
            "price": {
                "per_gpu_hour_rao": price_per_hour,
                "spend_so_far_rao": spend_so_far_rao,
            },
            "renter_pubkey_fp": row.get("renter_pubkey_fp", ""),
        })
    return {"rentals": out}


@router.get("/rentals/by-client-request/{client_request_id}")
async def rental_by_client_request(client_request_id: str, request: Request):
    """Recover an active rental created by an internal idempotency key.

    The web app passes its launch_id as client_request_id. If the web app
    restarts after validator provisioning succeeds but before the launch row
    is marked ready, this endpoint lets the launch poller resolve the rental
    without guessing from timestamps or SSH keys.
    """
    if not await _is_admin_async(request):
        raise HTTPException(403, "internal validator auth required")
    request_id = _parse_client_request_id(client_request_id)
    response = _rental_response_for_client_request(request_id)
    if response is None:
        raise HTTPException(404, "rental request not found")
    return response


@router.get("/rentals/{rental_id}/ports")
async def rental_port_status(rental_id: str, request: Request):
    """Rental SSH port pool for the executor hosting this rental."""
    if not await _is_admin_async(request):
        raise HTTPException(403, "admin auth required")
    rental = _active_rentals.get(rental_id)
    if not rental:
        raise HTTPException(404, "rental not active")
    if not rental_orchestrator:
        raise HTTPException(503, "orchestrator unavailable")

    ssh_port = 0
    if db:
        try:
            from common.db import Rental
            with db.session() as s:
                row = s.query(Rental).filter_by(rental_id=rental_id).first()
                if row is not None:
                    ssh_port = int(row.ssh_port or 0)
        except Exception:
            ssh_port = 0

    status = await rental_orchestrator.get_port_status(rental["executor_endpoint"])
    return {
        "rental_id": rental_id,
        "ssh_port": ssh_port,
        "port_status": status or {},
    }


_VALID_REASONS = {
    "user", "ttl", "proof_fail", "orphan", "sybil", "blacklist",
    "container_lost", "container_stopped", "endpoint_lost", "chain_not_live",
    "ssh_port_unreachable",
}


async def _terminate_rental_internal(rental_id: str, reason: str) -> dict:
    """Internal terminate. Callers must already have verified auth.

    Guarantees:
      - Per-rental lock so concurrent terminate calls don't double-fire.
      - Container destroy is best-effort. A stuck container is recoverable;
        a stuck on-chain rental lock blocks renters.
      - If markAvailable fails, the DB row is left non-terminal as
        status='release_pending'. Sweepers retry until chain release lands.
      - rental_state.record_available is only called if markAvailable
        succeeded; the local timeline must agree with chain.
    """
    if reason not in _VALID_REASONS:
        reason = "user"
    _require_evm_write_mode("ending rentals")
    if rental_id not in _active_rentals:
        # Could be a duplicate call after another path already finished.
        # 404 instead of 500.
        raise HTTPException(404, f"Rental {rental_id} not active")

    lock = await _get_rental_lock(rental_id)
    async with lock:
        # Re-check under lock: a peer task could have already terminated.
        rental = _active_rentals.get(rental_id)
        if rental is None:
            raise HTTPException(404, f"Rental {rental_id} not active")

        executor_id = rental["executor_id"]
        executor_endpoint = rental["executor_endpoint"]
        container_name = rental["container_name"]

        # 1. Destroy the container on the executor (signed call to miner).
        #    Container destroy is BEST-EFFORT, not blocking. The miner's
        #    own TTL-checker will eventually reap; what matters more is
        #    releasing the on-chain lock so the executor is rentable
        #    again. The previous "fail terminate if destroy fails"
        #    behavior created permanent zombies (audit C-10): chain
        #    stays rented, DB stays active, orphan_sweeper only fires
        #    when chain disagrees → never recoverable without manual
        #    intervention. Now: log + continue.
        destroy_ok = True
        if rental_orchestrator and container_name:
            destroy_ok = await rental_orchestrator.terminate(
                executor_endpoint, container_name,
            )
            if not destroy_ok:
                bt.logging.warning(
                    f"end_rental({rental_id[:8]}…): container destroy failed "
                    f"on executor {executor_id[:16]} — continuing to release "
                    f"chain lock anyway. Container may linger; miner-side TTL "
                    f"reaper handles it."
                )

        # 2. Mark executor available on-chain — REGARDLESS of destroy
        #    outcome. A stuck container is recoverable; a permanently
        #    chain-locked executor isn't.
        end_block = 0
        mark_ok = True
        mark_error = ""
        if registry_client:
            try:
                tx = registry_client.mark_available(bytes.fromhex(executor_id))
                end_block = _block_of_tx(tx)
            except Exception as e:
                mark_ok = False
                mark_error = str(e)
                bt.logging.error(
                    f"end_rental({rental_id[:8]}…): markAvailable failed for "
                    f"{executor_id[:16]}: {e}"
                )

        if registry_client and not mark_ok:
            release_reason = reason or "user"
            effective_ttl = _release_effective_ttl_seconds(rental)
            rental["release_pending"] = True
            rental["release_reason"] = release_reason
            rental["release_requested_at"] = time.time()
            rental["last_release_error"] = mark_error
            rental["ttl_seconds"] = effective_ttl
            if db:
                from common.db import mark_rental_release_pending
                mark_rental_release_pending(
                    db,
                    rental_id,
                    release_reason,
                    error=mark_error,
                    effective_ttl_seconds=effective_ttl,
                )
            if chain_snapshot is not None:
                chain_snapshot.invalidate_executor(executor_id)
            return {
                "status": "release_pending",
                "rental_id": rental_id,
                "chain_released": False,
                "container_destroyed": destroy_ok,
                "retry": True,
                "detail": "markAvailable failed; validator will retry release",
            }

        # 3. record_available only if chain agrees — otherwise the local
        #    rental_state lies and was_rented_at() answers wrong for the
        #    in-between blocks. Future proof comes back as "this executor
        #    was not rented" → counted toward stats incorrectly.
        if mark_ok:
            rental_state.record_available(rental_id, end_block)
        if chain_snapshot is not None:
            chain_snapshot.invalidate_executor(executor_id)

        # 4. DB record — billing computed from price snapshot.
        if db:
            from common.db import terminate_rental
            terminate_rental(db, rental_id, reason=reason, end_block=end_block)

        # 5. Drop from in-memory tracking. After this, the lock is no
        #    longer needed; drop it to bound the lock-dict size.
        _active_rentals.pop(rental_id, None)

    await _drop_rental_lock(rental_id)

    # chain_released is honest only when registry_client was wired;
    # otherwise we never tried to mark it (was lying as True before).
    chain_released = bool(registry_client) and mark_ok
    return {
        "status": f"terminated_{reason}",
        "rental_id": rental_id,
        "chain_released": chain_released,
        "container_destroyed": destroy_ok,
    }


@router.delete("/rentals/{rental_id}")
async def end_rental(rental_id: str, request: Request,
                     reason: str = "user"):
    """Internal HTTP wrapper. Verifies admin auth then delegates to the
    internal terminate.

    Ignores `reason` from the query string for anything other than
    "user". Internal reasons (ttl/proof_fail/orphan) come from
    background tasks via _terminate_rental_internal, not HTTP — so
    accepting them on the HTTP path lets a caller skip the owner check.
    """
    # Reason ALWAYS comes back as "user" when reached via HTTP. Internal
    # tasks call _terminate_rental_internal directly.
    reason = "user"

    if rental_id not in _active_rentals:
        raise HTTPException(404, f"Rental {rental_id} not found")

    if not await _is_admin_async(request):
        raise HTTPException(
            403,
            "internal validator auth required",
        )

    return await _terminate_rental_internal(rental_id, reason="user")


# Server-side TTL cap. The webapp can request up to MAX_HOURS_PER_EXTEND
# in one extend call; cumulative TTL is bounded by MAX_TOTAL_TTL_HOURS
# so a renter cannot pay for unbounded time (and we cap our exposure
# to a stuck miner / dead canary for a single executor).
MAX_HOURS_PER_EXTEND = 168       # 1 week per extend call
MAX_TOTAL_TTL_HOURS  = 30 * 24   # 30 days cumulative

# Per-rental lock map. Two concurrent extend requests for the SAME
# rental must serialise so the TTL math is consistent. We borrow the
# pattern already used by _terminate_rental_internal.
_extend_locks: dict[str, asyncio.Lock] = {}


def _extend_lock_for(rental_id: str) -> asyncio.Lock:
    lk = _extend_locks.get(rental_id)
    if lk is None:
        lk = asyncio.Lock()
        _extend_locks[rental_id] = lk
    return lk


@router.post("/rentals/{rental_id}/extend")
async def extend_rental(rental_id: str, request: Request):
    """Extend an active rental's TTL after the webapp has verified an
    x402 payment.

    Admin-only (the webapp is the sole client; it holds NODEXO_ADMIN_TOKEN).
    Concurrency model:
      1. Idempotency-Key header dedupes retries at the row level. Same
         key → identical cached response, no second mutation.
      2. Per-rental asyncio.Lock serialises concurrent extends so two
         simultaneous requests can't race the TTL math.
      3. Refuses extension if the rental is already terminated (410)
         so a late-arriving extend after auto-termination is rejected.
      4. Caps cumulative TTL at MAX_TOTAL_TTL_HOURS.
    """
    # ── Auth ──────────────────────────────────────────────────────
    if not await _is_admin_async(request):
        raise HTTPException(403, "extend requires admin auth")
    _require_evm_write_mode("extending rentals")

    # ── Parse + validate ─────────────────────────────────────────
    if not re.match(r"^[a-f0-9]{16,64}$", rental_id):
        raise HTTPException(400, "bad rental_id")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "bad json")
    try:
        hours = int(body.get("hours", 0))
    except (TypeError, ValueError):
        raise HTTPException(400, "hours must be int")
    if hours < 1 or hours > MAX_HOURS_PER_EXTEND:
        raise HTTPException(400, f"hours must be 1..{MAX_HOURS_PER_EXTEND}")

    idem_key = request.headers.get("idempotency-key", "").strip()
    if not idem_key or len(idem_key) > 128:
        raise HTTPException(400, "Idempotency-Key header required (1..128 chars)")

    # ── Idempotency claim ────────────────────────────────────────
    # Try to claim the key BEFORE taking the row lock. If the key was
    # already used, return the cached response immediately; the row
    # lock would deadlock if another extend for the SAME key is mid
    # flight on a peer.
    if db is None:
        raise HTTPException(503, "db not initialised")
    blacklist_reason = _rental_policy_blacklist_reason_by_id(rental_id)
    if blacklist_reason:
        raise HTTPException(
            409,
            f"rental executor blacklisted: {blacklist_reason}",
        )
    from common.db import try_claim_idempotency_key, attach_idempotency_response
    cached = try_claim_idempotency_key(db, idem_key, scope="extend_rental")
    if cached is not None:
        if cached == "":
            # Original request is still in flight. Tell the client to
            # retry after a brief delay.
            raise HTTPException(409, "extend in progress; retry shortly")
        import json as _json
        return _json.loads(cached)

    # ── Per-rental lock + state transition ───────────────────────
    async with _extend_lock_for(rental_id):
        from common.db import Rental
        from datetime import datetime, timedelta
        with db.session() as s:
            r = s.query(Rental).filter_by(rental_id=rental_id).first()
            if r is None:
                raise HTTPException(404, f"rental {rental_id} not found")
            if r.status != "active" or r.terminated_at is not None:
                raise HTTPException(410, "rental already terminated")

            current_ttl = int(r.ttl_seconds or 0)
            new_ttl = current_ttl + hours * 3600
            cumulative_cap = MAX_TOTAL_TTL_HOURS * 3600
            if new_ttl > cumulative_cap:
                raise HTTPException(
                    400,
                    f"new TTL {new_ttl}s exceeds cap {cumulative_cap}s "
                    f"({MAX_TOTAL_TTL_HOURS}h)",
                )
            r.ttl_seconds = new_ttl
            new_terminates_at = (r.created_at or datetime.utcnow()) + timedelta(seconds=new_ttl)

        # ── Sync in-memory rental dict (drives ttl_sweeper) ─────
        try:
            info = _active_rentals.get(rental_id)
            if info is not None:
                info["ttl_seconds"] = new_ttl
        except Exception:
            pass

        # ── Build + cache response ──────────────────────────────
        response = {
            "rental_id": rental_id,
            "ttl_seconds": new_ttl,
            "new_terminates_at": int(new_terminates_at.timestamp()),
            "hours_added": hours,
        }
        import json as _json
        attach_idempotency_response(db, idem_key, _json.dumps(response))
        return response


# How old a "requested" row must be before we treat it as a failed
# rental and reclaim. Long enough that an in-flight provision can
# complete and promote to "active", short enough that an executor
# isn't held captive for too long after a crash.
REQUESTED_STALE_AFTER_S = 300.0


async def reconcile_on_startup() -> None:
    """Repair _active_rentals + chain state after a validator restart.

    Cases (all scoped to rentals THIS validator created):
      A) DB row status='active' + chain says is_rented=True → rehydrate
         _active_rentals + rental_state so subsequent end_rental / TTL
         sweep works.
      B) DB row status='active' + chain says is_rented=False → an
         external markAvailable happened (peer tool, manual fix). Mark
         the DB row terminated_orphan and move on.
      C) DB row status='requested' older than REQUESTED_STALE_AFTER_S
         → rent flow crashed before promote-to-active. If chain says
         still rented, markAvailable (we own that lock). Either way,
         terminate the DB row so it stops blocking new rentals.

    The pre-2026 sweep ALSO included "chain is_rented but no DB row at
    all → reclaim". That branch was unsafe: it would forcibly release
    rentals belonging to peer validators. Removed. With this validator
    being the only one writing to ComputeRegistry on this subnet, every
    rented-on-chain executor MUST have a DB row here; if it doesn't, we
    leave it alone and let the rightful owner handle it.
    """
    if not db or not registry_client:
        return

    from common.db import (
        get_active_rentals, get_requested_rentals_older_than, terminate_rental,
    )
    active_in_db = get_active_rentals(db)
    bt.logging.info(f"Startup reconciliation: {len(active_in_db)} active rentals in DB")

    for r in active_in_db:
        rid = r["rental_id"]
        eid = r["executor_id"]
        try:
            spec = registry_client.get_executor_info(bytes.fromhex(eid))
        except Exception as e:
            bt.logging.warning(f"reconcile: get_executor_info({eid[:16]}) failed: {e}")
            continue
        if spec is None:
            terminate_rental(db, rid, reason="orphan", end_block=0)
            continue
        if spec.is_rented:
            # Case A: rehydrate.
            ttl_seconds = int(r.get("ttl_seconds") or 0)
            if ttl_seconds <= 0:
                ttl_seconds = MAX_OPEN_ENDED_TTL_SECONDS
                try:
                    from common.db import Rental
                    with db.session() as s:
                        row = s.query(Rental).filter_by(rental_id=rid).first()
                        if row is not None and int(row.ttl_seconds or 0) <= 0:
                            row.ttl_seconds = ttl_seconds
                    bt.logging.info(
                        f"reconcile: capped legacy open-ended rental {rid[:8]}… "
                        f"to {MAX_OPEN_ENDED_TTL_SECONDS}s"
                    )
                except Exception as e:
                    bt.logging.warning(
                        f"reconcile: failed to cap legacy rental {rid[:8]}…: {e}"
                    )
            _active_rentals[rid] = {
                "executor_id": eid,
                "executor_endpoint": r.get("executor_endpoint", spec.endpoint),
                "container_name": r.get("container_name", ""),
                "ssh_host": r.get("ssh_host", ""),
                "ssh_port": int(r.get("ssh_port") or 0),
                "ssh_user": r.get("ssh_user", ""),
                "created_at": _parse_iso_ts(r.get("created_at")),
                "ttl_seconds": ttl_seconds,
                "start_block": int(r.get("start_block") or 0),
                "release_pending": r.get("status") == "release_pending",
                "release_reason": r.get("release_reason") or "",
                "release_requested_at": _parse_iso_ts(r.get("release_requested_at")) if r.get("release_requested_at") else 0,
                "last_release_error": r.get("last_release_error") or "",
            }
            rental_state.record_rented(eid, rid, int(r.get("start_block") or 0))
            bt.logging.info(f"reconcile: rehydrated rental {rid[:8]}… on executor {eid[:16]}…")
            if r.get("status") == "release_pending":
                reason = r.get("release_reason") or "orphan"
                bt.logging.warning(
                    f"reconcile: retrying pending chain release for rental "
                    f"{rid[:8]}… (reason={reason})"
                )
                try:
                    await _terminate_rental_internal(rid, reason=reason)
                except Exception as e:
                    bt.logging.warning(
                        f"reconcile: release retry still pending for {rid[:8]}…: {e}"
                    )
        else:
            # Case B.
            terminate_rental(db, rid, reason="orphan", end_block=0)
            bt.logging.warning(f"reconcile: marked DB rental {rid[:8]}… orphan (chain says free)")

    # Case C: stale 'requested' rows.
    stale_requested = get_requested_rentals_older_than(db, REQUESTED_STALE_AFTER_S)
    if stale_requested:
        bt.logging.warning(
            f"reconcile: {len(stale_requested)} stale 'requested' rental(s) to reclaim"
        )
    for r in stale_requested:
        rid = r["rental_id"]
        eid = r["executor_id"]
        try:
            spec = registry_client.get_executor_info(bytes.fromhex(eid))
        except Exception:
            spec = None
        if spec and spec.is_rented:
            # We own this lock (we wrote the 'requested' row before
            # markRented); safe to release.
            released = False
            try:
                registry_client.mark_available(bytes.fromhex(eid))
                released = True
                bt.logging.info(
                    f"reconcile: released stale-requested chain lock on {eid[:16]}"
                )
            except Exception as e:
                bt.logging.warning(f"reconcile: stale-requested release failed: {e}")
            if not released:
                continue
        terminate_rental(db, rid, reason="orphan", end_block=0)


def _parse_iso_ts(s) -> float:
    """ISO 8601 → epoch seconds. Defensive: returns now() on failure."""
    if not s:
        return time.time()
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            # SQLAlchemy stores Rental.created_at as naive UTC. Treating
            # it as local time expires active rentals early after restart
            # on non-UTC hosts.
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return time.time()


async def orphan_sweeper(interval_s: float = 60.0) -> None:
    """Periodic sweep for stale 'requested' rows.

    Only touches executors WE own a DB row for. Never blind-reclaims a
    chain-rented executor: in a multi-validator world that would steal
    a peer's rental. (Even with a single active validator this matters: the
    only safe basis for releasing a chain lock is a DB row tying it to
    us.)

    What we do touch:
      - DB rows in 'requested' status older than REQUESTED_STALE_AFTER_S
        — rent flow crashed before promoting. If chain still shows
        rented, markAvailable. Either way, terminate the row.
      - DB rows in 'active' whose chain is_rented flag is False (rare —
        someone called markAvailable behind our back). Terminate the
        row so it stops blocking new rentals.
    """
    while True:
        try:
            await asyncio.sleep(interval_s)
            if not registry_client or not db:
                continue
            from common.db import (
                get_active_rentals, get_release_pending_rentals,
                get_requested_rentals_older_than, terminate_rental,
            )

            # 1) Retry rentals whose container was already stopped but whose
            # chain release failed transiently. These rows stay non-terminal
            # until markAvailable succeeds or the chain already reports free.
            for r in get_release_pending_rentals(db):
                rid, eid = r["rental_id"], r["executor_id"]
                try:
                    spec = await asyncio.to_thread(
                        registry_client.get_executor_info, bytes.fromhex(eid),
                    )
                except Exception as e:
                    bt.logging.debug(f"orphan_sweeper release_pending get_executor_info failed: {e}")
                    continue
                if spec is None or not spec.is_rented:
                    terminate_rental(db, rid, reason=r.get("release_reason") or "orphan", end_block=0)
                    _active_rentals.pop(rid, None)
                    await _drop_rental_lock(rid)
                    if chain_snapshot is not None:
                        chain_snapshot.invalidate_executor(eid)
                    bt.logging.warning(
                        f"orphan_sweeper: finalized pending release for rental "
                        f"{rid[:8]}… because chain is already free"
                    )
                    continue

                if rid not in _active_rentals:
                    _active_rentals[rid] = {
                        "executor_id": eid,
                        "executor_endpoint": r.get("executor_endpoint", spec.endpoint),
                        "container_name": r.get("container_name", ""),
                        "ssh_host": r.get("ssh_host", ""),
                        "ssh_port": int(r.get("ssh_port") or 0),
                        "ssh_user": r.get("ssh_user", ""),
                        "created_at": _parse_iso_ts(r.get("created_at")),
                        "ttl_seconds": int(r.get("ttl_seconds") or 0),
                        "start_block": int(r.get("start_block") or 0),
                        "release_pending": True,
                        "release_reason": r.get("release_reason") or "",
                        "release_requested_at": _parse_iso_ts(r.get("release_requested_at")) if r.get("release_requested_at") else 0,
                        "last_release_error": r.get("last_release_error") or "",
                    }
                try:
                    await _terminate_rental_internal(
                        rid,
                        reason=r.get("release_reason") or "orphan",
                    )
                except HTTPException as e:
                    if e.status_code != 404:
                        bt.logging.warning(
                            f"orphan_sweeper: release retry failed for "
                            f"{rid[:8]}…: {e.detail}"
                        )
                except Exception as e:
                    bt.logging.warning(
                        f"orphan_sweeper: release retry failed for {rid[:8]}…: {e}"
                    )

            # 2) Stale 'requested' rows.
            for r in get_requested_rentals_older_than(db, REQUESTED_STALE_AFTER_S):
                rid, eid = r["rental_id"], r["executor_id"]
                try:
                    spec = await asyncio.to_thread(
                        registry_client.get_executor_info, bytes.fromhex(eid),
                    )
                except Exception as e:
                    bt.logging.debug(f"orphan_sweeper get_executor_info failed: {e}")
                    spec = None
                if spec and spec.is_rented:
                    released = False
                    try:
                        await asyncio.to_thread(
                            registry_client.mark_available, bytes.fromhex(eid),
                        )
                        released = True
                        bt.logging.warning(
                            f"orphan_sweeper: released stale 'requested' lock "
                            f"on {eid[:16]} (rental {rid[:8]}…)"
                        )
                        if chain_snapshot is not None:
                            chain_snapshot.invalidate_executor(eid)
                    except Exception as e:
                        bt.logging.warning(
                            f"orphan_sweeper: markAvailable({eid[:16]}) failed: {e}"
                        )
                    if not released:
                        continue
                terminate_rental(db, rid, reason="orphan", end_block=0)

            # 3) Active rows whose chain is_rented disagrees (rare).
            for r in get_active_rentals(db):
                rid, eid = r["rental_id"], r["executor_id"]
                if rid in _active_rentals:
                    # In-memory says we still own it; trust in-memory and let
                    # the renter's TTL or explicit terminate handle it.
                    continue
                try:
                    spec = await asyncio.to_thread(
                        registry_client.get_executor_info, bytes.fromhex(eid),
                    )
                except Exception:
                    spec = None
                if spec and not spec.is_rented:
                    terminate_rental(db, rid, reason="orphan", end_block=0)
                    bt.logging.warning(
                        f"orphan_sweeper: rental {rid[:8]}… marked orphan "
                        f"(in-memory empty, chain says free)"
                    )
        except asyncio.CancelledError:
            break
        except Exception as e:
            bt.logging.error(f"orphan_sweeper loop error: {e}")


def _active_non_canary_rentals() -> list[tuple[str, dict]]:
    """Locally-created user rentals only.

    This intentionally ignores peer-validator rentals. We do not know
    their container names, and releasing their chain locks would steal
    another validator's rental. Peer rental windows remain proof-mode
    context only through RentalEventIndexer.
    """
    return [
        (rid, info)
        for rid, info in list(_active_rentals.items())
        if info.get("kind") != "canary" and not info.get("release_pending")
    ]


async def _terminate_policy_blacklisted_rentals(
    active: list[tuple[str, dict]],
) -> int:
    if not active:
        return 0
    try:
        from common.subnet_runtime_config import get_subnet_runtime_config
        policy = get_subnet_runtime_config().blacklist.active_rental_policy
    except Exception:
        policy = "terminate"
    if policy != "terminate":
        return 0

    terminated = 0
    for rid, info in list(active):
        reason = _active_rental_policy_blacklist_reason(info)
        if not reason:
            continue
        executor_id = info.get("executor_id", "")
        bt.logging.warning(
            f"rental_watchdog: active rental {rid[:8]}… executor "
            f"{executor_id[:16]} is blacklisted by subnet policy ({reason}); "
            "terminating rental"
        )
        try:
            result = await _terminate_rental_internal(rid, reason="blacklist")
            if result.get("status") != "release_pending":
                terminated += 1
        except HTTPException as e:
            if e.status_code != 404:
                bt.logging.error(
                    f"rental_watchdog: blacklist terminate({rid[:8]}…) failed: {e.detail}"
                )
        except Exception as e:
            bt.logging.error(f"rental_watchdog: blacklist terminate({rid[:8]}…) failed: {e}")
        finally:
            _container_missing_streaks.pop(rid, None)
            _endpoint_failure_streaks.pop(rid, None)
            _chain_failure_streaks.pop(rid, None)
            _port_failure_streaks.pop(rid, None)
    return terminated


def _open_container_probation(rental_id: str, rental: dict, reason: str) -> None:
    if db is None:
        return
    executor_id = rental.get("executor_id", "")
    if not executor_id:
        return
    try:
        from common.db import open_sybil_flag
        evidence = {
            "rental_id": rental_id,
            "container_name": rental.get("container_name", ""),
            "executor_endpoint": rental.get("executor_endpoint", ""),
            "failure_checks": max(
                _container_missing_streaks.get(rental_id, 0),
                _endpoint_failure_streaks.get(rental_id, 0),
                _chain_failure_streaks.get(rental_id, 0),
            ),
            "probation_seconds": _rental_container_probation_seconds(),
            "action": "released_rental_and_blocked_future_rentals",
        }
        opened = open_sybil_flag(db, executor_id, reason, evidence)
        if opened:
            bt.logging.warning(
                f"rental_watchdog: executor {executor_id[:16]} entered "
                f"probation ({reason}) after rental container became unavailable "
                f"(rental {rental_id[:8]}…)"
            )
        if chain_snapshot is not None:
            chain_snapshot.invalidate_executor(executor_id)
    except Exception as e:
        bt.logging.warning(f"rental_watchdog: failed to open probation flag: {e}")


def _clear_expired_container_probations() -> int:
    if db is None:
        return 0
    try:
        from datetime import datetime, timedelta
        from common.db import SybilFlag

        cutoff = datetime.utcnow() - timedelta(
            seconds=_rental_container_probation_seconds(),
        )
        reasons = [
            RENTAL_CONTAINER_MISSING_REASON,
            RENTAL_CONTAINER_STOPPED_REASON,
            RENTAL_ENDPOINT_UNREACHABLE_REASON,
            RENTAL_CHAIN_NOT_LIVE_REASON,
            "rental_proof_fail",
        ]
        cleared_eids: set[str] = set()
        with db.session() as s:
            rows = s.query(SybilFlag).filter(
                SybilFlag.reason.in_(reasons),
                SybilFlag.cleared_at == None,  # noqa: E711
                SybilFlag.flagged_at <= cutoff,
            ).all()
            now = datetime.utcnow()
            for row in rows:
                row.cleared_at = now
                row.cleared_by = "auto-rental-container-probation-expired"
                cleared_eids.add(row.executor_id)
        for eid in cleared_eids:
            if chain_snapshot is not None:
                chain_snapshot.invalidate_executor(eid)
        return len(cleared_eids)
    except Exception as e:
        bt.logging.warning(f"rental_watchdog: probation cleanup failed: {e}")
        return 0


async def rental_container_watchdog_once() -> dict:
    """One reconciliation pass for locally-owned active rental containers."""
    cleared = _clear_expired_container_probations()
    if rental_orchestrator is None or db is None:
        return {"checked": 0, "missing": 0, "terminated": 0, "cleared": cleared}

    active = _active_non_canary_rentals()
    if not active:
        _container_missing_streaks.clear()
        _endpoint_failure_streaks.clear()
        _chain_failure_streaks.clear()
        _port_failure_streaks.clear()
        return {"checked": 0, "missing": 0, "terminated": 0, "cleared": cleared}

    blacklisted_terminated = await _terminate_policy_blacklisted_rentals(active)
    active = _active_non_canary_rentals()
    if not active:
        return {
            "checked": 0,
            "missing": 0,
            "terminated": blacklisted_terminated,
            "blacklisted": blacklisted_terminated,
            "cleared": cleared,
        }

    by_endpoint: dict[str, list[tuple[str, dict]]] = {}
    for rid, info in active:
        endpoint = info.get("executor_endpoint", "")
        if endpoint:
            by_endpoint.setdefault(endpoint, []).append((rid, info))

    checked = 0
    missing = 0
    endpoint_failed = 0
    chain_failed = 0
    port_failed = 0
    terminated = blacklisted_terminated
    threshold = _rental_container_missing_threshold()
    endpoint_threshold = _rental_endpoint_failure_threshold()
    chain_threshold = _rental_chain_failure_threshold()
    port_threshold = _rental_port_failure_threshold()

    for endpoint, rentals in by_endpoint.items():
        try:
            if hasattr(rental_orchestrator, "list_containers_strict"):
                containers = await rental_orchestrator.list_containers_strict(endpoint)
            else:
                containers = await rental_orchestrator.list_containers(endpoint)
        except Exception as e:
            bt.logging.debug(f"rental_watchdog: list_containers {endpoint} failed: {e}")
            containers = None

        for rid, info in rentals:
            checked += 1
            if rid not in _active_rentals:
                _container_missing_streaks.pop(rid, None)
                _endpoint_failure_streaks.pop(rid, None)
                _chain_failure_streaks.pop(rid, None)
                _port_failure_streaks.pop(rid, None)
                continue

            if _rental_ttl_expired(info):
                bt.logging.info(
                    f"rental_watchdog: rental {rid[:8]}… TTL expired; "
                    "terminating without container probation"
                )
                try:
                    result = await _terminate_rental_internal(rid, reason="ttl")
                    if result.get("status") != "release_pending":
                        terminated += 1
                except HTTPException as e:
                    if e.status_code != 404:
                        bt.logging.error(
                            f"rental_watchdog: ttl terminate({rid[:8]}…) failed: {e.detail}"
                        )
                except Exception as e:
                    bt.logging.error(
                        f"rental_watchdog: ttl terminate({rid[:8]}…) failed: {e}"
                    )
                finally:
                    _container_missing_streaks.pop(rid, None)
                    _endpoint_failure_streaks.pop(rid, None)
                    _chain_failure_streaks.pop(rid, None)
                    _port_failure_streaks.pop(rid, None)
                continue

            executor_id = info.get("executor_id", "")
            chain_kind = ""
            if executor_id and _rental_age_seconds(info) >= _rental_chain_grace_seconds():
                chain_kind = _chain_failure_kind(executor_id)
            if _update_chain_failure_streak(rid, chain_kind, chain_threshold):
                direct_kind = await asyncio.to_thread(
                    _direct_chain_failure_kind,
                    executor_id,
                )
                if not direct_kind:
                    _chain_failure_streaks.pop(rid, None)
                    continue
                chain_kind = direct_kind
                # "not_rented" means the chain lock was already released
                # externally. Clean local state, but do not blame the miner.
                if chain_kind == "not_rented":
                    reason = "orphan"
                    bt.logging.warning(
                        f"rental_watchdog: rental {rid[:8]}… chain no longer "
                        "shows rented; cleaning local active state"
                    )
                else:
                    reason = "chain_not_live"
                    _open_container_probation(
                        rid, info, RENTAL_CHAIN_NOT_LIVE_REASON,
                    )
                    bt.logging.error(
                        f"rental_watchdog: active rental {rid[:8]}… chain "
                        f"failure={chain_kind}; terminating rental and entering "
                        "executor probation"
                    )
                try:
                    result = await _terminate_rental_internal(rid, reason=reason)
                    if result.get("status") != "release_pending":
                        terminated += 1
                    if reason == "chain_not_live":
                        chain_failed += 1
                except HTTPException as e:
                    if e.status_code != 404:
                        bt.logging.error(
                            f"rental_watchdog: terminate({rid[:8]}…) failed: {e.detail}"
                        )
                except Exception as e:
                    bt.logging.error(f"rental_watchdog: terminate({rid[:8]}…) failed: {e}")
                finally:
                    _container_missing_streaks.pop(rid, None)
                    _endpoint_failure_streaks.pop(rid, None)
                    _chain_failure_streaks.pop(rid, None)
                    _port_failure_streaks.pop(rid, None)
                continue

            if containers is None:
                if not _update_endpoint_failure_streak(
                    rid, containers, endpoint_threshold,
                ):
                    continue
                endpoint_failed += 1
                _open_container_probation(
                    rid, info, RENTAL_ENDPOINT_UNREACHABLE_REASON,
                )
                bt.logging.error(
                    f"rental_watchdog: active rental {rid[:8]}… endpoint "
                    f"{endpoint!r} unreachable for {endpoint_threshold} checks; "
                    "terminating rental and entering executor probation"
                )
                try:
                    result = await _terminate_rental_internal(rid, reason="endpoint_lost")
                    if result.get("status") != "release_pending":
                        terminated += 1
                except HTTPException as e:
                    if e.status_code != 404:
                        bt.logging.error(
                            f"rental_watchdog: terminate({rid[:8]}…) failed: {e.detail}"
                        )
                except Exception as e:
                    bt.logging.error(f"rental_watchdog: terminate({rid[:8]}…) failed: {e}")
                finally:
                    _container_missing_streaks.pop(rid, None)
                    _endpoint_failure_streaks.pop(rid, None)
                    _chain_failure_streaks.pop(rid, None)
                    _port_failure_streaks.pop(rid, None)
                continue
            _update_endpoint_failure_streak(rid, containers, endpoint_threshold)
            container_name = info.get("container_name", "")
            failure_kind = _container_failure_kind(container_name, containers)
            if not _update_container_missing_streak(
                rid, container_name, containers, threshold,
            ):
                port_reachable = await _active_rental_ssh_port_reachable(info)
                if not _update_port_failure_streak(
                    rid, port_reachable, port_threshold,
                ):
                    continue
                port_failed += 1
                _open_container_probation(
                    rid, info, RENTAL_PORT_UNREACHABLE_REASON,
                )
                host = info.get("ssh_host") or "unknown"
                port = info.get("ssh_port") or "unknown"
                bt.logging.error(
                    f"rental_watchdog: active rental {rid[:8]}… SSH port "
                    f"{host}:{port} unreachable for {port_threshold} checks; "
                    "terminating rental and entering executor probation"
                )
                try:
                    result = await _terminate_rental_internal(
                        rid, reason="ssh_port_unreachable",
                    )
                    if result.get("status") != "release_pending":
                        terminated += 1
                except HTTPException as e:
                    if e.status_code != 404:
                        bt.logging.error(
                            f"rental_watchdog: terminate({rid[:8]}…) failed: {e.detail}"
                        )
                except Exception as e:
                    bt.logging.error(f"rental_watchdog: terminate({rid[:8]}…) failed: {e}")
                finally:
                    _container_missing_streaks.pop(rid, None)
                    _endpoint_failure_streaks.pop(rid, None)
                    _chain_failure_streaks.pop(rid, None)
                    _port_failure_streaks.pop(rid, None)
                continue
            missing += 1
            if failure_kind == "missing":
                _open_container_probation(
                    rid, info, RENTAL_CONTAINER_MISSING_REASON,
                )
            else:
                bt.logging.warning(
                    f"rental_watchdog: rental {rid[:8]}… container "
                    f"{container_name!r} is {failure_kind}; releasing rental "
                    "and entering executor probation"
                )
                _open_container_probation(
                    rid, info, RENTAL_CONTAINER_STOPPED_REASON,
                )
            bt.logging.error(
                f"rental_watchdog: active rental {rid[:8]}… container "
                f"{container_name!r} failure={failure_kind}; terminating "
                "rental and releasing chain lock"
            )
            try:
                reason = (
                    "container_lost"
                    if failure_kind == "missing"
                    else "container_stopped"
                )
                result = await _terminate_rental_internal(rid, reason=reason)
                if result.get("status") != "release_pending":
                    terminated += 1
            except HTTPException as e:
                if e.status_code != 404:
                    bt.logging.error(
                        f"rental_watchdog: terminate({rid[:8]}…) failed: {e.detail}"
                    )
            except Exception as e:
                bt.logging.error(f"rental_watchdog: terminate({rid[:8]}…) failed: {e}")
            finally:
                _container_missing_streaks.pop(rid, None)
                _port_failure_streaks.pop(rid, None)

    return {
        "checked": checked,
        "missing": missing,
        "endpoint_failed": endpoint_failed,
        "chain_failed": chain_failed,
        "port_failed": port_failed,
        "terminated": terminated,
        "blacklisted": blacklisted_terminated,
        "cleared": cleared,
    }


async def rental_container_watchdog(interval_s: float = 30.0) -> None:
    """Keep local active rentals honest against miner Docker state."""
    bt.logging.info(
        f"RentalContainerWatchdog starting (interval={interval_s:.0f}s, "
        f"threshold={_rental_container_missing_threshold()}, "
        f"probation={_rental_container_probation_seconds()}s)"
    )
    while True:
        try:
            await asyncio.sleep(interval_s)
            await rental_container_watchdog_once()
        except asyncio.CancelledError:
            break
        except Exception as e:
            bt.logging.error(f"rental_watchdog loop error: {e}")


async def ttl_sweeper(interval_s: float = 60.0) -> None:
    """Background task: terminate rentals whose TTL has elapsed."""
    while True:
        try:
            now = time.time()
            expired = []
            for rid, info in list(_active_rentals.items()):
                if info.get("release_pending"):
                    continue
                if _rental_ttl_expired(info, now):
                    expired.append(rid)
            for rid in expired:
                bt.logging.info(f"TTL sweeper: rental {rid[:8]}… expired, terminating")
                try:
                    await _terminate_rental_internal(rid, reason="ttl")
                except Exception as e:
                    bt.logging.error(f"TTL sweeper terminate({rid[:8]}) failed: {e}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            bt.logging.error(f"TTL sweeper loop error: {e}")
        await asyncio.sleep(interval_s)


async def _check_internal_rental_access(rental_id: str, request: Request) -> dict:
    """Return the in-memory rental dict after admin auth, else raise 403.

    Public renter authorization is handled by the web app. The validator
    only trusts its admin token/hotkey for rental control operations.
    """
    if rental_id not in _active_rentals:
        raise HTTPException(404, f"Rental {rental_id} not active")
    if not await _is_admin_async(request):
        raise HTTPException(
            403,
            "internal validator auth required",
        )
    return _active_rentals[rental_id]


@router.post("/rentals/{rental_id}/ssh_keys")
async def rental_add_ssh_key(rental_id: str, request: Request):
    """Add an SSH pubkey to a live rental's container.

    Internal/admin only. The web app authorizes the renter via account
    session or recovery token before calling this route.
    """
    rental = await _check_internal_rental_access(rental_id, request)
    body = await request.json()
    pubkey = body.get("ssh_pub_key", "")
    if not pubkey:
        raise HTTPException(400, "ssh_pub_key required")
    if rental_orchestrator is None:
        raise HTTPException(503, "Orchestrator not initialized")
    ok = await rental_orchestrator.add_ssh_key(
        rental["executor_endpoint"], rental["container_name"], pubkey,
    )
    if not ok:
        raise HTTPException(502, "Miner refused to add SSH key")
    return {"status": "ok"}


@router.delete("/rentals/{rental_id}/ssh_keys")
async def rental_remove_ssh_key(rental_id: str, request: Request):
    """Remove an SSH pubkey from a live rental's container.

    Matches by key body (the AAAA… field), so a key with an alternate
    comment still gets removed. Idempotent.
    """
    rental = await _check_internal_rental_access(rental_id, request)
    body = await request.json()
    pubkey = body.get("ssh_pub_key", "")
    if not pubkey:
        raise HTTPException(400, "ssh_pub_key required")
    if rental_orchestrator is None:
        raise HTTPException(503, "Orchestrator not initialized")
    ok = await rental_orchestrator.remove_ssh_key(
        rental["executor_endpoint"], rental["container_name"], pubkey,
    )
    if not ok:
        raise HTTPException(502, "Miner refused to remove SSH key")
    return {"status": "ok"}


@router.get("/executors/{executor_id}/active_containers")
async def list_executor_active_containers(executor_id: str, request: Request):
    """Active rental container names this validator owns for the given
    executor.

    Used by the miner's reap_orphans cross-check. Sensitive because the
    timestamp-suffixed container names reveal rental cadence; making
    this public was a leak (audit C-9). Gated two ways:

      1. Admin auth (X-Admin-Token or NODEXO_ADMIN_HOTKEYS sig) — for
         operators / ops tooling.
      2. Same-miner sig — the miner attests via SR25519 over the path
         using the executor's bound hotkey. The validator already
         tracks the executor→hotkey binding (proofs.py) so we know
         which key SHOULD be signing for this executor.

    Without one of those, refuse.
    """
    # Admin path.
    if await _is_admin_async(request):
        pass
    else:
        # Same-miner path: signature over (HTTP method + path) by the
        # executor's bound hotkey.
        from common.crypto import (
            check_and_record_nonce,
            extract_signature_headers,
            verify_signature,
        )
        try:
            extracted = extract_signature_headers(dict(request.headers))
            if not extracted:
                raise HTTPException(403, "auth required")
            sig, hk, ts, nonce = extracted
            # Body is empty for GET — sign over the path string instead.
            body = f"GET /executors/{executor_id}/active_containers".encode()
            if not verify_signature(body, sig, hk, ts, nonce):
                raise HTTPException(403, "invalid signature")
            if not check_and_record_nonce("executor-active-containers", hk, nonce, body):
                raise HTTPException(403, "replayed signature")
            # Hotkey must match the executor's bound hotkey on this validator.
            from neurons.validator.api.routes.proofs import _executor_hotkey_binding
            bound = _executor_hotkey_binding.get(executor_id, "")
            if not bound or bound != hk:
                raise HTTPException(403, "hotkey not bound to this executor")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(403, "auth failed")

    out = []
    for rid, info in _active_rentals.items():
        if info.get("executor_id") == executor_id:
            cn = info.get("container_name", "")
            if cn:
                out.append(cn)
    return {"executor_id": executor_id, "container_names": out}


@router.get("/sybil/flags")
async def list_sybil_flags(request: Request):
    """Admin-gated view of all open sybil flags + evidence.

    Operator audits this after any newly-flagged executors to decide
    whether to clear (false-positive) or escalate. Renters never see
    this endpoint — banned executors simply don't appear in
    marketplace/executors listings.
    """
    if not await _is_admin_async(request):
        raise HTTPException(403, "admin auth required")
    if not db:
        return {"flags": []}
    from common.db import list_open_sybil_flags
    return {"flags": list_open_sybil_flags(db)}


@router.post("/sybil/scan")
async def trigger_sybil_scan(request: Request):
    """Force an immediate sybil scan (admin only).

    Bypasses the 10-min periodic interval. Useful for ops (when an
    operator wants to confirm a new flag would catch a known bad
    executor) and for the test path. Returns the count of NEW flags
    opened by this scan.
    """
    if not await _is_admin_async(request):
        raise HTTPException(403, "admin auth required")
    if not db or not chain_snapshot:
        raise HTTPException(503, "db or snapshot not initialized")
    from neurons.validator.sybil.scanner import _scan_once
    from common.db import get_active_rentals as _gar, list_open_sybil_flags
    before = len(list_open_sybil_flags(db))
    await _scan_once(
        db=db,
        get_active_rentals_callable=lambda: _gar(db),
        terminate_callable=_terminate_rental_internal,
        chain_snapshot=chain_snapshot,
    )
    after = len(list_open_sybil_flags(db))
    return {"status": "ok", "new_flags": max(0, after - before), "total_open": after}


@router.post("/sybil/flags/{flag_id}/clear")
async def clear_sybil_flag_endpoint(flag_id: int, request: Request):
    """Manually clear a sybil flag (admin only).

    Use after reviewing evidence — e.g., a falsely-detected nvml_digest
    outlier turned out to be a legitimate new driver release. Cleared
    flags stay in the DB for audit but no longer ban the executor.
    """
    if not await _is_admin_async(request):
        raise HTTPException(403, "admin auth required")
    if not db:
        raise HTTPException(503, "no db")
    from common.db import clear_sybil_flag
    cleared_by = request.headers.get("x-admin-user", "admin")
    ok = clear_sybil_flag(db, flag_id, cleared_by=cleared_by)
    if not ok:
        raise HTTPException(404, "flag not found or already cleared")
    return {"status": "cleared", "flag_id": flag_id}


@router.get("/rentals/history")
async def rental_history(request: Request,
                         since_ts: float = 0, limit: int = 100):
    """Historical view of every rental (incl. terminated).

    Internal/admin only. Historical records include SSH host/port and
    billing details; public users access scoped history through the web app.
    """
    if not await _is_admin_async(request):
        raise HTTPException(
            403,
            "internal validator auth required",
        )
    if not db:
        return {"rentals": []}
    from common.db import get_rental_history
    return {
        "rentals": get_rental_history(
            db, since_ts=since_ts, limit=min(limit, 1000),
        )
    }


@router.get("/rentals/history/{rental_id}")
async def rental_history_by_id(rental_id: str, request: Request):
    """Direct historical rental lookup.

    Used by settlement workers so an old rental is never misclassified just
    because it fell outside the newest-N history page.
    """
    if not await _is_admin_async(request):
        raise HTTPException(
            403,
            "internal validator auth required",
        )
    if not db:
        raise HTTPException(503, "no db")
    from common.db import get_rental_by_id
    row = get_rental_by_id(db, rental_id)
    if row is None:
        raise HTTPException(404, "rental not found")
    return {"rental": row}


MAX_OPEN_ENDED_TTL_SECONDS = 7 * 86400   # 7-day operator cap on open-ended rentals


def _parse_duration(duration: str) -> int:
    """Parse duration string to seconds. Special values: '0', 'unlimited',
    'open', '' are treated as metered/open-ended UX, but still receive the
    operator safety cap MAX_OPEN_ENDED_TTL_SECONDS.

    Normal forms: '30s', '15m', '2h', '1d'.
    """
    duration = (duration or "").strip().lower()
    if duration in ("", "0", "unlimited", "open", "none"):
        return MAX_OPEN_ENDED_TTL_SECONDS
    try:
        if duration.endswith("h"):
            return int(float(duration[:-1]) * 3600)
        if duration.endswith("d"):
            return int(float(duration[:-1]) * 86400)
        if duration.endswith("m"):
            return int(float(duration[:-1]) * 60)
        if duration.endswith("s"):
            return int(float(duration[:-1]))
        return int(duration)
    except (ValueError, IndexError):
        return 7200  # Default 2h


def _probe_port(host: str, port: int, timeout: float = 2.5) -> bool:
    """Open a quick TCP socket to (host, port) to confirm reachability.

    Validator uses this just before returning rental info to the renter,
    so we don't hand out an unreachable SSH port (provider firewall not
    open, miner host misconfigured). Failure → markAvailable + 503.
    """
    from neurons.validator.rental.ports import probe_tcp_port
    return probe_tcp_port(host, port, timeout=timeout)
