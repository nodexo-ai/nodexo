"""
Validator-side intake for monitor reports.

The monitor container (neurons/monitor) POSTs signed reports here at
regular intervals. We:
  - Verify the signature (sr25519 against the per-monitor pubkey the
    monitor self-generated and published in its first report).
  - Trust-on-first-use: the first report from a (executor_id) pair
    establishes the monitor's pubkey. Subsequent reports must match
    unless the replacement monitor key is authorized by the executor
    owner's miner hotkey.
  - Persist the report + accumulate per-monitor state used by the
    sybil scanner.

Phase 1 storage: in-memory accumulator. The sybil scanner reads from
this in-process state. Phase 2 will add a MonitorReport DB table for
audit + cross-validator correlation.

What we ACT on:
  - extra_proofs_seen > 0 over multiple reports → open a soft
    sybil_flag(reason="behavioral_extra_proof") for operator review.
  - rental_containers_visible cross-correlated across executors: if
    monitor for executor A reports the same rental container name
    that monitor for executor B reports, that's a double-rent
    sybil signal.
  - Monitor silent for too long → open
    sybil_flag(reason="monitor_silent"). Soft signal (could be a
    legit network blip); operator review unblocks.
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import bittensor as bt
from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(tags=["monitor"])

# Set by validator daemon at startup
db_instance = None  # common.db.Database


# In-memory monitor state: per-executor most-recent report + first-seen
# pubkey + extra-proof streak counter.
#   - _reports: ephemeral (last report cached for fast read)
#   - _pubkeys: DB-backed (audit C-7) — restart wipe used to drop TOFU
#               bindings, enabling silent monitor pubkey rotation
#   - _extra_streak: ephemeral (streak resets on restart, fine —
#                    fresh observations re-establish the streak)
_lock = threading.Lock()
_reports: dict[str, dict] = {}            # executor_id → latest report dict
_pubkeys: dict[str, str] = {}             # DB-backed cache
_extra_streak: dict[str, int] = {}
_extra_last_cycle: dict[str, int] = {}
# Consecutive clean-cycle counter per executor. Used to auto-clear
# behavioral_extra_proof flags once the monitor classifier settles —
# the signal is noisy on the edge of the jitter window and recovers
# on its own when chain context refreshes, so we don't want a single
# bad streak to permanently brand an honest executor.
_clean_streak: dict[str, int] = {}


def load_monitor_pubkeys_from_db() -> None:
    """Hydrate _pubkeys at validator startup. Without this, every
    restart re-opens a TOFU window for every executor.
    """
    global _pubkeys
    if db_instance is None:
        return
    try:
        from common.db import get_all_monitor_pubkeys
        rows = get_all_monitor_pubkeys(db_instance)
        with _lock:
            _pubkeys.update(rows)
        bt.logging.info(f"Loaded {len(rows)} monitor TOFU bindings from DB")
    except Exception as e:
        bt.logging.warning(f"load_monitor_pubkeys_from_db: {e}")


def _persist_monitor_pubkey(executor_id: str, monitor_ss58: str) -> None:
    if db_instance is None:
        return
    try:
        from common.db import set_monitor_pubkey
        set_monitor_pubkey(db_instance, executor_id, monitor_ss58)
    except Exception as e:
        bt.logging.warning(f"_persist_monitor_pubkey: {e}")

# Reports with this many consecutive "extra proofs" events flag the
# executor. The monitor classifier is sporadically noisy on cycles
# where chain context drifts at the jitter-window edge (we've seen
# bursts of extra=1-11 followed by clean cycles); a small threshold
# generated false positives on honest miners. Five consecutive cycles
# of extras (~15 minutes at 180s/cycle) is enough to rule out
# classifier noise and still well below an attacker's profitable
# horizon.
EXTRA_PROOF_STREAK_THRESHOLD = 5
# Clean-cycle threshold for auto-clearing an open extra_proof flag.
# Once the streak breaks, this many consecutive clean cycles
# (~6 min) marks the signal as transient noise rather than active
# sybil activity, and we drop the flag.
EXTRA_PROOF_CLEAR_AFTER_CLEAN = 2


def get_latest_report(executor_id: str) -> Optional[dict]:
    """Read-only access for the sybil scanner."""
    with _lock:
        return _reports.get(executor_id)


def all_reports() -> dict[str, dict]:
    """Snapshot of all latest reports — sybil scanner uses this for
    cross-executor correlation (rental_containers_visible overlap)."""
    with _lock:
        return dict(_reports)


def _verify_monitor_owner_binding(eid: str, monitor_hotkey: str, headers: dict) -> None:
    """Authorize the first monitor key for an executor.

    The monitor's own key proves report integrity after binding. The initial
    bind also needs a miner-hotkey signature so an unrelated caller cannot
    claim someone else's executor_id before the real sidecar reports.
    """
    from common.crypto import monitor_binding_body, verify_signature
    from neurons.validator.api.routes.proofs import _executor_hotkey_binding

    owner_hotkey = headers.get("x-monitor-owner-hotkey", "")
    owner_sig = headers.get("x-monitor-owner-signature", "")
    owner_ts = headers.get("x-monitor-owner-timestamp", "")
    owner_nonce = headers.get("x-monitor-owner-nonce", "")
    if not (owner_hotkey and owner_sig and owner_ts and owner_nonce):
        raise HTTPException(403, "monitor owner binding proof required")

    bound_hotkey = _executor_hotkey_binding.get(eid, "")
    if not bound_hotkey:
        raise HTTPException(403, "executor hotkey binding not established")
    if owner_hotkey != bound_hotkey:
        raise HTTPException(403, "monitor owner hotkey does not own executor")

    body = monitor_binding_body(eid, monitor_hotkey)
    # This is a scoped delegation from miner hotkey to monitor key, not an
    # HTTP request authorization. It is reusable until the monitor key rotates.
    if not verify_signature(
        body, owner_sig, owner_hotkey, owner_ts, owner_nonce, max_age=None,
    ):
        raise HTTPException(403, "invalid monitor owner binding proof")


@router.post("/monitor/report")
async def post_monitor_report(request: Request):
    """Ingest one signed monitor report.

    Verification:
      - Headers X-Monitor-{Signature,Hotkey,Timestamp,Nonce} present.
      - Signature verifies against the body bytes.
      - First report from an executor → record (executor_id → ss58)
        binding. Subsequent reports must match the bound ss58, or include
        a miner-owner monitor binding proof to rotate the monitor key.
        Unapproved rotation is rejected with 403 and raises a sybil flag.
    """
    body_bytes = await request.body()
    sig = request.headers.get("x-monitor-signature", "")
    hk = request.headers.get("x-monitor-hotkey", "")
    ts = request.headers.get("x-monitor-timestamp", "")
    nonce = request.headers.get("x-monitor-nonce", "")
    if not (sig and hk and ts and nonce):
        raise HTTPException(401, "missing monitor signature headers")

    # Reuse the common verifier (same digest scheme as the
    # proof/signing path).
    try:
        from common.crypto import check_and_record_nonce, verify_signature
        if not verify_signature(body_bytes, sig, hk, ts, nonce):
            raise HTTPException(401, "invalid monitor signature")
        if not check_and_record_nonce("monitor", hk, nonce, body_bytes):
            raise HTTPException(401, "replayed monitor report")
    except HTTPException:
        raise
    except Exception as e:
        bt.logging.warning(f"monitor sig verify error: {e}")
        raise HTTPException(401, "signature verify failed")

    import json as _json
    try:
        payload = _json.loads(body_bytes)
    except Exception:
        raise HTTPException(400, "invalid json")
    eid = payload.get("executor_id", "")
    if not eid:
        raise HTTPException(400, "executor_id required")
    if payload.get("monitor_pubkey") and payload.get("monitor_pubkey") != hk:
        raise HTTPException(400, "monitor_pubkey does not match signing key")
    health = payload.get("monitor_health") or {}
    if not isinstance(health, dict):
        health = {}
    nvml_error = str(payload.get("nvml_error") or health.get("nvml_error") or "")
    monitor_unhealthy = bool(nvml_error) or health.get("nvml_ok") is False

    with _lock:
        existing_binding = _pubkeys.get(eid)
    if existing_binding is None:
        _verify_monitor_owner_binding(eid, hk, dict(request.headers))
        rotation_authorized = False
    elif existing_binding != hk:
        try:
            _verify_monitor_owner_binding(eid, hk, dict(request.headers))
            rotation_authorized = True
        except HTTPException:
            rotation_authorized = False
    else:
        rotation_authorized = False

    with _lock:
        bound = _pubkeys.get(eid)
        if bound is None:
            _pubkeys[eid] = hk
            _persist_monitor_pubkey(eid, hk)
            bt.logging.info(
                f"monitor: TOFU pubkey bound for executor={eid[:16]} ss58={hk[:12]}…"
            )
        elif bound != hk:
            if rotation_authorized:
                _pubkeys[eid] = hk
                _persist_monitor_pubkey(eid, hk)
                if db_instance is not None:
                    try:
                        from common.db import clear_open_sybil_flag
                        clear_open_sybil_flag(
                            db_instance,
                            eid,
                            "monitor_pubkey_rotation",
                            cleared_by="auto-authorized-monitor-rotation",
                        )
                    except Exception as e:
                        bt.logging.debug(f"monitor: rotation flag auto-clear failed: {e}")
                bt.logging.info(
                    f"monitor: authorized pubkey rotation for executor={eid[:16]} "
                    f"old={bound[:12]}… new={hk[:12]}…"
                )
            else:
                bt.logging.warning(
                    f"monitor: pubkey mismatch for executor={eid[:16]} "
                    f"(expected {bound[:12]}…, got {hk[:12]}…). "
                    f"Likely monitor replacement."
                )
                # Open a sybil flag on this executor.
                if db_instance is not None:
                    try:
                        from common.db import open_sybil_flag
                        open_sybil_flag(db_instance, eid, "monitor_pubkey_rotation", {
                            "expected_ss58": bound,
                            "got_ss58": hk,
                        })
                    except Exception as e:
                        bt.logging.warning(f"monitor: open_sybil_flag failed: {e}")
                raise HTTPException(403, "monitor pubkey mismatch")
        _reports[eid] = {
            "payload": payload,
            "received_ts": time.time(),
            "monitor_hotkey": hk,
        }

        # Recovery: a report means the monitor is not silent. A healthy
        # report also clears monitor_unhealthy. A blind/NVML-broken monitor
        # remains a hard blocker, but it must not be confused with proof
        # suppression; otherwise an honest operator with a broken sidecar gets
        # permanent fraud-looking evidence.
        if db_instance is not None:
            try:
                from common.db import (
                    SybilFlag, clear_open_sybil_flag, open_sybil_flag,
                )
                from datetime import datetime as _dt
                with db_instance.session() as s:
                    for row in s.query(SybilFlag).filter(
                        SybilFlag.executor_id == eid,
                        SybilFlag.reason == "monitor_silent",
                        SybilFlag.cleared_at == None,  # noqa: E711
                    ).all():
                        row.cleared_at = _dt.utcnow()
                        row.cleared_by = "auto-monitor-recovery"
                if monitor_unhealthy:
                    open_sybil_flag(db_instance, eid, "monitor_unhealthy", {
                        "cycle_id": int(payload.get("cycle_id") or 0),
                        "report_seq": int(payload.get("report_seq") or 0),
                        "nvml_error": nvml_error[:300],
                        "monitor_health": health,
                    })
                else:
                    clear_open_sybil_flag(
                        db_instance, eid, "monitor_unhealthy",
                        cleared_by="auto-monitor-health-recovered",
                    )
            except Exception as e:
                bt.logging.debug(f"monitor: health-flag update failed: {e}")

        try:
            cycle_id = int(payload.get("cycle_id") or 0)
        except Exception:
            cycle_id = 0

        # Persist per-cycle observation for the proof-suppression
        # cross-correlation signal. Skip if both fields are None
        # (telemetry-only mode — nothing to compare against later) or
        # cycle_id=0 (no usable chain context).
        if db_instance is not None and cycle_id > 0:
            try:
                from common.db import record_monitor_cycle_observation
                record_monitor_cycle_observation(
                    db_instance,
                    eid,
                    cycle_id,
                    None if monitor_unhealthy else payload.get("expected_proof_seen"),
                    None if monitor_unhealthy else payload.get("extra_proofs_seen"),
                )
            except Exception as e:
                bt.logging.debug(f"monitor: record_observation: {e}")

        # Streak tracking for extra-proof anomaly. None = telemetry-only
        # mode (monitor running without chain wiring); treat that as
        # "no signal" rather than an integer comparison (which crashes
        # when the value is JSON null). Reset the streak too — a
        # transition to telemetry-only is not a violation, just an
        # absence of measurement.
        extra = payload.get("extra_proofs_seen")
        # cycle_id=0 means the monitor had no usable chain context. Treat it
        # as telemetry-only; otherwise a validator restart can leave the
        # monitor using a stale proof window and falsely count honest future
        # proof PIDs as extra.
        extra_measured = (not monitor_unhealthy) and extra is not None and cycle_id > 0
        if extra_measured and int(extra) > 0:
            # Count distinct proof cycles, not repeated 30s reports from the
            # same cycle. One observed extra PID must not become five strikes
            # just because the report interval is shorter than the proof epoch.
            if _extra_last_cycle.get(eid) != cycle_id:
                _extra_streak[eid] = _extra_streak.get(eid, 0) + 1
                _extra_last_cycle[eid] = cycle_id
                _clean_streak[eid] = 0
            if _extra_streak.get(eid, 0) >= EXTRA_PROOF_STREAK_THRESHOLD:
                if db_instance is not None:
                    try:
                        from common.db import (
                            open_sybil_flag, has_open_sybil_flag,
                        )
                        was_open = has_open_sybil_flag(
                            db_instance, eid, "behavioral_extra_proof",
                        )
                        open_sybil_flag(
                            db_instance, eid, "behavioral_extra_proof",
                            {
                                "streak": _extra_streak[eid],
                                "cycle_id": cycle_id,
                                "last_extra_count":
                                    int(payload.get("extra_proofs_seen", 0)),
                            },
                        )
                        if not was_open:
                            bt.logging.warning(
                                f"monitor: extra_proof streak={_extra_streak[eid]} "
                                f"executor={eid[:16]} — opening sybil flag"
                            )
                        else:
                            bt.logging.debug(
                                f"monitor: extra_proof streak={_extra_streak[eid]} "
                                f"executor={eid[:16]} (flag already open)"
                            )
                    except Exception as e:
                        bt.logging.warning(f"monitor: open_sybil_flag: {e}")
        else:
            # Streak broke. If a flag was open, count clean cycles
            # towards auto-clear (signal is self-healing — classifier
            # noise lasts a few cycles, real sybil activity persists).
            if not extra_measured:
                _extra_last_cycle.pop(eid, None)
                return {"status": "ok"}
            if _extra_last_cycle.get(eid) == cycle_id:
                return {"status": "ok"}
            _extra_last_cycle[eid] = cycle_id
            if _extra_streak.get(eid, 0) > 0:
                _extra_streak[eid] = 0
            _clean_streak[eid] = _clean_streak.get(eid, 0) + 1
            if _clean_streak[eid] >= EXTRA_PROOF_CLEAR_AFTER_CLEAN and db_instance is not None:
                try:
                    from common.db import clear_open_sybil_flag
                    cleared = clear_open_sybil_flag(
                        db_instance, eid, "behavioral_extra_proof",
                        cleared_by="auto-extra-proof-recovery",
                    )
                    if cleared:
                        bt.logging.info(
                            f"monitor: behavioral_extra_proof cleared "
                            f"executor={eid[:16]} after "
                            f"{_clean_streak[eid]} clean cycles"
                        )
                except Exception as e:
                    bt.logging.debug(
                        f"monitor: extra_proof auto-clear failed: {e}"
                    )

    return {"status": "ok"}


_chain_ctx_cache: dict = {"ts": 0.0, "epoch_interval": 0, "data": None}
_chain_ctx_lock = threading.Lock()
_chain_ctx_tasks: dict[int, asyncio.Task] = {}
CHAIN_CTX_TTL_S = 1.0  # beacon-cache reads are cheap; keep miner proof starts
                       # close to the beacon boundary.
_chain_ctx_hits: dict[str, tuple[int, int]] = {}
_chain_ctx_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="nodexo-chain-context",
)
_chain_ctx_worker_lock = threading.Lock()
_chain_ctx_worker_future = None


def _chain_context_rpc_timeout_s() -> float:
    import os as _os

    try:
        return max(0.5, float(_os.environ.get("CHAIN_CONTEXT_RPC_TIMEOUT_S", "4")))
    except Exception:
        return 4.0


async def _to_thread_timeout(fn, *args, timeout_s: float):
    global _chain_ctx_worker_future

    loop = asyncio.get_running_loop()
    with _chain_ctx_worker_lock:
        if _chain_ctx_worker_future is not None and not _chain_ctx_worker_future.done():
            raise asyncio.TimeoutError()
        future = loop.run_in_executor(_chain_ctx_executor, fn, *args)
        _chain_ctx_worker_future = future

        def _clear_done(done, *, expected=future):
            global _chain_ctx_worker_future
            with _chain_ctx_worker_lock:
                if _chain_ctx_worker_future is expected:
                    _chain_ctx_worker_future = None

        future.add_done_callback(_clear_done)

    return await asyncio.wait_for(asyncio.shield(future), timeout=timeout_s)


def _fresh_chain_context_beacon(
    network: str,
    netuid: int,
    epoch_start_block: int,
    block_number: int,
    max_offset: int,
) -> tuple[str, int, int]:
    """Resolve the monitor beacon using an isolated chain client.

    Do not use the validator-wide `app.state.rpc` object here. Testnet
    websocket calls can hang while holding that object's internal lock; if this
    public monitor helper shares the lock, a stuck `/chain/context` request can
    make proof/heartbeat ownership checks miss the miner broadcast deadline.
    """
    from common.chain.rpc import SubtensorRPC
    from common.proof_schedule import derive_beacon_offset_blocks

    fresh = SubtensorRPC(network=network, netuid=netuid)
    try:
        anchor = fresh.get_block_hash(epoch_start_block)
        beacon_offset_blocks = derive_beacon_offset_blocks(anchor, max_offset)
        beacon_block = epoch_start_block + beacon_offset_blocks
        if beacon_block > block_number:
            raise RuntimeError(
                f"beacon block {beacon_block} not reached yet "
                f"(current={block_number})"
            )
        beacon = anchor if beacon_offset_blocks == 0 else fresh.get_block_hash(beacon_block)
    finally:
        closer = getattr(fresh, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
    if isinstance(beacon, bytes):
        beacon_hex = beacon.hex()
    elif isinstance(beacon, str):
        beacon_hex = beacon.removeprefix("0x")
    else:
        beacon_hex = ""
    return beacon_hex, beacon_block, beacon_offset_blocks


def _chain_context_rate_limit(request: Request) -> None:
    """Small per-IP guard for the public monitor helper endpoint.

    This endpoint cannot be admin-gated because miner-side monitor containers
    need it before posting their signed reports. It is still bounded so random
    internet clients cannot bypass the cache with a query storm.
    """
    import os as _os
    import time as _time

    try:
        max_per_min = int(_os.environ.get("CHAIN_CONTEXT_RATE_LIMIT_PER_MIN", "2400"))
    except Exception:
        max_per_min = 2400
    if max_per_min <= 0:
        return
    ip = (request.client.host if request.client else "") or "unknown"
    minute = int(_time.time() // 60)
    cur_minute, count = _chain_ctx_hits.get(ip, (minute, 0))
    if cur_minute != minute:
        cur_minute, count = minute, 0
    count += 1
    _chain_ctx_hits[ip] = (cur_minute, count)
    if len(_chain_ctx_hits) > 10000:
        for key, (seen_minute, _count) in list(_chain_ctx_hits.items()):
            if seen_minute < minute:
                _chain_ctx_hits.pop(key, None)
    if count > max_per_min:
        raise HTTPException(429, "chain context rate limited")


def _chain_context_cache_hit(request: Request, epoch_interval: int, now: float) -> dict | None:
    with _chain_ctx_lock:
        if (
            _chain_ctx_cache["data"] is not None
            and _chain_ctx_cache["epoch_interval"] == epoch_interval
            and now - _chain_ctx_cache["ts"] < CHAIN_CTX_TTL_S
        ):
            data = _chain_ctx_cache["data"]
        else:
            data = None
    if data is None:
        return None
    try:
        cache = getattr(getattr(request.app, "state", None), "beacon_cache", None)
        if cache is not None:
            head = cache.head(max_age_s=_chain_context_cache_max_age_s())
            if head and int(head) // int(epoch_interval) != int(data.get("cycle_id") or -1):
                return None
    except Exception:
        return None
    out = dict(data)
    out["context_at_ts"] = float(now)
    try:
        beacon_available_at = float(out.get("beacon_available_at_ts") or 0.0)
        if beacon_available_at > 0:
            out["beacon_age_s"] = max(0.0, float(now) - beacon_available_at)
    except Exception:
        pass
    return out


def _chain_context_cache_max_age_s() -> float:
    import os as _os

    try:
        return max(1.0, float(_os.environ.get("CHAIN_CONTEXT_CACHE_MAX_AGE_S", "30")))
    except Exception:
        return 30.0


def _chain_context_rpc_fallback_enabled() -> bool:
    import os as _os

    return _os.environ.get("NODEXO_CHAIN_CONTEXT_RPC_FALLBACK", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _cache_chain_context(epoch_interval: int, data: dict) -> dict:
    with _chain_ctx_lock:
        _chain_ctx_cache["ts"] = time.time()
        _chain_ctx_cache["epoch_interval"] = epoch_interval
        _chain_ctx_cache["data"] = data
    return data


def _chain_context_from_beacon_cache(request: Request, epoch_interval: int) -> dict | None:
    import os as _os

    cache = getattr(getattr(request.app, "state", None), "beacon_cache", None)
    if cache is None:
        return None

    try:
        head_state = getattr(cache, "head_state", None)
        if callable(head_state):
            block_number, block_number_at_ts = head_state(
                max_age_s=_chain_context_cache_max_age_s()
            )
        else:
            block_number = cache.head(max_age_s=_chain_context_cache_max_age_s())
            block_number_at_ts = time.time() if block_number else 0.0
    except Exception:
        return None
    if not block_number:
        return None

    cycle_id = int(block_number) // int(epoch_interval)
    try:
        from common.proof_schedule import DEFAULT_BEACON_MAX_OFFSET_BLOCKS
        max_offset = int(_os.environ.get(
            "PROOF_BEACON_MAX_OFFSET_BLOCKS",
            str(DEFAULT_BEACON_MAX_OFFSET_BLOCKS),
        ))
        ctx = cache.context_for_epoch(
            cycle_id,
            epoch_interval,
            max_offset,
            int(block_number),
        )
    except Exception:
        return None
    if not ctx:
        return None

    _head, beacon, beacon_available_at, beacon_block = ctx
    if isinstance(beacon, bytes):
        beacon_hex = beacon.hex()
    elif isinstance(beacon, str):
        beacon_hex = beacon.removeprefix("0x")
    else:
        return None

    epoch_start_block = cycle_id * int(epoch_interval)
    now = time.time()
    return {
        "block_number": int(block_number),
        "block_number_at_ts": float(block_number_at_ts or time.time()),
        "context_at_ts": float(now),
        "epoch_interval": int(epoch_interval),
        "cycle_id": int(cycle_id),
        "epoch_start_block": int(epoch_start_block),
        "beacon_block": int(beacon_block),
        "beacon_offset_blocks": int(beacon_block) - int(epoch_start_block),
        "beacon_hex": beacon_hex,
        "beacon_available_at_ts": float(beacon_available_at or now),
        "beacon_age_s": max(0.0, float(now) - float(beacon_available_at or now)),
    }


async def _refresh_chain_context(request: Request, epoch_interval: int) -> dict:
    import os as _os
    import time as _time

    cached_context = _chain_context_from_beacon_cache(request, epoch_interval)
    if cached_context is not None:
        return _cache_chain_context(epoch_interval, cached_context)
    if not _chain_context_rpc_fallback_enabled():
        raise HTTPException(503, "beacon cache unavailable")

    try:
        from neurons.validator.api.routes import rent as rent_route
        registry = rent_route.registry_client
    except Exception:
        raise HTTPException(503, "chain not initialized")
    if registry is None:
        raise HTTPException(503, "chain not initialized")

    timeout_s = _chain_context_rpc_timeout_s()
    try:
        block_number = await _to_thread_timeout(
            registry.block_number,
            timeout_s=timeout_s,
        )
    except asyncio.TimeoutError:
        raise HTTPException(503, "block number unavailable: timeout")
    except Exception as e:
        raise HTTPException(503, f"block number unavailable: {e}")

    now = _time.time()
    cycle_id = block_number // epoch_interval
    epoch_start_block = cycle_id * epoch_interval
    beacon_hex = ""
    beacon_block = epoch_start_block
    beacon_offset_blocks = 0
    try:
        rpc = getattr(request.app.state, "rpc", None)
        if rpc is not None:
            from common.proof_schedule import DEFAULT_BEACON_MAX_OFFSET_BLOCKS
            max_offset = int(_os.environ.get(
                "PROOF_BEACON_MAX_OFFSET_BLOCKS",
                str(DEFAULT_BEACON_MAX_OFFSET_BLOCKS),
            ))
            network = str(
                getattr(rpc, "network", "")
                or _os.environ.get("SUBTENSOR_NETWORK", "finney")
            )
            netuid = int(
                getattr(rpc, "netuid", 0)
                or _os.environ.get("NETUID", "0")
                or 0
            )
            beacon_hex, beacon_block, beacon_offset_blocks = await _to_thread_timeout(
                _fresh_chain_context_beacon,
                network,
                netuid,
                epoch_start_block,
                block_number,
                max_offset,
                timeout_s=timeout_s,
            )
    except Exception as e:
        bt.logging.debug(f"chain_context beacon unavailable: {e}")

    if not beacon_hex:
        # Do not cache or serve half-context. The monitor treats HTTP 503 as
        # "no behavioral signal for this tick" and clears its proof window.
        raise HTTPException(503, "beacon unavailable")

    data = {
        "block_number": block_number,
        # Wall-clock time when block_number was observed by us. The
        # monitor uses this (NOT its own time.time()) to compute the
        # cycle's epoch_start_ts — otherwise the chain-ctx cache's
        # staleness propagates into a wrong jitter-window calculation
        # and the legitimate proof gets classified as `extra`.
        "block_number_at_ts": now,
        "context_at_ts": now,
        "epoch_interval": epoch_interval,
        "cycle_id": cycle_id,
        "epoch_start_block": epoch_start_block,
        "beacon_block": beacon_block,
        "beacon_offset_blocks": beacon_offset_blocks,
        "beacon_hex": beacon_hex,
        # Fallback path has no rolling-cache first-seen record. Use the time
        # this validator resolved the beacon hash so miners align with the
        # validator-owned availability anchor.
        "beacon_available_at_ts": now,
        "beacon_age_s": 0.0,
    }
    return _cache_chain_context(epoch_interval, data)


def _get_or_create_chain_context_task(request: Request, epoch_interval: int) -> asyncio.Task:
    loop = asyncio.get_running_loop()
    with _chain_ctx_lock:
        task = _chain_ctx_tasks.get(epoch_interval)
        if task is None or task.done():
            task = loop.create_task(_refresh_chain_context(request, epoch_interval))
            _chain_ctx_tasks[epoch_interval] = task

            def _clear_done(done: asyncio.Task, *, key=epoch_interval):
                with _chain_ctx_lock:
                    if _chain_ctx_tasks.get(key) is done:
                        _chain_ctx_tasks.pop(key, None)

            task.add_done_callback(_clear_done)
        return task


@router.get("/chain/context")
async def chain_context(
    request: Request,
    epoch_interval: int = Query(15, ge=1, le=360),
):
    """Current chain context. Monitors poll this every NVML tick (~5Hz)
    to compute the expected micro-proof jitter window.

    Cached for CHAIN_CTX_TTL_S seconds so multi-tick monitors don't
    hammer the upstream chain RPC. Without this cache testnet's
    public endpoint rate-limits us into HTTP 429s within minutes
    (observed in production today — see commit history).
    """
    import time as _time

    _chain_context_rate_limit(request)
    now = _time.time()
    cached = _chain_context_cache_hit(request, epoch_interval, now)
    if cached is not None:
        return cached

    task = _get_or_create_chain_context_task(request, epoch_interval)
    return await asyncio.shield(task)


@router.get("/monitor/reports")
async def list_monitor_reports(request: Request):
    """Admin-gated view of all latest monitor reports (one per
    executor). Used for ops/debug; not exposed to renters.
    """
    from neurons.validator.api.routes.rent import _is_admin_async
    if not await _is_admin_async(request):
        raise HTTPException(403, "admin auth required")
    with _lock:
        return {"reports": {
            eid: {
                "received_ts": v["received_ts"],
                "monitor_hotkey": v["monitor_hotkey"],
                "payload": v["payload"],
            }
            for eid, v in _reports.items()
        }}
