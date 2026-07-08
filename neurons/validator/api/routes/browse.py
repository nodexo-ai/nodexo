"""
Browse API — list available instances for rental.

Two data sources:
  /instances   — from validator's local DB (fast, includes hardware + live metrics)
  /executors   — from ComputeRegistry on-chain (authoritative but slower)
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)
router = APIRouter(tags=["browse"])

# Set by validator daemon at startup
registry_client = None
scoring_data = None  # dict[executor_id, ExecutorScore]
db_instance = None   # common.db.Database
chain_snapshot = None  # neurons.validator.state.chain_snapshot.ChainSnapshot
endpoint_health = None  # neurons.validator.health.prober.EndpointHealthProber


async def _require_internal_api(request: Request) -> None:
    """Require validator admin auth for control-plane browse data.

    Public renters should consume the web app's `/api/*` endpoints. The
    validator may expose miner/proof ingress to the network, but raw browse
    data includes endpoints, hotkeys, live metrics, and operational state.
    """
    from neurons.validator.api.routes.rent import _is_admin_async
    if not await _is_admin_async(request):
        raise HTTPException(403, "internal validator auth required")


# Hash-prefix → readable GPU name. Built once at import. New miners register
# the canonical model hash; older miners may have registered raw aliases.
from common.config import (
    GPU_VRAM_GB,
    canonical_gpu_model_name,
    gpu_model_hash_lookup,
    gpu_model_name_for_hash,
)
from common.proof_timing import is_timing_model_calibrated

_GPU_HASH_TO_NAME: dict[str, str] = gpu_model_hash_lookup()


def _name_for_hash(hash_hex: str) -> str:
    return gpu_model_name_for_hash(hash_hex)


def _resolve_gpu_model_name(gpu_model: str) -> str:
    resolved = canonical_gpu_model_name(gpu_model)
    return resolved if resolved in GPU_VRAM_GB else gpu_model.strip()


@router.get("/pricing/quote")
async def pricing_quote(
    request: Request,
    gpu_model: str = Query(..., min_length=1, max_length=96),
    gpu_count: int = Query(1, ge=1, le=8),
):
    """Quote the validator-authoritative renter price for a GPU shape.

    `/marketplace/tiers` intentionally only lists currently rentable
    inventory. Active-rental top-ups still need a price when the only GPU
    of that tier is already rented, so expose the same scorer price table
    without implying availability.
    """
    await _require_internal_api(request)
    from neurons.validator.scoring.scorer import lookup_usdc_price

    resolved = _resolve_gpu_model_name(gpu_model)
    return {
        "gpu_model": resolved,
        "gpu_count": gpu_count,
        "price_usdc_per_hour": lookup_usdc_price(resolved, gpu_count),
    }


def _canary_passed_executor_ids(executor_ids: list[str]) -> set[str]:
    """Executor ids that passed the production rental canary gate.

    `/marketplace/tiers` must advertise the same inventory that
    `POST /rent` can actually select. Listing chain-available but
    canary-pending executors causes renters to pay before the validator
    rejects provisioning.
    """
    if db_instance is None or not executor_ids:
        return set()
    try:
        from common.db import CanaryRecord
        with db_instance.session() as s:
            rows = s.query(CanaryRecord.executor_id).filter(
                CanaryRecord.executor_id.in_(executor_ids),
                CanaryRecord.last_status == "pass",
            ).all()
            return {row[0] for row in rows}
    except Exception as e:
        logger.warning("marketplace canary gate query failed: %s", e)
        return set()


def _healthy_executor_ids(executor_ids: list[str]) -> set[str]:
    """Executors currently healthy enough for public inventory/rent.

    This is deliberately independent from the incentive score. Score updates
    are periodic and used for ranking/weights; liveness must react directly to
    miner heartbeats and verifier status.
    """
    if not executor_ids or db_instance is None:
        return set()
    import calendar
    import os as _os
    import time as _time
    from datetime import datetime, timedelta
    from common.db import (
        Executor,
        ExecutorStats,
        ProofResult,
        RegistryExecutorState,
        validator_outage_overlap_seconds,
    )
    from common.subnet_runtime_config import get_subnet_runtime_config

    runtime_cfg = get_subnet_runtime_config().scoring
    try:
        heartbeat_recency_s = float(
            _os.environ.get(
                "LIVE_HEALTH_HEARTBEAT_RECENCY_S",
                str(runtime_cfg.heartbeat_recency_s),
            )
        )
    except Exception:
        heartbeat_recency_s = float(runtime_cfg.heartbeat_recency_s)
    bad_statuses = {"fail", "stale", "offline"}

    def _dt_to_unix(dt) -> float:
        if not dt:
            return 0.0
        if getattr(dt, "tzinfo", None) is not None:
            return float(dt.timestamp())
        return float(calendar.timegm(dt.timetuple()))

    now = _time.time()
    out: set[str] = set()
    recent_validator_skip_ids: set[str] = set()
    with db_instance.session() as s:
        rows = {
            row.executor_id: row
            for row in s.query(Executor).filter(Executor.executor_id.in_(executor_ids)).all()
        }
        stats = {
            row.executor_id: row
            for row in s.query(ExecutorStats).filter(ExecutorStats.executor_id.in_(executor_ids)).all()
        }
        registry_rows = {
            row.executor_id: row
            for row in s.query(RegistryExecutorState).filter(
                RegistryExecutorState.executor_id.in_(executor_ids),
            ).all()
        }
        default_proof_recency_s = runtime_cfg.proof_recency_s
        proof_recency_s = float(_os.environ.get(
            "PROOF_RECENCY_SECONDS", str(default_proof_recency_s),
        ))
        cutoff = datetime.utcnow() - timedelta(seconds=proof_recency_s)
        recent_validator_skip_ids = {
            row[0]
            for row in s.query(ProofResult.executor_id).filter(
                ProofResult.executor_id.in_(executor_ids),
                ProofResult.valid == True,  # noqa: E712
                ProofResult.tier == "skipped",
                ProofResult.verified_at >= cutoff,
            ).all()
        }

    for eid, row in rows.items():
        registry_row = registry_rows.get(eid)
        gpu_name = _name_for_hash(getattr(registry_row, "gpu_model_hash", "") or "")
        gpu_count = max(1, int(getattr(registry_row, "gpu_count", 1) or 1))
        if not is_timing_model_calibrated(gpu_name, gpu_count):
            continue
        if endpoint_health is not None:
            health_state = endpoint_health.state(eid)
            if health_state is None or health_state.status != "healthy":
                continue
        last_seen = _dt_to_unix(row.last_seen)
        outage_s = 0.0
        if last_seen:
            try:
                outage_s = validator_outage_overlap_seconds(db_instance, last_seen, now)
            except Exception:
                outage_s = 0.0
        if not last_seen or max(0.0, now - last_seen - outage_s) > heartbeat_recency_s:
            continue
        status = (getattr(stats.get(eid), "current_status", "") or "").lower()
        if status in bad_statuses:
            if status == "stale" and eid in recent_validator_skip_ids:
                out.add(eid)
                continue
            continue
        out.add(eid)
    return out


def _positive_score_executor_ids(executor_ids: list[str]) -> set[str]:
    """Executor ids with a positive validator score for public inventory."""
    if not executor_ids or not scoring_data:
        return set()
    out: set[str] = set()
    for executor_id in executor_ids:
        score = scoring_data.get(executor_id)
        if score is not None and getattr(score, "score", 0.0) > 0:
            out.add(executor_id)
    return out


def _executor_hotkeys(executor_ids: list[str]) -> dict[str, str]:
    if db_instance is None or not executor_ids:
        return {}
    try:
        from common.db import Executor
        with db_instance.session() as s:
            rows = s.query(Executor.executor_id, Executor.hotkey_ss58).filter(
                Executor.executor_id.in_(executor_ids),
            ).all()
            return {row[0]: row[1] or "" for row in rows}
    except Exception:
        return {}


def _policy_blacklist_reason_for_executor(executor, hotkey_ss58: str = "") -> str:
    from common.subnet_runtime_config import policy_blacklist_reason

    return policy_blacklist_reason(
        getattr(executor, "executor_id", "") or "",
        hotkey_ss58=hotkey_ss58,
        miner_address=getattr(executor, "miner_address", "") or "",
        miner_uid=getattr(executor, "miner_uid", None),
        endpoint=getattr(executor, "endpoint", "") or "",
    )


def _policy_blacklist_blocked_ids(executors: list) -> set[str]:
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


def _unhealthy_reason(executor_id: str) -> str:
    if endpoint_health is not None:
        health_state = endpoint_health.state(executor_id)
        if health_state is None:
            return "Endpoint health not confirmed yet"
        if health_state.status != "healthy":
            return health_state.reason or "Endpoint health probe failed"
    return "Heartbeat stale or verifier status failed"


@router.get("/marketplace/tiers")
async def marketplace_tiers(request: Request):
    """Renter-facing marketplace view, aggregated by GPU tier.

    Hides individual executor identities so renters can't target a specific
    miner. They see "1× A6000 48GB — 3 available — pass-rate 99% — 0.05
    TAO/h"; the validator picks a real executor at rent time via weighted
    random over candidates.

    Source of truth is on-chain `get_all_active_executors` (already filters
    is_active AND expires_at > now AND not rented), so stale rows can't leak
    through like they do via /instances DB cache.
    """
    await _require_internal_api(request)
    # Read from the snapshot, NOT the chain — chain RPC per request causes
    # rate-limit cascades and stalls renters when the chain is slow.
    if chain_snapshot is None:
        return {"tiers": [], "snapshot_age_s": None}
    free = chain_snapshot.available_executors()

    # Belt-and-suspenders: also exclude executors in our local _active_rentals
    # dict. Without this filter, a renter who just rented can still see their
    # own GPU in the marketplace for up to one snapshot tick (30s) because the
    # snapshot might have just refreshed BEFORE our markRented tx landed.
    # _active_rentals reflects what THIS validator has reserved, which is the
    # source-of-truth for marketplace availability.
    try:
        from neurons.validator.api.routes import rent as rent_route
        active_eids = {info["executor_id"] for info in rent_route._active_rentals.values()}
        free = [e for e in free if e.executor_id not in active_eids]
    except Exception:
        pass

    policy_blocked = _policy_blacklist_blocked_ids(free)
    if policy_blocked:
        free = [e for e in free if e.executor_id not in policy_blocked]

    # Hard-exclude any executor with an open sybil flag. Banned executors
    # MUST NOT be presentable in the marketplace; the sybil scanner has
    # objective evidence they're not what they claim.
    if db_instance is not None:
        try:
            from common.db import get_banned_executor_ids
            banned = get_banned_executor_ids(db_instance)
            if banned:
                free = [e for e in free if e.executor_id not in banned]
        except Exception:
            pass

    # Match /rent's production gate: only advertise inventory that has
    # a successful canary record. Executors with no canary yet or a
    # last canary status of error/fail remain hidden until the scheduler
    # records a pass.
    passed_canary = _canary_passed_executor_ids([e.executor_id for e in free])
    if passed_canary:
        free = [e for e in free if e.executor_id in passed_canary]
    else:
        free = []

    # Match /rent's live-health gate. A stopped miner may still be active on
    # chain until its lease expires, but it must disappear from inventory once
    # heartbeat/proof health fails.
    healthy = _healthy_executor_ids([e.executor_id for e in free])
    free = [e for e in free if e.executor_id in healthy]

    # Match /rent's score gate. Chain-free and live is not enough: hard
    # production gates such as insufficient alpha stake must keep the tier out
    # of public renter inventory.
    positive_score = _positive_score_executor_ids([e.executor_id for e in free])
    free = [e for e in free if e.executor_id in positive_score]

    # Pull per-executor stats + hardware from the DB to enrich each tier
    # with renter-meaningful fields. Validator's local DB is updated on
    # every proof verification and heartbeat — much cheaper than chain.
    stats_by_id: dict[str, dict] = {}
    hw_by_id: dict[str, dict] = {}
    if db_instance is not None:
        try:
            from common.db import get_all_executor_stats, get_all_executor_hardware
            for row in get_all_executor_stats(db_instance):
                stats_by_id[row["executor_id"]] = row
            hw_by_id = get_all_executor_hardware(db_instance)
        except Exception:
            pass

    import time as _t
    now = _t.time()

    # Group by (gpu_model_name, vram_mb, gpu_count) — multi-GPU configs are
    # their own tier because the renter genuinely needs to know "this is a
    # 4x card box vs a 1x card box". Each rental binds to exactly one
    # executor, so a renter who wants 4 GPUs needs a 4-GPU executor; they
    # can't compose four 1-GPU executors into one rental.
    tiers: dict[tuple[str, int, int], dict] = {}
    for ex in free:
        name = _name_for_hash(ex.gpu_model_hash)
        key = (name, ex.vram_mb, ex.gpu_count)
        t = tiers.setdefault(key, {
            "gpu_model": name,
            "vram_gb": round(ex.vram_mb / 1024),
            "gpu_count": ex.gpu_count,
            "available": 0,
            # Aggregates we'll average / min over at the end:
            "pass_rate_sum": 0.0,
            "pass_rate_count": 0,
            "samples_sum": 0,
            "uptime_days_sum": 0.0,
            "uptime_days_count": 0,
            "min_price_rao": None,
            "attestation": "ZkGEMM",
            # Host hardware aggregates (min/max for ranges in the UI).
            "cpu_cores_min": None, "cpu_cores_max": None,
            "ram_gb_min": None, "ram_gb_max": None,
            "disk_gb_min": None, "disk_gb_max": None,
            "disk_total_gb_min": None, "disk_total_gb_max": None,
            # Curated image-cache availability among currently rentable
            # executors in this GPU tier. Filled from heartbeat DB state,
            # never by calling miners during a marketplace/rent request.
            "image_cache": {},
        })
        t["available"] += 1

        # Hardware enrichment
        hw = hw_by_id.get(ex.executor_id, {})
        def _agg_range(field, value):
            if value is None or value <= 0:
                return
            lo = t[f"{field}_min"]
            hi = t[f"{field}_max"]
            t[f"{field}_min"] = value if lo is None else min(lo, value)
            t[f"{field}_max"] = value if hi is None else max(hi, value)
        _agg_range("cpu_cores", hw.get("cpu_cores"))
        _agg_range("ram_gb",   hw.get("ram_gb"))
        _agg_range("disk_gb",  hw.get("disk_gb"))
        _agg_range("disk_total_gb", hw.get("disk_total_gb"))
        cached_images = hw.get("cached_images") or {}
        for image_ref in hw.get("image_catalog") or []:
            if not image_ref:
                continue
            img = t["image_cache"].setdefault(
                image_ref,
                {"cached": 0, "checked": 0, "digest_verified": 0},
            )
            img["checked"] += 1
            if image_ref in cached_images:
                img["cached"] += 1
                if (cached_images.get(image_ref) or {}).get("digest_verified"):
                    img["digest_verified"] += 1

        st = stats_by_id.get(ex.executor_id, {})
        pr = st.get("pass_rate_24h")
        if pr is not None and st.get("samples_24h", 0) > 0:
            t["pass_rate_sum"] += pr
            t["pass_rate_count"] += 1
            t["samples_sum"] += int(st.get("samples_24h") or 0)
        # Uptime: days since first_seen (best signal we have for "this miner
        # has been around"). Validator records first_seen at first heartbeat.
        first_seen = st.get("first_seen")
        if first_seen:
            try:
                from datetime import datetime
                fs = datetime.fromisoformat(first_seen).timestamp()
                t["uptime_days_sum"] += (now - fs) / 86400.0
                t["uptime_days_count"] += 1
            except Exception:
                pass
        if ex.price_per_gpu_hour > 0:
            cur = t["min_price_rao"]
            t["min_price_rao"] = ex.price_per_gpu_hour if cur is None else min(cur, ex.price_per_gpu_hour)

    out = []
    for t in tiers.values():
        pr = t["pass_rate_sum"] / t["pass_rate_count"] if t["pass_rate_count"] else None
        samples = t["samples_sum"]
        # Confidence is a renter-meaningful summary of how well-tested this
        # tier is. New tiers with <5 samples get 'new' — render as "unrated"
        # so renters know not to over-index on a 100% pass rate from 2 cycles.
        if samples >= 100:
            confidence = "high"
        elif samples >= 20:
            confidence = "medium"
        elif samples >= 5:
            confidence = "low"
        else:
            confidence = "new"
        uptime = (t["uptime_days_sum"] / t["uptime_days_count"]) if t["uptime_days_count"] else None

        def _range_str(lo, hi, unit=""):
            if lo is None and hi is None:
                return None
            if lo == hi:
                return f"{lo}{unit}"
            return f"{lo}-{hi}{unit}"

        # Renter-facing USDC price is validator-authoritative subnet policy,
        # not miner-self-reported `min_price_rao`.
        from neurons.validator.scoring.scorer import lookup_usdc_price
        price_usdc_per_hour = lookup_usdc_price(t["gpu_model"], t["gpu_count"])

        out.append({
            "gpu_model": t["gpu_model"],
            "vram_gb": t["vram_gb"],
            "gpu_count": t["gpu_count"],
            "available": t["available"],
            "image_cache": {
                image_ref: {
                    "cached": int(info.get("cached") or 0),
                    "checked": int(info.get("checked") or 0),
                    "digest_verified": int(info.get("digest_verified") or 0),
                    "available": t["available"],
                }
                for image_ref, info in sorted(t["image_cache"].items())
            },
            "attestation": t["attestation"],
            "reliability_24h": pr,
            "reliability_samples": samples,
            "reliability_confidence": confidence,
            "uptime_days": uptime,
            # Renter-facing USDC price for the full tier (already
            # multiplied by gpu_count). The webapp displays this
            # directly; the validator's /rent route recomputes it
            # server-side to avoid trusting the renter.
            "price_usdc_per_hour": price_usdc_per_hour,
            # Legacy miner-self-reported field; not authoritative.
            "min_price_rao": t["min_price_rao"],
            # Host specs (ranges within the tier; identical → single value)
            "cpu_cores": _range_str(t["cpu_cores_min"], t["cpu_cores_max"]),
            "ram_gb":    _range_str(t["ram_gb_min"],   t["ram_gb_max"]),
            # Usable free Docker storage for placement/rental UX. This is
            # intentionally not total disk capacity because cached images and
            # existing layers consume part of the filesystem before a renter
            # starts writing data.
            "disk_gb":   _range_str(t["disk_gb_min"],  t["disk_gb_max"]),
            "disk_total_gb": _range_str(t["disk_total_gb_min"], t["disk_total_gb_max"]),
        })
    # Sort: per-GPU VRAM desc → count desc → model alpha.
    #
    # Per-GPU VRAM first keeps same-class GPUs bundled together (1×, 2×,
    # 4×, 8× of the same model all in one cluster, not scattered across
    # the table by total VRAM). Within a per-GPU bucket, higher counts
    # float up. Across models with identical per-GPU VRAM (e.g. 48 GB
    # for L40S / RTX 6000 Ada / RTX A6000), alphabetical for determinism.
    #
    # Result fleet order (L4 → B300):
    #   1× B300, 1× B200, 1× H200, 8× H100, 2× A100-80, 1× A100-80,
    #   1× H100, 4× RTX A6000, 1× L40S, 1× RTX A6000, 1× RTX 5090,
    #   1× L4, 1× RTX 4090
    out.sort(key=lambda x: (
        -x["vram_gb"],
        -x["gpu_count"],
        x["gpu_model"],
    ))
    return {
        "tiers": out,
        "snapshot_age_s": round(chain_snapshot.age_seconds() or 0, 1) if chain_snapshot.age_seconds() is not None else None,
    }


@router.get("/executors")
async def list_executors(
    request: Request,
    gpu: Optional[str] = Query(None, description="Filter by GPU model substring"),
    available_only: bool = Query(True, description="Only show unrented executors"),
    min_score: float = Query(0.0, description="Minimum composite score"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List available executors with their specs and scores.

    Users query this to find GPUs to rent. Randomized selection happens
    server-side at rental time (POST /rent), not here.
    """
    await _require_internal_api(request)
    # Read from snapshot, never chain directly.
    if chain_snapshot is None:
        return {"executors": [], "total": 0}
    all_executors = chain_snapshot.available_executors() if available_only else chain_snapshot.all_executors()

    # Fraud-hidden executors are removed from public browse/network views.
    # Rental-probation executors are only shown when the caller asks for all
    # executors; they remain excluded from rentable inventory.
    blocked: set[str] = set()
    if db_instance is not None:
        try:
            from common.db import get_banned_executor_ids, get_public_hidden_executor_ids
            blocked = (
                get_banned_executor_ids(db_instance)
                if available_only
                else get_public_hidden_executor_ids(db_instance)
            )
        except Exception:
            pass
    policy_blocked = _policy_blacklist_blocked_ids(all_executors)
    if available_only:
        blocked.update(policy_blocked)

    if available_only:
        passed_canary = _canary_passed_executor_ids([e.executor_id for e in all_executors])
        all_executors = [e for e in all_executors if e.executor_id in passed_canary]

    # Filter
    filtered = []
    healthy = (
        _healthy_executor_ids([e.executor_id for e in all_executors])
        if available_only else set()
    )
    for ex in all_executors:
        if ex.executor_id in blocked:
            continue
        if available_only and ex.is_rented:
            continue
        if available_only and ex.executor_id not in healthy:
            continue
        if gpu:
            # Filter against the human-readable name resolved from the
            # GPU model hash — NOT the raw hash hex, which would never
            # match a string like "4090" except by coincidence.
            name = _name_for_hash(ex.gpu_model_hash).lower()
            if gpu.lower() not in name:
                continue
        # Enrich with scoring
        score = scoring_data.get(ex.executor_id, None) if scoring_data else None
        val = score.score if score else 0.0
        if val < min_score:
            continue

        score_payload = None
        if score:
            score_payload = {
                "value": val,
                "base_price": score.base_price,
                "util_factor": score.util_factor,
                "zero_reason": score.zero_reason,
                "miner_stake_alpha": score.miner_stake_alpha,
                "executor_required_stake_alpha": score.executor_required_stake_alpha,
                "miner_required_stake_alpha": score.miner_required_stake_alpha,
                "stake_grace_until_ts": score.stake_grace_until_ts,
                "stake_ok": score.stake_ok,
                "stake_in_grace": score.stake_in_grace,
            }

        filtered.append({
            "executor_id": ex.executor_id,
            "gpu_model_hash": ex.gpu_model_hash,
            "gpu_count": ex.gpu_count,
            "vram_mb": ex.vram_mb,
            "price_per_gpu_hour": ex.price_per_gpu_hour,
            "is_rented": ex.is_rented,
            "expires_at": ex.expires_at,
            "endpoint": ex.endpoint,
            "score": score_payload,
        })

    # Sort by score descending (None → 0)
    filtered.sort(
        key=lambda x: (x.get("score") or {}).get("value", 0),
        reverse=True,
    )

    total = len(filtered)
    page = filtered[offset:offset + limit]
    return {"executors": page, "total": total}


@router.get("/executors/{executor_id}")
async def get_executor(executor_id: str, request: Request):
    """Get detailed info for a specific executor (from snapshot)."""
    await _require_internal_api(request)
    if chain_snapshot is None:
        return {"error": "Snapshot not available"}
    spec = chain_snapshot.get(executor_id)
    if not spec:
        return {"error": "Executor not found"}

    score = scoring_data.get(executor_id) if scoring_data else None

    return {
        "executor_id": spec.executor_id,
        "miner_address": spec.miner_address,
        "endpoint": spec.endpoint,
        "gpu_model_hash": spec.gpu_model_hash,
        "gpu_count": spec.gpu_count,
        "vram_mb": spec.vram_mb,
        "price_per_gpu_hour": spec.price_per_gpu_hour,
        "is_active": spec.is_active,
        "is_rented": spec.is_rented,
        "expires_at": spec.expires_at,
        "score": {
            "value": score.score,
            "base_price": score.base_price,
            "util_factor": score.util_factor,
            "zero_reason": score.zero_reason,
            "gates_passed": score.gates_passed,
            "gates_failed": score.gates_failed,
            "miner_stake_alpha": score.miner_stake_alpha,
            "executor_required_stake_alpha": score.executor_required_stake_alpha,
            "miner_required_stake_alpha": score.miner_required_stake_alpha,
            "stake_grace_until_ts": score.stake_grace_until_ts,
            "stake_ok": score.stake_ok,
            "stake_in_grace": score.stake_in_grace,
        } if score else None,
    }


# ── /instances: local DB-backed view (fast, rich data) ────────

@router.get("/instances")
async def list_instances(
    request: Request,
    status: Optional[str] = Query(None, description="Filter by status (ok/stale/offline/fail)"),
    gpu: Optional[str] = Query(None, description="Filter by GPU model substring"),
    mig: Optional[bool] = Query(None, description="If true, only MIG-isolated executors"),
    hotkey: Optional[str] = Query(None, description="Filter by miner hotkey"),
    executor_id: Optional[str] = Query(None, description="Filter by executor ID"),
    include_inactive: bool = Query(False, description="Include DB-stale executors no longer active on chain"),
    include_unhealthy: bool = Query(False, description="Include executors with stale heartbeat or failed verifier status"),
    include_hidden: bool = Query(False, description="Include operator diagnostic rows hidden from public inventory"),
    limit: int = Query(50, ge=1, le=200),
):
    """List all known instances with hardware specs + live metrics.

    Reads from the validator's local DB (populated by heartbeats + proofs).
    This is the primary endpoint for users browsing available compute.
    External-facing name: 'instances' (cloud provider convention).
    """
    await _require_internal_api(request)
    if db_instance is None:
        return {"instances": [], "total": 0}

    from common.db import ExecutorStats, ExecutorHardware, Executor, CanaryRecord, ProofResult

    with db_instance.session() as s:
        try:
            from common.db import get_public_hidden_executor_ids, get_rental_probation_executor_ids
            hidden = get_public_hidden_executor_ids(db_instance)
            probation = get_rental_probation_executor_ids(db_instance)
        except Exception:
            hidden = set()
            probation = set()

        hotkey_filter = (hotkey or "").strip()
        executor_filter = (executor_id or "").strip()
        operator_scoped = bool(hotkey_filter or executor_filter)
        allow_hidden_diagnostics = include_hidden and operator_scoped

        active_ids: set[str] | None = None
        if chain_snapshot is not None and not include_inactive:
            active_ids = {ex.executor_id for ex in chain_snapshot.all_executors()}

        query = s.query(ExecutorStats)
        if hotkey_filter:
            query = query.filter(ExecutorStats.hotkey_ss58 == hotkey_filter)
        if executor_filter:
            query = query.filter(ExecutorStats.executor_id == executor_filter)
        if status:
            query = query.filter(ExecutorStats.current_status == status)
        if hidden and not allow_hidden_diagnostics:
            query = query.filter(~ExecutorStats.executor_id.in_(hidden))
        if active_ids is not None:
            if not active_ids:
                return {"instances": [], "total": 0}
            query = query.filter(ExecutorStats.executor_id.in_(active_ids))
        stats_rows = query.limit(limit).all()
        executor_ids = [st.executor_id for st in stats_rows]
        import os as _os
        canary_required = _os.environ.get("CANARY_ENABLED", "0") == "1"
        canary_status: dict[str, str] = {}
        canary_inflight: set[str] = set()
        healthy_ids = _healthy_executor_ids(executor_ids)
        recent_validator_skip_ids: set[str] = set()
        if executor_ids:
            for row in s.query(
                CanaryRecord.executor_id,
                CanaryRecord.last_status,
                CanaryRecord.inflight_at,
            ).filter(CanaryRecord.executor_id.in_(executor_ids)).all():
                canary_status[row[0]] = row[1] or ""
                if row[2] is not None:
                    canary_inflight.add(row[0])
            try:
                from datetime import datetime, timedelta
                from common.subnet_runtime_config import get_subnet_runtime_config
                default_proof_recency_s = get_subnet_runtime_config().scoring.proof_recency_s
                proof_recency_s = float(_os.environ.get(
                    "PROOF_RECENCY_SECONDS", str(default_proof_recency_s),
                ))
                cutoff = datetime.utcnow() - timedelta(seconds=proof_recency_s)
                recent_validator_skip_ids = {
                    row[0]
                    for row in s.query(ProofResult.executor_id).filter(
                        ProofResult.executor_id.in_(executor_ids),
                        ProofResult.valid == True,  # noqa: E712
                        ProofResult.tier == "skipped",
                        ProofResult.verified_at >= cutoff,
                    ).all()
                }
            except Exception:
                recent_validator_skip_ids = set()

        import json as _json
        instances = []
        for st in stats_rows:
            # Get hardware info
            hw = s.query(ExecutorHardware).filter_by(executor_id=st.executor_id).first()
            ex = s.query(Executor).filter_by(executor_id=st.executor_id).first()
            chain_spec = chain_snapshot.get(st.executor_id) if chain_snapshot else None
            endpoint = (
                (chain_spec.endpoint if chain_spec else "")
                or st.endpoint
                or (ex.endpoint if ex else "")
            )
            policy_blacklist_reason = ""
            if chain_spec is not None:
                policy_blacklist_reason = _policy_blacklist_reason_for_executor(
                    chain_spec,
                    hotkey_ss58=st.hotkey_ss58 or (ex.hotkey_ss58 if ex else ""),
                )
            else:
                from common.subnet_runtime_config import policy_blacklist_reason as _pbr
                policy_blacklist_reason = _pbr(
                    st.executor_id,
                    hotkey_ss58=st.hotkey_ss58 or (ex.hotkey_ss58 if ex else ""),
                    miner_address=ex.miner_address if ex else "",
                    endpoint=endpoint,
                )

            # MIG state lives in the hw_static JSON blob (no dedicated
            # column). Parse it lazily here so the marketplace can
            # surface a hardware-isolation tier without a schema change.
            mig_capable_any = False
            mig_enabled_any = False
            _hw_blob = {}
            if hw and hw.hw_static_json:
                try:
                    _hw_blob = _json.loads(hw.hw_static_json)
                    mig_capable_any = bool(_hw_blob.get("mig_capable_any", False))
                    mig_enabled_any = bool(_hw_blob.get("mig_enabled_any", False))
                except Exception:
                    _hw_blob = {}
            _live_blob = {}
            if st.hw_live_json:
                try:
                    _live_blob = _json.loads(st.hw_live_json)
                except Exception:
                    _live_blob = {}

            storage = _hw_blob.get("storage") or {}
            docker_storage = storage.get("docker") or {}
            live_storage = (_live_blob.get("storage") or {}).get("docker") or {}
            network_speedtest = _live_blob.get("network_speedtest") or {}

            alloc_state = st.alloc_state
            try:
                from neurons.validator.api.routes import rent as rent_route
                if (
                    (chain_spec is not None and chain_spec.is_rented)
                    or rent_route.rental_state.has_active_rental(st.executor_id)
                ):
                    alloc_state = "rented"
                elif chain_spec is not None:
                    alloc_state = "free"
            except Exception:
                pass
            if alloc_state != "rented" and (
                st.executor_id in probation
                or policy_blacklist_reason
                or
                st.executor_id in canary_inflight
                or (canary_required and canary_status.get(st.executor_id, "") != "pass")
            ):
                alloc_state = "audit"

            if st.executor_id not in healthy_ids:
                if not include_unhealthy:
                    continue
                display_status = st.current_status
                display_reason = st.current_reason
                if display_status not in ("fail", "offline", "stale"):
                    display_status = "offline"
                    display_reason = _unhealthy_reason(st.executor_id)
            else:
                display_status = st.current_status
                display_reason = st.current_reason
                if display_status == "stale" and st.executor_id in recent_validator_skip_ids:
                    display_status = "ok"
                    display_reason = "Verification delayed; previous proof remains valid"

            inst = {
                "instance_id": st.executor_id,
                "status": display_status,
                "status_reason": display_reason,
                "endpoint": endpoint,
                "hotkey": st.hotkey_ss58,
                "uid": getattr(chain_spec, "miner_uid", None) if chain_spec else None,
                "miner_uid": getattr(chain_spec, "miner_uid", None) if chain_spec else None,
                "alloc_state": alloc_state,

                # Hardware
                "hardware": {
                    "gpu_name": hw.primary_gpu_name if hw else "",
                    "gpu_count": hw.gpu_count if hw else 0,
                    "gpu_vram_bytes": hw.gpu_vram_total_bytes if hw else 0,
                    "cpu_model": hw.cpu_model if hw else "",
                    "cpu_cores": hw.cpu_cores_logical if hw else 0,
                    "ram_bytes": hw.ram_total_bytes if hw else 0,
                    "disk_bytes": (
                        docker_storage.get("total_bytes")
                        or (hw.disk_total_bytes if hw else 0)
                        or st.disk_total_bytes
                    ),
                    "disk_free_bytes": (
                        live_storage.get("available_bytes")
                        or live_storage.get("free_bytes")
                        or _live_blob.get("disk_free_bytes")
                        or docker_storage.get("available_bytes")
                        or docker_storage.get("free_bytes")
                        or 0
                    ),
                    "disk_kind": docker_storage.get("kind") or (hw.primary_disk_type if hw else ""),
                    "disk_fstype": docker_storage.get("fstype") or "",
                    "disk_source": docker_storage.get("source") or "",
                    "disk_mountpoint": docker_storage.get("mountpoint") or "",
                    "storage": storage,
                    "driver_version": hw.driver_version if hw else "",
                    # MIG-isolated executors offer hardware-enforced
                    # tenant isolation — promoted to a higher tier in
                    # the marketplace for confidential workloads.
                    "mig_capable": mig_capable_any,
                    "mig_enabled": mig_enabled_any,
                    "isolation_tier": (
                        "mig" if mig_enabled_any
                        else "sysbox" if hw and hw.sysbox_detected
                        else "container"
                    ) if hw else "unknown",
                } if hw else None,

                # Live metrics
                "live": {
                    "cpu_pct": st.cpu_usage_pct,
                    "ram_used_bytes": st.ram_used_bytes,
                    "gpu_pct": st.gpu_usage_pct,
                    "gpu_vram_used_bytes": st.gpu_vram_used_bytes,
                    "gpu_temp_c": st.gpu_temp_c,
                    "gpu_power_w": st.gpu_power_w,
                    "disk_used_bytes": live_storage.get("used_bytes") or st.disk_used_bytes,
                    "disk_total_bytes": live_storage.get("total_bytes") or st.disk_total_bytes,
                    "disk_free_bytes": (
                        live_storage.get("available_bytes")
                        or live_storage.get("free_bytes")
                        or _live_blob.get("disk_free_bytes")
                        or 0
                    ),
                    "net_rx_bytes": st.net_rx_bytes,
                    "net_tx_bytes": st.net_tx_bytes,
                    "network_speedtest": network_speedtest,
                },

                # Proof stats
                "proof": {
                    "pass_rate_1h": st.pass_rate_1h,
                    "pass_rate_24h": st.pass_rate_24h,
                    "samples_1h": st.samples_1h,
                    "samples_24h": st.samples_24h,
                    "last_epoch": st.last_epoch_id,
                    "trust_level": st.trust_level,
                },

                # Scoring snapshot — fed by _scoring_loop into scoring_data.
                # Always emit a dict (never None) so renters / UIs can read
                # `value` and `zero_reason` unconditionally.
                "score": (
                    {
                        "value": scoring_data[st.executor_id].score,
                        "base_price": scoring_data[st.executor_id].base_price,
                        "util_factor": scoring_data[st.executor_id].util_factor,
                        "zero_reason": scoring_data[st.executor_id].zero_reason,
                        "miner_stake_alpha": scoring_data[st.executor_id].miner_stake_alpha,
                        "executor_required_stake_alpha": scoring_data[st.executor_id].executor_required_stake_alpha,
                        "miner_required_stake_alpha": scoring_data[st.executor_id].miner_required_stake_alpha,
                        "stake_grace_until_ts": scoring_data[st.executor_id].stake_grace_until_ts,
                        "stake_ok": scoring_data[st.executor_id].stake_ok,
                        "stake_in_grace": scoring_data[st.executor_id].stake_in_grace,
                    }
                    if scoring_data and st.executor_id in scoring_data
                    else {"value": 0.0, "base_price": 0.0, "util_factor": 0.0,
                          "zero_reason": "no_score_yet",
                          "miner_stake_alpha": 0.0,
                          "executor_required_stake_alpha": 0.0,
                          "miner_required_stake_alpha": 0.0,
                          "stake_grace_until_ts": 0.0,
                          "stake_ok": True,
                          "stake_in_grace": False}
                ),

                "age_sec": st.age_sec,
                "first_seen": st.first_seen.isoformat() if st.first_seen else None,
                "updated_at": st.updated_at.isoformat() if st.updated_at else None,
            }

            if gpu and hw and gpu.lower() not in hw.primary_gpu_name.lower():
                continue
            if mig is True and not mig_enabled_any:
                continue

            instances.append(inst)

    return {"instances": instances, "total": len(instances)}


@router.get("/scores")
async def get_scores(request: Request):
    """CLI-compat shim. Returns per-executor pass-rate / trust / status.

    Reuses the same data source as /instances but reshapes to the legacy
    schema the nodexo CLI's `--validator-url` mode expects.
    """
    await _require_internal_api(request)
    if not db_instance:
        return {"scores": {}}
    from common.db import ExecutorStats
    out: dict[str, dict] = {}
    with db_instance.session() as s:
        rows = s.query(ExecutorStats).all()
        for st in rows:
            out[st.executor_id] = {
                "pass_rate_24h": st.pass_rate_24h,
                "status": st.current_status,
                "avg_delta": st.avg_delta_24h,
                "trust_level": st.trust_level,
            }
    return {"scores": out}
