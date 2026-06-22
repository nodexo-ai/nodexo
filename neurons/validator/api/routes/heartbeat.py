"""
Heartbeat endpoint — miners push hardware info periodically.

POST /heartbeat
  Body: {
    "executor_id": "...",
    "hw_static": { cpu, ram, gpus, disk, system_uuid, ... },
    "hw_live": { cpu_pct, ram_used, gpu_pct, gpu_temp, disk_used, ... },
    "endpoint": "https://miner:8090",
    "software_version": "0.1.0"
  }

The validator records this in the executor DB for liveness tracking
and hardware verification. Miners push every ~60s.

Reference: see protocol design notes
"""
from __future__ import annotations

import time

import bittensor as bt
from fastapi import APIRouter, HTTPException, Request
router = APIRouter(tags=["heartbeat"])

# Set by validator startup
db_instance = None           # common.db.Database
registry_client = None       # ComputeRegistryClient

# Rate limit: max 1 heartbeat per executor per 30s
_last_heartbeat: dict[str, float] = {}
_HB_INTERVAL = 30


@router.post("/heartbeat")
async def receive_heartbeat(request: Request):
    """Receive hardware heartbeat from an executor."""
    body_bytes = await request.body()
    body = __import__("json").loads(body_bytes)

    executor_id = body.get("executor_id", "")
    if not executor_id:
        raise HTTPException(400, "executor_id required")

    # Signature verification (same as proofs — skip in dev mode)
    if registry_client is not None:
        from neurons.validator.api.routes.proofs import _verify_request_signature
        await _verify_request_signature(request, body_bytes, executor_id)

    # Rate limit after auth. Otherwise an unauthenticated caller could burn the
    # heartbeat slot and suppress a legitimate miner report for _HB_INTERVAL.
    now = time.time()
    last = _last_heartbeat.get(executor_id, 0)
    if now - last < _HB_INTERVAL:
        raise HTTPException(429, f"Heartbeat too frequent (wait {_HB_INTERVAL}s)")
    _last_heartbeat[executor_id] = now

    # Extract fields
    hw_static = body.get("hw_static")
    hw_live = body.get("hw_live")
    endpoint = str(body.get("endpoint") or "").strip()
    hotkey_ss58 = ""

    # Get hotkey from signature headers if present
    from common.crypto import extract_signature_headers
    headers = dict(request.headers)
    sig_parts = extract_signature_headers(headers)
    if sig_parts:
        hotkey_ss58 = sig_parts[1]

    # ── VRAM consistency check (Gap 1 closure) ──────────────────
    # Compare the miner's reported VRAM against the GPU model's spec.
    # A miner who under-reports total VRAM (claims 40 GB on an A6000
    # to hide 8 GB of sybil) trips this check immediately, without
    # needing a canary fill. NVML's reported total is slightly less
    # than the marketing capacity (driver carveout for ECC/reserve),
    # so we tolerate a small negative drift; over-claims and large
    # under-claims both flag.
    if hw_static and db_instance:
        try:
            _maybe_flag_vram_inconsistency(executor_id, hw_static)
        except Exception as e:
            bt.logging.debug(f"vram-consistency check failed: {e}")

    # Update DB
    if db_instance:
        from common.db import update_executor_on_heartbeat
        update_executor_on_heartbeat(
            db_instance, executor_id,
            hw_static=hw_static, hw_live=hw_live,
            hotkey_ss58=hotkey_ss58, endpoint=endpoint,
        )

    bt.logging.debug(f"Heartbeat: executor={executor_id[:16]} endpoint={endpoint[:32]}")
    return {"status": "accepted"}


# Acceptable deviation between miner-reported VRAM and the GPU model's
# spec-sheet capacity. NVML's `total_memory` runs ~1-2% below the
# nominal capacity (driver reserve + ECC overhead). We allow:
#   reported / spec >= 0.92   (catches under-reports of >8% — a sybil
#                              hiding 4 GB on an A6000 leaves a >8%
#                              gap, definitely caught)
#   reported / spec <= 1.02   (small over-claim margin for rounding)
_VRAM_LOWER_RATIO = 0.92
_VRAM_UPPER_RATIO = 1.02


def _maybe_flag_vram_inconsistency(executor_id: str, hw_static: dict) -> None:
    """Compare miner-reported per-GPU VRAM against GPU_VRAM_GB spec.
    Opens a HARD `vram_underreport` flag on mismatch.

    The check is per-GPU because a miner can mix-and-match models in
    a multi-GPU rig, and our spec table is keyed by model name.
    """
    from common.config import GPU_VRAM_GB, canonical_gpu_model_name
    from common.db import open_sybil_flag

    gpus = hw_static.get("gpus", []) or []
    for g in gpus:
        name = g.get("name", "")
        reported_bytes = int(g.get("vram_bytes") or 0)
        if not name or reported_bytes <= 0:
            continue
        canonical_name = canonical_gpu_model_name(name)
        spec_gb = GPU_VRAM_GB.get(canonical_name)
        if spec_gb is None:
            # Unknown GPU model — not necessarily a violation; could be
            # a new card we haven't added to the table. Log at DEBUG so
            # we can spot gaps without false-flagging.
            bt.logging.debug(
                f"vram-consistency: unknown GPU model {name!r} from "
                f"executor={executor_id[:16]}; skipping check"
            )
            continue
        spec_bytes = spec_gb * 1024 * 1024 * 1024
        ratio = reported_bytes / spec_bytes
        if ratio < _VRAM_LOWER_RATIO or ratio > _VRAM_UPPER_RATIO:
            evidence = {
                "gpu_name": name,
                "canonical_gpu_name": canonical_name,
                "reported_bytes": reported_bytes,
                "spec_bytes": spec_bytes,
                "ratio": round(ratio, 4),
                "lower_ratio": _VRAM_LOWER_RATIO,
                "upper_ratio": _VRAM_UPPER_RATIO,
                "gpu_index": g.get("index", -1),
            }
            opened = open_sybil_flag(
                db_instance, executor_id,
                "vram_underreport" if ratio < _VRAM_LOWER_RATIO else "vram_overclaim",
                evidence,
            )
            if opened:
                bt.logging.warning(
                    f"vram-consistency: executor={executor_id[:16]} "
                    f"gpu={name} reported={reported_bytes // (1024**3)}GB "
                    f"spec={spec_gb}GB ratio={ratio:.3f} — opened sybil flag"
                )
