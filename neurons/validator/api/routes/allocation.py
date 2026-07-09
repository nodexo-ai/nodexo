"""Allocation authority API."""
from __future__ import annotations

import json
import os
import time
from typing import Any

import bittensor as bt
from fastapi import APIRouter, HTTPException, Request

from common.crypto import (
    check_and_record_nonce,
    extract_signature_headers,
    sign_payload,
    verify_signature,
)

router = APIRouter(tags=["allocation"])

db_instance = None
authority_hotkey_seed: bytes | None = None
permitted_reader_hotkeys: set[str] = set()


def set_permitted_reader_hotkeys(hotkeys: set[str]) -> None:
    global permitted_reader_hotkeys
    permitted_reader_hotkeys = {str(h).strip() for h in hotkeys if str(h).strip()}


def _canonical_request_bytes(method: str, path: str, body: bytes) -> bytes:
    return method.upper().encode() + b" " + path.encode() + b"\n" + body


def _path_with_query(request: Request) -> str:
    path = request.url.path
    if request.url.query:
        path += "?" + request.url.query
    return path


def _runtime_allowed() -> tuple[set[str], set[str], set[str], str]:
    try:
        from common.subnet_runtime_config import get_subnet_runtime_config
        cfg = get_subnet_runtime_config()
        allocation = cfg.allocation
        writers = set(allocation.writer_hotkeys)
        admins = set(allocation.admin_hotkeys)
        readers = set(cfg.network.rental_validator_hotkeys) | permitted_reader_hotkeys
        authority = allocation.authority_hotkey or (
            cfg.network.rental_validator_hotkeys[0]
            if cfg.network.rental_validator_hotkeys else ""
        )
        if authority:
            writers.add(authority)
            readers.add(authority)
        readers |= writers
        readers |= admins
        return readers, writers, admins, authority
    except Exception:
        return set(), set(), set(), ""


def _env_hotkeys(name: str) -> set[str]:
    return {h.strip() for h in os.environ.get(name, "").split(",") if h.strip()}


async def _verify_signed_request(
    request: Request,
    *,
    allow_readers: bool = False,
    allow_writers: bool = False,
    allow_admins: bool = False,
    allow_executor_owner: str = "",
) -> tuple[str, bool]:
    extracted = extract_signature_headers(dict(request.headers))
    if not extracted:
        raise HTTPException(403, "signature required")
    sig, hotkey, ts, nonce = extracted
    raw_body = await request.body()
    body = _canonical_request_bytes(request.method, _path_with_query(request), raw_body)
    if not verify_signature(body, sig, hotkey, ts, nonce):
        raise HTTPException(403, "invalid signature")
    if not check_and_record_nonce("allocation-api", hotkey, nonce, body):
        raise HTTPException(403, "replayed signature")

    readers, writers, admins, _authority = _runtime_allowed()
    admins |= _env_hotkeys("NODEXO_ADMIN_HOTKEYS")
    writers |= _env_hotkeys("NODEXO_ALLOCATION_WRITER_HOTKEYS")
    readers |= _env_hotkeys("NODEXO_ALLOCATION_READER_HOTKEYS")
    readers |= writers | admins
    admin = hotkey in admins
    writer = hotkey in writers
    reader = hotkey in readers
    if allow_admins and admin:
        return hotkey, True
    if allow_writers and writer:
        return hotkey, False
    if allow_readers and reader:
        return hotkey, False
    if allow_executor_owner:
        if db_instance is None:
            raise HTTPException(503, "allocation db unavailable")
        try:
            from common.db import Executor, ExecutorHotkeyBinding
            with db_instance.session() as s:
                bound = s.query(ExecutorHotkeyBinding).filter_by(
                    executor_id=allow_executor_owner,
                ).first()
                executor = s.query(Executor).filter_by(
                    executor_id=allow_executor_owner,
                ).first()
                expected = (
                    (bound.hotkey_ss58 if bound else "")
                    or (executor.hotkey_ss58 if executor else "")
                )
        except Exception:
            expected = ""
        if expected and expected == hotkey:
            return hotkey, False
    raise HTTPException(403, "not authorized")


def _payload_response(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload.setdefault("generated_at", time.time())
    payload.setdefault("source", "allocation-authority")
    if not authority_hotkey_seed:
        return {"payload": payload, "signature": None}
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    headers = sign_payload(body, authority_hotkey_seed)
    return {
        "payload": payload,
        "signature": {
            "hotkey": headers.get("X-Nodexo-Hotkey", ""),
            "timestamp": headers.get("X-Nodexo-Timestamp", ""),
            "nonce": headers.get("X-Nodexo-Nonce", ""),
            "signature": headers.get("X-Nodexo-Signature", ""),
        },
    }


def _require_authority_signing_ready() -> None:
    _readers, _writers, _admins, authority = _runtime_allowed()
    if authority and authority_hotkey_seed is None:
        raise HTTPException(503, "allocation authority signing unavailable")


def _sanitize_lock(row: dict) -> dict:
    return {
        "executor_id": row.get("executor_id", ""),
        "lock_id": row.get("lock_id", ""),
        "busy": row.get("state") in {"active", "release_pending"},
        "state": "busy" if row.get("state") in {"active", "release_pending"} else "released",
        "start_block": int(row.get("start_block") or 0),
        "end_block": int(row.get("end_block") or 0),
        "container_name": row.get("container_name", ""),
        "revision": int(row.get("revision") or 0),
    }


@router.get("/api/allocation/health")
async def health():
    return {"status": "ok", "service": "nodexo-allocation"}


@router.get("/api/allocation/status")
async def status(request: Request, since_block: int = 0, since_revision: int = 0):
    await _verify_signed_request(request, allow_readers=True, allow_admins=True)
    _require_authority_signing_ready()
    if db_instance is None:
        raise HTTPException(503, "allocation db unavailable")
    from common.db import allocation_latest_revision, list_allocation_locks
    latest_revision = allocation_latest_revision(db_instance)
    limit = 10000
    active_locks = list_allocation_locks(
        db_instance,
        active_only=True,
        limit=limit,
    )
    window_locks = list_allocation_locks(
        db_instance,
        since_block=max(0, int(since_block or 0)),
        since_revision=max(0, int(since_revision or 0)),
        limit=limit,
        newest_first=not int(since_block or 0) and not int(since_revision or 0),
    )
    active = [row for row in active_locks if row.get("state") in {"active", "release_pending"}]
    returned_window_max_revision = max(
        [int(row.get("revision") or 0) for row in window_locks] or [0]
    )
    revision = latest_revision
    if int(since_revision or 0) > 0 and len(window_locks) >= limit:
        revision = min(latest_revision, returned_window_max_revision)
    windows = [
        _sanitize_lock(row)
        for row in window_locks
        if int(row.get("start_block") or 0) > 0
        or (
            int(since_revision or 0) > 0
            and row.get("state") not in {"active", "release_pending"}
        )
    ]
    return _payload_response({
        "revision": revision,
        "active": [_sanitize_lock(row) for row in active],
        "windows": windows,
    })


@router.get("/api/allocation/executors/{executor_id}")
async def executor_status(executor_id: str, request: Request):
    eid = executor_id.lower().removeprefix("0x")
    if len(eid) != 64:
        raise HTTPException(400, "invalid executor_id")
    await _verify_signed_request(request, allow_admins=True, allow_executor_owner=eid)
    _require_authority_signing_ready()
    if db_instance is None:
        raise HTTPException(503, "allocation db unavailable")
    from common.db import active_allocation_container_names, list_allocation_locks
    active = list_allocation_locks(db_instance, active_only=True, executor_id=eid, limit=100)
    return _payload_response({
        "executor_id": eid,
        "busy": bool(active),
        "container_names": active_allocation_container_names(db_instance, eid),
        "revision": max([int(row.get("revision") or 0) for row in active] or [0]),
    })


@router.post("/api/allocation/lock")
async def lock(request: Request):
    hotkey, _admin = await _verify_signed_request(
        request,
        allow_writers=True,
        allow_admins=True,
    )
    _require_authority_signing_ready()
    if db_instance is None:
        raise HTTPException(503, "allocation db unavailable")
    body = await request.json()
    try:
        from common.db import create_allocation_lock
        row = create_allocation_lock(
            db_instance,
            lock_id=str(body.get("lock_id") or body.get("rental_id") or ""),
            executor_id=str(body.get("executor_id") or ""),
            owner_validator_hotkey=hotkey,
            kind=str(body.get("kind") or "rental"),
            start_block=int(body.get("start_block") or 0),
            container_name=str(body.get("container_name") or ""),
            chain_locked=bool(body.get("chain_locked")),
            idempotency_key=str(body.get("idempotency_key") or ""),
        )
    except ValueError as e:
        raise HTTPException(409, str(e))
    return _payload_response({"lock": _sanitize_lock(row)})


@router.post("/api/allocation/release")
async def release(request: Request):
    hotkey, admin = await _verify_signed_request(
        request,
        allow_writers=True,
        allow_admins=True,
    )
    _require_authority_signing_ready()
    if db_instance is None:
        raise HTTPException(503, "allocation db unavailable")
    body = await request.json()
    try:
        from common.db import release_allocation_lock
        row = release_allocation_lock(
            db_instance,
            lock_id=str(body.get("lock_id") or body.get("rental_id") or ""),
            executor_id=str(body.get("executor_id") or ""),
            end_block=int(body.get("end_block") or 0),
            reason=str(body.get("reason") or "user"),
            owner_validator_hotkey=hotkey,
            admin=admin,
            pending=bool(body.get("pending")),
        )
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    if row is None:
        raise HTTPException(404, "allocation lock not found")
    return _payload_response({"lock": _sanitize_lock(row)})


@router.post("/api/allocation/container")
async def container(request: Request):
    hotkey, admin = await _verify_signed_request(
        request,
        allow_writers=True,
        allow_admins=True,
    )
    _require_authority_signing_ready()
    if db_instance is None:
        raise HTTPException(503, "allocation db unavailable")
    body = await request.json()
    try:
        from common.db import update_allocation_container
        row = update_allocation_container(
            db_instance,
            str(body.get("lock_id") or body.get("rental_id") or ""),
            container_name=str(body.get("container_name") or ""),
            owner_validator_hotkey=hotkey,
            admin=admin,
        )
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    if row is None:
        raise HTTPException(404, "allocation lock not found")
    return _payload_response({"lock": _sanitize_lock(row)})


@router.post("/api/allocation/admin/recover")
async def recover(request: Request):
    hotkey, admin = await _verify_signed_request(request, allow_admins=True)
    if not admin:
        raise HTTPException(403, "admin auth required")
    _require_authority_signing_ready()
    if db_instance is None:
        raise HTTPException(503, "allocation db unavailable")
    body = await request.json()
    from common.db import release_allocation_lock
    row = release_allocation_lock(
        db_instance,
        lock_id=str(body.get("lock_id") or body.get("rental_id") or ""),
        executor_id=str(body.get("executor_id") or ""),
        end_block=int(body.get("end_block") or 0),
        reason=str(body.get("reason") or "admin_recovery"),
        owner_validator_hotkey=hotkey,
        admin=True,
    )
    if row is None:
        raise HTTPException(404, "allocation lock not found")
    bt.logging.warning(
        f"allocation recovery by {hotkey[:12]} for "
        f"{str(body.get('executor_id') or body.get('lock_id') or '')[:16]}"
    )
    return _payload_response({"lock": _sanitize_lock(row)})
