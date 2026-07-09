"""Client helpers for the Nodexo allocation authority."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import aiohttp
import bittensor as bt

from common.crypto import sign_payload, verify_signature
from common.subnet_runtime_config import get_subnet_runtime_config


@dataclass
class AllocationSnapshot:
    active: dict[str, dict] = field(default_factory=dict)
    windows: dict[str, list[dict]] = field(default_factory=dict)
    revision: int = 0
    generated_at: float = 0.0
    received_at: float = 0.0
    last_error: str = ""

    def age_seconds(self) -> float | None:
        if not self.generated_at:
            return None
        return max(0.0, time.time() - self.generated_at)

    def is_fresh(self) -> bool:
        max_age = get_subnet_runtime_config().allocation.max_snapshot_age_seconds
        age = self.age_seconds()
        return age is not None and age <= max_age and not self.last_error


def _merge_delta(base: AllocationSnapshot, delta: AllocationSnapshot) -> AllocationSnapshot:
    merged = AllocationSnapshot(
        active=dict(base.active),
        windows={eid: list(items) for eid, items in base.windows.items()},
        revision=max(int(base.revision or 0), int(delta.revision or 0)),
        generated_at=delta.generated_at,
        received_at=delta.received_at,
    )
    changed: list[dict] = []
    for item in delta.windows.values():
        changed.extend(item)
    for item in delta.active.values():
        changed.append(item)
    for item in changed:
        eid = str(item.get("executor_id") or "").lower().removeprefix("0x")
        lock_id = str(item.get("lock_id") or "")
        if not eid:
            continue
        if lock_id:
            merged.windows[eid] = [
                existing
                for existing in merged.windows.get(eid, [])
                if str(existing.get("lock_id") or "") != lock_id
            ]
        if int(item.get("start_block") or 0) > 0:
            merged.windows.setdefault(eid, []).append(item)
        if item.get("busy"):
            merged.active[eid] = item
        elif eid in merged.active and (
            not lock_id or str(merged.active[eid].get("lock_id") or "") == lock_id
        ):
            merged.active.pop(eid, None)
    return merged


_SNAPSHOT = AllocationSnapshot()
_LOCK = asyncio.Lock()


def _base_url() -> str:
    return get_subnet_runtime_config().allocation.status_url.rstrip("/")


def _path_for(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    return path


def _canonical_request_bytes(method: str, path: str, body: bytes) -> bytes:
    return method.upper().encode() + b" " + path.encode() + b"\n" + body


def _signed_headers(method: str, url: str, body: bytes, hotkey_seed: bytes | None) -> dict[str, str]:
    if not hotkey_seed:
        return {}
    return sign_payload(_canonical_request_bytes(method, _path_for(url), body), hotkey_seed)


def _verify_response(data: dict, *, max_age_s: int) -> dict:
    payload = data.get("payload")
    signature = data.get("signature")
    if not isinstance(payload, dict):
        raise ValueError("allocation response missing payload")
    cfg = get_subnet_runtime_config().allocation
    expected_hotkey = cfg.authority_hotkey or ""
    if not expected_hotkey:
        try:
            expected_hotkey = get_subnet_runtime_config().network.rental_validator_hotkeys[0]
        except Exception:
            expected_hotkey = ""
    if expected_hotkey:
        if not isinstance(signature, dict):
            raise ValueError("allocation response missing signature")
        hotkey = str(signature.get("hotkey") or "")
        if hotkey != expected_hotkey:
            raise ValueError("allocation response signed by unexpected hotkey")
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ok = verify_signature(
            body,
            str(signature.get("signature") or ""),
            hotkey,
            str(signature.get("timestamp") or ""),
            str(signature.get("nonce") or ""),
            max_age=max_age_s,
        )
        if not ok:
            raise ValueError("allocation response signature invalid")
    return payload


async def fetch_status(
    hotkey_seed: bytes | None,
    *,
    since_block: int = 0,
    since_revision: int = 0,
) -> AllocationSnapshot:
    cfg = get_subnet_runtime_config().allocation
    url = _base_url() + "/status"
    params = []
    if since_block > 0:
        params.append(("since_block", str(int(since_block))))
    if since_revision > 0:
        params.append(("since_revision", str(int(since_revision))))
    if params:
        query = "&".join(f"{k}={v}" for k, v in params)
        url = f"{url}?{query}"
    headers = _signed_headers("GET", url, b"", hotkey_seed)
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                raise RuntimeError(f"allocation status HTTP {resp.status}: {(await resp.text())[:160]}")
            payload = _verify_response(
                await resp.json(content_type=None),
                max_age_s=max(30, int(cfg.max_snapshot_age_seconds)),
            )
    snap = AllocationSnapshot(
        revision=int(payload.get("revision") or 0),
        generated_at=float(payload.get("generated_at") or time.time()),
        received_at=time.time(),
    )
    for item in payload.get("active") or []:
        eid = str(item.get("executor_id") or "").lower().removeprefix("0x")
        if eid:
            snap.active[eid] = item
    for item in payload.get("windows") or []:
        eid = str(item.get("executor_id") or "").lower().removeprefix("0x")
        if eid:
            snap.windows.setdefault(eid, []).append(item)
    return snap


async def post_lock(
    hotkey_seed: bytes | None,
    *,
    lock_id: str,
    executor_id: str,
    kind: str,
    start_block: int,
    container_name: str = "",
    chain_locked: bool = False,
    idempotency_key: str = "",
) -> dict:
    return await _post_json(
        "/lock",
        hotkey_seed,
        {
            "lock_id": lock_id,
            "executor_id": executor_id,
            "kind": kind,
            "start_block": int(start_block or 0),
            "container_name": container_name,
            "chain_locked": bool(chain_locked),
            "idempotency_key": idempotency_key or lock_id,
        },
    )


async def post_container(
    hotkey_seed: bytes | None,
    *,
    lock_id: str,
    container_name: str,
) -> dict:
    return await _post_json(
        "/container",
        hotkey_seed,
        {
            "lock_id": lock_id,
            "container_name": container_name,
        },
    )


async def post_release(
    hotkey_seed: bytes | None,
    *,
    lock_id: str = "",
    executor_id: str = "",
    end_block: int = 0,
    reason: str = "user",
    pending: bool = False,
) -> dict:
    return await _post_json(
        "/release",
        hotkey_seed,
        {
            "lock_id": lock_id,
            "executor_id": executor_id,
            "end_block": int(end_block or 0),
            "reason": reason,
            "pending": bool(pending),
        },
    )


async def _post_json(path: str, hotkey_seed: bytes | None, payload: dict) -> dict:
    if not hotkey_seed:
        raise RuntimeError("allocation write signing seed unavailable")
    cfg = get_subnet_runtime_config().allocation
    url = _base_url() + path
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    headers = _signed_headers("POST", url, body, hotkey_seed)
    headers["content-type"] = "application/json"
    timeout = aiohttp.ClientTimeout(total=8)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, data=body, headers=headers) as resp:
            if resp.status != 200:
                raise RuntimeError(f"allocation write HTTP {resp.status}: {(await resp.text())[:160]}")
            return _verify_response(
                await resp.json(content_type=None),
                max_age_s=max(30, int(cfg.max_snapshot_age_seconds)),
            )


async def refresh_loop(hotkey_seed: bytes | None, *, interval_s: float = 30.0) -> None:
    global _SNAPSHOT
    while True:
        try:
            current = get_snapshot()
            since_revision = int(current.revision or 0) if current.is_fresh() else 0
            snap = await fetch_status(hotkey_seed, since_revision=since_revision)
            if since_revision > 0:
                snap = _merge_delta(current, snap)
            async with _LOCK:
                _SNAPSHOT = snap
        except asyncio.CancelledError:
            raise
        except Exception as e:
            async with _LOCK:
                _SNAPSHOT.last_error = str(e)
            bt.logging.debug(f"allocation status refresh failed: {e}")
        await asyncio.sleep(max(5.0, interval_s))


def get_snapshot() -> AllocationSnapshot:
    return _SNAPSHOT


def is_effectively_busy(executor_id: str) -> bool:
    cfg = get_subnet_runtime_config().allocation
    if cfg.read_mode == "chain":
        return False
    eid = str(executor_id or "").lower().removeprefix("0x")
    if not eid:
        return False
    snap = get_snapshot()
    if not snap.is_fresh():
        return True
    return eid in snap.active


def was_allocated_at(executor_id: str, block: int) -> bool | None:
    cfg = get_subnet_runtime_config().allocation
    if cfg.read_mode == "chain":
        return False
    eid = str(executor_id or "").lower().removeprefix("0x")
    if not eid or block <= 0:
        return None
    snap = get_snapshot()
    if not snap.is_fresh():
        return None
    for window in snap.windows.get(eid, []):
        start = int(window.get("start_block") or 0)
        end = int(window.get("end_block") or 0)
        if start <= block and (end <= 0 or block < end):
            return True
    return False


async def fetch_executor_status(
    executor_id: str,
    hotkey_seed: bytes | None,
) -> dict | None:
    cfg = get_subnet_runtime_config().allocation
    if cfg.read_mode == "chain":
        return None
    eid = str(executor_id or "").lower().removeprefix("0x")
    if len(eid) != 64:
        return None
    url = _base_url() + f"/executors/{eid}"
    headers = _signed_headers("GET", url, b"", hotkey_seed)
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                raise RuntimeError(f"allocation executor HTTP {resp.status}")
            return _verify_response(
                await resp.json(content_type=None),
                max_age_s=max(30, int(cfg.max_snapshot_age_seconds)),
            )


def fetch_executor_status_sync(
    executor_id: str,
    hotkey_seed: bytes | None,
    *,
    timeout_s: float = 5.0,
) -> dict | None:
    cfg = get_subnet_runtime_config().allocation
    if cfg.read_mode == "chain":
        return None
    eid = str(executor_id or "").lower().removeprefix("0x")
    if len(eid) != 64:
        return None
    import urllib.request

    url = _base_url() + f"/executors/{eid}"
    headers = _signed_headers("GET", url, b"", hotkey_seed)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return _verify_response(
        data,
        max_age_s=max(30, int(cfg.max_snapshot_age_seconds)),
    )
