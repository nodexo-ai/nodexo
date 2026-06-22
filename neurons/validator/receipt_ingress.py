"""Dedicated validator receipt ingress service.

This process owns only the timing-critical /proofs/receipt path. It keeps
receipt timestamps independent from the main validator's chain, verification,
indexer, scoring, and weight-setting work. The main validator later binds these
durable rows to the final recipe roots/T_commit values before scoring.
"""
from __future__ import annotations

import asyncio
import string
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial

import bittensor as bt
from fastapi import FastAPI, HTTPException, Request

from common.db import Database, default_db_url, store_proof_receipt


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


_CRYPTO_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, min(128, _env_int("NODEXO_RECEIPT_CRYPTO_WORKERS", 16))),
    thread_name_prefix="receipt-ingress-crypto",
)
_STORE_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, min(32, _env_int("NODEXO_RECEIPT_STORE_WORKERS", 4))),
    thread_name_prefix="receipt-ingress-store",
)
_db: Database | None = None
_MAX_RECEIPT_BODY_BYTES = max(512, _env_int("NODEXO_RECEIPT_MAX_BODY_BYTES", 4096))


async def _read_limited_body(request: Request, max_bytes: int = _MAX_RECEIPT_BODY_BYTES) -> bytes:
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


def _clean_hex(value: object, *, expected_len: int = 64) -> str:
    out = str(value or "").strip().lower().removeprefix("0x")
    if len(out) != expected_len or any(c not in string.hexdigits for c in out):
        return ""
    return out


def _verify_and_record_signature(
    body_bytes: bytes,
    headers: dict,
) -> tuple[bool, bool, str]:
    from common.crypto import (
        check_and_record_nonce,
        extract_signature_headers,
        verify_signature,
    )

    sig_parts = extract_signature_headers(headers)
    if sig_parts is None:
        return False, False, ""
    sig_hex, hotkey_ss58, timestamp, nonce = sig_parts
    if not verify_signature(body_bytes, sig_hex, hotkey_ss58, timestamp, nonce):
        return False, False, hotkey_ss58
    if not check_and_record_nonce(
        "validator-ingress",
        hotkey_ss58,
        nonce,
        body_bytes,
    ):
        return False, True, hotkey_ss58
    return True, False, hotkey_ss58


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db
    db_url = (
        os.environ.get("DB_URL")
        or os.environ.get("NODEXO_VALIDATOR_DB_URL")
        or default_db_url()
    )
    _db = Database(db_url)
    bt.logging.info("Receipt ingress database initialized")
    try:
        yield
    finally:
        _CRYPTO_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _STORE_EXECUTOR.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Nodexo Receipt Ingress", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "nodexo-receipt-ingress"}


@app.post("/proofs/receipt")
async def receive_receipt(request: Request):
    recv_time = time.time()
    body_bytes = await _read_limited_body(request)
    try:
        body = json.loads(body_bytes)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Receipt JSON must be an object")

    executor_id = _clean_hex(body.get("executor_id"))
    root = _clean_hex(body.get("root"))
    t_commit = _clean_hex(body.get("T_commit") or body.get("t_commit"))
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

    loop = asyncio.get_running_loop()
    ok, replay, hotkey_ss58 = await loop.run_in_executor(
        _CRYPTO_EXECUTOR,
        _verify_and_record_signature,
        body_bytes,
        dict(request.headers),
    )
    if replay:
        raise HTTPException(401, "Replay detected")
    if not ok:
        raise HTTPException(401, "Invalid signature or expired request")

    if _db is None:
        raise HTTPException(503, "receipt store unavailable")
    try:
        await loop.run_in_executor(
            _STORE_EXECUTOR,
            partial(
                store_proof_receipt,
                _db,
                executor_id=executor_id,
                epoch_id=epoch_id,
                gpu_index=gpu_index,
                u=u,
                recv_time=recv_time,
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

    bt.logging.debug(
        f"Receipt ingress: executor={executor_id[:16]} "
        f"epoch={epoch_id} gpu={gpu_index} u={u}"
    )
    return {"status": "accepted"}


if __name__ == "__main__":
    import uvicorn

    bt.logging.enable_info()
    uvicorn.run(
        app,
        host=os.environ.get("NODEXO_RECEIPT_BIND_HOST", "127.0.0.1"),
        port=int(os.environ.get("NODEXO_RECEIPT_PORT", "19444")),
        log_level=os.environ.get("NODEXO_RECEIPT_LOG_LEVEL", "warning"),
        access_log=False,
        loop="asyncio",
    )
