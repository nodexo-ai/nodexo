"""Shared EVM RPC provider helpers."""
from __future__ import annotations

import os

from web3 import Web3


def evm_rpc_timeout_seconds() -> float:
    """HTTP timeout for Bittensor EVM JSON-RPC calls.

    Miners start their API only after EVM registration. Without a bounded
    timeout, a stale or firewalled RPC endpoint can make a healthy miner look
    dead before it ever begins serving `/health`.
    """
    raw = os.environ.get("EVM_RPC_TIMEOUT_SECONDS", "12")
    try:
        return max(1.0, float(raw))
    except Exception:
        return 12.0


def make_http_provider(rpc_url: str):
    return Web3.HTTPProvider(
        rpc_url,
        request_kwargs={"timeout": evm_rpc_timeout_seconds()},
    )
