"""
Substrate RPC client for Bittensor chain interaction.

Provides block hash, block number, and metagraph access.
Used by both executor (proof seed derivation) and validator (seed verification).

Reference: see protocol design notes
Reference: see protocol design notes
"""
from __future__ import annotations

import json
import os
import threading
import time
from functools import lru_cache
from typing import Optional
from urllib import request as urllib_request

import bittensor as bt

# bittensor v10+ uses PascalCase; v8 uses lowercase. Support both.
_SubtensorCls = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
_WalletCls = getattr(bt, "Wallet", None) or getattr(bt, "wallet")


def _http_block_hash_rpc_url(network: str) -> str:
    network_key = str(network or "").upper().replace("-", "_")
    return (
        os.environ.get(f"NODEXO_SUBTENSOR_HTTP_RPC_URL_{network_key}")
        or os.environ.get("NODEXO_SUBTENSOR_HTTP_RPC_URL")
        or os.environ.get(f"SUBTENSOR_HTTP_RPC_URL_{network_key}")
        or os.environ.get("SUBTENSOR_HTTP_RPC_URL")
        or ""
    ).strip()


def _subtensor_endpoint_from_env() -> str:
    return (
        os.environ.get("NODEXO_SUBTENSOR_ENDPOINT")
        or os.environ.get("SUBTENSOR_ENDPOINT")
        or ""
    ).strip()


def _http_rpc_timeout_s() -> float:
    return max(0.001, float(os.environ.get("NODEXO_SUBTENSOR_HTTP_RPC_TIMEOUT_S", "8")))


def _get_block_hash_http(rpc_url: str, block_number: int) -> str | None:
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "chain_getBlockHash",
        "params": [int(block_number)],
    }).encode("utf-8")
    req = urllib_request.Request(
        rpc_url,
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=_http_rpc_timeout_s()) as resp:
        raw = resp.read()
    body = json.loads(raw.decode("utf-8"))
    if body.get("error"):
        raise RuntimeError(body["error"])
    result = body.get("result")
    return str(result) if result else None


class SubtensorRPC:
    """Thin wrapper around bittensor subtensor for RPC calls.

    All chain-touching methods are serialized by `_lock` because
    bittensor's underlying substrate websocket is not thread-safe:
    two threads calling on the same Subtensor instance simultaneously
    raise "cannot call recv while another thread is already running recv".
    The lock is cheap (calls are sub-second when chain is healthy) and
    prevents the cycle-boundary race between the proof loop and the
    periodic metagraph stats logger.
    """

    def __init__(
        self,
        network: str = "finney",
        netuid: int = 0,
        subtensor=None,
        chain_endpoint: str | None = None,
    ):
        self.network = network
        self.chain_endpoint = (chain_endpoint or _subtensor_endpoint_from_env()).strip()
        self.netuid = netuid
        self._subtensor = subtensor  # Accept pre-created instance
        self._metagraph = None
        self._metagraph_updated_at: float = 0
        self._metagraph_ttl: float = 180  # Refresh every 3 min
        self._hotkey_uid_index: dict[str, int] = {}
        self._hotkey_uid_index_at: float = 0
        self._subnet_owner_hotkey: Optional[str] = None
        self._subnet_owner_uid: Optional[int] = None
        self._lock = threading.Lock()

    @property
    def subtensor(self):
        if self._subtensor is None:
            self._subtensor = _SubtensorCls(network=self.chain_endpoint or self.network)
        return self._subtensor

    def close(self) -> None:
        """Close the underlying bittensor/substrate client if this wrapper owns one."""
        with self._lock:
            sub = self._subtensor
            self._subtensor = None
        if sub is None:
            return
        closer = getattr(sub, "close", None)
        if callable(closer):
            try:
                closer()
                return
            except Exception:
                pass
        substrate = getattr(sub, "substrate", None) or getattr(sub, "_substrate", None)
        closer = getattr(substrate, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass

    def get_current_block(self) -> int:
        """Get the current finalized block number."""
        with self._lock:
            return self.subtensor.get_current_block()

    def get_block_hash(self, block_number: int) -> bytes:
        """Get the block hash for a specific block number.

        Returns 32 bytes. The hash is used as the random beacon for
        proof seed derivation.

        Note: Substrate only stores hashes for the last ~256 blocks.
        With 15-block epochs, the epoch start is always within this window.
        """
        last_error: Exception | None = None
        try:
            with self._lock:
                hash_hex = self.subtensor.get_block_hash(block_number)
        except Exception as e:
            last_error = e
            hash_hex = None
        if hash_hex is None:
            # Bittensor/Substrate websocket sessions can temporarily return
            # None for recent historical hashes while a fresh connection can
            # read them. Proof generation is keyed to those hashes, so use a
            # fresh fallback before declaring the cycle unavailable.
            fresh = None
            try:
                fresh = _SubtensorCls(network=self.chain_endpoint or self.network)
                for _attempt in range(3):
                    hash_hex = fresh.get_block_hash(block_number)
                    if hash_hex is not None:
                        break
                    time.sleep(1.0)
            except Exception as e:
                last_error = e
            finally:
                closer = getattr(fresh, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:
                        pass
        if hash_hex is None:
            # Optional operator fallback for rate-limited websocket endpoints.
            # This is intentionally opt-in so release defaults do not route
            # chain reads through a Nodexo-owned service.
            rpc_url = _http_block_hash_rpc_url(self.network)
            if rpc_url:
                try:
                    hash_hex = _get_block_hash_http(rpc_url, block_number)
                except Exception as e:
                    last_error = e
        if hash_hex is None:
            if last_error is not None:
                raise ValueError(
                    f"Block hash not available for block {block_number}: {last_error}"
                ) from last_error
            raise ValueError(f"Block hash not available for block {block_number}")
        # Remove 0x prefix if present
        if isinstance(hash_hex, str):
            hash_hex = hash_hex.replace("0x", "")
        return bytes.fromhex(hash_hex)

    def get_metagraph(self, force_refresh: bool = False, lite: bool = True) -> bt.metagraph:
        """Get the metagraph, with caching.

        Refreshes every _metagraph_ttl seconds unless force_refresh is True.
        Cache hits are lock-free (read-only); only the actual chain refresh
        is serialized through `_lock`.

        `lite=True` (default) skips heavy extras (bonds, weights, identities,
        emissions_split). On bt v10 the non-lite path makes ~10 sequential
        RPC calls and can hang on slow chain connections. We don't use any
        of those fields anywhere; lite has hotkeys/incentive/emission/stake/
        validator_trust/dividends/consensus which is what we need.
        """
        now = time.time()
        needs_refresh = (
            self._metagraph is None
            or force_refresh
            or now - self._metagraph_updated_at > self._metagraph_ttl
        )
        if needs_refresh:
            with self._lock:
                # Double-check inside the lock — another waiter may have refreshed.
                now = time.time()
                if (
                    self._metagraph is None
                    or force_refresh
                    or now - self._metagraph_updated_at > self._metagraph_ttl
                ):
                    self._metagraph = self.subtensor.metagraph(netuid=self.netuid, lite=lite)
                    self._metagraph_updated_at = now
                    self._hotkey_uid_index = {
                        hk: uid for uid, hk in enumerate(getattr(self._metagraph, "hotkeys", []))
                    }
                    self._hotkey_uid_index_at = self._metagraph_updated_at
                    bt.logging.debug(f"Metagraph refreshed ({self._metagraph.n.item()} neurons, lite={lite})")
        return self._metagraph

    def get_uid_for_hotkey(self, hotkey_ss58: str) -> Optional[int]:
        """Resolve a hotkey SS58 address to a UID on this subnet."""
        mg = self.get_metagraph()
        if self._hotkey_uid_index_at != self._metagraph_updated_at:
            self._hotkey_uid_index = {
                hk: uid for uid, hk in enumerate(getattr(mg, "hotkeys", []))
            }
            self._hotkey_uid_index_at = self._metagraph_updated_at
        return self._hotkey_uid_index.get(hotkey_ss58)

    def get_cached_uid_for_hotkey(
        self,
        hotkey_ss58: str,
        *,
        max_age_s: float | None = None,
    ) -> tuple[bool, Optional[int]]:
        """Return a cached hotkey->UID lookup without refreshing metagraph.

        The boolean is True only when the local metagraph index is populated
        and fresh enough for the caller. A True/None result means the fresh
        index proves the hotkey is not currently registered on the subnet.
        """
        if not self._hotkey_uid_index_at:
            return False, None
        if max_age_s is None:
            max_age_s = self._metagraph_ttl
        try:
            if float(max_age_s) >= 0 and time.time() - self._hotkey_uid_index_at > float(max_age_s):
                return False, None
        except Exception:
            return False, None
        return True, self._hotkey_uid_index.get(hotkey_ss58)

    def get_subnet_owner_uid(self) -> Optional[int]:
        """Resolve the subnet owner hotkey to its UID on this subnet.

        This is the emission-redirect target used by the validator weight loop.
        It is cached because the owner hotkey is subnet-level metadata and does
        not need to be re-read every weight-setting attempt.
        """
        if self._subnet_owner_uid is not None:
            return self._subnet_owner_uid
        with self._lock:
            if self._subnet_owner_hotkey is None:
                getter = getattr(self.subtensor, "get_subnet_owner_hotkey", None)
                if getter is None:
                    return None
                self._subnet_owner_hotkey = getter(self.netuid)
        uid = self.get_uid_for_hotkey(self._subnet_owner_hotkey)
        if uid is not None:
            self._subnet_owner_uid = int(uid)
            bt.logging.info(
                f"Subnet owner UID resolved: {self._subnet_owner_uid} "
                f"({self._subnet_owner_hotkey})"
            )
        return self._subnet_owner_uid

    def is_validator(self, uid: int) -> bool:
        """Check if a UID has validator permit."""
        mg = self.get_metagraph()
        return bool(mg.validator_permit[uid])

    def set_weights_result(
        self,
        wallet,
        uids: list[int],
        weights: list[float],
        version_key: int = 0,
        wait_for_inclusion: bool = True,
    ) -> tuple[bool, str]:
        """Set weights on the Bittensor chain.

        This is the core action that distributes emission to miners.
        Called every tempo (~360 blocks / 72 min).

        Args:
            wallet: Validator's Bittensor wallet.
            uids: List of UIDs to set weights for.
            weights: Corresponding weight values (will be normalized).
            version_key: Software version for compatibility gating.
            wait_for_inclusion: Wait for the extrinsic to be included in a block.
        """
        import torch
        with self._lock:
            success, msg = self.subtensor.set_weights(
                wallet=wallet,
                netuid=self.netuid,
                uids=torch.tensor(uids, dtype=torch.int64),
                weights=torch.tensor(weights, dtype=torch.float32),
                version_key=version_key,
                wait_for_inclusion=wait_for_inclusion,
            )
        return bool(success), str(msg or "")

    def set_weights(
        self,
        wallet,
        uids: list[int],
        weights: list[float],
        version_key: int = 0,
        wait_for_inclusion: bool = True,
    ) -> bool:
        """Set weights on the Bittensor chain and return success only."""
        success, msg = self.set_weights_result(
            wallet=wallet,
            uids=uids,
            weights=weights,
            version_key=version_key,
            wait_for_inclusion=wait_for_inclusion,
        )
        if success:
            bt.logging.info(f"Weights set successfully for {len(uids)} UIDs")
        else:
            bt.logging.error(f"Failed to set weights: {msg}")
        return success
