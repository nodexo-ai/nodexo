"""
Nodexo — Python client for the ComputeRegistry contract.

Handles EVM registration, executor management, heartbeats, and discovery.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Callable, Optional

import bittensor as bt
from web3 import Web3
from web3.contract import Contract

from common.config import ChainConfig
from common.chain.evm_rpc import make_http_provider
from common.types import ExecutorSpec

# ABI search paths (Foundry build output or checked-in ABI file)
_ABI_SEARCH_PATHS = [
    Path(__file__).parent.parent.parent / "contracts" / "out" / "ComputeRegistry.sol" / "ComputeRegistry.json",
    Path(__file__).parent.parent.parent / "abis" / "ComputeRegistry.json",
    Path(__file__).parent / "abis" / "ComputeRegistry.json",
]

_abi_cache: list | None = None
_RPC_THROTTLE_LOCK = threading.Lock()
_RPC_THROTTLE_LAST_AT = 0.0


class SubmittedTransactionTimeout(RuntimeError):
    def __init__(self, label: str, tx_hash: str, tx_block: int = 0):
        self.label = label or "tx"
        self.tx_hash = tx_hash
        self.tx_block = int(tx_block or 0)
        block_detail = f", block={self.tx_block}" if self.tx_block else ""
        super().__init__(
            f"{self.label}: submitted (hash={tx_hash}{block_detail}) "
            "but receipt timed out"
        )

_METAGRAPH_ABI = [
    {
        "inputs": [
            {"internalType": "uint16", "name": "netuid", "type": "uint16"},
            {"internalType": "uint16", "name": "uid", "type": "uint16"},
        ],
        "name": "getHotkey",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function",
    },
]


def _rpc_min_interval_s() -> float:
    """Minimum spacing between EVM JSON-RPC calls in this process.

    Multiple ComputeRegistryClient instances live inside one validator process,
    so any configured throttle is process-wide.
    """
    raw = os.environ.get(
        "COMPUTE_REGISTRY_RPC_MIN_INTERVAL_S",
        os.environ.get("EVM_RPC_MIN_INTERVAL_S", "0"),
    )
    try:
        return max(0.0, float(raw))
    except Exception:
        return 0.0


def _view_rpc_retries() -> int:
    raw = os.environ.get("COMPUTE_REGISTRY_VIEW_RPC_RETRIES", "4")
    try:
        return max(1, int(raw))
    except Exception:
        return 4


def _replacement_gas_price_floor_wei() -> int:
    """Minimum legacy gas price for replacing an invisible known transaction."""
    raw = os.environ.get("EVM_TX_REPLACEMENT_MIN_GWEI", "30")
    try:
        return max(0, int(float(raw) * 1_000_000_000))
    except Exception:
        return 30_000_000_000


def _is_transient_rpc_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "429" in msg
        or "too many requests" in msg.lower()
        or "timeout" in msg.lower()
        or "connection" in msg.lower()
    )


def _load_abi() -> list:
    """Load contract ABI from Foundry build output or checked-in ABI file."""
    global _abi_cache
    if _abi_cache is not None:
        return _abi_cache

    for path in _ABI_SEARCH_PATHS:
        if path.exists():
            data = json.loads(path.read_text())
            _abi_cache = data.get("abi", data) if isinstance(data, dict) else data
            return _abi_cache

    raise FileNotFoundError(
        "ComputeRegistry ABI not found. Run 'forge build' in contracts/ "
        "or place the ABI in abis/ComputeRegistry.json"
    )


def _tx_hash_hex(value) -> str:
    """Return a tx hash as a normalized 0x-prefixed hex string."""
    if isinstance(value, str):
        return value if value.startswith("0x") else f"0x{value}"
    try:
        return Web3.to_hex(value)
    except Exception:
        raw = value.hex() if hasattr(value, "hex") else str(value)
        return raw if raw.startswith("0x") else f"0x{raw}"


def _bytes32_topic(value: bytes) -> str:
    return "0x" + bytes(value).hex().rjust(64, "0")


def _address_topic(value: str) -> str:
    raw = Web3.to_checksum_address(value).removeprefix("0x").lower()
    return "0x" + raw.rjust(64, "0")


class ComputeRegistryClient:
    """Python client for interacting with the ComputeRegistry contract."""

    def __init__(self, config: ChainConfig, private_key: Optional[str] = None):
        self.config = config
        self.w3 = Web3(make_http_provider(config.rpc_url))
        self.contract: Contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(config.compute_registry_address),
            abi=_load_abi(),
        )
        self._batch_view_disabled_until: float = 0.0
        self._active_view_disabled_until: float = 0.0
        self._last_executor_total: int = 0
        self._confirmed_tx_blocks: dict[str, int] = {}
        self._private_key = private_key
        self._account = None
        if private_key:
            from eth_account import Account
            self._account = Account.from_key(private_key)

    def _rpc_call(self, fn, label: str = "rpc"):
        global _RPC_THROTTLE_LAST_AT
        min_interval = _rpc_min_interval_s()
        with _RPC_THROTTLE_LOCK:
            if min_interval > 0:
                now = time.monotonic()
                wait = (_RPC_THROTTLE_LAST_AT + min_interval) - now
                if wait > 0:
                    time.sleep(wait)
            try:
                return fn()
            finally:
                if min_interval > 0:
                    _RPC_THROTTLE_LAST_AT = time.monotonic()

    @property
    def address(self) -> str:
        return self._account.address if self._account else ""

    def _send_and_wait(
        self,
        fn,
        gas: int,
        label: str = "",
        *,
        event_signature: str | None = None,
        indexed_topics: list[str] | None = None,
        state_confirm: Callable[[], bool] | None = None,
    ) -> str:
        """Build, sign, send, wait. RAISES on revert (status != 1).

        Two-phase retry:
          (a) BEFORE submit (build / nonce / send_raw_transaction): retry
              on 429 / transport errors. No tx has reached the chain yet,
              so retrying is safe.
          (b) AFTER submit (wait_for_transaction_receipt): poll-only retry
              on 429 — we already have a tx_hash and re-sending would
              create a duplicate tx, which can land in addition to the
              first one. Just keep polling until the receipt arrives or
              operation-specific confirmation succeeds.
        """
        import time as _time

        if not self._account:
            raise RuntimeError(
                f"{label or getattr(fn, 'fn_name', 'transaction')}: "
                "EVM private key not configured"
            )

        # Phase a: get to a submitted tx_hash, with retries.
        last_exc: Exception | None = None
        tx_hash = None
        replacement_gas_floor = 0
        for attempt in range(5):
            signed_hash = None
            try:
                gas_price = int(self._rpc_call(
                    lambda: self.w3.eth.gas_price,
                    f"{label}:gas_price",
                ))
                if attempt:
                    # Same nonce + higher gas replaces a stuck tx if one is
                    # actually pending; if the node merely cached a dropped raw
                    # tx, the changed gas price also changes the raw hash.
                    gas_price = int(gas_price * (1.15 ** attempt))
                if replacement_gas_floor:
                    gas_price = max(gas_price, replacement_gas_floor)
                tx = fn.build_transaction({
                    "from": self.address,
                    "nonce": self._rpc_call(
                        lambda: self.w3.eth.get_transaction_count(self.address, "pending"),
                        f"{label}:nonce",
                    ),
                    "gas": gas,
                    "gasPrice": gas_price,
                })
                signed = self._account.sign_transaction(tx)
                raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
                signed_hash = getattr(signed, "hash", None) or Web3.keccak(raw)
                tx_hash = self._rpc_call(
                    lambda: self.w3.eth.send_raw_transaction(raw),
                    f"{label}:send_raw_transaction",
                )
                break
            except Exception as e:
                last_exc = e
                msg = str(e)
                already_known = "already known" in msg.lower() or "known transaction" in msg.lower()
                if already_known and signed_hash is not None:
                    try:
                        receipt = self._rpc_call(
                            lambda: self.w3.eth.get_transaction_receipt(signed_hash),
                            f"{label}:known_tx_receipt",
                        )
                        if receipt is not None:
                            tx_hex = _tx_hash_hex(receipt.transactionHash)
                            if receipt.status != 1:
                                raise RuntimeError(f"{label or fn.fn_name} reverted (tx={tx_hex})")
                            return tx_hex
                    except RuntimeError:
                        raise
                    except Exception:
                        pass
                    if attempt < 4:
                        replacement_gas_floor = _replacement_gas_price_floor_wei()
                        bt.logging.warning(
                            f"{label or getattr(fn, 'fn_name', 'tx')} already known by RPC "
                            "but no receipt is visible; retrying with gas bump"
                        )
                        _time.sleep(2 ** attempt)
                        continue
                    bt.logging.warning(
                        f"{label or getattr(fn, 'fn_name', 'tx')} already known by RPC; "
                        "polling signed transaction hash instead"
                    )
                    tx_hash = signed_hash
                    break
                transient = ("429" in msg or "Too Many Requests" in msg or
                             "timeout" in msg.lower() or "Connection" in msg)
                if not transient or attempt == 4:
                    raise
                _time.sleep(2 ** attempt)
        if tx_hash is None:
            raise last_exc or RuntimeError(f"{label}: failed to submit tx")

        # Phase b: poll for receipt and, for state-changing rental calls,
        # accept the exact emitted event plus final contract state as
        # confirmation. Some subtensor node modes expose logs before receipts.
        raw_timeout = os.environ.get("EVM_TX_CONFIRM_TIMEOUT_S", "45")
        try:
            timeout_s = max(10.0, float(raw_timeout))
        except Exception:
            timeout_s = 45.0
        deadline = _time.time() + timeout_s
        submitted_hex = _tx_hash_hex(tx_hash)
        while _time.time() < deadline:
            try:
                receipt = self._rpc_call(
                    lambda: self.w3.eth.get_transaction_receipt(tx_hash),
                    f"{label}:get_transaction_receipt",
                )
                if receipt is None:
                    pass
                else:
                    tx_hex = _tx_hash_hex(receipt.transactionHash)
                    if receipt.status != 1:
                        raise RuntimeError(f"{label or fn.fn_name} reverted (tx={tx_hex})")
                    return tx_hex
            except RuntimeError:
                raise
            except Exception:
                pass
            if event_signature and indexed_topics is not None and state_confirm is not None:
                tx_block = self.transaction_event_block(
                    submitted_hex,
                    event_signature,
                    indexed_topics,
                    lookback_blocks=96,
                )
                if tx_block > 0:
                    try:
                        if state_confirm():
                            self._confirmed_tx_blocks[submitted_hex.lower()] = tx_block
                            bt.logging.info(
                                f"{label or 'tx'} event-confirmed "
                                f"(tx={submitted_hex}, block={tx_block})"
                            )
                            return submitted_hex
                    except Exception:
                        pass
            _time.sleep(2)
        # Timed out waiting. The tx may still land, and some subtensor node
        # modes expose the containing EVM block before they expose a
        # transaction receipt. Surface the hash as structured data so
        # operation-specific callers can confirm the resulting contract state.
        bt.logging.warning(
            f"{label or 'tx'} submitted as {submitted_hex} "
            "but receipt not received in time."
        )
        raise SubmittedTransactionTimeout(label or "tx", submitted_hex, tx_block=0)

    # ── EVM Registration ───────────────────────────────────────────

    def is_evm_registered(self, address: str) -> bool:
        return self._rpc_call(
            lambda: self.contract.functions.evmRegistered(
                Web3.to_checksum_address(address)
            ).call(),
            "evmRegistered",
        )

    def register_evm(self, uid: int, sig_r: bytes, sig_s: bytes) -> str:
        """Register EVM address with SR25519 hotkey ownership proof."""
        return self._send_and_wait(
            self.contract.functions.registerEvm(uid, sig_r, sig_s),
            gas=300_000, label="registerEvm",
        )

    # ── Executor Registration ──────────────────────────────────────

    def register_executor(
        self,
        executor_id: bytes,
        endpoint: str,
        gpu_model_hash: bytes,
        gpu_count: int,
        vram_mb: int,
        price_per_gpu_hour: int,
    ) -> str:
        return self._send_and_wait(
            self.contract.functions.registerExecutor(
                executor_id, endpoint, gpu_model_hash, gpu_count, vram_mb, price_per_gpu_hour,
            ),
            gas=500_000, label="registerExecutor",
        )

    def update_endpoint(self, executor_id: bytes, new_endpoint: str) -> str:
        """Update the on-chain endpoint for an already-registered executor."""
        return self._send_and_wait(
            self.contract.functions.updateEndpoint(executor_id, new_endpoint),
            gas=200_000, label="updateEndpoint",
        )

    def deregister_executor(self, executor_id: bytes) -> str:
        return self._send_and_wait(
            self.contract.functions.deregisterExecutor(executor_id),
            gas=200_000, label="deregisterExecutor",
        )

    # ── Heartbeat ──────────────────────────────────────────────────

    def renew_executor(self, executor_id: bytes) -> str:
        return self._send_and_wait(
            self.contract.functions.renewExecutor(executor_id),
            gas=100_000, label="renewExecutor",
        )

    # ── Rental state ───────────────────────────────────────────────

    def mark_rented(self, executor_id: bytes) -> str:
        try:
            return self._send_and_wait(
                self.contract.functions.markRented(executor_id),
                gas=150_000, label="markRented",
                event_signature="RentalStarted(bytes32,address)",
                indexed_topics=[_bytes32_topic(executor_id), _address_topic(self.address)],
                state_confirm=lambda: bool(
                    (spec := self.get_executor_info(executor_id)) is not None
                    and spec.is_rented
                ),
            )
        except SubmittedTransactionTimeout as e:
            tx_block = self.transaction_event_block(
                e.tx_hash,
                "RentalStarted(bytes32,address)",
                [_bytes32_topic(executor_id), _address_topic(self.address)],
            )
            spec = self.get_executor_info(executor_id)
            if tx_block > 0 and spec is not None and spec.is_rented:
                self._confirmed_tx_blocks[_tx_hash_hex(e.tx_hash).lower()] = tx_block
                bt.logging.warning(
                    f"markRented state-confirmed after receipt timeout "
                    f"(tx={e.tx_hash}, block={tx_block})"
                )
                return e.tx_hash
            raise

    def mark_available(self, executor_id: bytes) -> str:
        try:
            return self._send_and_wait(
                self.contract.functions.markAvailable(executor_id),
                gas=150_000, label="markAvailable",
                event_signature="RentalEnded(bytes32,address)",
                indexed_topics=[_bytes32_topic(executor_id), _address_topic(self.address)],
                state_confirm=lambda: bool(
                    (spec := self.get_executor_info(executor_id)) is not None
                    and not spec.is_rented
                ),
            )
        except SubmittedTransactionTimeout as e:
            tx_block = self.transaction_event_block(
                e.tx_hash,
                "RentalEnded(bytes32,address)",
                [_bytes32_topic(executor_id), _address_topic(self.address)],
            )
            spec = self.get_executor_info(executor_id)
            if tx_block > 0 and spec is not None and not spec.is_rented:
                self._confirmed_tx_blocks[_tx_hash_hex(e.tx_hash).lower()] = tx_block
                bt.logging.warning(
                    f"markAvailable state-confirmed after receipt timeout "
                    f"(tx={e.tx_hash}, block={tx_block})"
                )
                return e.tx_hash
            raise

    # ── Discovery (read-only) ──────────────────────────────────────

    @staticmethod
    def _bytes32_hex(value) -> str:
        if isinstance(value, str):
            return value.removeprefix("0x")
        if isinstance(value, (bytes, bytearray)):
            return bytes(value).hex()
        return bytes(value).hex()

    @staticmethod
    def _row_get(row, idx: int, name: str):
        if isinstance(row, dict):
            return row[name]
        if hasattr(row, name):
            return getattr(row, name)
        return row[idx]

    def get_executor_info(self, executor_id: bytes) -> Optional[ExecutorSpec]:
        try:
            result = self._rpc_call(
                lambda: self.contract.functions.getExecutorInfo(executor_id).call(),
                "getExecutorInfo",
            )
            miner_address = result[0]
            miner_uid: int | None = None
            miner_registered: bool | None = None
            uid_owner_match: bool | None = None
            try:
                owner = Web3.to_checksum_address(miner_address)
                miner_registered = bool(
                    self._rpc_call(
                        lambda: self.contract.functions.evmRegistered(owner).call(),
                        "evmRegistered",
                    )
                )
                if miner_registered:
                    miner_uid = int(self._rpc_call(
                        lambda: self.contract.functions.evmToUid(owner).call(),
                        "evmToUid",
                    ))
                    uid_evm = self._rpc_call(
                        lambda: self.contract.functions.uidToEvm(miner_uid).call(),
                        "uidToEvm",
                    )
                    uid_owner_match = str(uid_evm).lower() == str(owner).lower()
            except Exception as e:
                bt.logging.debug(
                    f"get_executor_info identity metadata unavailable for "
                    f"{executor_id.hex()[:16]}: {e}"
                )
            return ExecutorSpec(
                executor_id=executor_id.hex(),
                miner_address=miner_address,
                endpoint=result[1],
                gpu_model_hash=result[2].hex(),
                gpu_count=result[3],
                vram_mb=result[4],
                price_per_gpu_hour=result[5],
                expires_at=result[6],
                is_active=result[7],
                is_rented=result[8],
                miner_uid=miner_uid,
                miner_registered=miner_registered,
                uid_owner_match=uid_owner_match,
            )
        except Exception as e:
            if _is_transient_rpc_error(e):
                raise
            return None

    def get_executor_count(self) -> int:
        return self._rpc_call(
            lambda: self.contract.functions.getExecutorCount().call(),
            "getExecutorCount",
        )

    def block_number(self) -> int:
        return int(self._rpc_call(lambda: self.w3.eth.block_number, "eth_blockNumber"))

    def transaction_receipt(self, tx_hash: str):
        return self._rpc_call(
            lambda: self.w3.eth.get_transaction_receipt(tx_hash),
            "getTransactionReceipt",
        )

    def confirmed_transaction_block(self, tx_hash: str) -> int:
        return int(self._confirmed_tx_blocks.get(_tx_hash_hex(tx_hash).lower(), 0))

    def transaction_event_block(
        self,
        tx_hash: str,
        event_signature: str,
        indexed_topics: list[str],
        lookback_blocks: int = 256,
    ) -> int:
        """Find an expected event for a submitted tx in one logs query."""
        target = _tx_hash_hex(tx_hash).lower()
        try:
            head = self.block_number()
        except Exception:
            return 0
        topic0 = Web3.to_hex(Web3.keccak(text=event_signature))
        topics = [topic0, *indexed_topics]
        from_block = max(0, head - max(1, int(lookback_blocks)))
        try:
            logs = self._rpc_call(
                lambda: self.w3.eth.get_logs({
                    "fromBlock": from_block,
                    "toBlock": head,
                    "address": self.contract.address,
                    "topics": topics,
                }),
                "eth_getLogs",
            )
        except Exception:
            return 0
        for log in logs or []:
            try:
                log_tx = log.get("transactionHash", "")
            except Exception:
                log_tx = getattr(log, "transactionHash", "")
            if _tx_hash_hex(log_tx).lower() == target:
                try:
                    return int(log.get("blockNumber", 0))
                except Exception:
                    return int(getattr(log, "blockNumber", 0) or 0)
        return 0

    def transaction_has_event(
        self,
        tx_hash: str,
        block_number: int,
        event_signature: str,
        indexed_topics: list[str],
    ) -> bool:
        if block_number <= 0:
            return False
        target = _tx_hash_hex(tx_hash).lower()
        topic0 = Web3.to_hex(Web3.keccak(text=event_signature))
        topics = [topic0, *indexed_topics]
        try:
            logs = self._rpc_call(
                lambda: self.w3.eth.get_logs({
                    "fromBlock": int(block_number),
                    "toBlock": int(block_number),
                    "address": self.contract.address,
                    "topics": topics,
                }),
                "eth_getLogs",
            )
        except Exception:
            return False
        for log in logs or []:
            try:
                if _tx_hash_hex(log.get("transactionHash", "")).lower() == target:
                    return True
            except Exception:
                if _tx_hash_hex(getattr(log, "transactionHash", "")).lower() == target:
                    return True
        return False

    def evm_to_uid(self, address: str) -> int:
        checksum = Web3.to_checksum_address(address)
        return int(self._rpc_call(
            lambda: self.contract.functions.evmToUid(checksum).call(),
            "evmToUid",
        ))

    def hotkey_for_uid(self, uid: int) -> bytes:
        """Return the current metagraph hotkey AccountId for a subnet UID.

        This reads the same Bittensor metagraph precompile that the registry
        contract uses for EVM registration, but from Python. Validators use it
        to prove proof-ingress hotkey ownership without refreshing the full
        Subtensor metagraph on the HTTP hot path.
        """
        last_exc: Exception | None = None
        for attempt in range(_view_rpc_retries()):
            try:
                meta_addr = self._rpc_call(lambda: self.contract.functions.META().call(), "META")
                meta = self.w3.eth.contract(address=Web3.to_checksum_address(meta_addr), abi=_METAGRAPH_ABI)
                value = self._rpc_call(
                    lambda: meta.functions.getHotkey(int(self.config.netuid), int(uid)).call(),
                    "Metagraph.getHotkey",
                )
                break
            except Exception as e:
                last_exc = e
                if not _is_transient_rpc_error(e) or attempt >= _view_rpc_retries() - 1:
                    raise
                time.sleep(min(2.0 * (attempt + 1), 6.0))
        else:
            raise last_exc or RuntimeError("Metagraph.getHotkey failed")
        if isinstance(value, str):
            return bytes.fromhex(value.removeprefix("0x"))
        return bytes(value)

    def get_active_executor_count(self) -> Optional[int]:
        """Return active-list count when the deployed registry supports it."""
        import time as _time

        if _time.time() < self._active_view_disabled_until:
            return None
        try:
            ready = bool(self._rpc_call(
                lambda: self.contract.functions.activeExecutorIndexReady().call(),
                "activeExecutorIndexReady",
            ))
            if not ready:
                return None
            return int(self._rpc_call(
                lambda: self.contract.functions.getActiveExecutorCount().call(),
                "getActiveExecutorCount",
            ))
        except Exception as e:
            self._active_view_disabled_until = _time.time() + 300
            bt.logging.debug(f"ComputeRegistry active executor view unavailable: {e}")
            return None

    def get_executor_ids_paginated(self, offset: int, limit: int) -> list[bytes]:
        return self._rpc_call(
            lambda: self.contract.functions.getExecutorIdsPaginated(offset, limit).call(),
            "getExecutorIdsPaginated",
        )

    def get_active_executors_paginated(self, offset: int, limit: int) -> Optional[list[ExecutorSpec]]:
        """Fetch active-listed executor details in one call when supported."""
        import time as _time

        if _time.time() < self._active_view_disabled_until:
            return None
        try:
            rows = self._rpc_call(
                lambda: self.contract.functions.getActiveExecutorsPaginated(offset, limit).call(),
                "getActiveExecutorsPaginated",
            )
        except Exception as e:
            self._active_view_disabled_until = _time.time() + 300
            bt.logging.debug(f"ComputeRegistry active executor view unavailable: {e}")
            return None
        return self._rows_to_specs(rows)

    def get_executors_paginated(self, offset: int, limit: int) -> Optional[list[ExecutorSpec]]:
        """Fetch a page of executor details in one call when the deployed
        registry supports the batched view.

        Older test deployments do not have this view yet. In that case we
        disable the batch attempt briefly and let get_all_active_executors()
        fall back to the legacy id+info path.
        """
        import time as _time

        if _time.time() < self._batch_view_disabled_until:
            return None
        try:
            rows = self._rpc_call(
                lambda: self.contract.functions.getExecutorsPaginated(offset, limit).call(),
                "getExecutorsPaginated",
            )
        except Exception as e:
            self._batch_view_disabled_until = _time.time() + 300
            bt.logging.debug(f"ComputeRegistry batch executor view unavailable: {e}")
            return None

        return self._rows_to_specs(rows)

    def _rows_to_specs(self, rows) -> list[ExecutorSpec]:
        specs: list[ExecutorSpec] = []
        for row in rows:
            owner = self._row_get(row, 0, "miner")
            executor_id = self._bytes32_hex(self._row_get(row, 1, "executorId"))
            if not owner or str(owner).lower() == "0x0000000000000000000000000000000000000000":
                continue
            specs.append(ExecutorSpec(
                executor_id=executor_id,
                miner_address=owner,
                endpoint=self._row_get(row, 2, "endpoint"),
                gpu_model_hash=self._bytes32_hex(self._row_get(row, 3, "gpuModelHash")),
                gpu_count=int(self._row_get(row, 4, "gpuCount")),
                vram_mb=int(self._row_get(row, 5, "vramMb")),
                price_per_gpu_hour=int(self._row_get(row, 6, "pricePerGpuHour")),
                expires_at=int(self._row_get(row, 7, "expiresAt")),
                is_active=bool(self._row_get(row, 8, "isActive")),
                is_rented=bool(self._row_get(row, 9, "isRented")),
                miner_uid=int(self._row_get(row, 10, "minerUid")),
                miner_registered=bool(self._row_get(row, 11, "minerRegistered")),
                uid_owner_match=bool(self._row_get(row, 12, "uidOwnerMatches")),
            ))
        return specs

    @property
    def last_executor_total(self) -> int:
        return self._last_executor_total

    def get_all_active_executors(self, page_size: int = 100) -> list[ExecutorSpec]:
        """Fetch all live executors (active flag set AND lease not expired).

        An executor is live only when the active flag is set and the heartbeat
        lease has not expired. The contract's `isActive` flag is sticky: only a
        `reportOffline` quorum or self-deregister flips it. Filtering on
        `expires_at > now` drops stale registrations from validator listings,
        scoring, and rentals.
        """
        import time as _time
        now = int(_time.time())
        active_total = self.get_active_executor_count()
        if active_total is not None:
            self._last_executor_total = active_total
            executors = []
            for offset in range(0, active_total, page_size):
                batch = self.get_active_executors_paginated(offset, page_size)
                if batch is None:
                    break
                for spec in batch:
                    if spec.is_active and spec.expires_at > now:
                        executors.append(spec)
            else:
                return executors

        total = self.get_executor_count()
        self._last_executor_total = total
        executors = []
        for offset in range(0, total, page_size):
            batch = self.get_executors_paginated(offset, page_size)
            if batch is not None:
                for spec in batch:
                    if spec.is_active and spec.expires_at > now:
                        executors.append(spec)
                continue
            ids = self.get_executor_ids_paginated(offset, page_size)
            for eid in ids:
                spec = self.get_executor_info(eid)
                if spec and spec.is_active and spec.expires_at > now:
                    executors.append(spec)
        return executors

    def is_rented(self, executor_id: bytes) -> bool:
        """Check if an executor is currently rented."""
        spec = self.get_executor_info(executor_id)
        return spec.is_rented if spec else False

    @staticmethod
    def _get_event_logs(event, from_block: int, to_block: int):
        """web3.py renamed event log kwargs between v6 and v7.

        The validator venv currently runs web3 v6 (`fromBlock`/`toBlock`),
        while newer environments use v7 (`from_block`/`to_block`). Keep the
        chain client compatible with both so rental timeline indexing does not
        depend on a specific minor dependency line.
        """
        import inspect

        params = inspect.signature(event.get_logs).parameters
        if "from_block" in params:
            return event.get_logs(
                from_block=int(from_block),
                to_block=int(to_block),
            )
        return event.get_logs(
            fromBlock=int(from_block),
            toBlock=int(to_block),
        )

    def get_rental_events(self, from_block: int, to_block: int) -> list[dict]:
        """Return RentalStarted/RentalEnded logs in block/log order.

        Validators use this to build a historical rental timeline for proof
        verification. That is strictly better than checking live `isRented`
        at verification time because proofs are verified several blocks after
        their beacon.
        """
        events: list[dict] = []
        for name, kind in (
            ("RentalStarted", "started"),
            ("RentalEnded", "ended"),
        ):
            event_cls = getattr(self.contract.events, name)
            for log in self._rpc_call(
                lambda: self._get_event_logs(event_cls(), from_block, to_block),
                f"{name}:get_logs",
            ):
                args = log.get("args", {})
                raw_eid = args.get("executorId")
                if isinstance(raw_eid, str):
                    executor_id = raw_eid.removeprefix("0x")
                else:
                    executor_id = bytes(raw_eid).hex()
                txh = log.get("transactionHash")
                events.append({
                    "kind": kind,
                    "executor_id": executor_id,
                    "validator": str(args.get("validator", "")),
                    "block_number": int(log.get("blockNumber", 0)),
                    "log_index": int(log.get("logIndex", 0)),
                    "tx_hash": txh.hex() if hasattr(txh, "hex") else str(txh or ""),
                })
        events.sort(key=lambda e: (e["block_number"], e["log_index"]))
        return events

    def get_registry_events(self, from_block: int, to_block: int) -> list[dict]:
        """Return ComputeRegistry lifecycle logs in block/log order.

        New deployments emit ExecutorStateChanged with the complete current
        row. Older deployments only emit narrow lifecycle events; those carry
        needs_info=True so the indexer can fetch getExecutorInfo once for
        sparse changes instead of polling the full registry on every API read.
        """
        events: list[dict] = []
        full_state_keys: set[tuple[str, str]] = set()

        def _event_cls(name: str):
            try:
                return getattr(self.contract.events, name)
            except Exception:
                return None

        state_cls = _event_cls("ExecutorStateChanged")
        if state_cls is not None:
            try:
                for log in self._rpc_call(
                    lambda: self._get_event_logs(state_cls(), from_block, to_block),
                    "ExecutorStateChanged:get_logs",
                ):
                    args = log.get("args", {})
                    raw_eid = args.get("executorId")
                    raw_hash = args.get("gpuModelHash")
                    txh = log.get("transactionHash")
                    tx_hex = txh.hex() if hasattr(txh, "hex") else str(txh or "")
                    executor_id = self._bytes32_hex(raw_eid)
                    full_state_keys.add((tx_hex, executor_id))
                    events.append({
                        "kind": "executor_state",
                        "executor_id": executor_id,
                        "miner": str(args.get("miner", "")),
                        "action": int(args.get("action") or 0),
                        "endpoint": str(args.get("endpoint", "")),
                        "gpu_model_hash": self._bytes32_hex(raw_hash),
                        "gpu_count": int(args.get("gpuCount") or 0),
                        "vram_mb": int(args.get("vramMb") or 0),
                        "price_per_gpu_hour": int(args.get("pricePerGpuHour") or 0),
                        "expires_at": int(args.get("expiresAt") or 0),
                        "is_active": bool(args.get("isActive")),
                        "is_rented": bool(args.get("isRented")),
                        "block_number": int(log.get("blockNumber", 0)),
                        "log_index": int(log.get("logIndex", 0)),
                        "tx_hash": tx_hex,
                    })
            except Exception as e:
                bt.logging.debug(f"ExecutorStateChanged log fetch failed: {e}")

        for name, kind, needs_info in (
            ("ExecutorRegistered", "executor_registered", True),
            ("ExecutorRenewed", "executor_renewed", True),
            ("ExecutorSpecsUpdated", "executor_specs_updated", True),
            ("ExecutorDeregistered", "executor_deregistered", False),
            ("ExecutorDeactivated", "executor_deactivated", False),
            ("RentalStarted", "rental_started", False),
            ("RentalEnded", "rental_ended", False),
        ):
            event_cls = _event_cls(name)
            if event_cls is None:
                continue
            try:
                logs = self._rpc_call(
                    lambda: self._get_event_logs(event_cls(), from_block, to_block),
                    f"{name}:get_logs",
                )
            except Exception as e:
                bt.logging.debug(f"{name} log fetch failed: {e}")
                continue
            for log in logs:
                args = log.get("args", {})
                raw_eid = args.get("executorId")
                txh = log.get("transactionHash")
                tx_hex = txh.hex() if hasattr(txh, "hex") else str(txh or "")
                executor_id = self._bytes32_hex(raw_eid)
                if kind in {
                    "executor_registered",
                    "executor_renewed",
                    "executor_specs_updated",
                    "executor_deactivated",
                } and (tx_hex, executor_id) in full_state_keys:
                    continue
                event = {
                    "kind": kind,
                    "executor_id": executor_id,
                    "block_number": int(log.get("blockNumber", 0)),
                    "log_index": int(log.get("logIndex", 0)),
                    "tx_hash": tx_hex,
                    "needs_info": needs_info,
                }
                if "miner" in args:
                    event["miner"] = str(args.get("miner", ""))
                if "validator" in args:
                    event["validator"] = str(args.get("validator", ""))
                if "reason" in args:
                    event["reason"] = str(args.get("reason", ""))
                events.append(event)

        events.sort(key=lambda e: (e["block_number"], e["log_index"]))
        return events

    # ── Offline reporting ──────────────────────────────────────────

    def report_offline(self, executor_id: bytes) -> str:
        return self._send_and_wait(
            self.contract.functions.reportOffline(executor_id),
            gas=150_000,
            label="reportOffline",
        )
