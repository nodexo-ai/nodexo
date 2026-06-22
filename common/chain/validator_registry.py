"""
Python client for the ValidatorRegistry contract.

Used by executors to discover validator endpoints for proof broadcasting.
Used by validators to register their own proxy endpoints.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Optional

import bittensor as bt
from web3 import Web3
from web3.contract import Contract

from common.chain.evm_rpc import make_http_provider
from common.config import ChainConfig
from common.types import ValidatorEndpoint

_ABI_SEARCH_PATHS = [
    Path(__file__).parent.parent.parent / "contracts" / "out" / "ValidatorRegistry.sol" / "ValidatorRegistry.json",
    Path(__file__).parent.parent.parent / "abis" / "ValidatorRegistry.json",
]

_abi_cache: list | None = None
_RPC_THROTTLE_LOCK = threading.Lock()
_RPC_THROTTLE_LAST_AT = 0.0


def _rpc_min_interval_s() -> float:
    raw = os.environ.get(
        "VALIDATOR_REGISTRY_RPC_MIN_INTERVAL_S",
        os.environ.get("EVM_RPC_MIN_INTERVAL_S", "1.05"),
    )
    try:
        return max(0.0, float(raw))
    except Exception:
        return 1.05


def _rpc_retries() -> int:
    raw = os.environ.get("VALIDATOR_REGISTRY_RPC_RETRIES", "5")
    try:
        return max(1, int(raw))
    except Exception:
        return 5


def _is_transient_rpc_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "429" in msg
        or "too many requests" in msg.lower()
        or "timeout" in msg.lower()
        or "connection" in msg.lower()
    )


def _tx_hash_hex(value) -> str:
    if isinstance(value, str):
        return value if value.startswith("0x") else f"0x{value}"
    try:
        return Web3.to_hex(value)
    except Exception:
        raw = value.hex() if hasattr(value, "hex") else str(value)
        return raw if raw.startswith("0x") else f"0x{raw}"


def _load_abi() -> list:
    global _abi_cache
    if _abi_cache is not None:
        return _abi_cache
    for path in _ABI_SEARCH_PATHS:
        if path.exists():
            data = json.loads(path.read_text())
            _abi_cache = data.get("abi", data) if isinstance(data, dict) else data
            return _abi_cache
    raise FileNotFoundError("ValidatorRegistry ABI not found. Run 'forge build'.")


class ValidatorRegistryClient:
    """Python client for the ValidatorRegistry contract."""

    def __init__(self, config: ChainConfig, private_key: Optional[str] = None):
        self.config = config
        self.w3 = Web3(make_http_provider(config.rpc_url))
        self.contract: Contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(config.validator_registry_address),
            abi=_load_abi(),
        )
        self._private_key = private_key
        self._account = None
        if private_key:
            from eth_account import Account
            self._account = Account.from_key(private_key)

    @property
    def address(self) -> str:
        return self._account.address if self._account else ""

    def _rpc_call(self, fn, label: str = "rpc"):
        global _RPC_THROTTLE_LAST_AT
        last_exc: Exception | None = None
        for attempt in range(_rpc_retries()):
            try:
                with _RPC_THROTTLE_LOCK:
                    min_interval = _rpc_min_interval_s()
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
            except Exception as exc:
                last_exc = exc
                if not _is_transient_rpc_error(exc) or attempt + 1 >= _rpc_retries():
                    raise
                bt.logging.warning(
                    f"ValidatorRegistry {label} RPC transient failure; retrying: {exc}"
                )
                time.sleep(min(2 ** attempt, 8))
        raise last_exc or RuntimeError(f"ValidatorRegistry {label} RPC failed")

    def _send_and_wait(self, fn, gas: int, label: str = "") -> str:
        """Build, sign, send, wait for receipt. RAISES on revert (status==0).

        Without the status check the previous wrapper silently swallowed
        contract reverts, which left validators logging "Registered on
        ValidatorRegistry" while the on-chain row was still empty.
        """
        if not self._account:
            raise RuntimeError(
                f"{label or getattr(fn, 'fn_name', 'transaction')}: "
                "EVM private key not configured"
            )

        last_exc: Exception | None = None
        tx_hash = None
        for attempt in range(5):
            signed_hash = None
            try:
                gas_price = int(self._rpc_call(
                    lambda: self.w3.eth.gas_price,
                    f"{label}:gas_price",
                ))
                if attempt:
                    gas_price = int(gas_price * (1.15 ** attempt))
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
                        bt.logging.warning(
                            f"{label or getattr(fn, 'fn_name', 'tx')} already known by RPC "
                            "but no receipt is visible; retrying with gas bump"
                        )
                        time.sleep(2 ** attempt)
                        continue
                    tx_hash = signed_hash
                    break
                if not _is_transient_rpc_error(e) or attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        if tx_hash is None:
            raise last_exc or RuntimeError(f"{label}: failed to submit tx")

        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                receipt = self._rpc_call(
                    lambda: self.w3.eth.get_transaction_receipt(tx_hash),
                    f"{label}:get_transaction_receipt",
                )
                if receipt is None:
                    time.sleep(2)
                    continue
                tx_hex = _tx_hash_hex(receipt.transactionHash)
                if receipt.status != 1:
                    raise RuntimeError(f"{label or fn.fn_name} reverted (tx={tx_hex})")
                return tx_hex
            except RuntimeError:
                raise
            except Exception as e:
                msg = str(e)
                if "not found" in msg.lower() or _is_transient_rpc_error(e):
                    time.sleep(3)
                    continue
                time.sleep(2)

        submitted_hex = _tx_hash_hex(tx_hash)
        bt.logging.warning(
            f"{label or 'tx'} submitted as {submitted_hex} "
            "but receipt not received in time; it may still land."
        )
        raise RuntimeError(
            f"{label or 'tx'}: submitted (hash={submitted_hex}) "
            "but receipt timed out (RPC may be throttled)"
        )

    def is_evm_registered(self, address: str) -> bool:
        """Check whether an EVM address is bound to a UID on this registry."""
        return bool(self._rpc_call(
            lambda: self.contract.functions.evmRegistered(
                Web3.to_checksum_address(address)
            ).call(),
            "evmRegistered",
        ))

    def evm_uid(self, address: str) -> int:
        """Return the UID currently bound to an EVM address.

        The contract returns 0 for both "not registered" and UID 0, so callers
        must combine this with `is_evm_registered`.
        """
        return int(self._rpc_call(
            lambda: self.contract.functions.evmToUid(
                Web3.to_checksum_address(address)
            ).call(),
            "evmToUid",
        ))

    def evm_for_uid(self, uid: int) -> str:
        return self._rpc_call(
            lambda: self.contract.functions.uidToEvm(int(uid)).call(),
            "uidToEvm",
        )

    def is_evm_registered_for_uid(self, address: str, uid: int) -> bool:
        checksum = Web3.to_checksum_address(address)
        if not self.is_evm_registered(checksum):
            return False
        try:
            if self.evm_uid(checksum) != int(uid):
                return False
            return Web3.to_checksum_address(self.evm_for_uid(uid)) == checksum
        except Exception:
            return False

    def get_validator(self, address: str) -> ValidatorEndpoint:
        """Return a validator's registry row, active or inactive."""
        checksum = Web3.to_checksum_address(address)
        row = self._rpc_call(
            lambda: self.contract.functions.getValidator(checksum).call(),
            "getValidator",
        )
        return ValidatorEndpoint(
            address=checksum,
            proxy_endpoint=str(row[0] or ""),
            uid=int(row[1] or 0),
            is_active=bool(row[4]),
        )

    def get_active_validators(
        self,
        *,
        raise_on_error: bool = False,
    ) -> list[ValidatorEndpoint]:
        """Get all active validators with their endpoints.

        This is what executors call to know where to broadcast proofs.
        """
        try:
            addrs, endpoints, uids = self._rpc_call(
                lambda: self.contract.functions.getActiveValidators().call(),
                "getActiveValidators",
            )
            validators = []
            for addr, ep, uid in zip(addrs, endpoints, uids):
                validators.append(ValidatorEndpoint(
                    address=addr,
                    proxy_endpoint=ep,
                    uid=uid,
                    is_active=True,
                ))
            return validators
        except Exception as e:
            bt.logging.error(f"Failed to get active validators: {e}")
            if raise_on_error:
                raise
            return []

    def register_evm(self, uid: int, sig_r: bytes, sig_s: bytes) -> str:
        return self._send_and_wait(
            self.contract.functions.registerEvm(uid, sig_r, sig_s),
            gas=300_000,
            label="validatorRegisterEvm",
        )

    def register(self, proxy_endpoint: str) -> str:
        return self._send_and_wait(
            self.contract.functions.register(proxy_endpoint),
            gas=300_000,
            label="validatorRegister",
        )

    def update_endpoint(self, new_endpoint: str) -> str:
        return self._send_and_wait(
            self.contract.functions.updateEndpoint(new_endpoint),
            gas=200_000,
            label="validatorUpdateEndpoint",
        )

    def deactivate(self) -> str:
        """Deactivate this validator's own registry row."""
        return self._send_and_wait(
            self.contract.functions.deactivate(),
            gas=120_000,
            label="validatorDeactivate",
        )

    def deactivate_validator(self, address: str) -> str:
        """Owner/admin deactivation for a validator registry row.

        Requires a ValidatorRegistry implementation that exposes
        deactivateValidator(address); older deployments only support
        self-deactivation through deactivate().
        """
        checksum = Web3.to_checksum_address(address)
        return self._send_and_wait(
            self.contract.functions.deactivateValidator(checksum),
            gas=140_000,
            label="validatorDeactivateValidator",
        )
