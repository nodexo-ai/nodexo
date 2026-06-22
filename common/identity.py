"""Endpoint identity challenge helpers."""

from __future__ import annotations

IDENTITY_PREFIX = b"NODEXO_IDENTITY_V1"


def normalize_evm_address(address: str) -> str:
    addr = (address or "").strip()
    if not addr.startswith("0x"):
        addr = "0x" + addr
    if len(addr) != 42:
        raise ValueError("EVM address must be 20 bytes")
    int(addr[2:], 16)
    return "0x" + addr[2:].lower()


def normalize_executor_id(executor_id: str) -> str:
    eid = (executor_id or "").strip().lower().removeprefix("0x")
    if len(eid) != 64:
        raise ValueError("executor_id must be 32 bytes")
    int(eid, 16)
    return eid


def build_identity_message(
    nonce: bytes,
    evm_address: str,
    executor_id: str,
) -> bytes:
    if len(nonce) != 32:
        raise ValueError("nonce must be 32 bytes")
    addr = normalize_evm_address(evm_address)
    eid = normalize_executor_id(executor_id)
    return (
        IDENTITY_PREFIX
        + nonce
        + bytes.fromhex(addr[2:])
        + bytes.fromhex(eid)
    )


def sign_identity_challenge(
    nonce: bytes,
    evm_address: str,
    executor_id: str,
    private_key: str,
) -> str:
    from eth_account import Account
    from eth_account.messages import encode_defunct

    msg = build_identity_message(nonce, evm_address, executor_id)
    signed = Account.sign_message(encode_defunct(primitive=msg), private_key)
    return signed.signature.hex()


def recover_identity_challenge(
    nonce: bytes,
    evm_address: str,
    executor_id: str,
    signature_hex: str,
) -> str:
    from eth_account import Account
    from eth_account.messages import encode_defunct

    msg = build_identity_message(nonce, evm_address, executor_id)
    sig = signature_hex[2:] if signature_hex.startswith("0x") else signature_hex
    recovered = Account.recover_message(
        encode_defunct(primitive=msg), signature=bytes.fromhex(sig),
    )
    return normalize_evm_address(recovered)
