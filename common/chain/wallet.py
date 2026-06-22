"""
Nodexo — EVM key derivation from Bittensor hotkey.

EVM private keys are derived deterministically from the Bittensor hotkey seed,
never stored in config files.
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from web3 import Web3

if TYPE_CHECKING:
    from eth_account.signers.local import LocalAccount


def sign_evm_registration(
    hotkey_seed: bytes,
    evm_address: str,
    uid: int,
    netuid: int,
    contract_address: str,
) -> tuple[bytes, bytes]:
    """Sign an EVM registration message with the SR25519 hotkey.

    The contract verifies this via the Sr25519Verify precompile (0x403).
    Message: keccak256(abi.encodePacked(msg.sender, uid, netuid, address(this)))

    """
    from substrateinterface import Keypair

    evm_bytes = bytes.fromhex(evm_address[2:])     # 20 bytes
    uid_bytes = uid.to_bytes(2, "big")              # uint16
    netuid_bytes = netuid.to_bytes(2, "big")        # uint16
    contract_bytes = bytes.fromhex(contract_address[2:])  # 20 bytes

    packed = evm_bytes + uid_bytes + netuid_bytes + contract_bytes
    message = Web3.keccak(packed)

    keypair = Keypair.create_from_seed(hotkey_seed[:32].hex())
    signature = keypair.sign(message)

    sig_bytes = signature if isinstance(signature, bytes) else bytes.fromhex(
        signature.replace("0x", "")
    )
    assert len(sig_bytes) == 64, f"Expected 64-byte SR25519 signature, got {len(sig_bytes)}"
    return sig_bytes[:32], sig_bytes[32:]


def derive_evm_private_key(hotkey_seed: bytes) -> str:
    """Derive an EVM private key from a Bittensor hotkey seed.

    Args:
        hotkey_seed: The raw 32-byte seed of the Bittensor hotkey.

    Returns:
        Hex-encoded EVM private key (without 0x prefix).
    """
    evm_pk = Web3.keccak(hotkey_seed)
    return evm_pk.hex()


def derive_evm_account(hotkey_seed: bytes) -> LocalAccount:
    """Derive a full EVM account (address + signing capability) from hotkey seed."""
    from eth_account import Account
    pk_hex = derive_evm_private_key(hotkey_seed)
    return Account.from_key(pk_hex)


def get_ss58_mirror_address(evm_address: str) -> str:
    """Compute the SS58 mirror address for an EVM address.

    This is the Substrate address to which you transfer TAO to fund
    the EVM wallet. Uses Substrate Frontier's HashedAddressMapping.

    mirror_account_id = blake2b_256(b"evm:" + h160_bytes)

    """
    from substrateinterface.utils.ss58 import ss58_encode

    h160 = bytes.fromhex(evm_address.lower().replace("0x", ""))
    assert len(h160) == 20, f"Invalid EVM address length: {len(h160)}"

    payload = b"evm:" + h160
    account_id = hashlib.blake2b(payload, digest_size=32).digest()
    return ss58_encode(account_id, ss58_format=42)


def load_hotkey_seed(wallet_name: str, hotkey_name: str = "default") -> bytes:
    """Load the hotkey seed from a Bittensor wallet on disk.

    Handles both old (substrateinterface) and new (bittensor-wallet Rust)
    Keypair implementations.

    Args:
        wallet_name: Bittensor wallet name (e.g., "miner").
        hotkey_name: Hotkey name within the wallet (e.g., "default").

    Returns:
        Raw 32-byte seed bytes.
    """
    import json
    from pathlib import Path

    # Direct file read approach — works regardless of bittensor version.
    # Hotkey files are JSON with "secretSeed" or the raw keypair data.
    bt_path = Path.home() / ".bittensor" / "wallets" / wallet_name / "hotkeys" / hotkey_name
    if bt_path.exists():
        try:
            data = json.loads(bt_path.read_text())
            # Standard bittensor keyfile format
            if "secretSeed" in data:
                seed_hex = data["secretSeed"]
                if seed_hex.startswith("0x"):
                    seed_hex = seed_hex[2:]
                return bytes.fromhex(seed_hex)
            # Encrypted keyfile — need bittensor to decrypt
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: use bittensor's wallet loader (handles encryption)
    import bittensor as bt
    wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
    keypair = wallet.hotkey

    # Try different attribute names across bittensor versions
    if hasattr(keypair, "private_key") and keypair.private_key is not None:
        return keypair.private_key[:32]
    if hasattr(keypair, "seed_hex"):
        return bytes.fromhex(keypair.seed_hex)
    if hasattr(keypair, "_seed"):
        return keypair._seed[:32]

    raise RuntimeError(
        f"Cannot extract hotkey seed from wallet '{wallet_name}/{hotkey_name}'. "
        "Ensure the hotkey file contains the private key (not just public key)."
    )
