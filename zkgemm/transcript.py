"""
Fiat-Shamir transcript for non-interactive proofs.

Converts an interactive protocol into a non-interactive one by replacing
the verifier's random challenges with deterministic hashes of the
protocol transcript so far.

Uses keccak256 (EVM-native) with a running hash state for efficient
on-chain verification.  Each absorb/squeeze updates a fixed 32-byte
state — no growing buffer.
"""

from __future__ import annotations

import struct

from Crypto.Hash import keccak as _keccak_mod

from zkgemm.field import P


def _keccak256(data: bytes) -> bytes:
    """EVM-compatible keccak256 (NOT NIST SHA3-256)."""
    h = _keccak_mod.new(digest_bits=256)
    h.update(data)
    return h.digest()


class Transcript:
    """Append-only Fiat-Shamir transcript with running keccak256 hash.

    Both prover and verifier maintain identical transcripts.
    Absorb values in the same order, squeeze challenges at the same points,
    and the derived challenges will match.

    Uses a fixed 32-byte running state (hash chaining) instead of an
    accumulating buffer.  This maps efficiently to Solidity where
    keccak256 is a native opcode (30 + 6*words gas) vs SHA-256 precompile
    (60 + 12*words + staticcall overhead).
    """

    def __init__(self, label: bytes = b"zkgemm-v1"):
        self._state: bytes = _keccak256(label)

    def absorb(self, label: bytes, value: int) -> None:
        """Absorb a labeled field element."""
        self._state = _keccak256(
            self._state + label + struct.pack("<Q", value)
        )

    def absorb_bytes(self, label: bytes, data: bytes) -> None:
        """Absorb raw bytes (e.g. a hash digest)."""
        self._state = _keccak256(self._state + label + data)

    def absorb_many(self, label: bytes, values: list[int]) -> None:
        """Absorb multiple field elements with a length prefix."""
        parts = [self._state, label, struct.pack("<I", len(values))]
        for v in values:
            parts.append(struct.pack("<Q", v))
        self._state = _keccak256(b"".join(parts))

    def squeeze(self, label: bytes) -> int:
        """Derive a challenge from the current transcript state.

        Hashes state || "squeeze:" || label to get a digest,
        feeds that digest back into the state (for chaining),
        and returns the first 8 bytes reduced modulo P.
        """
        digest = _keccak256(self._state + b"squeeze:" + label)
        # Feed digest back for chaining so successive squeezes differ
        self._state = _keccak256(self._state + digest)
        raw = int.from_bytes(digest[:8], "little")
        return raw % P

    def squeeze_n(self, label: bytes, n: int) -> list[int]:
        """Squeeze n independent challenges."""
        return [self.squeeze(label + str(i).encode()) for i in range(n)]
