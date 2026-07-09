"""Nodexo version constants.

Three independent versions gate protocol compatibility and role-specific
auto-updates:

``spec_version``
    On-chain weight gating. Validators pass this as ``version_key`` to
    ``subtensor.set_weights()``. It tracks the highest role version so the
    chain's ``weights_version`` can gate any miner or validator release.

``miner_version``
    Miner-side release gate. Miners running with auto-update only restart
    when the remote ``miner_version`` is higher than the local one.

``validator_version``
    Validator-side release gate. Validators running with auto-update only
    restart when the remote ``validator_version`` is higher.

Encoding: MAJOR * 1_000_000 + MINOR * 1_000 + PATCH.
"""

from __future__ import annotations

_VERSION_BASE = 1_000


def _encode(major: int, minor: int, patch: int) -> int:
    return major * _VERSION_BASE * _VERSION_BASE + minor * _VERSION_BASE + patch


def _version_str(major: int, minor: int, patch: int) -> str:
    return f"{major}.{minor}.{patch}"


MINER_MAJOR = 0
MINER_MINOR = 1
MINER_PATCH = 4

VALIDATOR_MAJOR = 0
VALIDATOR_MINOR = 1
VALIDATOR_PATCH = 4

miner_version: int = _encode(MINER_MAJOR, MINER_MINOR, MINER_PATCH)
validator_version: int = _encode(VALIDATOR_MAJOR, VALIDATOR_MINOR, VALIDATOR_PATCH)
spec_version: int = max(miner_version, validator_version)

if validator_version >= miner_version:
    SPEC_MAJOR = VALIDATOR_MAJOR
    SPEC_MINOR = VALIDATOR_MINOR
    SPEC_PATCH = VALIDATOR_PATCH
else:
    SPEC_MAJOR = MINER_MAJOR
    SPEC_MINOR = MINER_MINOR
    SPEC_PATCH = MINER_PATCH

version_str: str = _version_str(SPEC_MAJOR, SPEC_MINOR, SPEC_PATCH)
miner_version_str: str = _version_str(MINER_MAJOR, MINER_MINOR, MINER_PATCH)
validator_version_str: str = _version_str(
    VALIDATOR_MAJOR, VALIDATOR_MINOR, VALIDATOR_PATCH,
)

__version__ = version_str
