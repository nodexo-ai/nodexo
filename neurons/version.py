"""Nodexo version constants.

Three independent versions gate protocol compatibility and role-specific
auto-updates:

``spec_version``
    On-chain weight gating. Validators pass this as ``version_key`` to
    ``subtensor.set_weights()``. The subnet owner sets the chain's
    ``weights_version`` hyperparameter to the same value for breaking
    protocol releases.

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


SPEC_MAJOR = 0
SPEC_MINOR = 1
SPEC_PATCH = 0

MINER_MAJOR = 0
MINER_MINOR = 1
MINER_PATCH = 0

VALIDATOR_MAJOR = 0
VALIDATOR_MINOR = 1
VALIDATOR_PATCH = 0

spec_version: int = _encode(SPEC_MAJOR, SPEC_MINOR, SPEC_PATCH)
miner_version: int = _encode(MINER_MAJOR, MINER_MINOR, MINER_PATCH)
validator_version: int = _encode(VALIDATOR_MAJOR, VALIDATOR_MINOR, VALIDATOR_PATCH)

version_str: str = _version_str(SPEC_MAJOR, SPEC_MINOR, SPEC_PATCH)
miner_version_str: str = _version_str(MINER_MAJOR, MINER_MINOR, MINER_PATCH)
validator_version_str: str = _version_str(
    VALIDATOR_MAJOR, VALIDATOR_MINOR, VALIDATOR_PATCH,
)

__version__ = version_str
