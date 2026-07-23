"""Shared test setup.

``neurons.miner.services.hardware_service`` imports ``bittensor`` at module
import time and is not installable in this test environment. Install a
minimal stub here rather than inside a test module, so that the stub is
registered once during collection instead of depending on which test file
imports first.
"""
from __future__ import annotations

import sys
import types


def _install_bittensor_stub() -> None:
    if "bittensor" in sys.modules:
        return
    bt = types.ModuleType("bittensor")
    bt.logging = types.SimpleNamespace(
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        trace=lambda *a, **k: None,
        success=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    sys.modules["bittensor"] = bt


_install_bittensor_stub()
