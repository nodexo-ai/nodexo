"""
TDX (Intel Trust Domain Extensions) attestation scaffold.

DESIGN INTENT
=============

For H100 (or B100/B200) miners running on TDX-capable Intel CPUs
(Sapphire Rapids+ Xeon), enable end-to-end confidential compute:

  - Intel TDX seals the host kernel + container runtime inside a
    Confidential VM (CVM). The CPU's HSM signs an attestation quote
    over the VM's measurement.
  - NVIDIA Confidential Compute mode (H100+ with CC firmware)
    seals the GPU's VRAM. The GPU's hardware root of trust signs
    a separate attestation over the GPU identity + CC state.

Combined: a miner can prove cryptographically that:
  1. The executor binary they're running is exactly the code we
     published (firmware + kernel + container image attested).
  2. The GPU is a specific H100 chip in CC mode (NOT spoofed).
  3. The host operator CANNOT see inside the VM or the GPU memory.

This eliminates K-1 through K-5 in our threat model — the kernel
rootkit attack path doesn't apply when the kernel itself is sealed
inside the CVM and attested. Sybil defenses become cryptographic
instead of statistical for TDX-attested executors.

DEPLOYMENT — DSTACK (THE OPEN-SOURCE TDX RUNTIME)
=================================================

We adopt [dstack](https://github.com/Dstack-TEE/dstack) for TDX-VM
orchestration.

The miner operator's flow:
  1. Verify hardware: `./nodexo-cvm.sh check` — confirms TDX + SGX +
     KVM + H100 with CC firmware.
  2. Download OS image (signed): `./nodexo-cvm.sh download` — pulls the
     reproducible dstack guest OS image with our miner pre-installed.
  3. Configure: `nodexo-cvm.sh new my-miner` — generates a CVM config
     binding the GPU PCIe address, RAM, vCPUs.
  4. Boot: `./nodexo-cvm.sh run my-miner` — starts the TDX VM. At boot,
     the VM generates a TDX_QUOTE attesting to its initial state.
  5. The CVM-side nodexo-miner registers with the validator,
     including the TDX_QUOTE in the heartbeat.

The CVM is the executor — there's no "outer" miner the operator can
interfere with. The operator owns the metal; the validator+chain
verify the attestation.

PROTOCOL — MINER → VALIDATOR
=============================

Heartbeat payload extension:

  {
    ...existing hw_static fields...,
    "tee_type": "tdx" | "sev-snp" | "",        # "" = non-TEE miner
    "tdx_quote": "<base64 of TDX quote bytes>",
    "tdx_quote_collateral": "<aux bytes>",       # PCK certs, etc.
    "nvidia_cc_quote": "<base64 of GPU attestation>",
    "image_digest": "<sha256 of the CVM's docker compose hash>",
  }

VALIDATOR — VERIFICATION
========================

Roadmap validator attestation verifier:

  1. Parse TDX quote with Intel's PCCS / DCAP verifier.
  2. Check:
     - Quote signature valid (Intel HSM-rooted).
     - Quote's MRTD (firmware) measurement is in our whitelist.
     - Quote's RTMRs (runtime measurements) match expected CVM image.
     - Report data field includes the miner's hotkey ss58 (binds
       attestation to the specific miner identity).
  3. Parse NVIDIA CC quote via NRAS (NVIDIA Remote Attestation Service)
     or local verify with the H100's signed firmware cert.
     - Check GPU model, CC mode enabled, no debug state.
  4. If both verify: mark executor `tee_type=tdx, attestation_verified=True`.

WHITELIST FORMAT
================

```
TDX_WHITELIST = {
    "OS_IMAGE_HASH": { "<sha256>", ... },     # The guest OS image
    "COMPOSE_HASH": {
        "PROD": { "<sha256>", ... },          # nodexo-miner production
        "TESTNET": { "<sha256>", ... },       # public testnet preview
    },
    "RTMR0_KERNEL": { "<sha256>", ... },      # kernel image measurement
}
```

SCORING IMPACT
==============

TDX-verified executors get a TRUST PREMIUM in the composite score:
  - Higher tier in `light/spot/full` verification cadence (most cycles
    use cheap light verification because we trust the attestation).
  - Renter UX surface: marketplace shows "TEE" badge — renters who
    care about confidential workloads filter for these.
  - The sybil-detection signals SOFTEN for TDX-verified executors:
    a TDX miner can't run a sybil monitor (the VM is sealed; only
    our pinned-image miner is inside), and can't fake GPU UUID
    (NVIDIA CC attestation binds the chip identity).

NON-TDX EXECUTORS
=================

The system MUST keep working for non-TDX miners (RTX 4090, A6000,
5090, A100 without CC mode, etc.). The behavioral defenses
(monitor + proof-suppression + cross-monitor PID overlap +
cgroup-during-rental) remain authoritative for them.

TDX is opt-in. Non-TDX miners pay nothing for being non-TDX, but get
no trust premium either.

IMPLEMENTATION STATUS
=====================

Phase 0 (this file): protocol fields + design contract.
Phase 1: integrate dstack scripts into setup_miner.sh (operator
         opt-in flag). Miner emits tdx_quote in heartbeat when running
         in a CVM. Validator stores fields, doesn't verify.
Phase 2: build `neurons/validator/attestation/tdx_verifier.py` —
         integrates Intel PCCS / DCAP or remote attestation service.
         Verifies signed quotes. Populates `attestation_verified` field.
Phase 3: tie verification to scoring (trust premium).
Phase 4: NVIDIA CC integration via NRAS for H100+.

THIS MODULE PROVIDES
====================

Stubs / detection helpers:
  - `tdx_available()` — runtime check whether we're inside a TDX VM.
  - `read_tdx_quote(report_data)` — attempt to get a quote from the
    Linux TDX guest interface (/dev/tdx_guest). Returns bytes or None.
  - `tdx_supported_on_host()` — check whether the HOST CPU supports
    TDX (for setup-time decisions).

Real verification lives in `neurons/validator/attestation/`.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def tdx_supported_on_host() -> bool:
    """Whether the underlying CPU supports TDX. Used by setup scripts
    to decide whether to offer the CVM path.

    Cheap check via /proc/cpuinfo. False on non-Intel and on Intel CPUs
    pre-Sapphire-Rapids (Xeon 4th gen+).
    """
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
    except OSError:
        return False
    return "tdx" in cpuinfo.lower() or "tdx_guest" in cpuinfo.lower()


def tdx_available() -> bool:
    """Whether we're currently RUNNING inside a TDX guest VM.

    True if /dev/tdx_guest exists (the Linux interface to the
    TDX_MODULE for in-guest quote generation). The miner uses this
    at startup to decide whether to attempt quote generation; if
    False, falls back to non-TDX heartbeat shape.
    """
    return Path("/dev/tdx_guest").exists()


def read_tdx_quote(report_data: bytes) -> bytes | None:
    """Generate a TDX quote with `report_data` embedded (must be 64 bytes).

    Returns the raw quote bytes for inclusion in heartbeats. None on
    any failure (most commonly: not in a TDX guest).

    The full path uses dstack's quote-generation flow inside the CVM.
    Here we sketch the kernel ioctl path; real impl will call into
    dstack-sdk or use intel-tdx-tools.

    Returns None until Phase 2 lands the quote-generation integration.
    """
    if not tdx_available():
        return None
    if len(report_data) != 64:
        report_data = hashlib.sha512(report_data).digest()
    # Real implementation: ioctl /dev/tdx_guest with TDX_CMD_GET_QUOTE
    # struct, parse response. See dstack/sdk/python for the canonical
    # implementation.
    # Return None until Phase 2 lands the ioctl glue.
    return None


def heartbeat_attestation_payload(miner_hotkey_ss58: str) -> dict:
    """Construct the attestation block for inclusion in heartbeat
    payload. Always safe to call; returns a non-empty dict only when
    we're inside a TDX guest with a working quote path.

    The `miner_hotkey_ss58` is bound into the report_data field so
    the attestation cannot be replayed under a different miner identity.
    """
    out: dict = {"tee_type": "", "tdx_quote": "", "nvidia_cc_quote": ""}
    if not tdx_available():
        return out

    # Bind the quote to the miner's hotkey: report_data = sha256(ss58 ||
    # "nodexo-attest-v1"). Validator verifies by recomputing.
    import base64
    binding = hashlib.sha256(
        miner_hotkey_ss58.encode() + b"nodexo-attest-v1"
    ).digest()
    quote = read_tdx_quote(binding * 2)  # 64 bytes = 2 * sha256(32)
    if quote is None:
        return out
    out["tee_type"] = "tdx"
    out["tdx_quote"] = base64.b64encode(quote).decode("ascii")
    # nvidia_cc_quote is populated in Phase 4 via the NVIDIA Open
    # Kernel Module's nvmlSystemGetConfComputeAttestationReport.
    return out
