"""
Nodexo Monitor — attestable per-miner GPU observer.

ARCHITECTURE
============

One persistent monitor container per miner, running alongside the
miner daemon and any rental containers. The monitor is the
HARDEST-TO-FAKE signal source in the system; the miner daemon is
trusted to do its job but cryptographically distrusted as a witness.

Diagram (one physical box):

    miner host
    ├── miner_daemon         (proof generation, rental orchestration)
    │   ├── --gpus device=<UUID>          ← shared GPU passthrough
    ├── renter_container_N   (one per active rental)
    │   ├── --gpus device=<UUID>          ← same GPU passthrough
    └── monitor_container    (persistent, our pinned image)
        ├── --gpus device=<UUID>          ← same GPU passthrough
        ├── read-only NVML access
        ├── outbound HTTP to validator(s)
        ├── per-monitor signing keypair (generated inside, never
        │   reachable from the host)
        └── sleeps when nothing to attest; ticks on schedule

WHAT THE MONITOR DOES
======================

1. **Per-PID NVML polling.** Reads
   `nvmlDeviceGetComputeRunningProcesses` at 5–10 Hz to enumerate
   processes touching the GPU. Records: PID, process name, VRAM,
   util, start/end timestamps. The kernel-driver-reported view of
   ground truth.

2. **Proof fingerprint matching.** Knows the expected proof
   schedule for the executor_id this monitor is bound to (formula:
   block_number, beacon, derive_jitter). When proofs run, confirms
   a fresh gpu_worker process or a persistent gpu_worker daemon shows
   current-cycle proof-shaped GPU memory. Reports anomalies — extra
   proof-shaped processes = sybil signal.

3. **Live telemetry replacement.** Replaces the miner_daemon's
   self-reported `hw_live` heartbeat. The validator trusts the
   monitor's signed report instead of the miner's because:
     - Monitor reports are signed with a per-monitor keypair the
       miner can't trivially extract.
     - Monitor image digest is pinned; we know what code is running.
     - Monitor is a passive observer; it doesn't have the proof
       generation code to lie about.

4. **Challenge response.** Validator can hit the monitor with a
   periodic small CUDA workload challenge. Monitor runs it on the
   GPU, returns timing + result hash. Forging this requires a real
   second GPU (more expensive than the sybil pays).

5. **Renter-container correlation.** Reports the set of
   active rental container names visible on this GPU (by docker
   process name pattern AND by NVML-reported process). If TWO
   monitors bound to different executor_ids see the SAME rental
   container PID, the GPU is being double-rented.

WHAT THE MONITOR DOES NOT DO
============================

- Touch the renter's container (no shell, no SSH key injection, no
  network reachability). Privacy is preserved end-to-end. The renter
  doesn't see the monitor; the monitor doesn't see the renter's
  workload contents, only NVML's per-process accounting.
- Hold the miner's hotkey or any signing material that could be
  abused to impersonate the miner.
- Receive inbound HTTP. Outbound only. Reduces attack surface to
  nearly zero from the network side.

THREAT MODEL
============

What the monitor defends against:
  * Miner registers two executor_ids for one GPU and rents both
    (the "double-rent" attack) — monitor sees the second's
    extra proof-shaped GPU process, flags.
  * Miner runs a hidden process on a rented GPU stealing compute —
    monitor sees the extra PID, attributes the load, flags.
  * Miner lies about gpu_temp_c / gpu_util_pct in heartbeat to look
    healthier than it is — replaced by signed monitor telemetry.

What it does NOT defend against without TEE:
  * Miner with kernel-mode access patches the nvidia.ko module to
    lie selectively to different containers. Out of scope until
    we have hardware attestation (H100+ confidential compute).
  * Miner replaces the monitor image with a forked impostor that
    has the same digest. SHA256 collision is infeasible; this
    attack doesn't exist.
  * Miner runs the official monitor image but firewalls outbound
    HTTP so it can't report. Detection: validator notes "no
    monitor report from executor X for Y minutes" → soft flag.

DEPLOYMENT
==========

Phase 1 (current state — scaffold only):
  - This package exists; the daemon is implemented but not yet
    containerized. Operators wishing to run it manually can launch
    `python -m neurons.monitor.monitor --executor-id <id> --validator-url <url>`.
  - No pinned image yet; the validator accepts reports from any
    monitor identified by the per-monitor keypair (TOFU).

Phase 2 (planned):
  - Dockerfile in `neurons/monitor/Dockerfile`.
  - Published image at `ghcr.io/nodexo/monitor:<version>`,
    digest pinned in chain config.
  - `setup_miner.sh` pulls + starts the monitor alongside the miner.
  - Validator verifies the monitor's claimed image digest via the
    miner's signed `/containers` endpoint.

Phase 3 (planned):
  - Replace miner_daemon's `hw_live` heartbeat with monitor reports.
  - Add lightweight compute challenge (small matmul, known seed,
    expected timing+hash for the registered GPU model).

Phase 4 (planned, hardware-dependent):
  - TEE attestation for monitor identity on H100+ confidential
    compute hosts.

PROTOCOL — MONITOR ↔ VALIDATOR
==============================

Monitor → Validator: POST /monitor/report
  Body (JSON, separators=(',',':') so signature matches):
    {
      "executor_id": "...",
      "monitor_pubkey": "<sr25519 pubkey hex>",
      "report_ts": <epoch seconds>,
      "report_seq": <monotonic counter>,
      "gpu_live": {
        "util_pct": <float>,
        "mem_used_mb": <int>,
        "mem_total_mb": <int>,
        "temp_c": <int>,
        "power_w": <float>,
      },
      "processes": [
        {"pid": 123, "name": "gpu_worker.py",
         "mem_mb": 512, "util_pct": 95.2,
         "started_ts": <epoch>, "lifetime_s": 4.7},
        ...
      ],
      "expected_proof_seen": <bool>,
      "extra_proofs_seen": <int>,  # > 0 ⇒ sybil signal
      "rental_containers_visible": ["nodexo-rental-..." , ...],
    }
  Headers:
    X-Monitor-Signature: <hex sr25519 sig of canonical body>
    X-Monitor-Hotkey:    <ss58 of monitor's keypair>
    X-Monitor-Timestamp: <epoch>
    X-Monitor-Nonce:     <16 hex>

Validator stores reports and runs an analyzer pass cross-correlating
expected_proof_seen + extra_proofs_seen + rental_containers_visible
against the network-wide rental state.
"""
