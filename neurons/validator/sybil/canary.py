"""
Active sybil-attestation via canary rentals.

PURPOSE
=======

The behavioral classifier in `monitor.py` is a PASSIVE observer: it
watches PIDs and trusts NVML+/proc not to lie. A rootkit at kernel
level that selectively hides PIDs from NVML defeats it without our
ever finding out (audit K-1).

Canaries flip this from passive to ACTIVE attestation. The validator
periodically creates events it KNOWS the outcome of, then checks
whether the miner's monitor reported them faithfully.

THE GAME-THEORETIC TRAP
=======================

A rootkit must decide per-PID: "do I hide this from NVML or not?"
- Hide all extra processes → CANARY pid hidden too → caught by
  `monitor_suppression`.
- Hide nothing → sybil's own gpu_workers visible → caught by
  `cross_monitor_pid_overlap` (other monitors see them too).

No middle ground exists UNLESS the rootkit can distinguish canary
from sybil. Our job is to eliminate every distinguishing signal.

TELLS THE ATTACKER COULD USE (and the design countermeasures)
==============================================================

| Tell                          | Countermeasure                             |
|-------------------------------|--------------------------------------------|
| cmdline / process name        | Canary uses indistinguishable workload     |
| VRAM footprint                | Canary varies VRAM across runs (in proof   |
|                               | range, but jittered)                       |
| Timing                        | Canary fires at cryptographically random   |
|                               | offsets, unannounced                       |
| Source TCP / IP of rental req | Cross-validator canary swap (validator A   |
|                               | proxies through B's IP) + cycle through    |
|                               | random egress points                       |
| cgroup (in-container vs not)  | Run BOTH in-container canaries AND out-of- |
|                               | container canaries (new miner endpoint     |
|                               | `POST /attest/canary`)                     |
| Renter SSH key fingerprint    | Rotate per canary                          |
| Container name pattern        | Standard `nodexo-rental-<ts>` — no special   |
|                               | marker (would be a tell otherwise)         |
| Rental TTL pattern            | Canary TTL varies in normal range          |

PROTOCOL — IN-CONTAINER CANARY
==============================

  1. Validator picks target executor E (weighted toward less-tested).
  2. Validator generates a fresh canary ssh keypair K_c.
  3. Validator calls `POST /rent` with:
       - gpu_model from E's tier
       - ssh_pub_key = K_c.pub
       - duration = random short window (5-15 min)
       - renter_pubkey_fp = sha256(K_c.pub)[:16]  (looks like a renter)
     [If using cross-validator swap: routes the POST through peer
     validator B so the source IP is B's, not ours.]
  4. Validator gets the rental connection details.
  5. Validator SSHes in using K_c.priv, runs a small CUDA workload
     with a SEED chosen to give a predictable PID lifetime/VRAM/timing.
     Records: (host_pid, start_ts, end_ts, vram_mb, container_id).
  6. While the workload runs, validator polls monitor reports for E.
  7. Validator runs the sybil scanner immediately after.
  8. Checks:
       a. Did monitor E's report contain the canary's host_pid?
          NO → open `monitor_suppression` flag (HARD).
       b. Did ANY other executor's monitor report the canary's host_pid?
          YES → open `canary_pid_overlap` flag on that executor (HARD).
  9. Validator terminates the rental.

PROTOCOL — OUT-OF-CONTAINER CANARY
==================================

For cgroup-distinguishing rootkits, we need a way to spawn work on
the miner host but OUTSIDE any rental container.

  1. Validator signs an `attest_canary` request to the miner.
  2. Miner spawns a small CUDA workload as the miner user (not inside
     any rental cgroup), with arguments echoing a validator-supplied
     nonce so the PID's mapping back to "this canary" is provable.
  3. Miner replies with the spawned PID.
  4. Same monitor check as above.

Endpoint: `POST /attest/canary` on miner (gated by validator sig).
Body: {nonce: hex, vram_mb: int, duration_s: int}.
Reply: {pid: int, started_ts: float}.

If the miner refuses to implement this endpoint, the validator can
detect: a 404 on attest/canary is itself a (soft) signal of an
upgrade gap; over time, an executor that NEVER answers attest/canary
gets weight degradation.

PROTOCOL — CROSS-VALIDATOR CANARY SWAP (MAINNET)
================================================

For mainnet where ValidatorRegistry has N validators:
  1. Validator A picks target executor E (registered to some miner M).
  2. A asks validator B (via gossip / API call) to run a canary on E.
  3. B initiates the rent from B's IP — source IP looks normal to M.
  4. B runs the canary workload, records the PID lifetime.
  5. B sends the signed result back to A (or to a shared coordination
     channel: e.g., a `RentalState` chain event).
  6. A consults E's monitor reports to verify.

This makes IP-based whitelisting infeasible: M would have to whitelist
EVERY validator's IP to fool any of them, which means M accepts
canaries from every validator and the system works.

STATISTICAL DETECTION
=====================

Single canary: low confidence (rootkit might happen to allow this one).
N canaries: P(rootkit slips up at least once) → 1 as N → ∞.

If the rootkit has 90% accuracy in distinguishing canary from sybil:
  P(undetected after N canaries) = 0.9 ** N
  N=10 → 35%
  N=24 → 8%
  N=48 → 0.6%

At 1 canary/hour:
  90% accuracy → caught within ~2 days with 99%+ confidence
  95% accuracy → caught within ~4 days
  99% accuracy → caught within ~3 weeks

This is the K-1 mitigation: TEE buys instant guarantee; canaries buy
asymptotic detection.

IMPLEMENTATION STATUS
=====================

Phase 0 (today): this file is the design contract. Module is empty.
Phase 1 (next): in-container canary, single-validator (testnet).
Phase 2: out-of-container canary + miner `POST /attest/canary`.
Phase 3: cross-validator swap (requires multi-validator deployment).
Phase 4: tie weight degradation to attest_canary refusals + missed
  canary participation.

KEY DESIGN INVARIANTS
=====================

1. Canary identity NEVER provably maps back to "validator-originated"
   from anything the miner host can observe.
2. Canary runs are RANDOM in timing, duration, VRAM footprint, and
   workload pattern. The miner cannot predict.
3. Canaries are FREQUENT enough to overwhelm statistical
   distinguishing — at least 1/hr per executor.
4. Canary results are signed by the validator and STORED on chain or
   in a shared validator-coordination state, so a colluding validator
   can't suppress them.
5. The endpoint structure is symmetric: a real renter and a canary
   should be indistinguishable from the miner's network stack.
"""
