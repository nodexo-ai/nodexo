# Nodexo — full operational flow

One diagram, then the per-component cycle. Every loop's cadence and
every state transition is named.

---

## The big picture (cycle view)

```
                         ┌──────────────────────┐
                         │  Bittensor EVM chain │
                         │  ComputeRegistry +   │
                         │  ValidatorRegistry   │
                         └──┬─────────────────┬─┘
              register      │                 │  read isRented
              ┌─────────────┘                 └────────────┐
              ▼                                            │
  ┌───────────────────────────┐         ┌──────────────────┴─────┐
  │  Miner host (1 executor)  │         │      Validator         │
  │  ─────────────────────    │         │  ──────────────────    │
  │                           │         │                        │
  │  1) registerExecutor      │         │  loops, every cadence: │
  │     (one-time, on start)  │         │                        │
  │                           │         │   verify_loop (~tick)  │
  │  2) ProofService          │         │   ┌──────────────────┐ │
  │     every 15 blocks =     │ recipe  │   │ verify_recipe()  │ │
  │     180s:                 │────────►│   │ → proof_results  │ │
  │       read isRented       │ POST    │   └──────────────────┘ │
  │       heavy if not rented │ /proofs │                        │
  │       micro if rented     │         │   heartbeat (every 60s)│
  │       per-GPU subprocess  │         │   ┌──────────────────┐ │
  │       broadcast receipt + │ POST    │   │ VRAM consistency │ │
  │       recipe              │ /hb     │   │ check → flag if  │ │
  │                           │────────►│   │   spec mismatch  │ │
  │  3) MonitorContainer      │         │   └──────────────────┘ │
  │     sysbox sidecar        │ POST    │                        │
  │     signed NVML reports   │/monitor │   sybil_scanner (10min)│
  │     every ~5Hz            │────────►│   ┌──────────────────┐ │
  │                           │         │   │ cross-executor   │ │
  │                           │         │   │ UUID/IP/PID      │ │
  │                           │         │   │ correlations     │ │
  │                           │         │   └──────────────────┘ │
  │                           │         │                        │
  │                           │         │   canary_scheduler     │
  │                           │         │   (memoryless Poisson, │
  │                           │         │    mean=24h, tick=60s) │
  │                           │ rental  │   ┌──────────────────┐ │
  │                           │◄────────│   │ markRented →     │ │
  │                           │ provision   │ T_drain (240s) → │ │
  │                           │ /containers │ provision →      │ │
  │                           │         │   │ ssh+install+fill │ │
  │                           │         │   │ → evaluate →     │ │
  │                           │ ssh fill│   │ markAvailable    │ │
  │                           │◄────────│   └────────┬─────────┘ │
  │                           │         │            ▼           │
  │                           │         │   canary_records.write │
  │                           │         │                        │
  │                           │         │   scoring_loop (60s)   │
  │                           │         │   ┌──────────────────┐ │
  │                           │         │   │ score_one() per  │ │
  │                           │         │   │ executor; gates  │ │
  │                           │         │   │ + util_factor    │ │
  │                           │         │   └──────────────────┘ │
  │                           │         │                        │
  │                           │         │   weight_loop (~72min) │
  │                           │         │   ┌──────────────────┐ │
  │                           │ tx      │   │ aggregate per UID│ │
  │                           │◄────────│   │ normalise →      │ │
  │                           │ setWeights  │ rpc.set_weights  │ │
  │                           │         │   └──────────────────┘ │
  └───────────────────────────┘         └────────────────────────┘
              ▲                                            ▲
              │           ┌────────────────────┐           │
              │           │   real renter      │           │
              │ ssh into  │   POST /rent ─────►│ canary    │
              │ container │   (canary-gated)   │ records   │
              └───────────┤   markRented +     │ filter ────┘
                          │   provision +      │
                          │   return ssh creds │
                          └────────────────────┘
```

---

## State machine for one executor

```
                  registerExecutor()
                          │
                          ▼
                  ┌───────────────┐
                  │ CANARY_PENDING│  (no rentals, score=0)
                  └───────┬───────┘
                          │ at least 1 valid proof verified
                          │ AND scheduler picks (Poisson)
                          ▼
                       canary
                  ┌──────┴──────┐
                  │             │
              PASS                FAIL / error_streak ≥ 3
                  │             │
                  ▼             ▼
            ┌─────────┐   ┌─────────┐
            │RENTABLE │   │ BANNED  │   (score=0)
            └────┬────┘   └─────────┘
                 │              ▲
                 │              │
        rent OR canary           │ open HARD sybil flag
        markRented              │
                 │              │
                 ▼              │
            (proof switches     │
             to micro mode)     │
                 │              │
                 ▼              │
        markAvailable           │
                 │              │
                 ▼              │
            ┌─────────┐         │
            │RENTABLE │─────────┘
            └────┬────┘
                 │ no proof/heartbeat for N seconds
                 ▼
            ┌─────────┐
            │  STALE  │  (score=0; heartbeat resumes → back to RENTABLE)
            └─────────┘

                 (any state) ── deregister / lease expire ──► INACTIVE
```

State definitions live in `neurons/validator/scoring/scorer.py::executor_state()`.
The rental gate (`/rent`) and the scoring gate use the same predicates against
`canary_records.last_status`, so the rentable set equals the score-positive set.

---

## Loop cadences (production defaults)

| Loop | Cadence | Source |
|---|---|---|
| ProofService (miner) | 15 blocks ≈ 180s | `proof_service.py` |
| Heartbeat (miner → validator) | ~60s | `miner.py` heartbeat task |
| Monitor reports (sidecar → validator) | ~200ms tick, batched | `monitor.py` |
| Validator verification loop | continuous, per-recipe | `validator.py::_verification_loop` |
| Sybil scanner | 600s (10 min) | `validator.py::sybil_scanner_loop` |
| **Canary scheduler tick** | **60s** | `scheduler.py` |
| **Canary fire probability per tick per executor** | **60 / 86400 = 0.069%** (mean = 24h) | `scheduler.py` |
| Scoring loop | 60s | `validator.py::_scoring_loop` |
| Weight setting | 4320s (1 tempo = 360 blocks × 12s) | `validator.py::_weight_setting_loop` |
| TTL sweeper | 60s | `rent.py::ttl_sweeper` |
| Orphan sweeper | 60s | `rent.py::orphan_sweeper` |
| Rental container watchdog | 30s | `rent.py::rental_container_watchdog` |

The rental container watchdog is scoped to rentals this validator created
and stored locally. It does not inspect or release peer-validator rentals;
peer rental events are only indexed as proof-mode context.

---

## Canary deep dive — every phase + every clock

```
T = 0    scheduler tick fires for executor X
         (HMAC-Bernoulli passed, X is unrented + canary-eligible + has 1+ valid proof)
T = 0    runner: generate fresh ed25519 SSH key + 32-byte nonce
T = 0    runner: markRented(X) — chain tx
T ≈ 12s  chain tx confirmed, rental_block recorded
T = 12s  runner: sleep T_drain (240s)
         purpose: miner reads isRented from chain on next proof cycle,
         switches heavy → micro, releasing ~40% VRAM
T ≈ 252s runner: orchestrator.provision(image=ubuntu:22.04, ssh_pub=K_c)
         miner: docker run --rm --gpus all -v /dev/nvidia* sysbox-runc ubuntu:22.04
T ≈ 268s container ready, ssh port 20000-21000 mapped
T ≈ 268s runner: ssh -i K_priv user@host:port 'sh -c "<setup>"'
         setup: apt update + apt install python3 python3-pip
                + pip install numpy + pip install torch (cu121 wheel)
                + python3 -  (fill.py piped via stdin)
T ≈ 280s pip install done (~12s after container ready when cached)
T ≈ 280s python3 - reads stdin, executes fill.py
         CANARY_NONCE = <hex>
         CANARY_TARGET_MB = (GPU_VRAM_GB[gpu_model] - 1GB headroom) * 1024
         CANARY_HOLD_SECONDS = 180
         CANARY_VERIFY_MB = 16
T ≈ 280s fill.py: cuRAND chunks of 512MB until target or OOM
T ≈ 281s fill.py: device-side sum reduction (HBM bandwidth scan)
T ≈ 281s fill.py: numpy.PCG64(seed_from_nonce).bytes(16MB) → push to GPU →
                  read back → sha256 → verify_hash
T ≈ 282s fill.py: time.sleep(180) ← hold window, any concurrent sybil heavy fails
T ≈ 462s fill.py: print JSON to stdout, exit
T ≈ 462s runner: parse JSON, evaluator.evaluate_fill_result():
                 allocated_mb >= 0.95 * target ?
                 hbm_bandwidth_gb_s >= floor[gpu_model] ?
                 sha256(numpy.PCG64(seed).bytes()) == verify_hash ?
T ≈ 463s runner: open sybil_flag(s) if any check failed
T ≈ 463s runner: terminate container (DELETE /containers/<name>)
T ≈ 478s runner: markAvailable(X) — chain tx
T ≈ 490s runner: write canary_records row (last_status, last_summary)
                 update consecutive_errors (sabotage detection)
T ≈ 490s scheduler: ready for next tick

Total wall-clock: ~8-9 minutes per canary
```

---

## Production cadence configuration

Validator environment:

```bash
export CANARY_ENABLED=1                    # only the primary validator
export CANARY_MEAN_SECONDS=86400           # 24h mean inter-canary
export CANARY_EXCLUDE_EXECUTOR_IDS=""      # operator denylist (CSV)
```

The scheduler is OFF by default for other validators (no `CANARY_ENABLED=1`).

Per-tick (60s) probability = 60 / 86400 ≈ 0.069 %.
30-day simulation: 21 canaries per executor, gaps from 15 min to 8 days, 4.8 % under 1 hour.
No predictable safe window for cheating.

---

## Related Docs

- This file: the **flow + cycle visualisation**.
- [API_BOUNDARIES.md](API_BOUNDARIES.md): public API and internal validator boundaries.
