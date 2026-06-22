# Nodexo — Flow Diagrams

## Conceptual Overview

Each epoch (~180s), every GPU independently proves it exists via ZkGEMM.
Two types of messages flow from miner to validator:

- **RECEIPTS** (tiny, per GPU per pass): root + T_commit + timestamp.
  Purpose: TIMING. Validator records arrival on its own clock.
  Delta = arrival(u=1) - arrival(u=0) = GPU compute time (network latency cancels).

- **RECIPE** (big, ONE per executor): all GPU proofs with sumcheck + Merkle paths.
  Purpose: CRYPTOGRAPHIC VERIFICATION. Proves the roots are real GEMM outputs.
  Without this, a miner could send fake roots with perfect timing.

Both are required: receipts without recipe = no proof the GPU computed anything.
Recipe without receipts = no proof the GPU computed it NOW (could be pre-computed).

## Simplified Flow (4 GPUs)

```
Block mined → beacon
      │
      ├── GPU[0] ─┐
      ├── GPU[1] ─┤  PARALLEL subprocesses
      ├── GPU[2] ─┤  CUDA_VISIBLE_DEVICES isolation
      ├── GPU[3] ─┘  Each gets unique seed = hash(beacon + executor_id + gpu#)
      │
      │   Each GPU does:
      │     Pass 0: GEMM(seed) → root_0 → write file → parent sends RECEIPT(u=0)
      │     Pass 1: GEMM(hash(seed+root_0)) → root_1 → write file → parent sends RECEIPT(u=1)
      │     Challenges from root_1 → sumcheck proofs for both passes
      │
      │   After ALL GPUs finish:
      └── Send ONE RECIPE with all 4 GPU proofs


VALIDATOR receives:

  8 RECEIPTS (4 GPUs × 2 passes):
    GPU[0] u=0 → t=100.00    GPU[0] u=1 → t=111.02   delta=11.02s
    GPU[1] u=0 → t=100.05    GPU[1] u=1 → t=111.08   delta=11.03s
    GPU[2] u=0 → t=100.03    GPU[2] u=1 → t=111.05   delta=11.02s
    GPU[3] u=0 → t=100.08    GPU[3] u=1 → t=111.10   delta=11.02s
    All u=0 within 0.08s → PARALLEL ✓
    Mean delta 11.02s < deadline 28.3s → TIMING OK ✓

  1 RECIPE → 13 security checks:
    Global: run_id, beacon, epoch, GPU count, VRAM, wall-clock
    Per-GPU: seed, T_commit×2, chain integrity, challenges, Merkle, sumcheck
    → ALL PASS → VALID
```

## Cheater Detection (1 GPU faking 4)

```
  GPU[0] u=0 → t=100.00    GPU[0] u=1 → t=111.02
  GPU[1] u=0 → t=111.05    GPU[1] u=1 → t=122.07   ← 11s AFTER GPU[0]
  GPU[2] u=0 → t=122.10    GPU[2] u=1 → t=133.12
  GPU[3] u=0 → t=133.15    GPU[3] u=1 → t=144.17

  u=0 receipt spread = 33.15s (parallel would be <1s) → SEQUENTIAL → CHEATER
```

## Security Summary

| Attack | What stops it |
|---|---|
| No GPU | Sumcheck (can't fake the math) |
| Weaker GPU | VRAM check + timing delta |
| 1 GPU faking N | Receipt spread reveals sequential execution |
| Pre-compute | Beacon unpredictable until block mined |
| Replay | Seed bound to block hash + epoch freshness + run_id |
| Steal proof | Seed bound to executor_id + SR25519 signature |
| Easy blocks | Challenge re-derivation from root_1 |
| Skip pass_1 | Chain: seed_1 = hash(seed + root_0) |
| Fake timestamps | Receipts use validator's clock |
| Inject fake receipts | Registered-executor check + SR25519 |

## Detailed Miner Flow

```
STARTUP (miner.py:lifespan)
├─ Detect GPUs (pynvml) → generate executor_id (SHA256 of GPU UUIDs)
├─ Load wallet → derive EVM key → resolve UID from metagraph
├─ Self-register on EVM: registerEvm() + registerExecutor()
├─ Create broadcast service
│  └─ ValidatorRegistry endpoints + validator-permit native axons
├─ Build signed-rental validator allowlist
│  └─ subnet runtime config, env override, or ValidatorRegistry when non-strict
├─ Start proof loop + heartbeat + TTL checker
└─ Serve FastAPI: /health, /hardware, /proof/status, /containers

PROOF LOOP (every ~180s)
├─ Get beacon from chain → derive jitter → compute n from VRAM
├─ Spawn GPU workers (subprocess per GPU, CUDA_VISIBLE_DEVICES)
├─ Poll for receipt files → broadcast immediately when written
├─ Wait for workers → read results → build recipe → broadcast
└─ Cleanup temp dir

GPU WORKER (per-GPU subprocess)
├─ Pass 0: Philox PRF → INT8 GEMM (64 cuBLAS) → Merkle → root_0 → T_commit_0
├─ CPU offload (free GPU VRAM for pass_1)
├─ Pass 1: chained seed → GEMM → root_1 → T_commit_1
├─ Derive challenges from root_1 (unknowable until pass_1 done)
├─ GPU swap: generate sumcheck proofs for both passes
└─ Write final result JSON
```

## Detailed Validator Flow

```
STARTUP (validator.py:lifespan)
├─ Database init (SQLite or PostgreSQL)
├─ ProofAnalyzer + asyncio.Queue + ProcessPoolExecutor
├─ Chain clients (ComputeRegistry, ValidatorRegistry)
├─ EVM registration on ValidatorRegistry/ComputeRegistry
│  └─ skipped when VALIDATOR_NO_EVM=1
├─ Start 5 background loops

RECEIPT INGESTION (POST /proofs/receipt)
├─ Registered-executor check + SR25519 verify
└─ Record recv_time on validator's clock per (executor, gpu, pass)

RECIPE INGESTION (POST /proofs/recipe)
├─ Size + executor + signature + rate limit + dedup checks
└─ Queue for verification

VERIFICATION LOOP
├─ Finalization wait (5 blocks lag, prevents reorg issues)
├─ Build context: beacon from chain, VRAM/gpu_count from registry
├─ verify_recipe() in ProcessPool:
│   Global: run_id, beacon_meta, allocation, epoch, GPU count, VRAM, wall-clock
│   Per-GPU: seed, T_commit×2, chain, challenges, Merkle, sumcheck
├─ Timing enforcement: mean_delta ≤ deadline (per GPU model)
└─ Record result → DB + in-memory analyzer

SCORING (every 60s)
├─ Rolling pass rates (1h/24h/all) → proof_score
├─ Composite: proof(40%) + availability(25%) + hardware(25%) + trust(10%)
└─ × utilization bonus if rented

WEIGHT SETTING (every 72min)
├─ Map executor_id → miner_address → UID (via ComputeRegistry.evmToUid)
├─ Aggregate scores per UID, normalize to sum=1
└─ set_weights() on Bittensor chain

PRUNING (every 1h)
├─ Backup old proof_results + rentals to .jsonl.gz
├─ Delete from live DB (3 days proofs, 30 days rentals)
├─ Deactivate executors no longer on-chain
└─ Clean orphaned stats
```
