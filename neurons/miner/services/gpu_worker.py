"""
GPU Worker — standalone subprocess that runs proof generation on ONE GPU.

Spawned by ProofService with CUDA_VISIBLE_DEVICES=gpu_idx.
Each worker computes both passes (pass_0 + pass_1) sequentially on its
assigned GPU, captures ns-precision timestamps, computes T_commit,
derives challenges, generates sumcheck proofs, and writes all results
to an output JSON file.

The parent process reads the output after the worker exits.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m neurons.miner.services.gpu_worker \\
    --seed-hex abc123... --n 20480 --gpu-idx 0 --epoch-id 100 \\
    --executor-id def456... --num-challenges 4 --num-spot-checks 100 \\
    --output-dir /tmp/nodexo_proof_epoch100/

Reference: see protocol design notes worker_entry()
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import time
import os
import gc

# Domain separation (must match proof_service.py and verifier.py)
CHAIN_PREFIX = b"NODEXO_CHAIN_V1"
CHALLENGE_PREFIX = b"NODEXO_CHALLENGE_V1"
COMMIT_PREFIX = b"NODEXO_COMMIT_V1"


def compute_t_commit(seed: bytes, root: bytes, u: int, n: int, epoch_id: int) -> bytes:
    """Compute T_commit binding proof to beacon/seed/run context.

    T_commit = SHA256("NODEXO_COMMIT_V1" || seed || root || pack(u, n, epoch_id))

    This ties the proof to a specific seed, root, pass number, matrix size,
    and epoch. The validator re-derives T_commit and rejects mismatches.

    Reference: see protocol design notes T_commit in proof.py line 127-128
    """
    payload = (
        COMMIT_PREFIX
        + seed
        + root
        + struct.pack("<I", u)
        + struct.pack("<I", n)
        + struct.pack("<I", epoch_id)
    )
    return hashlib.sha256(payload).digest()


def derive_chain_seed(seed: bytes, root_0: bytes) -> bytes:
    """Derive chained seed for pass_1."""
    return hashlib.sha256(CHAIN_PREFIX + seed + root_0).digest()[:8]


def derive_challenge_indices(seed: bytes, root_1: bytes, num_challenges: int, total_blocks: int) -> list[int]:
    """Derive deterministic challenge block indices."""
    indices = []
    seen = set()
    counter = 0
    while len(indices) < num_challenges:
        payload = CHALLENGE_PREFIX + seed + root_1 + struct.pack("<I", counter)
        h = hashlib.sha256(payload).digest()
        idx = struct.unpack("<I", h[:4])[0] % total_blocks
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
        counter += 1
    return indices


def write_json_atomic(path: str, payload: dict) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def release_cuda_cache(torch_module) -> None:
    try:
        torch_module.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
    try:
        torch_module.cuda.empty_cache()
    except Exception:
        pass


def run_worker(args):
    """Main worker function — runs on one GPU."""
    worker_t0 = time.perf_counter()
    seed = bytes.fromhex(args.seed_hex)
    seed_int = struct.unpack("<Q", seed)[0]
    n = args.n
    bs = args.block_size
    gpu_idx = args.gpu_idx
    epoch_id = args.epoch_id
    num_challenges = args.num_challenges
    num_spot_checks = args.num_spot_checks
    output_dir = args.output_dir

    # Import CUDA extension (CUDA_VISIBLE_DEVICES already set by parent)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "zkgemm", "cuda"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    import_t0 = time.perf_counter()
    from zkgemm.prover import ZkGemmBlockProver
    import_s = time.perf_counter() - import_t0

    result = {
        "gpu_index": gpu_idx,
        "seed": args.seed_hex,
        "n": n,
        "epoch_id": epoch_id,
        "status": "running",
    }

    try:
        # ── Pass 0 ─────────────────────────────────────────────
        pass0_t0 = time.perf_counter()
        ts_before_pass0 = time.time_ns()

        prover_0 = ZkGemmBlockProver(
            seed=seed_int, n=n, block_size=bs,
            device=0,  # Always device 0 (CUDA_VISIBLE_DEVICES handles mapping)
            num_spot_checks=num_spot_checks,
        )
        root_0 = prover_0.phase1_commit()

        ts_after_pass0 = time.time_ns()
        pass0_s = time.perf_counter() - pass0_t0

        T_commit_0 = compute_t_commit(seed, root_0, 0, n, epoch_id)

        # Write pass_0 result immediately (parent can read for receipt broadcasting)
        pass0_result = {
            "gpu_index": gpu_idx,
            "u": 0,
            "root": root_0.hex(),
            "T_commit": T_commit_0.hex(),
            "ts_ns": ts_after_pass0,
        }
        pass0_path = os.path.join(output_dir, f"gpu_{gpu_idx}_pass0.json")
        write_json_atomic(pass0_path, pass0_result)

        # ── Preserve only pass_0 C for later challenged blocks ──
        # A and B are PRF-derived public matrices. Copying pass_0 C to CPU
        # changes the calibrated timing profile, so the shipped proof path keeps
        # C resident on GPU and treats OOM as a failed cycle.
        import torch
        prover_0.A = None
        prover_0.B = None
        release_cuda_cache(torch)

        # ── Pass 1 (chained) ───────────────────────────────────
        seed_1 = derive_chain_seed(seed, root_0)
        seed_1_int = struct.unpack("<Q", seed_1)[0]

        pass1_t0 = time.perf_counter()
        ts_before_pass1 = time.time_ns()

        def _build_pass1_prover():
            p = None
            try:
                p = ZkGemmBlockProver(
                    seed=seed_1_int, n=n, block_size=bs,
                    device=0,
                    num_spot_checks=num_spot_checks,
                )
                r = p.phase1_commit()
                return p, r
            except Exception:
                if p is not None:
                    p.A = None
                    p.B = None
                    p.C = None
                    p.flat_tree = None
                release_cuda_cache(torch)
                raise

        prover_1, root_1 = _build_pass1_prover()

        ts_after_pass1 = time.time_ns()
        pass1_s = time.perf_counter() - pass1_t0

        T_commit_1 = compute_t_commit(seed_1, root_1, 1, n, epoch_id)

        # Write pass_1 result (parent can read for receipt broadcasting)
        pass1_result = {
            "gpu_index": gpu_idx,
            "u": 1,
            "root": root_1.hex(),
            "T_commit": T_commit_1.hex(),
            "ts_ns": ts_after_pass1,
        }
        pass1_path = os.path.join(output_dir, f"gpu_{gpu_idx}_pass1.json")
        write_json_atomic(pass1_path, pass1_result)

        # ── Challenge derivation + proof generation ────────────
        blocks_per_row = n // bs
        total_blocks = blocks_per_row * blocks_per_row
        challenge_indices = derive_challenge_indices(seed, root_1, num_challenges, total_blocks)
        challenged_blocks = [(idx // blocks_per_row, idx % blocks_per_row) for idx in challenge_indices]

        # Prove pass_1 while it is still resident on GPU.  The old flow
        # offloaded pass_1 to CPU, reloaded pass_0, then reloaded pass_1
        # again, adding a full A+B+C device round trip to every cycle.
        proof1_t0 = time.perf_counter()
        block_proofs_1 = prover_1.phase2_prove(challenged_blocks)
        proof1_s = time.perf_counter() - proof1_t0
        prover_1.A = None
        prover_1.B = None
        prover_1.C = None
        torch.cuda.empty_cache()

        # Prove pass_0 from CPU C + PRF-regenerated A/B slices.  Output order
        # stays pass_0, pass_1 below, so validators and API consumers see the
        # same schema.
        proof0_t0 = time.perf_counter()
        block_proofs_0 = prover_0.phase2_prove(challenged_blocks)
        proof0_s = time.perf_counter() - proof0_t0

        ts_proofs_done = time.time_ns()

        # ── Serialize proofs ───────────────────────────────────
        # We serialize to a format that broadcast_service can read
        def serialize_block_proof(bp):
            zp = bp.zkgemm_proof
            sp = zp.sumcheck_proof
            return {
                "bi": bp.bi, "bj": bp.bj,
                "leaf_hash": bp.leaf_hash.hex(),
                "merkle_path": [[h.hex(), is_left] for h, is_left in bp.merkle_path],
                "bs": zp.bs, "n_inner": zp.n_inner,
                "claimed_sum": sp.claimed_sum,
                "sumcheck_rounds": [list(r.evals) for r in sp.rounds],
                "final_A_eval": zp.final_A_eval,
                "final_B_eval": zp.final_B_eval,
                "spot_A": [list(t) for t in zp.spot_A],
                "spot_B": [list(t) for t in zp.spot_B],
            }

        # ── Write final result ─────────────────────────────────
        serialize_t0 = time.perf_counter()
        result = {
            "gpu_index": gpu_idx,
            "seed": args.seed_hex,
            "n": n,
            "epoch_id": epoch_id,
            "status": "ok",
            "root_0": root_0.hex(),
            "root_1": root_1.hex(),
            "T_commit_0": T_commit_0.hex(),
            "T_commit_1": T_commit_1.hex(),
            "ts_pass0_ns": ts_after_pass0,
            "ts_pass1_ns": ts_after_pass1,
            "ts_proofs_ns": ts_proofs_done,
            "pass_0": {
                "merkle_root": root_0.hex(),
                "block_proofs": [serialize_block_proof(bp) for bp in block_proofs_0],
            },
            "pass_1": {
                "merkle_root": root_1.hex(),
                "block_proofs": [serialize_block_proof(bp) for bp in block_proofs_1],
            },
        }
        serialize_s = time.perf_counter() - serialize_t0
        result["timings"] = {
            "import_s": import_s,
            "pass0_s": pass0_s,
            "pass1_s": pass1_s,
            "proof1_s": proof1_s,
            "proof0_s": proof0_s,
            "serialize_s": serialize_s,
            "total_s": time.perf_counter() - worker_t0,
        }

    except Exception as e:
        result = {
            "gpu_index": gpu_idx,
            "seed": args.seed_hex,
            "status": "error",
            "error": str(e),
            "timings": {
                "import_s": import_s if "import_s" in locals() else 0.0,
                "total_s": time.perf_counter() - worker_t0,
            },
        }

    # Write final result
    output_path = os.path.join(output_dir, f"gpu_{gpu_idx}.json")
    write_json_atomic(output_path, result)


def main():
    parser = argparse.ArgumentParser(description="Nodexo GPU Worker")
    parser.add_argument("--daemon", action="store_true",
                        help="Long-lived mode: read JSON commands from "
                             "stdin, run cycles in-process, reuse CUDA "
                             "context across cycles. Eliminates per-cycle "
                             "subprocess + CUDA init overhead.")
    parser.add_argument("--seed-hex", help="8-byte seed as hex (one-shot mode)")
    parser.add_argument("--n", type=int, help="Matrix dimension")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu-idx", type=int, help="GPU device index")
    parser.add_argument("--executor-id", default="", help="Executor id for stale-worker cleanup attribution")
    parser.add_argument("--epoch-id", type=int)
    parser.add_argument("--num-challenges", type=int, default=4)
    parser.add_argument("--num-spot-checks", type=int, default=100)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    if args.daemon:
        return _run_daemon()

    # Backwards-compatible one-shot mode (kept for tests / manual runs).
    required = ["seed_hex", "n", "gpu_idx", "epoch_id", "output_dir"]
    missing = [k for k in required if getattr(args, k.replace("_", "_")) is None]
    if missing:
        parser.error(f"Missing required args for one-shot mode: {missing}")
    os.makedirs(args.output_dir, exist_ok=True)
    run_worker(args)


def _run_daemon() -> int:
    """Long-lived worker loop.

    Reads one JSON command per line from stdin. Two ops:
      {"op": "run",
       "seed_hex": "...", "n": int, "block_size": int, "gpu_idx": int,
       "epoch_id": int, "num_challenges": int, "num_spot_checks": int,
       "output_dir": "..."}
      {"op": "exit"}

    After each "run", emits a JSON line to stdout:
      {"op": "done", "epoch_id": int, "status": "ok"|"error", "error": "?"}

    CRITICAL: this is the production hot path. Importing torch, loading the
    ZkGEMM extension, and creating the CUDA context can cost several seconds
    per fresh process. Keeping the worker alive eliminates that tax from every
    proof cycle.
    """
    import sys
    import types
    # Pre-warm imports + CUDA context once. Subsequent run_worker calls
    # reuse them.
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        sys.path.insert(0, os.path.join(project_root, "zkgemm", "cuda"))
        sys.path.insert(0, project_root)

        import torch
        from zkgemm.prover import ZkGemmBlockProver  # noqa: F401 - prewarm import cache
        torch.cuda.init()
        torch.cuda.synchronize()
    except Exception as e:
        sys.stderr.write(f"daemon: cuda init failed: {e}\n")
        sys.stderr.flush()
        return 1
    sys.stdout.write(json.dumps({"op": "ready"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except Exception as e:
            sys.stderr.write(f"daemon: bad json: {e}\n")
            sys.stderr.flush()
            continue
        op = cmd.get("op")
        if op == "exit":
            return 0
        if op != "run":
            sys.stderr.write(f"daemon: unknown op {op!r}\n")
            sys.stderr.flush()
            continue
        cycle_args = types.SimpleNamespace(
            seed_hex=cmd["seed_hex"],
            n=int(cmd["n"]),
            block_size=int(cmd.get("block_size", 256)),
            gpu_idx=int(cmd["gpu_idx"]),
            executor_id=str(cmd.get("executor_id") or ""),
            epoch_id=int(cmd["epoch_id"]),
            num_challenges=int(cmd.get("num_challenges", 4)),
            num_spot_checks=int(cmd.get("num_spot_checks", 100)),
            output_dir=cmd["output_dir"],
        )
        os.makedirs(cycle_args.output_dir, exist_ok=True)
        try:
            run_worker(cycle_args)
            resp = {"op": "done", "epoch_id": cycle_args.epoch_id,
                    "status": "ok"}
        except Exception as e:
            resp = {"op": "done", "epoch_id": cycle_args.epoch_id,
                    "status": "error", "error": str(e)[:300]}
            sys.stderr.write(f"daemon: cycle {cycle_args.epoch_id} error: {e}\n")
            sys.stderr.flush()
        # Free VRAM from this cycle's tensors so the next cycle starts
        # with a clean allocator pool. Doesn't release the CUDA context
        # (that stays cached for the next cycle).
        try:
            import torch as _t
            _t.cuda.empty_cache()
        except Exception:
            pass
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    main()
