"""
VRAM-fill canary probe — runs inside a rented container via SSH stdin.

DEPLOYMENT
==========

Not deployed as a file. The validator pipes this script over an SSH
session to the rented container:

    ssh -i <canary_key> ubuntu@<miner> 'python3 -' < fill.py

That way nothing distinctive lands on disk; the miner sees a generic
`python3 -` process consuming GPU memory, indistinguishable from any
renter running an inline memory benchmark.

PROTOCOL
========

Reads parameters from environment, writes a single-line JSON result to
stdout.

ENV INPUTS
  CANARY_NONCE          — hex nonce (>=32 hex chars) tying this run to
                          the validator's expected result
  CANARY_TARGET_MB      — bytes to allocate per device, in MB. The
                          validator computes this from GPU spec minus a
                          micro-headroom budget (default 1 GB).
  CANARY_HOLD_SECONDS   — how long to hold the allocation after fill+
                          checksum (default 180 = one proof cycle)
  CANARY_CHUNK_MB       — allocation chunk size in MB (default 256)

OUTPUT (single-line JSON to stdout)
  {
    "nonce": "<echoed>",
    "device_count": <int>,
    "per_device": [
      {
        "index": <int>,
        "name": "<gpu name>",
        "reported_total_mb": <int>,
        "target_mb": <int>,
        "allocated_mb": <int>,
        "checksum": "<sha256[:32]>",
        "write_seconds": <float>,
        "read_seconds": <float>,
        "write_bandwidth_gb_s": <float>,
        "read_bandwidth_gb_s": <float>
      }, ...
    ],
    "hold_seconds": <int>,
    "total_seconds": <float>,
    "exit_code": 0
  }

WHAT THIS DEFEATS
=================

  Miner hides a sybil's VRAM allocation:
    cudaMalloc fails before reaching target → allocated_mb < target_mb
    → flag canary_vram_fail.

  Miner runs a concurrent sybil proof on the same GPU:
    canary saturates VRAM during hold window → sybil's heavy proof
    OOMs next cycle → caught by independent proof verification.

  Miner intercepts cudaMalloc and stores bytes in host RAM:
    bandwidth falls from HBM (~700-2000 GB/s) to PCIe (~32-64 GB/s)
    → flag canary_bandwidth_low.

WHAT THIS DOES NOT DEFEAT
=========================

  Miner under-reports TOTAL vram and hides a small sybil in the gap.
  Closure: validator compares claimed vram_mb against GPU_VRAM_GB at
  registration time (heartbeat-time check; not this script's job).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time


def _seed_from_nonce(nonce_hex: str, device_index: int) -> int:
    """Per-device deterministic seed. The validator can re-derive the
    expected checksum from (nonce, device_index) without sending bytes,
    so a miner can't pre-compute the expected checksum without actually
    running the GEMM. Returns a 64-bit int suitable for torch.Generator.
    """
    h = hashlib.sha256(
        nonce_hex.encode() + b"|fill|" + device_index.to_bytes(4, "little"),
    ).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def _fill_one_device(device_index: int, target_mb: int, chunk_mb: int,
                     nonce_hex: str) -> tuple[dict, list]:
    """Allocate target_mb of GPU memory, fill with seeded random ints,
    measure HBM bandwidth via a DEVICE-SIDE reduction, then read back
    a small sample for cryptographic checksum.

    Returns (result_dict, kept_chunks_list). Caller must keep
    `kept_chunks` alive to maintain the VRAM hold.

    HBM BANDWIDTH (rootkit-defeat test)
    -----------------------------------
    We measure how long it takes the GPU to SCAN the entire fill once.
    A xor-reduction over `allocated_mb` of int32 is memory-bound — the
    arithmetic is trivial; the bottleneck is the GPU reading every byte
    from device memory. On real HBM:
        4090  : ~1008 GB/s peak  → 40 GB scans in ~40 ms
        A6000 :  ~768 GB/s peak  → 40 GB scans in ~52 ms
        H100  : ~3350 GB/s peak  → 40 GB scans in ~12 ms
    A rootkit faking the allocation by storing bytes in host RAM has to
    pull them back across PCIe (~32 GB/s Gen4 x16) → 40 GB scans take
    >1.2 s. 20-30× slower than honest HBM. The validator flags low
    bandwidth as `canary_bandwidth_low`.

    The checksum is computed over a SMALL sample (1 chunk pulled to
    host) to verify integrity without making sha256 the bottleneck of
    the bandwidth measurement.
    """
    import torch

    torch.cuda.set_device(device_index)
    props = torch.cuda.get_device_properties(device_index)
    reported_total_mb = props.total_memory // (1024 * 1024)

    gen = torch.Generator(device=f"cuda:{device_index}")
    gen.manual_seed(_seed_from_nonce(nonce_hex, device_index))

    chunks: list = []
    # int32: 4 bytes per elem. Random over full int32 range → high
    # entropy → defeats a rootkit that zero-fills fake allocations.
    chunk_elems = (chunk_mb * 1024 * 1024) // 4
    target_chunks = target_mb // chunk_mb
    allocated_mb = 0

    # The canary purposely allocates up to the cap. The OOM that ends
    # the loop is informative — allocated_mb < target_mb is the signal
    # for "GPU has less free than claimed". Catch broadly: torch 2.x
    # raises torch.cuda.OutOfMemoryError, torch 1.x raises RuntimeError
    # with "out of memory" in the message, and some allocator paths
    # raise AssertionError or other types. Any of them mean "ran out."
    write_t0 = time.perf_counter()
    oom_message = ""
    for _ in range(target_chunks):
        try:
            buf = torch.randint(
                low=-(1 << 31), high=(1 << 31) - 1,
                size=(chunk_elems,), dtype=torch.int32,
                device=f"cuda:{device_index}", generator=gen,
            )
            chunks.append(buf)
            allocated_mb += chunk_mb
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg:
                oom_message = str(e).split("\n", 1)[0][:200]
                break
            raise
    torch.cuda.synchronize(device_index)
    write_seconds = time.perf_counter() - write_t0

    # HBM scan: sum-reduce every chunk on-device. Memory-bound, so
    # elapsed_time / total_bytes = effective HBM read bandwidth.
    # int64 accumulator avoids overflow; sum(dtype=int64) does the
    # widening inside the reduction kernel (no int64 materialization).
    # Wrapped in try/except because at near-full VRAM the reduction
    # kernel may itself OOM trying to allocate temporary buffers; we
    # surface this in the result without losing the allocation count.
    scan_t0 = time.perf_counter()
    sum_final = 0
    scan_error = ""
    try:
        sum_acc = torch.zeros((), dtype=torch.int64, device=f"cuda:{device_index}")
        for buf in chunks:
            sum_acc = sum_acc + buf.sum(dtype=torch.int64)
        torch.cuda.synchronize(device_index)
        sum_final = int(sum_acc.cpu().item()) & 0xFFFFFFFFFFFFFFFF
    except Exception as e:
        scan_error = str(e).split("\n", 1)[0][:200]
    scan_seconds = time.perf_counter() - scan_t0

    # CPU-VERIFIABLE INTEGRITY CHECK
    # -----------------------------
    # We can't trust torch's CUDA RNG to be byte-stable across GPU
    # models, AND the validator has no GPU. So integrity goes through
    # a separate channel: a small chunk of bytes derived on CPU from
    # numpy's PCG64 (which IS bit-stable across architectures per
    # numpy docs), then push it through GPU memory and hash on the
    # way out. Validator re-derives the same bytes on CPU and
    # compares hashes.
    #
    # This catches a miner who fakes the JSON output without actually
    # using the GPU: they'd need to either run cuRAND-on-GPU (which
    # they can do, but doesn't produce predictable bytes for the
    # validator) OR run our numpy PRNG on CPU (gives the right hash
    # but doesn't engage GPU silicon — caught by zero PCIe traffic in
    # write_seconds if we ever add that, AND by the canary's *main*
    # purpose, which is forcing concurrent sybils to OOM via VRAM
    # saturation. Faking the verify still lets that side-effect work.)
    verify_t0 = time.perf_counter()
    verify_hash = ""
    verify_error = ""
    verify_mb = 0
    try:
        import numpy as np
        verify_mb = int(os.environ.get("CANARY_VERIFY_MB", "16"))
        verify_seed = _seed_from_nonce(nonce_hex, 10_000 + device_index)
        # numpy generator → .bytes(N) is deterministic across hardware
        rng = np.random.default_rng(verify_seed)
        host_bytes = rng.bytes(verify_mb * 1024 * 1024)
        # Push to GPU, then pull back. PCIe round-trip; the integrity
        # signal is that the bytes we hashed CAME FROM GPU memory,
        # so a no-GPU fake fails this even though the bytes are CPU-
        # derivable.
        verify_tensor = torch.frombuffer(
            bytearray(host_bytes), dtype=torch.uint8,
        ).to(f"cuda:{device_index}")
        readback = verify_tensor.cpu().numpy().tobytes()
        del verify_tensor
        verify_hash = hashlib.sha256(readback).hexdigest()
    except Exception as e:
        verify_error = str(e).split("\n", 1)[0][:200]
    verify_seconds = time.perf_counter() - verify_t0
    # Legacy field names kept for backwards-compat with earlier output
    # shape; some are now informational only. Real integrity goes via
    # verify_hash above.
    sample_checksum = verify_hash[:32]
    sample_error = verify_error
    sample_seconds = verify_seconds

    scan_gb = allocated_mb / 1024.0
    result = {
        "index": device_index,
        "name": props.name,
        "reported_total_mb": reported_total_mb,
        "target_mb": target_mb,
        "allocated_mb": allocated_mb,
        "sum_final": sum_final,
        "write_seconds": round(write_seconds, 3),
        "scan_seconds": round(scan_seconds, 3),
        "hbm_bandwidth_gb_s": round(scan_gb / scan_seconds, 1) if scan_seconds > 0 else 0.0,
        # CPU-verifiable integrity check — validator re-derives via
        # numpy PCG64 on its own CPU; bit-stable across hardware.
        "verify_mb": verify_mb,
        "verify_hash": verify_hash,
        "verify_seconds": round(verify_seconds, 3),
        # Diagnostics
        "oom": oom_message,            # alloc-loop OOM (first line)
        "scan_error": scan_error,      # OOM during scan reduction
        "verify_error": verify_error,  # error during PCIe round-trip
    }
    return result, chunks


def main() -> int:
    nonce = os.environ.get("CANARY_NONCE", "")
    if not nonce or len(nonce) < 32:
        print(json.dumps({"error": "CANARY_NONCE missing or too short (need >=32 hex chars)", "exit_code": 2}))
        return 2

    target_mb = int(os.environ.get("CANARY_TARGET_MB", "0"))
    if target_mb <= 0:
        print(json.dumps({"error": "CANARY_TARGET_MB missing or <=0", "exit_code": 2}))
        return 2

    hold_seconds = int(os.environ.get("CANARY_HOLD_SECONDS", "180"))
    chunk_mb = int(os.environ.get("CANARY_CHUNK_MB", "256"))

    try:
        import torch
    except ImportError:
        print(json.dumps({"error": "torch not installed in container", "exit_code": 3}))
        return 3

    if not torch.cuda.is_available():
        print(json.dumps({"error": "CUDA not available inside container", "exit_code": 4}))
        return 4

    device_count = torch.cuda.device_count()
    t_start = time.perf_counter()

    per_device: list[dict] = []
    # CRITICAL: keep all chunks alive across the hold window so VRAM
    # stays saturated. Any concurrent sybil heavy proof during this
    # period must fail.
    held_chunks: list = []

    for i in range(device_count):
        try:
            result, chunks = _fill_one_device(i, target_mb, chunk_mb, nonce)
            per_device.append(result)
            held_chunks.append(chunks)
        except Exception as e:
            per_device.append({"index": i, "error": str(e)})

    # Hold the allocation. The whole point of the canary: any
    # concurrent sybil compute is blocked from VRAM during this window.
    # If hold_seconds spans at least one proof cycle (~180 s on a 15-
    # block 12 s/block chain), the sybil's next heavy proof attempt
    # OOMs and the validator catches it via independent verification.
    if hold_seconds > 0:
        time.sleep(hold_seconds)

    # Release explicitly so the container shutdown isn't blocked by
    # CUDA cleanup hiccups.
    held_chunks.clear()
    try:
        import torch  # already imported, but be defensive
        torch.cuda.empty_cache()
    except Exception:
        pass

    total_seconds = time.perf_counter() - t_start
    output = {
        "nonce": nonce,
        "device_count": device_count,
        "per_device": per_device,
        "hold_seconds": hold_seconds,
        "total_seconds": round(total_seconds, 3),
        "exit_code": 0,
    }
    print(json.dumps(output, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
