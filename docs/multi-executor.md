# GPU Subset Selection (and why splitting a host is not supported yet)

> **Status: the GPU subset filter ships; splitting one host across several
> miner instances does NOT work yet and will get both executors banned.**
> Read "What still blocks a host split" before you plan around this.

A renter who needs 3 GPUs should not have to rent your whole 8-GPU node. The
intended shape of that is **one miner process per GPU subset**, and the piece
that was missing is a way to make each process *see* only its own GPUs: NVML
enumerates every device on the box and ignores `CUDA_VISIBLE_DEVICES`, so
without a filter two processes would derive the **same** `executor_id`.

This change adds that filter. It is necessary for a host split but, on its
own, not sufficient — see below.

## The filter

| Variable | Meaning |
| --- | --- |
| `NODEXO_GPU_UUIDS` | Comma-separated GPU UUIDs to claim. Case-insensitive; the `GPU-` prefix is optional. **Preferred.** |
| `NODEXO_GPU_INDICES` | Comma-separated NVML indices to claim. Convenient, but indices can move across driver or hardware changes. |

Set **at most one** of them. Setting both, naming a UUID or index that is not
present, repeating an entry, or selecting nothing is a hard error: the miner
logs `GPU subset selection failed: …` and refuses to start rather than
silently registering a GPU set you did not ask for. With neither set,
behaviour is exactly as before — all GPUs, one executor.

### The subset must cover physical indices 0..n-1

The miner refuses to start on any other selection. `NODEXO_GPU_INDICES=0,1,2`
is accepted; `4,5,6,7` and `0,2` are rejected with

```
GPU subset selection failed: GPU subset {4,5,6,7} is not supported: proof
workers are pinned by ordinal position (CUDA_VISIBLE_DEVICES=<slot>, slot in
range(gpu_count)), not by physical index, ...
```

The reason is in `neurons/miner/services/proof_service.py`. `ProofService` is
constructed with a GPU **count**, not a device list
(`gpu_count=len(gpus)` in `neurons/miner/miner.py`). It then iterates
`for gpu_idx in range(self.gpu_count)` and launches each worker with
`CUDA_VISIBLE_DEVICES=str(gpu_idx)` — an absolute overwrite you cannot
pre-empt by exporting `CUDA_VISIBLE_DEVICES` on the parent. So **proof slot k
always runs on CUDA device k**, whichever physical GPUs this executor
selected. An executor holding physical 4..7 would prove on physical 0..3: its
neighbour's silicon.

Those proofs would still *verify* — the validator re-derives the seed from
the reported 0-based slot (`neurons/validator/proof/verifier.py`), and miner
and validator agree on `0..N-1` — but the wrong hardware did the work. The
guard exists so that configuration cannot be reached.

So what this filter buys you **today** is "advertise and mine on only the
first N of my GPUs", not a host split.

Physical `GpuInfo.index` values are still preserved rather than renumbered
inside a subset, because that index is the key into the per-physical-GPU MIG
map assembled in `build_hw_static()`. It is *not* what pins proof workers.

## What still blocks a host split

Every item below is outside this change and must land before two miner
instances can share a box. None of them is a configuration mistake you can
avoid — they fire on the *correct* setup.

1. **`cross_monitor_pid_overlap` (HARD ban, both executors).** Each miner
   instance spawns its own monitor sidecar, launched with `--gpus all` and
   `--pid host` (`neurons/miner/miner.py`). `_sample_nvml()` in
   `neurons/monitor/monitor.py` is called with no `gpu_index`, so it walks
   `range(nvmlDeviceGetCount())` — every GPU on the host — and the monitor
   reads no GPU-subset env var. Both monitors therefore report an identical
   host-wide process list, and scanner signal 3b
   (`neurons/validator/sybil/scanner.py`) flags every executor sharing a PID.
   `cross_monitor_pid_overlap` is in `HARD_SYBIL_REASONS` (`common/db.py`),
   so both executors are removed from the rentable set and scored zero. The
   trigger is simply "any GPU process is running on the host" — your own
   `gpu_worker` guarantees that.

2. **`canary_pid_overlap` (HARD ban, the innocent sibling).** When the
   validator self-rents executor A and runs a known-PID CUDA workload,
   executor B's host-wide monitor reports that PID, and
   `neurons/validator/sybil/canary.py` step 8b flags B. Canaries are
   scheduled routinely, so a split host gets banned as a matter of course.

3. **`gpu_process_outside_rental` (HARD ban).** Scanner signal 0b compares
   docker-cgroup GPU processes in an executor's monitor report against the
   containers *that* miner reported. `containers.json` is per-instance, so
   instance A has no knowledge of instance B's renter containers and flags
   them.

4. **Rental and canary containers get `--gpus all`.** No validator caller
   supplies `gpu_uuids` on provision
   (`neurons/validator/api/routes/rent.py` and
   `neurons/validator/canary/runner.py` both call `provision(...)` without
   it), and the miner falls back to `--gpus all`
   (`neurons/miner/services/docker_service.py`) without clamping to its own
   detected set. A paying renter on instance A would receive all 8 physical
   GPUs. Worse, the canary's VRAM-fill workload
   (`neurons/validator/canary/fill.py`) allocates on every device
   `torch.cuda.device_count()` reports and holds it, which is designed to
   OOM a concurrent sybil proof — on a split host it OOMs the honest
   neighbour instead.

Fixing these means scoping the monitor to the executor's GPU subset, teaching
scanner signal 3b and the canary to skip peers with disjoint `gpu_uuids`
sets, clamping container GPU passthrough miner-side, and passing the selected
device list into `ProofService`. Until then, do not run two instances on one
host.

## Subsets must be disjoint — but that is not the main risk

If you do get a host split working, never list the same GPU UUID under two
executors. The scanner treats a GPU UUID appearing under two `executor_id`s
as sybil evidence and flags a shared `system_uuid` when the GPU-UUID sets
overlap (`neurons/validator/sybil/scanner.py`, signals 1 and 2).

Note the actual severity: `gpu_uuid_dup` and `system_uuid_dup` are in
`SOFT_SYBIL_REASONS` (`common/db.py`) — self-reported and fakeable, so they
sit in the table for operator review rather than auto-banning. The hard-ban
reasons listed in the previous section are the ones that zero your score, and
they fire on the *correct* configuration, not the overlapping one.

Nothing detects host-locally that a sibling instance already claims a GPU.
Two instances given the same subset and the same `system_uuid` derive the
**same** `executor_id` and will flap one registration between two endpoints,
with no sybil flag at all. Double-check the launch lines.

After a GPU swap or driver upgrade, re-read `nvidia-smi` and update the
lists — a changed GPU set changes the `executor_id` and requires
re-registration.

## Check your GPU count is calibrated first

Partitioning changes the GPU-count group your executor advertises, and the
miner **refuses to start** on an uncalibrated count:

```
GPU model/count 'NVIDIA …' × 7 has no calibrated timing model.
```

Coverage is a range check against `calibrated_gpu_counts` in
`common/proof_timing.py` (`min(...) <= count <= max(...)`), and the shipped
rows vary by model: some are `(1, 2, 4, 8)`, others `(1, 4)`, `(1, 2, 4)`, or
`(1,)`. A 4×+4× split of an H100 host is inside `(1, 2, 4, 8)`; a 7-GPU
instance on a model calibrated `(1, 4)` will refuse to start outright and
take a working miner offline. Look up your model's row before you plan a
split, and prefer counts that already appear in it.

`NODEXO_ALLOW_UNCALIBRATED_GPU=1` bypasses the check. It is for calibration
runs only, not a production workaround — the timing model is what the
validator scores you against.

Second-order: validator deadlines are computed from `expected_gpu_count`
against a model fitted on unpartitioned hosts
(`neurons/validator/validator.py`). Two instances contending for host CPU,
RAM bandwidth, and PCIe push measured deltas toward the tail of that fit, so
a partitioned host sits closer to the rejection boundary than the calibration
data represents.

## Worked example (for when the blockers above are fixed)

List the UUIDs first:

```bash
nvidia-smi --query-gpu=index,uuid,name --format=csv
```

```
index, uuid, name
0, GPU-1a2b3c4d-..., NVIDIA H100 80GB HBM3
...
7, GPU-9c0d1e2f-..., NVIDIA H100 80GB HBM3
```

A 4×+4× split — both counts are in the H100 row's `(1, 2, 4, 8)`:

**Instance A — physical GPUs 0-3:**

```bash
NODEXO_DATA_DIR=$HOME/.nodexo-a \
NODEXO_GPU_UUIDS=GPU-1a2b3c4d-...,GPU-...,GPU-...,GPU-... \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
PYTHONPATH=$PWD:$PWD/zkgemm/cuda \
.venv/bin/python -u -m neurons.miner.miner \
  --wallet nodexo_miner --hotkey default \
  --subtensor-network finney \
  --port 18091 --bind-host 127.0.0.1 \
  --endpoint http://YOUR_PUBLIC_IP:8091 \
  --rental-port-start 20000 --rental-port-end 20049
```

**Instance B — physical GPUs 4-7:** the same, with `NODEXO_DATA_DIR`
`$HOME/.nodexo-b`, the other four UUIDs, `--port 18092`,
`--endpoint http://YOUR_PUBLIC_IP:8092`, and
`--rental-port-start 20050 --rental-port-end 20099`.

Instance B is **currently rejected at startup** by the 0..n-1 guard, since it
holds physical 4-7. That guard comes off when `ProofService` takes the
selected device list.

Per instance, these must all differ:

- `NODEXO_DATA_DIR` — holds `executor_identity.json` and `containers.json`.
  Sharing it would make the two processes fight over one identity.
- The GPU subset (`NODEXO_GPU_UUIDS`).
- The miner API port (`--port`) and its public `--endpoint`.
- The rental port range — the ranges must **not overlap**, or the two
  instances will hand the same host port to two different renter containers.

The same wallet/hotkey can back both executors — the validator scores each
`executor_id` separately and sums under the hotkey
(`neurons/validator/scoring/scorer.py`, `aggregate_by_hotkey`). Under PM2,
copy the `miner-nodexo` app in `ecosystem.config.example.cjs` once per
instance and give each copy its own `name`, `env`, and port arguments.

### Reaching both instances from outside

`--port 18091` with `--endpoint http://YOUR_PUBLIC_IP:8091` is not a typo:
the miner **listens** on loopback 18091 and **advertises** public 8091, and a
reverse proxy bridges the two. That is the hardened pattern from
`docs/MINER_QUICKSTART.md`, and without the proxy step nothing listens on
8091 publicly — validators cannot reach you, your executor goes stale, and
you score zero.

`scripts/setup_endpoint_proxy.sh` is **single-instance**: it writes a fixed
`nodexo-miner.conf`, so running it a second time for 8092 overwrites instance
A's vhost and takes A offline too. For a second instance, hand-write an
additional nginx `server` block pointing 8092 at `127.0.0.1:18092`. If you
would rather not, drop `--bind-host 127.0.0.1` and bind each instance
directly on its public port.

Open **both** public endpoint ports and **both** rental port ranges on the
firewall; the port-range preflight only checks the range that instance was
given.

## `CUDA_DEVICE_ORDER=PCI_BUS_ID`

Set it on every instance. CUDA's default ordering (`FASTEST_FIRST`) does not
have to match NVML's enumeration order, and the subset is selected by NVML
UUID/index while proof workers are pinned by CUDA ordinal. With
`CUDA_DEVICE_ORDER=PCI_BUS_ID` the two orderings agree, which is what makes
the 0..n-1 guard above meaningful. (If `ProofService` is later changed to
pin by `GPU-<uuid>` rather than by integer, this requirement disappears —
UUID pinning is immune to both `CUDA_DEVICE_ORDER` and driver reordering.)

## Isolation caveat: this is not MIG

Splitting a host by whole GPUs gives each tenant its own devices, but the
tenants still share the PCIe fabric, host RAM bandwidth, CPU cores, storage,
and network — and, on NVLink/NVSwitch systems, the interconnect. A noisy
neighbour on instance B can measurably slow a job on instance A. That is a
performance boundary, not a hardware isolation boundary.

MIG is the stronger guarantee: it partitions SM clusters, L2, and memory
controllers behind the GPU's own MMU, and each slice gets its own UUID (see
`detect_mig_for_gpu` in `neurons/miner/services/hardware_service.py`). If you
advertise partitioned capacity, be explicit with renters about which of the
two they are getting.

## Co-tenancy caveat: proofs and rentals contend

Each instance decides whether to run a heavy proof by looking at *its own*
container state — `DockerService.has_active_containers()` is per-instance and
reads that instance's `containers.json`. Instance A therefore does not know
that instance B has a live rental, and may start a heavy proof beside it.

Do not assume disjoint GPU subsets keep that proof off the renter's device.
Under the blockers above they do not: proof workers are pinned by ordinal
slot, and rental containers are launched with `--gpus all`. Even once both
are fixed, the two tenants still compete for CPU, host RAM, and PCIe
bandwidth.

`/hardware` telemetry is scoped to this executor's subset
(`get_gpu_utilization()`), as is the MIG tier advertisement in `hw_static`,
so a renter on instance A is not shown instance B's GPUs. The monitor
sidecar's reporting is **not** yet scoped — that is blocker 1.
