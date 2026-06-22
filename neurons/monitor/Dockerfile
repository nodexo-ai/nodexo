# Nodexo Monitor — pinned-image GPU observer.
#
# Built and pushed to ghcr.io/nodexo/monitor:<version>. The image
# digest is what the validator pins; operators run by digest, not tag.
#
# Sysbox-compatible (no privileged mode required); NVIDIA passthrough
# via --gpus on `docker run`. Uses pynvml for per-PID accounting and
# substrate-interface for SR25519 signing.
#
# Build:
#   docker build -t nodexo-monitor:dev -f neurons/monitor/Dockerfile .
#
# Run (manual; in production the miner daemon spawns this automatically):
#   docker run -d --name nodexo-monitor --gpus all --runtime sysbox-runc \
#     -e NODEXO_EXECUTOR_ID=<hex> \
#     -e NODEXO_VALIDATOR_URLS=http://validator:8443 \
#     -e NODEXO_SUBTENSOR_NETWORK=test \
#     -e NODEXO_NETUID=468 \
#     -v nodexo_monitor_data:/data \
#     nodexo-monitor:dev
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal runtime — python3, pip, and the deps the monitor itself uses.
# No build toolchain, no compilers, no shell utilities the miner could
# use to pivot. Surface area kept narrow on purpose.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what the monitor needs. The whole repo isn't shipped — the
# validator's image-digest check would have to pin the entire tree if
# we did. Keep it small.
COPY common/__init__.py        common/__init__.py
COPY common/crypto.py          common/crypto.py
COPY common/proof_schedule.py  common/proof_schedule.py
COPY neurons/__init__.py       neurons/__init__.py
COPY neurons/monitor/          neurons/monitor/

# Pin runtime deps. Intentionally narrow: pynvml for per-PID NVML
# accounting, aiohttp for outbound HTTP, substrate-interface for the
# SR25519 keypair (signing reports). NO bittensor — pulling it would
# drag async-substrate-interface which conflicts with scalecodec
# (py-scale-codec, which substrate-interface depends on), and we
# don't need any bittensor-specific functionality here. Chain-anchored
# behavioral classification falls back gracefully to telemetry-only
# mode in this image.
RUN pip3 install --no-cache-dir \
        pynvml==11.5.0 \
        aiohttp==3.9.5 \
        substrate-interface==1.7.10

# Keypair + state persist here. Mount a docker volume to keep the
# monitor's identity stable across restarts.
RUN mkdir -p /data
VOLUME /data

# No inbound ports. The monitor only makes outbound HTTPS to the
# validator. This is intentional — reduces attack surface to zero
# from the network side.

# Run unprivileged. Even with sysbox the monitor never needs root.
RUN useradd -u 1000 -m monitor && chown -R monitor:monitor /app /data
USER monitor

ENTRYPOINT ["python3", "-m", "neurons.monitor.monitor"]
