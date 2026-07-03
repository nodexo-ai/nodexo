# Miner Quickstart

This guide is for GPU operators joining Nodexo.

## Install

One-command install:

```bash
curl -fsSL https://raw.githubusercontent.com/nodexo-ai/nodexo/main/install.sh | bash
```

Manual install:

```bash
git clone https://github.com/nodexo-ai/nodexo.git
cd nodexo
bash scripts/setup_miner.sh
```

Commands below assume you are in the repo root after setup. Use the venv
commands (`.venv/bin/btcli`, `.venv/bin/nodexo`) unless you explicitly activate
the venv.

## Prerequisites

- Ubuntu/Debian GPU host
- [Supported NVIDIA GPU](SUPPORTED_GPUS.md)
- NVIDIA driver installed; `nvidia-smi` must work before setup
- 4 CPU cores, 16 GB RAM, and 40 GB free Docker storage minimum
- 8+ CPU cores, 32+ GB RAM, and 100+ GB free Docker storage recommended
- Synchronized system clock; signed validator traffic uses a short freshness
  window and setup verifies NTP before miner startup
- Public miner API port open, default `8091`
- Rental TCP port range open, default `20000-20100`
- Bittensor wallet/hotkey registered on the target subnet

The setup script installs Docker, NVIDIA container toolkit, Sysbox, PM2,
Python dependencies, required warm images, and the prebuilt ZkGEMM proof
extension. Required renter images are digest-checked. Optional catalog images
are pulled only while enough Docker storage remains for the minimum rental
profile.

Setup also installs a persistent `systemd-timesyncd` drop-in for Nodexo's NTP
servers and falls back to `chrony` if the host is not synchronized quickly.
Do not start public miners with an unsynchronized clock. Set
`NODEXO_SKIP_CLOCK_SYNC_CHECK=1` only when the host clock is managed by the
provider or another external time service.

Setup checks the detected GPU model before registration. Unsupported models
stop before they are advertised on chain.

## 1. Install Host Dependencies

From the repository root:

```bash
bash scripts/setup_miner.sh
```

The script uses the shipped artifact manifest by default:

```text
https://pub-ef00d9a98f734d94af3c8904eba0eb11.r2.dev/zkgemm/v0.1.2/manifest.json
```

Override only for custom releases by setting `ZKGEMM_CUDA_MANIFEST_URL` to the
custom manifest before running `scripts/setup_miner.sh`.

Setup refuses to continue without `sysbox-runc`. Public miners must provide a
servable rental endpoint, the required warm images, and enough Docker storage
for the minimum rental profile before they can enter the rental marketplace.

Sysbox installation may reconfigure or restart Docker. If containers are
already running, setup refuses to continue unless you explicitly allow the
restart:

```bash
NODEXO_SYSBOX_ALLOW_DOCKER_RESTART=1 bash scripts/setup_miner.sh
```

## 2. Create Or Import Wallet

```bash
.venv/bin/btcli wallet new_coldkey --wallet-name nodexo_miner
.venv/bin/btcli wallet new_hotkey  --wallet-name nodexo_miner --hotkey default
```

If you already have a registered miner hotkey, import or copy it into the
standard Bittensor wallet path before starting the miner.

## 3. Register On Subnet

Public testnet preview:

```bash
.venv/bin/btcli subnet register \
  --wallet-name nodexo_miner \
  --hotkey default \
  --netuid 468 \
  --network test
```

Mainnet:

```bash
.venv/bin/btcli subnet register \
  --wallet-name nodexo_miner \
  --hotkey default \
  --netuid 106 \
  --network finney
```

| Target | Runtime flag | Netuid | Chain config |
|---|---|---:|---|
| Public testnet preview | `--subtensor-network test` | `468` | `chain_config_testnet.json` |
| Paid mainnet | `--subtensor-network finney` | `106` | `chain_config_mainnet.json` |

Mainnet requires `chain_config_mainnet.json` to be published in the release
repo after mainnet contracts are deployed. The same miner code path is used for
both networks.

## 4. Start With PM2

Recommended production path:

```bash
bash scripts/setup_endpoint_proxy.sh --role miner --public-port 8091
cp ecosystem.config.example.cjs ecosystem.config.cjs
```

Edit:

```text
--wallet
--hotkey
--subtensor-network
--subtensor-endpoint optional private/local RPC for that network
--endpoint
```

Then start:

```bash
pm2 start ecosystem.config.cjs --only miner-nodexo
pm2 logs miner-nodexo --lines 80
pm2 save
```

The miner derives its EVM mirror address from the Bittensor hotkey seed,
registers that EVM identity, registers the executor, renews its lease, reports
hardware, and starts proof generation.

The hotkey EVM mirror needs a small TAO balance for EVM registration and
executor lease transactions. Check the derived EVM address and SS58 mirror:

```bash
python scripts/show_evm_info.py --wallet nodexo_miner --hotkey default \
  --subtensor-network test
```

Then fund it from the coldkey:

```bash
.venv/bin/nodexo --wallet nodexo_miner --hotkey default \
  --subtensor-network test fund --amount 0.05 --yes
```

Use `--subtensor-network finney` for mainnet after mainnet launch.

## Validator Discovery And Rental Control

Miners use two separate validator paths:

- Proof, heartbeat, and monitor traffic goes to active validators discovered
  from `ValidatorRegistry`, plus validator-permit neurons that publish a native
  Bittensor axon endpoint.
- Rental lifecycle routes (`POST /containers`, SSH-key updates, termination)
  require a signed request from an allowed validator hotkey.

Miner discovery re-checks the current metagraph validator-permit flag for both
`ValidatorRegistry` rows and native axon endpoints before broadcasting proofs.
Stale registry rows do not remain trusted after their UID loses permit.
If an advertised validator endpoint remains permitted but stops responding, the
miner records hard failures and temporarily quarantines that endpoint so a dead
validator does not block proof, heartbeat, or monitor broadcast. Quarantine is a
fallback for miner liveness; validators should still deactivate or update stale
chain-advertised endpoints.

For production, keep strict mode enabled. The primary validator hotkey is
loaded from subnet runtime config; `NODEXO_ALLOWED_VALIDATOR_HOTKEYS` is only a
local/emergency override:

```env
NODEXO_STRICT_ALLOWLIST=1
```

This keeps rental and canary authority limited to the primary validator even if
other validators join the subnet. No-EVM validators can still receive proof
traffic when they have validator permit and serve a native axon endpoint on
subtensor. They do not receive container authority.

Miners should also fetch subnet runtime config from the public validator relay.
That keeps worker timeouts and cycle-skipping preflight aligned with the
validator timing policy. The release also ships a bundled last-known config
snapshot so first startup has a sane fallback if the relay is briefly
unavailable:

```env
NODEXO_SUBNET_CONFIG_URL=https://validator.nodexo.ai/subnet-config
```

If the subnet alpha-stake gate is active, use `nodexo fleet` or the web
operator dashboard to see current hotkey stake, required stake, and any grace
deadline. The requirement is computed from all active executors attached to the
miner hotkey.

## Supported GPUs

Production miners must run a supported GPU model. Setup verifies the detected
GPU before registration; unsupported models stop before being advertised.

The current list is published in the web docs under **Supported GPUs** and in
`docs/SUPPORTED_GPUS.md`.

## Bootstrap Shortcut

For a one-command first start, setup can write and start a PM2 launcher:

```bash
bash scripts/setup_miner.sh --start \
  --wallet nodexo_miner \
  --hotkey default \
  --subtensor-network test \
  --endpoint http://YOUR_PUBLIC_IP:8091
```

That direct shortcut binds the miner API on public port `8091`. For a hardened
host with nginx in front, create the proxy first and bind the Python daemon to
the loopback backend:

```bash
bash scripts/setup_endpoint_proxy.sh --role miner --public-port 8091
bash scripts/setup_miner.sh --start \
  --wallet nodexo_miner \
  --hotkey default \
  --subtensor-network test \
  --port 18091 \
  --bind-host 127.0.0.1 \
  --endpoint http://YOUR_PUBLIC_IP:8091
```

Mainnet uses the same command with `--subtensor-network finney`, netuid `106`
and `chain_config_mainnet.json` resolved from that network, and a
miner-reachable HTTPS endpoint after the mainnet chain config is published.
Use `--subtensor-endpoint` only when overriding the RPC transport for the
selected network.

This is convenient for testnet testing. For production, keep an edited
`ecosystem.config.cjs` under operator control.

## Ports

Open these in the host firewall and provider firewall:

```text
8091/tcp        miner API endpoint
20000-20100/tcp rental ports
```

When using nginx, the miner daemon listens on the loopback backend
`127.0.0.1:18091`. Do not open `18091/tcp` to the Internet.

Each rental uses one port for SSH and maps the remaining assigned TCP ports
one-to-one into the container. The setup/miner startup preflights a sample of
the rental port range before registration. If the sample is unreachable, strict
mode refuses to register.

## Image Cache

The default warm images are PyTorch CUDA runtime, CUDA runtime, and Ubuntu.
They are pulled before registration and must match the approved registry
digest. Setup fails if a required image cannot be cached. Miner startup defaults
to `MINER_RENTAL_IMAGE_PREFLIGHT=strict`, which verifies required images are
already cached and refuses to start if any are missing. Optional catalog images
such as PyTorch devel, CUDA devel, and vLLM are warmed opportunistically during
setup; runtime only warns when they are not cached.

Useful overrides:

```text
MINER_RENTAL_IMAGE_PREFLIGHT=pull|strict|warn|off
MINER_OPTIONAL_IMAGE_PREFLIGHT=pull|warn|off
MINER_ALLOW_COLD_IMAGE_PULL=0|1
MINER_MIN_RENTAL_STORAGE_GB=30
MINER_RENTAL_STORAGE_HEADROOM_GB=5
SETUP_PULL_OPTIONAL_IMAGES=0
CANARY_IMAGES=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime
```

`strict` fails immediately without pulling. `pull` attempts to cache required
images, then fails if any are still missing. `warn` and `off` are explicit
operator overrides and should not be used for production rental hosts. Rental
provisioning itself is cached-image-only by default; set
`MINER_ALLOW_COLD_IMAGE_PULL=1` only for controlled cold-pull testing.

Custom image catalogs can be configured with `NODEXO_IMAGE_CATALOG` and
`NODEXO_REQUIRED_RENTAL_IMAGES`. Add `NODEXO_IMAGE_DIGESTS` for custom images that
should be digest-pinned. Canary images must include Python, torch, and numpy.

## CLI After Install

The CLI is not required for installation. It is useful for operations:

```bash
nodexo fleet
nodexo inventory
nodexo fleet --chain-direct
```

If the venv is not active, use `.venv/bin/nodexo`.

Use `nodexo fleet --chain-direct` for operator diagnostics that read chain
state directly.

Public API rental commands that sign x402 payments use the optional Node helper
in `scripts/public-cli.mjs`:

```bash
npm install
nodexo quote --gpu A6000 --duration 1h
nodexo rent --gpu A6000 --duration 1h --ssh-key ~/.ssh/id_ed25519.pub
```

During public testnet preview, rent, extend, and top-up commands are blocked
before payment signing, credit reserve, or provisioning.

Credit-backed API-key rentals use the same helper with
`NODEXO_API_KEY=vc_...`.

See `docs/CLI_REFERENCE.md` for the full renter, account API-key, and operator
command surface.

## Link Miner Hotkey To Account

The Operator page can link this miner hotkey to a signed-in web account. It
creates a one-time challenge and shows a complete
`nodexo operator-claim sign ...` command with that challenge already filled in.
Run the generated command on a machine that has the miner hotkey.

This proves control of the miner hotkey without moving the hotkey into a
browser wallet.

## Artifact Format

The installer supports both raw `.so` artifacts and wheels in the same manifest.
If both match the host ABI, it prefers the wheel. The current testnet artifact
release is raw `.so` artifacts for:

```text
cp310 / cp311 / cp312
Torch 2.10
CUDA 12.8
linux_x86_64
```

Setup verifies SHA-256, imports the extension, and runs a native/CUDA smoke test
before the miner can register.

## 5. Harden The Public Endpoint

Production miners should put nginx in front of the Python daemon and bind the
daemon to loopback:

```bash
bash scripts/setup_endpoint_proxy.sh --role miner --public-port 8091
```

Then run the miner with:

```text
--bind-host 127.0.0.1
--port 18091
--endpoint http://YOUR_PUBLIC_IP_OR_DOMAIN:8091
```

If you have a real domain and certificate, terminate HTTPS at nginx:

```bash
bash scripts/setup_endpoint_proxy.sh --role miner \
  --tls existing \
  --server-name gpu1.example.com \
  --public-port 443 \
  --cert-path /etc/letsencrypt/live/gpu1.example.com/fullchain.pem \
  --key-path /etc/letsencrypt/live/gpu1.example.com/privkey.pem
```

Use `MINER_STRICT_PUBLIC_ENDPOINT=1` only with a validator-trusted HTTPS
certificate. Self-signed HTTPS is useful for local testing, but validators using
standard TLS verification will reject it unless explicitly configured otherwise.

For Cloudflare-protected domains, prefer `https://gpu.example.com` on port 443
for the miner API and keep nginx bound to the loopback backend. Do not rely on
standard Cloudflare HTTP proxy for rental TCP ports; the `20000-20100/tcp`
range must remain directly reachable unless you deliberately add a TCP proxy
product. See `docs/ENDPOINT_SECURITY.md`.
