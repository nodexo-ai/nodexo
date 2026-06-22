# Validator Quickstart

This guide is for operators running a Nodexo validator. Validators verify
proofs, maintain the local executor projection, compute scores, and set
Bittensor weights.

## Install

One-command validator install:

```bash
curl -fsSL https://raw.githubusercontent.com/nodexo-ai/nodexo/main/install.sh | bash -s -- --validator
```

Manual install:

```bash
git clone https://github.com/nodexo-ai/nodexo.git
cd nodexo
bash scripts/setup_validator.sh
```

Commands below assume you are in the repo root after setup. Use the venv
commands (`.venv/bin/btcli`, `.venv/bin/nodexo`) unless you explicitly activate
the venv.

## Prerequisites

- Ubuntu/Debian host with a stable public endpoint
- Python 3.10+
- CPU-only is fine. A validator does not need an NVIDIA GPU, Docker GPU access,
  or `nvidia-smi`.
- Synchronized system clock. Validator proof timing uses the host wall clock,
  so setup refuses to continue when NTP cannot be verified.
- Bittensor wallet/hotkey registered on the target subnet
- Chain config file present in the repo root. The validator selects
  `chain_config_testnet.json` from `--subtensor-network test` and
  `chain_config_mainnet.json` from `--subtensor-network finney`.
- Durable validator state directory
- A funded validator EVM mirror only if this validator should publish a custom
  HTTPS/domain endpoint in `ValidatorRegistry`

## 1. Install Dependencies

```bash
bash scripts/setup_validator.sh
```

The setup script installs Python dependencies, creates the venv, tightens local
state-file permissions, installs and validates the native `zkgemm_cuda`
verifier helper artifact, creates the canary SSH key if missing, and probes the
configured subtensor RPC.

Setup also installs a persistent `systemd-timesyncd` drop-in for Nodexo's NTP
servers and falls back to `chrony` if the host is not synchronized quickly. Do
not run a public validator with an unsynchronized clock; it can mis-score proof
timing. Set `NODEXO_SKIP_CLOCK_SYNC_CHECK=1` only when the host clock is
managed by the provider or another external time service.

The `zkgemm_cuda` name is historical: validators use the artifact's CPU-native
PRF/slice helpers, while miners use its CUDA proof kernels. Validator setup runs
the native CPU smoke check only. It does not require a GPU.

## 2. Create Or Import Wallet

```bash
.venv/bin/btcli wallet new_coldkey --wallet-name nodexo_vali
.venv/bin/btcli wallet new_hotkey  --wallet-name nodexo_vali --hotkey default
```

Register the hotkey on the subnet:

Public testnet preview:

```bash
.venv/bin/btcli subnet register \
  --wallet-name nodexo_vali \
  --hotkey default \
  --netuid 468 \
  --network test
```

Mainnet:

```bash
.venv/bin/btcli subnet register \
  --wallet-name nodexo_vali \
  --hotkey default \
  --netuid 106 \
  --network finney
```

| Target | Runtime flag | Netuid | Chain config |
|---|---|---:|---|
| Public testnet preview | `--subtensor-network test` | `468` | `chain_config_testnet.json` |
| Paid mainnet | `--subtensor-network finney` | `106` | `chain_config_mainnet.json` |

Mainnet requires `chain_config_mainnet.json` to be published in the release
repo after mainnet contracts are deployed. The same validator code path is used
for both networks.

## 3. Configure State

`scripts/setup_validator.sh` provisions a local Postgres database by default and
writes the validator DB settings to:

```text
/etc/nodexo/validator.env
```

The PM2 ecosystem example reads that file automatically. The generated local
Postgres config uses this shape:

```env
NODEXO_VALIDATOR_DB_URL=postgresql+psycopg2://nodexo:REDACTED@127.0.0.1:5432/nodexo_validator
DB_URL=postgresql+psycopg2://nodexo:REDACTED@127.0.0.1:5432/nodexo_validator
VALIDATOR_REQUIRE_POSTGRES=1
```

For managed or external Postgres, rerun setup with `--db-url` or edit the env
file before starting PM2. SQLite is only a local/dev fallback and requires an
explicit setup opt-in:

```bash
bash scripts/setup_validator.sh --sqlite
```

## 4. Choose Validator Role

Recommended role defaults:

- Validator discovery is permit-gated for both publication paths. Miners
  broadcast only to validator-permit UIDs, whether the endpoint comes from
  `ValidatorRegistry` or native Bittensor axon metadata.
- No-EVM verifier: `VALIDATOR_NO_EVM=1`; it publishes native
  Bittensor axon metadata.
- EVM endpoint validator: EVM enabled only to publish a custom
  `ValidatorRegistry` endpoint and enable validator EVM write authority.
- Low-resource validator: follower mode; it sets weights from published
  network state instead of running local proof verification.

### No-EVM Validator

Use this when an operator wants to verify proofs and set weights without
funding an EVM mirror:

```text
VALIDATOR_NO_EVM=1
# or pass --no-evm to neurons.validator.validator
NODEXO_VALIDATOR_DISCOVERY_MODE=native
```

No-EVM validators read `ComputeRegistry`, verify signed proof and heartbeat
traffic, score miners, and set Bittensor weights. They skip EVM registration,
rental control, rental release sweepers, and on-chain `reportOffline`.

Because a no-EVM validator does not publish itself in `ValidatorRegistry`, it is
discovered through native Bittensor axon metadata instead. Start it with a
public IP endpoint, for example `http://203.0.113.10:9443`; startup publishes
that external IP/port as native axon metadata without binding a second public
listener. Miners broadcast proof, heartbeat, and monitor traffic only to axons
whose UID has validator permit. This does not grant rental or container
authority.

### EVM Endpoint Validator

Use this only when the validator needs to advertise a custom HTTPS/domain URL
through `ValidatorRegistry`:

```text
VALIDATOR_NO_EVM=0
NODEXO_VALIDATOR_DISCOVERY_MODE=evm
```

The default `auto` discovery mode resolves to `evm` for this role, so it does
not also publish a native axon unless `NODEXO_VALIDATOR_DISCOVERY_MODE=both` is
set explicitly. If both are intentionally published, miners prefer the
`ValidatorRegistry` endpoint for that UID and skip the duplicate axon target.
The endpoint still has to pass the same current validator-permit gate as native
axon discovery. If the UID cannot activate its registry/control authority, EVM
write-mode startup exits instead of serving a misleading healthy API.
Keep `VALIDATOR_DEACTIVATE_ON_SHUTDOWN=1` unless an operator is intentionally
debugging registry writes. On normal shutdown the validator attempts to
deactivate its `ValidatorRegistry` endpoint so miners do not keep discovering a
dead EVM endpoint; miners also quarantine repeated endpoint failures as a
defensive fallback. The generated PM2 ecosystem gives the validator a 60 second
graceful-stop window, which is long enough for the default 45 second
deactivation receipt wait. If you write a custom process manager config, keep
that same relationship.

For low-resource validators that do not want to verify proofs locally, use
follower mode instead.

## 5. Harden The Public Endpoint

Bind the Python daemon to loopback and put nginx in front of it:

```bash
bash scripts/setup_endpoint_proxy.sh --role validator --public-port 9443
```

The proxy defaults to public `9443` and backend `127.0.0.1:19443`. The
validator process should bind the backend port, while `--endpoint` remains the
public URL that miners can reach. For validators, the proxy also sends exact
`/proofs/receipt` requests to the dedicated receipt ingress service on
`127.0.0.1:19444`; this keeps timing receipts independent from verification and
chain/indexer work in the main validator process.

The validator proxy exposes only miner ingress and status routes:

```text
/health
/version
/subnet-config
/heartbeat
/monitor/report
/chain/context
/proofs/receipt -> receipt ingress
/proofs/*
```

Browse, rent, sybil, history, and admin routes should be reached by the web app
or an operator network, not through the public listener.

If you have a real domain and certificate, terminate HTTPS at nginx and set
`VALIDATOR_STRICT_PUBLIC_ENDPOINT=1`.

## 6. Start With PM2

```bash
cp ecosystem.config.example.cjs ecosystem.config.cjs
```

Edit the validator app:

```text
--wallet
--hotkey
--subtensor-network
--subtensor-endpoint optional private/local RPC for that network
--port 19443
--bind-host 127.0.0.1
--endpoint
--no-evm for proof-verifying validators
```

Leave `DB_URL` in `/etc/nodexo/validator.env` unless you intentionally use an
external Postgres URL.

Start:

```bash
pm2 start ecosystem.config.cjs --only vali-nodexo
pm2 start ecosystem.config.cjs --only receipt-ingress
pm2 logs vali-nodexo --lines 100
pm2 logs receipt-ingress --lines 100
pm2 save
```

The validator resolves netuid and chain config from `--subtensor-network`.
Use `--subtensor-endpoint` only to override the RPC transport for that same
network, for example:

```text
--subtensor-network test --subtensor-endpoint ws://127.0.0.1:9944
```

Pass `--chain-config` only for custom deployments.

If logs show `Weights commit-reveal pending`, proof verification is still live.
It means Subtensor rejected a weight commit because this validator already has
unrevealed weight commits pending. The validator backs off to the configured
weight cadence and tries again after the chain window clears.

When changing role env such as `VALIDATOR_NO_EVM` or
`NODEXO_VALIDATOR_DISCOVERY_MODE`, start from the edited ecosystem file again
so PM2 does not reuse stale saved environment:

```bash
pm2 delete vali-nodexo
pm2 delete receipt-ingress
pm2 start ecosystem.config.cjs --only vali-nodexo
pm2 start ecosystem.config.cjs --only receipt-ingress
```

## Follower Validator Mode

Follower mode is optional and disabled by default. It is for low-resource
validators that want to set weights from the primary validator's published
executor state instead of running local proof verification.

```env
VALIDATOR_FOLLOWER_MODE=1
VALIDATOR_FOLLOWER_STATE_URL=https://nodexo.ai/api/instances
```

In follower mode the validator still uses its own Bittensor hotkey to set
weights. It does not run rental orchestration, offline reports, sybil scans, or
the local proof verification pool. The imported state is refreshed every 30
seconds by default (`VALIDATOR_FOLLOWER_POLL_INTERVAL_S`).

## Operational Checks

```bash
curl -fsS http://127.0.0.1:19443/health
curl -fsS http://127.0.0.1:9443/health
pm2 logs vali-nodexo --lines 100

# Optional chain diagnostics for the validator hotkey.
# This calls public subtensor RPCs and can be rate-limited on testnet.
.venv/bin/nodexo --wallet nodexo_vali --hotkey default \
  fleet --chain-direct --subtensor-network test

# Mainnet chain diagnostic after mainnet launch readiness.
.venv/bin/nodexo --wallet nodexo_vali --hotkey default \
  fleet --chain-direct --subtensor-network finney
```

Confirm PM2 stays online, the configured public endpoint answers `/health`,
and logs show the validator resolved its UID, served the native axon or
registered its `ValidatorRegistry` endpoint, and can read `ComputeRegistry`.

See `docs/CLI_REFERENCE.md` for fleet, inventory, rental, and validator-control
debug commands.
