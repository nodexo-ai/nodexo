# API Boundaries

Nodexo has two API surfaces with different trust models.

## Public Renter API

The public API is the web app backend:

```text
https://nodexo.ai/api/*
```

Renters, CLI tools, scripts, and agents use this API. It is responsible for:

- x402 payment verification and settlement
- credit-balance accounting
- account sessions and wallet login
- recovery-token authorization
- renter-facing rate limits
- billing and rental ledgers
- forwarding authorized provisioning requests to the validator

During public testnet preview, the launch gate allows read, login, account, and
admin-controlled test rental paths. Non-admin rental, extension, and credit
top-up writes are blocked before payment settlement, credit reserve, or
provisioning.

The public API then calls the validator with `VALIDATOR_ADMIN_TOKEN`.

## Validator API

The validator daemon is an internal control plane and miner-ingress service. It
is not the public renter API.

Internal/admin routes require `X-Admin-Token` or an allowed validator hotkey
signature:

- `POST /rent`
- `GET /rentals`
- `DELETE /rentals/{id}`
- `POST /rentals/{id}/extend`
- `GET /rentals/{id}/ports`
- `POST|DELETE /rentals/{id}/ssh_keys`
- `GET /rentals/history`
- `GET /marketplace/tiers`
- `GET /pricing/quote`
- `GET /instances`
- `GET /executors`
- `GET /executors/{id}`
- `GET /scores`
- `GET /monitor/reports`
- `/sybil/*`

Miner and monitor ingress remains direct-to-validator:

- `POST /proofs/commit`
- `POST /proofs/receipt`
- `POST /proofs/recipe`
- `POST /heartbeat`
- `POST /monitor/report`
- `GET /chain/context`

Those routes are direct because miners must broadcast proof and telemetry to
validators. Proof and heartbeat routes verify SR25519 signatures and on-chain
executor ownership. Monitor reports verify monitor signatures and persistent
TOFU bindings. `/chain/context` is bounded and rate-limited because monitors
need it before sending reports.

The URL published in `ValidatorRegistry` must be a real miner-reachable
validator URL. A loopback URL with a reverse SSH tunnel is only acceptable for a
single local testnet workstation; production validators publish HTTPS and keep
admin routes restricted by the reverse proxy/firewall.

Validators started with `VALIDATOR_NO_EVM=1` or `--no-evm` are read/verify
validators. They can verify proofs and set weights, but they do not publish a
`ValidatorRegistry` endpoint and cannot serve web app rental control routes.
Miners discover them through native Bittensor axon metadata only when the UID
has validator permit. Miners also re-check current validator permit for
`ValidatorRegistry` rows before proof broadcast, so stale active registry rows
are not trusted after a UID loses permit. Point the web app only at an
EVM-enabled primary validator.

Miner rental control is a separate signed path. Production miners should run
with `NODEXO_STRICT_ALLOWLIST=1`; the primary validator hotkey is loaded
from subnet runtime config. `NODEXO_ALLOWED_VALIDATOR_HOTKEYS` is a local override,
not the normal production setup.

## Production Network Policy

Recommended deployment:

```text
Internet
  -> web app /api/*
  -> validator miner-ingress routes only

web app backend
  -> validator admin/control routes with X-Admin-Token

miners/monitors
  -> validator proof, heartbeat, monitor, chain-context routes
```

Do not expose validator admin/control routes through a public reverse proxy. If
the validator port is reachable, the routes still require admin auth, but the
network should also enforce the boundary.

`NODEXO_ADMIN_TOKEN` must be a high-entropy per-environment secret and must never
be committed, logged, or returned to clients.
