# Endpoint Security

Nodexo has three different HTTP surfaces. Treat them differently.

## Public Web API

Renter integrations call the web app:

```text
https://nodexo.ai/api/*
```

The web app owns wallet login, x402, account API keys, credit balances, public
inventory, billing, and rental recovery. This is the API that agents and renter
scripts should use.

## Validator API

The validator API is an internal control plane plus signed miner ingress.

Public listener for validator APIs:

```text
/health
/version
/subnet-config
/heartbeat
/monitor/report
/chain/context
/proofs/*
```

Private validator routes:

```text
/rent
/rentals/*
/executors/*
/marketplace/*
/sybil/*
/scores
/monitor/reports
```

Those private routes either mutate rental state, expose operational state, or
duplicate web-app functionality. Keep them behind the web app, VPN, or an
operator network. On the primary validator they require
admin signatures or `X-Admin-Token`. Validators without rental/control duties
can run without admin auth; those routes simply return 403.

Recommended production shape:

```env
VALIDATOR_BIND_HOST=127.0.0.1
VALIDATOR_PORT=19443
VALIDATOR_ENDPOINT=http://validator-host.example.com:9443
VALIDATOR_ALLOW_LOCAL_ENDPOINT=0
```

Use `VALIDATOR_STRICT_PUBLIC_ENDPOINT=1` only when the endpoint is a real HTTPS
URL with a certificate trusted by miners.

## Miner API

The miner API is public because validators must provision and audit containers
on the host. Sensitive routes are SR25519-signed by validator hotkeys:

```text
POST   /containers
GET    /containers
DELETE /containers/{name}
GET    /ports
POST   /containers/{name}/ssh_keys
DELETE /containers/{name}/ssh_keys
```

Public status and identity routes:

```text
/health
/version
/identity/challenge
/hardware
/proof/status
/proof/latest
```

Recommended production shape:

```env
MINER_BIND_HOST=127.0.0.1
MINER_PORT=18091
MINER_ENDPOINT=http://miner-host.example.com:8091
MINER_ALLOW_LOCAL_ENDPOINT=0
NODEXO_STRICT_ALLOWLIST=1
```

The public endpoint must be reachable by validators. Do not register
loopback or wildcard-only addresses on chain.

Primary validator hotkeys are loaded from subnet runtime config. Use
`NODEXO_ALLOWED_VALIDATOR_HOTKEYS` only as a local/emergency override. No-EVM
validators are discovered from native Bittensor axon metadata and must have
validator permit; miners do not choose proof validators from local endpoint
lists.

## nginx Setup

Use the shared proxy script:

```bash
bash scripts/setup_endpoint_proxy.sh --role miner --public-port 8091
bash scripts/setup_endpoint_proxy.sh --role validator --public-port 9443
```

By default nginx listens on the public port and proxies to loopback backends:

```text
miner:     public 8091 -> 127.0.0.1:18091
validator: public 9443 -> 127.0.0.1:19443
```

Keep those backend ports closed to the Internet. The proxy script also writes
shared nginx guardrails to `/etc/nginx/conf.d/nodexo-limits.conf`:

- a generous per-IP baseline for public protocol endpoints
- a separate `/proofs/receipt` bucket for timing-critical receipt ingress
- a tighter `/chain/context` bucket, because honest clients should not poll it
  at high frequency
- per-IP connection caps

These limits are an edge safety net only. Application-level SR25519 signatures,
replay checks, executor ownership checks, and registry validation remain the
authority for accepting protocol messages.

For real HTTPS:

```bash
bash scripts/setup_endpoint_proxy.sh --role miner \
  --tls existing \
  --server-name gpu1.example.com \
  --public-port 443 \
  --cert-path /etc/letsencrypt/live/gpu1.example.com/fullchain.pem \
  --key-path /etc/letsencrypt/live/gpu1.example.com/privkey.pem
```

Self-signed TLS is fine for local smoke tests, but it is not a production
default because normal HTTP clients reject self-signed certificates.

## Cloudflare-Protected Domains

Recommended production pattern:

- Use a real domain for miner and validator API endpoints.
- Put the API endpoint on `443/tcp` where possible.
- Terminate HTTPS at nginx on the host, then proxy to the loopback Python
  daemon.
- Cloudflare proxy/WAF/rate limits are useful for the HTTP API path.
- Keep Cloudflare Access login pages, browser challenges, bot fights, and
  interactive challenges disabled on protocol endpoints. Validators, miners,
  and monitors are non-browser clients.
- Disable caching and request/response mutation for protocol routes.
- If the hostname is orange-cloud proxied, restrict the origin HTTPS port to
  Cloudflare IP ranges at the firewall.

Certificate choices:

- Let's Encrypt is the safest default because it works for both direct clients
  and Cloudflare-proxied clients.
- Cloudflare Origin Certificates are acceptable only when the hostname is
  always reached through Cloudflare. Direct validator/miner clients will not
  trust them.

Important limitation:

- Standard Cloudflare HTTP proxy covers HTTP/HTTPS API traffic only.
- It does not proxy the rental SSH range `20000-20100/tcp`.
- Rental SSH ports must remain directly reachable, or use a deliberate TCP
  proxy product/design such as Cloudflare Spectrum.

## Validator Ingress Protection

Validator public endpoints should use the same nginx proxy path:

```bash
bash scripts/setup_endpoint_proxy.sh --role validator --public-port 9443
```

The validator proxy exposes only status, subnet-config, proof, heartbeat,
monitor, and chain-context routes. Rental/admin/browse routes stay behind the
web app or an operator network. The subnet-config route is public read-only and
serves the validator's last accepted config cache. Proof and heartbeat routes
then apply application-level checks:

- request body size limits
- SR25519 miner-hotkey signature verification
- replay nonce checks
- executor existence and lease checks against `ComputeRegistry`
- hotkey-to-executor ownership checks
- per-executor submission rate limits

The exact `/proofs/receipt` route should go to the dedicated
`receipt-ingress` process on loopback while other proof routes go to the main
validator backend. The shared setup script configures that split by default for
validators: public `9443` -> main validator `127.0.0.1:19443`, receipt ingress
`127.0.0.1:19444`.

Network-level DDoS still has to be handled at the host/provider edge. Use a
provider firewall, nginx, and, for domain endpoints, Cloudflare or equivalent
L7 protection. Native Bittensor axon discovery stores IP/port only; use the
EVM `ValidatorRegistry` path when a validator needs to advertise a domain or a
Cloudflare-protected HTTPS endpoint.

Example HTTPS miner API behind a real domain:

```bash
bash scripts/setup_endpoint_proxy.sh --role miner \
  --tls existing \
  --server-name gpu1.example.com \
  --public-port 443 \
  --cert-path /etc/letsencrypt/live/gpu1.example.com/fullchain.pem \
  --key-path /etc/letsencrypt/live/gpu1.example.com/privkey.pem
```

Advertise:

```text
--endpoint https://gpu1.example.com
```
