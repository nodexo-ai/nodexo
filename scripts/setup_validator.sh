#!/usr/bin/env bash
# =============================================================================
# Nodexo Validator — one-shot environment setup
# =============================================================================
#
# Idempotent. Prepares a Linux host to run a nodexo validator daemon.
# Smaller than setup_miner.sh because validators don't host containers; they
# only verify proofs, score executors, set chain weights, and (optionally)
# orchestrate canary probes.
#
#   1. System packages    (python venv tooling, Postgres, curl, PM2)
#   2. Clock sync         Enable/verify NTP before validator startup
#   3. Python venv        + project deps installed
#   4. ZkGEMM extension   Validate the native CPU verifier helpers from the
#                          shipped zkgemm_cuda artifact. Validators do not
#                          need a GPU, but they must not fall back to pure
#                          Python PRF verification for production proofs.
#   5. SSH key check      (validator initiates canary SSH into rental
#                          containers; a key at ~/.ssh/id_ed25519 is the
#                          conventional one)
#   6. Connectivity probe (bittensor.Subtensor connects to the configured
#                          network — catches 429-rate-limit at install time
#                          rather than at PM2 restart time)
#
# Usage:
#   bash scripts/setup_validator.sh                         # full install, local Postgres
#   bash scripts/setup_validator.sh --sqlite                # local/dev fallback
#   bash scripts/setup_validator.sh --probe-only
#   ZKGEMM_CUDA_MANIFEST_URL=https://pub-ef00d9a98f734d94af3c8904eba0eb11.r2.dev/zkgemm/v0.1.2/manifest.json \
#     bash scripts/setup_validator.sh
# =============================================================================

set -euo pipefail

PROBE_ONLY=false
SKIP_ZKGEMM_CHECK=false
USE_POSTGRES=true
VALIDATOR_ENV_FILE="${NODEXO_VALIDATOR_ENV_FILE:-/etc/nodexo/validator.env}"
VALIDATOR_DB_URL="${NODEXO_VALIDATOR_DB_URL:-${DB_URL:-}}"
VALIDATOR_EVM_RPC_URL="${NODEXO_EVM_RPC_URL:-${EVM_RPC_URL:-}}"
VALIDATOR_SUBTENSOR_NETWORK="${NODEXO_VALIDATOR_SUBTENSOR_NETWORK:-${NODEXO_SUBTENSOR_NETWORK:-${SUBTENSOR_NETWORK:-test}}}"
VALIDATOR_SUBTENSOR_ENDPOINT="${NODEXO_VALIDATOR_SUBTENSOR_ENDPOINT:-${NODEXO_SUBTENSOR_ENDPOINT:-${SUBTENSOR_ENDPOINT:-}}}"
VALIDATOR_SUBTENSOR_HTTP_RPC_URL="${NODEXO_SUBTENSOR_HTTP_RPC_URL:-${SUBTENSOR_HTTP_RPC_URL:-}}"
VALIDATOR_SUBTENSOR_HTTP_RPC_URL_TEST="${NODEXO_SUBTENSOR_HTTP_RPC_URL_TEST:-${SUBTENSOR_HTTP_RPC_URL_TEST:-}}"
VALIDATOR_SUBTENSOR_HTTP_RPC_URL_FINNEY="${NODEXO_SUBTENSOR_HTTP_RPC_URL_FINNEY:-${SUBTENSOR_HTTP_RPC_URL_FINNEY:-}}"
POSTGRES_DB="${NODEXO_POSTGRES_DB:-nodexo_validator}"
POSTGRES_USER="${NODEXO_POSTGRES_USER:-nodexo}"
POSTGRES_PASSWORD="${NODEXO_POSTGRES_PASSWORD:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --probe-only) PROBE_ONLY=true; shift ;;
        --skip-zkgemm-check) SKIP_ZKGEMM_CHECK=true; shift ;;
        --sqlite) USE_POSTGRES=false; shift ;;
        --db-url) VALIDATOR_DB_URL="${2:?--db-url requires a value}"; shift 2 ;;
        --postgres-db) POSTGRES_DB="${2:?--postgres-db requires a value}"; shift 2 ;;
        --postgres-user) POSTGRES_USER="${2:?--postgres-user requires a value}"; shift 2 ;;
        --postgres-password) POSTGRES_PASSWORD="${2:?--postgres-password requires a value}"; shift 2 ;;
        --validator-env-file) VALIDATOR_ENV_FILE="${2:?--validator-env-file requires a value}"; shift 2 ;;
        --subtensor-network) VALIDATOR_SUBTENSOR_NETWORK="${2:?--subtensor-network requires a value}"; shift 2 ;;
        --subtensor-endpoint) VALIDATOR_SUBTENSOR_ENDPOINT="${2:?--subtensor-endpoint requires a value}"; shift 2 ;;
        -h|--help) sed -n '2,31p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1 (try --help)"; exit 2 ;;
    esac
done

if [[ "$VALIDATOR_DB_URL" == sqlite:* ]]; then
    USE_POSTGRES=false
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
NODEXO_DATA_DIR="${NODEXO_DATA_DIR:-$HOME/.nodexo}"
DEFAULT_ZKGEMM_CUDA_MANIFEST_URL="https://pub-ef00d9a98f734d94af3c8904eba0eb11.r2.dev/zkgemm/v0.1.2/manifest.json"
export ZKGEMM_CUDA_MANIFEST_URL="${ZKGEMM_CUDA_MANIFEST_URL:-$DEFAULT_ZKGEMM_CUDA_MANIFEST_URL}"

log()  { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[err]\033[0m %s\n' "$*" >&2; }

sh_run() {
    if [[ $EUID -eq 0 ]]; then "$@"; else sudo "$@"; fi
}

as_postgres() {
    if [[ $EUID -eq 0 ]]; then
        if command -v runuser >/dev/null 2>&1; then
            runuser -u postgres -- "$@"
        else
            su -s /bin/sh postgres -c "$(printf '%q ' "$@")"
        fi
    else
        sudo -u postgres "$@"
    fi
}

sql_ident_ok() {
    [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]{0,62}$ ]]
}

generate_password() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 24
    else
        python3 - <<'PY'
import secrets
print(secrets.token_hex(24))
PY
    fi
}

write_env_file() {
    local env_file=$1
    local tmp
    tmp="$(mktemp)"
    cat > "$tmp" <<EOF
# Auto-generated by scripts/setup_validator.sh.
# Edit wallet/hotkey/endpoint in ecosystem.config.cjs or your PM2 command.
NODEXO_DATA_DIR=$NODEXO_DATA_DIR
NODEXO_VALIDATOR_DB_URL=$VALIDATOR_DB_URL
DB_URL=$VALIDATOR_DB_URL
NODEXO_EVM_RPC_URL=$VALIDATOR_EVM_RPC_URL
NODEXO_VALIDATOR_SUBTENSOR_NETWORK=$VALIDATOR_SUBTENSOR_NETWORK
NODEXO_SUBTENSOR_NETWORK=$VALIDATOR_SUBTENSOR_NETWORK
SUBTENSOR_NETWORK=$VALIDATOR_SUBTENSOR_NETWORK
NODEXO_VALIDATOR_SUBTENSOR_ENDPOINT=$VALIDATOR_SUBTENSOR_ENDPOINT
NODEXO_SUBTENSOR_ENDPOINT=$VALIDATOR_SUBTENSOR_ENDPOINT
SUBTENSOR_ENDPOINT=$VALIDATOR_SUBTENSOR_ENDPOINT
NODEXO_SUBTENSOR_HTTP_RPC_URL=$VALIDATOR_SUBTENSOR_HTTP_RPC_URL
NODEXO_SUBTENSOR_HTTP_RPC_URL_TEST=$VALIDATOR_SUBTENSOR_HTTP_RPC_URL_TEST
NODEXO_SUBTENSOR_HTTP_RPC_URL_FINNEY=$VALIDATOR_SUBTENSOR_HTTP_RPC_URL_FINNEY
VALIDATOR_REQUIRE_POSTGRES=$([[ "$VALIDATOR_DB_URL" == sqlite:* ]] && printf '0' || printf '1')
VALIDATOR_DB_PRUNE_ENABLED=1
VALIDATOR_RETENTION_PROFILE=minimal
NODEXO_SUBNET_CONFIG_URL=https://nodexo.ai/api/subnet-config
NODEXO_SUBNET_CONFIG_REFRESH_SECONDS=600
EOF
    sh_run mkdir -p "$(dirname "$env_file")"
    sh_run install -m 600 "$tmp" "$env_file"
    rm -f "$tmp"
    ok "Validator env written: $env_file"
}

setup_postgres_db() {
    if [[ -n "$VALIDATOR_DB_URL" ]]; then
        case "$VALIDATOR_DB_URL" in
            postgresql://*|postgresql+*) ;;
            sqlite:*)
                USE_POSTGRES=false
                ;;
            *)
                err "Unsupported validator DB URL scheme. Use postgresql+psycopg2://... or pass --sqlite."
                exit 1
                ;;
        esac
        return 0
    fi

    sql_ident_ok "$POSTGRES_DB" || { err "Invalid Postgres database name: $POSTGRES_DB"; exit 1; }
    sql_ident_ok "$POSTGRES_USER" || { err "Invalid Postgres user name: $POSTGRES_USER"; exit 1; }
    if [[ -z "$POSTGRES_PASSWORD" ]]; then
        POSTGRES_PASSWORD="$(generate_password)"
    elif [[ ! "$POSTGRES_PASSWORD" =~ ^[A-Za-z0-9._~-]+$ ]]; then
        err "NODEXO_POSTGRES_PASSWORD contains URL/SQL-special characters. Use NODEXO_VALIDATOR_DB_URL for a custom password."
        exit 1
    fi

    log "Configuring local Postgres database $POSTGRES_DB..."
    sh_run systemctl enable --now postgresql >/dev/null 2>&1 || sh_run service postgresql start
    as_postgres psql -v ON_ERROR_STOP=1 <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '$POSTGRES_USER') THEN
    CREATE ROLE "$POSTGRES_USER" LOGIN PASSWORD '$POSTGRES_PASSWORD';
  ELSE
    ALTER ROLE "$POSTGRES_USER" WITH LOGIN PASSWORD '$POSTGRES_PASSWORD';
  END IF;
END
\$\$;
SQL
    if ! as_postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='$POSTGRES_DB'" | grep -qx 1; then
        as_postgres createdb -O "$POSTGRES_USER" "$POSTGRES_DB"
    fi
    VALIDATOR_DB_URL="postgresql+psycopg2://$POSTGRES_USER:$POSTGRES_PASSWORD@127.0.0.1:5432/$POSTGRES_DB"
    ok "Postgres database ready"
}

ensure_pm2() {
    command -v pm2 >/dev/null 2>&1 && return 0
    if $PROBE_ONLY; then
        warn "pm2 not installed; --probe-only will not install it."
        return 0
    fi
    log "Installing PM2..."
    if ! command -v npm >/dev/null 2>&1; then
        sh_run apt-get update -qq
        sh_run apt-get install -y -qq --no-install-recommends nodejs npm
    fi
    sh_run npm install -g pm2
    ok "PM2 installed"
}

clock_synchronized() {
    command -v timedatectl >/dev/null 2>&1 || return 1
    [[ "$(timedatectl show -p NTPSynchronized --value 2>/dev/null || true)" == "yes" ]]
}

configure_timesyncd_ntp() {
    command -v systemctl >/dev/null 2>&1 || return 0
    local tmp
    tmp="$(mktemp)"
    cat > "$tmp" <<'EOF'
[Time]
NTP=time.cloudflare.com time.google.com pool.ntp.org
FallbackNTP=0.pool.ntp.org 1.pool.ntp.org 2.pool.ntp.org 3.pool.ntp.org
EOF
    sh_run mkdir -p /etc/systemd/timesyncd.conf.d
    sh_run install -m 644 "$tmp" /etc/systemd/timesyncd.conf.d/nodexo.conf
    rm -f "$tmp"
    sh_run timedatectl set-ntp true >/dev/null 2>&1 || true
    sh_run systemctl restart systemd-timesyncd >/dev/null 2>&1 || true
}

ensure_clock_sync() {
    if [[ "${NODEXO_SKIP_CLOCK_SYNC_CHECK:-0}" == "1" ]]; then
        warn "NODEXO_SKIP_CLOCK_SYNC_CHECK=1 set; skipping system clock synchronization check."
        return 0
    fi
    if ! command -v timedatectl >/dev/null 2>&1; then
        err "timedatectl unavailable; cannot verify validator host clock synchronization."
        err "Validator proof timing depends on sane wall clocks. Install systemd/timedatectl or set NODEXO_SKIP_CLOCK_SYNC_CHECK=1 only if host time is managed externally."
        exit 1
    fi
    if clock_synchronized; then
        ok "System clock synchronized"
        return 0
    fi
    if $PROBE_ONLY; then
        err "System clock is not synchronized. Run the full setup or fix NTP/chrony before starting the validator."
        exit 1
    fi

    warn "System clock is not synchronized; validator proof timing can be wrong."
    warn "Installing persistent NTP configuration and waiting for synchronization."
    configure_timesyncd_ntp
    for _ in {1..15}; do
        sleep 2
        if clock_synchronized; then
            ok "System clock synchronized"
            return 0
        fi
    done

    warn "systemd-timesyncd did not synchronize quickly; installing chrony and forcing a time step."
    sh_run apt-get update -qq
    sh_run apt-get install -y -qq --no-install-recommends chrony
    sh_run systemctl enable --now chrony >/dev/null 2>&1 || true
    sh_run chronyc -a makestep >/dev/null 2>&1 || true
    sh_run timedatectl set-ntp true >/dev/null 2>&1 || true
    for _ in {1..15}; do
        sleep 2
        if clock_synchronized; then
            ok "System clock synchronized"
            return 0
        fi
    done

    err "System clock is still not synchronized."
    err "Fix NTP/chrony before starting the validator, or set NODEXO_SKIP_CLOCK_SYNC_CHECK=1 only if host time is managed externally."
    exit 1
}

# ── 1. System packages ───────────────────────────────────────────────────────
if ! $PROBE_ONLY; then
    log "Installing system packages..."
    sh_run apt-get update -qq
    packages=(ca-certificates curl git python3 python3-venv python3-pip openssh-client)
    if $USE_POSTGRES; then
        packages+=(postgresql postgresql-client)
    else
        packages+=(sqlite3)
    fi
    sh_run apt-get install -y -qq --no-install-recommends "${packages[@]}"
    ok "System packages installed"
    ensure_pm2
fi

# ── 2. Clock synchronization ─────────────────────────────────────────────────
ensure_clock_sync

# ── 3. Python venv ───────────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]] && ! $PROBE_ONLY; then
    log "Creating Python venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi
if [[ -d "$VENV_DIR" ]] && ! $PROBE_ONLY; then
    log "Installing project Python deps..."
    "$VENV_DIR/bin/pip" install -q --upgrade pip
    if [[ -f "$REPO_ROOT/requirements.txt" ]]; then
        "$VENV_DIR/bin/pip" install -q -r "$REPO_ROOT/requirements.txt"
    elif [[ -f "$REPO_ROOT/pyproject.toml" ]]; then
        "$VENV_DIR/bin/pip" install -q -e "$REPO_ROOT"
    else
        warn "No requirements.txt or pyproject.toml — skipping pip install."
    fi
    if [[ -f "$REPO_ROOT/pyproject.toml" ]]; then
        "$VENV_DIR/bin/pip" install -q --no-deps -e "$REPO_ROOT"
    fi
    ok "Python venv ready at $VENV_DIR"
fi

# ── 4. Native ZkGEMM verifier extension ─────────────────────────────────────
# Validators can be CPU-only, but they still need the native extension's CPU
# PRF/slice helpers. Without it, first-time/full verification of large GPUs can
# spend minutes in pure Python and starve the verifier pool.
if $SKIP_ZKGEMM_CHECK; then
    warn "--skip-zkgemm-check set; validator may be unable to verify full proofs safely."
elif [[ -d "$VENV_DIR" ]] && ! $PROBE_ONLY; then
    log "Validating native zkgemm verifier extension..."
    if PYTHONPATH="$REPO_ROOT:$REPO_ROOT/zkgemm/cuda" \
        "$VENV_DIR/bin/python" "$REPO_ROOT/scripts/verify_zkgemm_cuda.py"; then
        ok "zkgemm_cuda native verifier extension validates on this host"
    else
        log "Installing current zkgemm_cuda artifact"
        install_args=()
        if [[ -n "${ZKGEMM_CUDA_WHEEL_PATH:-}" ]]; then
            install_args+=(--wheel-path "$ZKGEMM_CUDA_WHEEL_PATH")
        elif [[ -n "${ZKGEMM_CUDA_WHEEL_URL:-}" ]]; then
            install_args+=(--wheel-url "$ZKGEMM_CUDA_WHEEL_URL")
        elif [[ -n "${ZKGEMM_CUDA_SO_PATH:-}" ]]; then
            install_args+=(--so-path "$ZKGEMM_CUDA_SO_PATH")
        elif [[ -n "${ZKGEMM_CUDA_SO_URL:-}" ]]; then
            install_args+=(--so-url "$ZKGEMM_CUDA_SO_URL")
        elif [[ -n "${ZKGEMM_CUDA_MANIFEST_PATH:-}" ]]; then
            install_args+=(--manifest-path "$ZKGEMM_CUDA_MANIFEST_PATH")
        else
            install_args+=(--manifest-url "$ZKGEMM_CUDA_MANIFEST_URL")
        fi
        if [[ -n "${ZKGEMM_CUDA_SHA256:-}" ]]; then
            install_args+=(--sha256 "$ZKGEMM_CUDA_SHA256")
        fi
        "$VENV_DIR/bin/python" "$REPO_ROOT/scripts/install_zkgemm_artifact.py" "${install_args[@]}"
        PYTHONPATH="$REPO_ROOT:$REPO_ROOT/zkgemm/cuda" \
            "$VENV_DIR/bin/python" "$REPO_ROOT/scripts/verify_zkgemm_cuda.py"
        ok "zkgemm_cuda artifact installed and validated"
    fi
fi

# ── 5a. Persistent state directory + DB file perms (defence in depth) ────────
# The validator's SQLite DB carries CanaryRecord rows; the canary
# reconcile loop fans the executor_id of any inflight_at-stamped row
# directly to `mark_available` on chain. An attacker with write access
# to the DB file can grief any executor across chain. Make the file
# readable only by the validator's UID. PostgreSQL DBs are externally
# protected and don't need this.
mkdir -p "$NODEXO_DATA_DIR"
chmod 700 "$NODEXO_DATA_DIR" 2>/dev/null && ok "chmod 700 $NODEXO_DATA_DIR" || \
    warn "could not chmod 700 $NODEXO_DATA_DIR"
for DB_FILE in "$REPO_ROOT/nodexo_validator.db" "$NODEXO_DATA_DIR/validator.db"; do
    if [[ -f "$DB_FILE" ]]; then
        chmod 600 "$DB_FILE" 2>/dev/null && ok "chmod 600 $DB_FILE" || \
            warn "could not chmod 600 $DB_FILE"
    fi
done
if [[ -f "$NODEXO_DATA_DIR/containers.json" ]]; then
    chmod 600 "$NODEXO_DATA_DIR/containers.json" 2>/dev/null || true
fi
if [[ -f "$NODEXO_DATA_DIR/executor_identity.json" ]]; then
    chmod 600 "$NODEXO_DATA_DIR/executor_identity.json" 2>/dev/null || true
fi

if ! $PROBE_ONLY; then
    if $USE_POSTGRES; then
        setup_postgres_db
    elif [[ -z "$VALIDATOR_DB_URL" ]]; then
        VALIDATOR_DB_URL="sqlite:///$NODEXO_DATA_DIR/validator.db"
        warn "SQLite validator DB selected explicitly. Use this only for local/dev testing."
    fi
    write_env_file "$VALIDATOR_ENV_FILE"
fi

# Also tighten the .env file if present.
if [[ -f "$REPO_ROOT/.env" ]]; then
    chmod 600 "$REPO_ROOT/.env" 2>/dev/null && ok "chmod 600 .env"
fi

# ── 5. SSH key for canary ────────────────────────────────────────────────────
if [[ ! -f "$HOME/.ssh/id_ed25519" ]]; then
    log "Generating SSH key for canary use ($HOME/.ssh/id_ed25519)..."
    mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
    ssh-keygen -t ed25519 -N '' -f "$HOME/.ssh/id_ed25519" -C "nodexo-validator@$(hostname)"
    ok "SSH key generated"
else
    ok "SSH key present at $HOME/.ssh/id_ed25519"
fi

# ── 6. Connectivity probe ────────────────────────────────────────────────────
log "Probing bittensor.Subtensor connectivity (catches HTTP 429 / network issues)..."
NET="$VALIDATOR_SUBTENSOR_NETWORK"
TARGET="${VALIDATOR_SUBTENSOR_ENDPOINT:-$NET}"
PYTHON_BIN="$VENV_DIR/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
        warn "Validator venv not present; using $PYTHON_BIN for probe-only connectivity check."
    else
        warn "python3 not available; skipping Subtensor connectivity probe."
        PYTHON_BIN=""
    fi
fi
if [[ -n "$PYTHON_BIN" ]] && NODEXO_PROBE_SUBTENSOR_TARGET="$TARGET" "$PYTHON_BIN" -c "
import bittensor as bt
import os
try:
    s = bt.Subtensor(network=os.environ['NODEXO_PROBE_SUBTENSOR_TARGET'])
    print('block=', s.get_current_block())
except Exception as e:
    print('FAIL:', type(e).__name__, str(e)[:200])
    raise SystemExit(1)
" 2>&1; then
    ok "Subtensor reachable on network=$NET${VALIDATOR_SUBTENSOR_ENDPOINT:+ endpoint=$VALIDATOR_SUBTENSOR_ENDPOINT}"
else
    warn "Subtensor probe failed. If you saw HTTP 429, wait ~15 min (the testnet"
    warn "RPC rate-limits aggressively) and retry. The validator daemon has a"
    warn "12-attempt backoff for this case, but resolving it now is friendlier."
fi

cat <<EOF

==============================================================================
Validator setup complete. Next steps:

  1. Wallet — use an existing validator wallet, or create one:
       .venv/bin/btcli wallet new_coldkey --wallet-name nodexo_vali
       .venv/bin/btcli wallet new_hotkey  --wallet-name nodexo_vali --hotkey default

  2. Subnet registration (subnet must already exist).

     Public testnet preview:
       .venv/bin/btcli subnet register --wallet-name nodexo_vali --hotkey default \\
         --netuid 468 --network test

     Mainnet:
       .venv/bin/btcli subnet register --wallet-name nodexo_vali --hotkey default \\
         --netuid 106 --network finney

  3. Chain RPC for validator runtime:
     Production validators should run against a dedicated subtensor RPC, for
     example:
       --subtensor-network test --subtensor-endpoint ws://127.0.0.1:9944
     A private RPC service or another non-shared subtensor endpoint for the same
     network is also valid.

     Public test/finney RPC endpoints are supported as a throttled fallback,
     but a cold validator cache may produce neutral validator_chain_unavailable
     skips until it warms.

  4. Fund the validator's EVM mirror address only if this validator should
     publish a custom endpoint in ValidatorRegistry. The mirror is derived
     from the validator hotkey seed and is not the hotkey SS58 address.
     Validator discovery is permit-gated for both ValidatorRegistry and native
     axon endpoints. Without current validator permit, miners will not use the
     endpoint for proof broadcasts.

     Read/verify-only validators can skip EVM funding and run with:
       VALIDATOR_NO_EVM=1
       # or pass --no-evm to neurons.validator.validator

  5. Harden the public endpoint. Production should bind the Python daemon to
     loopback and expose nginx:
       bash scripts/setup_endpoint_proxy.sh --role validator --public-port 9443
     The proxy defaults to public 9443 -> backend 127.0.0.1:19443 and exact
     /proofs/receipt -> receipt ingress 127.0.0.1:19444.

  6. Validator database:
       $VALIDATOR_ENV_FILE

     This setup writes DB_URL/NODEXO_VALIDATOR_DB_URL there. The default is a
     local Postgres database. Use --sqlite only for local-only development.

  6. Copy and edit the PM2 ecosystem file:
       cp ecosystem.config.example.cjs ecosystem.config.cjs

     Set wallet, hotkey, network, and endpoint.
     Proof-verifying validators without rental control normally set VALIDATOR_NO_EVM=1.
     If you change VALIDATOR_NO_EVM or NODEXO_VALIDATOR_DISCOVERY_MODE after a
     PM2 start, delete and start the app from this ecosystem file again so PM2
     does not reuse stale saved environment.
     Start:
       pm2 start ecosystem.config.cjs --only vali-nodexo
       pm2 start ecosystem.config.cjs --only receipt-ingress
       pm2 logs vali-nodexo --lines 100
       pm2 logs receipt-ingress --lines 100

  7. Backups:
       scripts/backup_validator_db.py
     Add it to cron once the validator DB path is final.

  Full docs:
       docs/VALIDATOR_QUICKSTART.md
       docs/ENDPOINT_SECURITY.md
==============================================================================
EOF
