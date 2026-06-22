#!/usr/bin/env bash
# =============================================================================
# Nodexo Miner — one-shot environment setup
# =============================================================================
#
# Idempotent. Re-running is safe. Prepares a fresh Linux host (or a stale one
# missing pieces) to run a nodexo miner:
#
#   1. System packages   (docker, python venv tooling, nvidia-smi sanity)
#   2. Docker GPU access (nvidia-container-toolkit, daemon restart if needed)
#   3. Sysbox-runc       (hardened rental container isolation)
#   4. Python venv       + project deps installed
#   5. ZkGEMM extension  Validate the shipped native proof extension against
#                        this Python/PyTorch/CUDA ABI.
#   6. Image pre-pull    Required warm renter images from common/images.py.
#                        Optional catalog images are pulled while the rental
#                        storage reserve allows, unless disabled.
#   7. Hardware probe    nvidia-smi works, NVML inside docker works
#
# By default this script does NOT register on-chain or touch wallets. With
# --start, it expects an existing Bittensor wallet/hotkey, writes a PM2 launcher,
# and starts the miner; the miner then performs its normal on-chain registration.
#
# Usage:
#   bash scripts/setup_miner.sh                 # full install
#   bash scripts/setup_miner.sh --skip-images   # everything except docker pull
#   bash scripts/setup_miner.sh --probe-only    # verify only; no apt/docker writes
#   bash scripts/setup_miner.sh --zkgemm-only   # install deps + validate proof artifact only
#   bash scripts/setup_miner.sh --launcher-only --wallet miner --port 18091 --bind-host 127.0.0.1 --endpoint http://host:8091
#   bash scripts/setup_miner.sh --start --wallet miner --subtensor-network test --port 18091 --bind-host 127.0.0.1 --endpoint http://host:8091
#   bash scripts/setup_miner.sh --start --wallet miner --subtensor-network finney --endpoint https://miner.example.com
#   bash scripts/setup_endpoint_proxy.sh --role miner --public-port 8091
# =============================================================================

set -euo pipefail

# ── Flags ────────────────────────────────────────────────────────────────────
SKIP_IMAGES=false
PROBE_ONLY=false
ZKGEMM_ONLY=false
SKIP_ZKGEMM_CHECK=false
START_MINER=false
LAUNCHER_ONLY=false
PM2_NAME="${NODEXO_MINER_PM2_NAME:-miner-nodexo}"
MINER_WALLET="${NODEXO_MINER_WALLET:-${WALLET:-}}"
MINER_HOTKEY="${NODEXO_MINER_HOTKEY:-${HOTKEY:-default}}"
MINER_NETUID="${NODEXO_MINER_NETUID:-${NETUID:-}}"
MINER_SUBTENSOR_NETWORK="${NODEXO_MINER_SUBTENSOR_NETWORK:-${NODEXO_SUBTENSOR_NETWORK:-${SUBTENSOR_NETWORK:-test}}}"
MINER_SUBTENSOR_ENDPOINT="${NODEXO_MINER_SUBTENSOR_ENDPOINT:-${NODEXO_SUBTENSOR_ENDPOINT:-${SUBTENSOR_ENDPOINT:-}}}"
MINER_CHAIN_CONFIG="${NODEXO_MINER_CHAIN_CONFIG:-${NODEXO_CHAIN_CONFIG:-${CHAIN_CONFIG:-}}}"
MINER_EVM_RPC_URL="${NODEXO_EVM_RPC_URL:-${EVM_RPC_URL:-}}"
MINER_EVM_RPC_TIMEOUT_SECONDS="${EVM_RPC_TIMEOUT_SECONDS:-12}"
MINER_MONITOR_IMAGE="${NODEXO_MONITOR_IMAGE:-nodexo-monitor:dev}"
MINER_PORT="${NODEXO_MINER_PORT:-8091}"
MINER_BIND_HOST="${NODEXO_MINER_BIND_HOST:-${MINER_BIND_HOST:-0.0.0.0}}"
MINER_ENDPOINT="${NODEXO_MINER_ENDPOINT:-${MINER_ENDPOINT:-}}"
MINER_RENTAL_PORT_START="${NODEXO_MINER_RENTAL_PORT_START:-20000}"
MINER_RENTAL_PORT_END="${NODEXO_MINER_RENTAL_PORT_END:-20100}"
MINER_LAUNCHER_PATH="${NODEXO_MINER_LAUNCHER_PATH:-}"
MINER_ALLOWED_VALIDATORS="${NODEXO_ALLOWED_VALIDATOR_HOTKEYS:-}"
MINER_STRICT_ALLOWLIST="${NODEXO_STRICT_ALLOWLIST:-1}"
DEFAULT_SUBNET_CONFIG_URL="${NODEXO_DEFAULT_SUBNET_CONFIG_URL:-https://validator.nodexo.ai/subnet-config}"
MINER_SUBNET_CONFIG_URL="${NODEXO_SUBNET_CONFIG_URL:-$DEFAULT_SUBNET_CONFIG_URL}"
MINER_AUTO_UPDATE="${MINER_AUTO_UPDATE:-0}"
MINER_ALLOW_UNCALIBRATED_GPU="${NODEXO_ALLOW_UNCALIBRATED_GPU:-}"
MINER_PROOF_ONLY_CALIBRATION="${NODEXO_PROOF_ONLY_CALIBRATION:-}"
MINER_PROOF_WORKER_TIMEOUT_SECONDS="${PROOF_WORKER_TIMEOUT_SECONDS:-}"
MINER_PROOF_SUBMIT_SAFETY_SECONDS="${PROOF_SUBMIT_SAFETY_SECONDS:-}"
MINER_VALIDATOR_POST_TIMEOUT_SECONDS="${NODEXO_VALIDATOR_POST_TIMEOUT_SECONDS:-30}"
MINER_VALIDATOR_CONNECT_TIMEOUT_SECONDS="${NODEXO_VALIDATOR_CONNECT_TIMEOUT_SECONDS:-}"
MINER_VALIDATOR_BROADCAST_DEADLINE_SECONDS="${NODEXO_VALIDATOR_BROADCAST_DEADLINE_SECONDS:-70}"
MINER_VALIDATOR_POST_MAX_ATTEMPTS="${NODEXO_VALIDATOR_POST_MAX_ATTEMPTS:-}"
MINER_RENTAL_PORT_PREFLIGHT_MODE="${MINER_RENTAL_PORT_PREFLIGHT:-strict}"
MINER_RENTAL_PORT_PREFLIGHT_SCOPE_MODE="${MINER_RENTAL_PORT_PREFLIGHT_SCOPE:-sample}"
MINER_RENTAL_IMAGE_PREFLIGHT_MODE="${MINER_RENTAL_IMAGE_PREFLIGHT:-strict}"
MINER_OPTIONAL_IMAGE_PREFLIGHT_MODE="${MINER_OPTIONAL_IMAGE_PREFLIGHT:-warn}"
DEFAULT_ZKGEMM_CUDA_MANIFEST_URL="https://pub-ef00d9a98f734d94af3c8904eba0eb11.r2.dev/zkgemm/v0.1.2/manifest.json"
SYSBOX_VERSION="${NODEXO_SYSBOX_VERSION:-0.7.0}"
SYSBOX_AMD64_SHA256="${NODEXO_SYSBOX_AMD64_SHA256:-eeff273671467b8fa351ab3d40709759462dc03d9f7b50a1b207b37982ce40a9}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-images) SKIP_IMAGES=true; shift ;;
        --probe-only)  PROBE_ONLY=true;  shift ;;
        --zkgemm-only) ZKGEMM_ONLY=true; shift ;;
        --skip-zkgemm-check) SKIP_ZKGEMM_CHECK=true; shift ;;
        --start) START_MINER=true; shift ;;
        --launcher-only) LAUNCHER_ONLY=true; shift ;;
        --pm2-name) PM2_NAME="${2:?--pm2-name requires a value}"; shift 2 ;;
        --wallet) MINER_WALLET="${2:?--wallet requires a value}"; shift 2 ;;
        --hotkey) MINER_HOTKEY="${2:?--hotkey requires a value}"; shift 2 ;;
        --netuid) MINER_NETUID="${2:?--netuid requires a value}"; shift 2 ;;
        --subtensor-network) MINER_SUBTENSOR_NETWORK="${2:?--subtensor-network requires a value}"; shift 2 ;;
        --subtensor-endpoint) MINER_SUBTENSOR_ENDPOINT="${2:?--subtensor-endpoint requires a value}"; shift 2 ;;
        --chain-config) MINER_CHAIN_CONFIG="${2:?--chain-config requires a value}"; shift 2 ;;
        --port) MINER_PORT="${2:?--port requires a value}"; shift 2 ;;
        --bind-host) MINER_BIND_HOST="${2:?--bind-host requires a value}"; shift 2 ;;
        --endpoint) MINER_ENDPOINT="${2:?--endpoint requires a value}"; shift 2 ;;
        --rental-port-start) MINER_RENTAL_PORT_START="${2:?--rental-port-start requires a value}"; shift 2 ;;
        --rental-port-end) MINER_RENTAL_PORT_END="${2:?--rental-port-end requires a value}"; shift 2 ;;
        --launcher-path) MINER_LAUNCHER_PATH="${2:?--launcher-path requires a value}"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1 (try --help)"; exit 2 ;;
    esac
done

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

default_netuid_for_network() {
    case "$MINER_SUBTENSOR_NETWORK" in
        finney) printf '106\n' ;;
        test) printf '468\n' ;;
        *) printf '\n' ;;
    esac
}

resolve_chain_config_for_network() {
    local explicit=${1:-}
    if [[ -n "$explicit" ]]; then
        printf '%s\n' "$explicit"
        return 0
    fi

    local candidates=()
    case "$MINER_SUBTENSOR_NETWORK" in
        finney)
            candidates=("$REPO_ROOT/chain_config_mainnet.json" "$REPO_ROOT/chain_config.json")
            ;;
        test)
            candidates=("$REPO_ROOT/chain_config_testnet.json" "$REPO_ROOT/chain_config.json")
            ;;
        *)
            candidates=("$REPO_ROOT/chain_config.json")
            ;;
    esac
    local p
    for p in "${candidates[@]}"; do
        if [[ -f "$p" ]]; then
            printf '%s\n' "$p"
            return 0
        fi
    done
    printf '%s\n' ""
}

if [[ -z "$MINER_NETUID" ]]; then
    MINER_NETUID="$(default_netuid_for_network)"
fi
if [[ -z "$MINER_LAUNCHER_PATH" ]]; then
    MINER_RUNTIME_DIR="${NODEXO_MINER_RUNTIME_DIR:-${NODEXO_RUNTIME_DIR:-${NODEXO_DATA_DIR:-$HOME/.nodexo}/runtime}}"
    MINER_LAUNCHER_PATH="${MINER_RUNTIME_DIR}/launch_miner_nodexo.sh"
fi
MINER_CHAIN_CONFIG="$(resolve_chain_config_for_network "$MINER_CHAIN_CONFIG")"
export ZKGEMM_CUDA_MANIFEST_URL="${ZKGEMM_CUDA_MANIFEST_URL:-$DEFAULT_ZKGEMM_CUDA_MANIFEST_URL}"

# ── Image stack ──────────────────────────────────────────────────────────────
# Warm rental images are defined in common/images.py so setup, miner heartbeat,
# and validator routing share the same catalog. Override with:
#   NODEXO_IMAGE_CATALOG=image1,image2,...
#   NODEXO_REQUIRED_RENTAL_IMAGES=image1,image2,...
REQUIRED_IMAGES=()
CATALOG_IMAGES=()
CANARY_IMAGE_LIST=()
MIN_RENTAL_STORAGE_GB="${MINER_MIN_RENTAL_STORAGE_GB:-30}"
RENTAL_STORAGE_HEADROOM_GB="${MINER_RENTAL_STORAGE_HEADROOM_GB:-5}"

# ── Helpers ──────────────────────────────────────────────────────────────────
log()  { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[err]\033[0m %s\n' "$*" >&2; }

load_image_catalog() {
    mapfile -t REQUIRED_IMAGES < <(
        PYTHONPATH="$REPO_ROOT" python3 - <<'PY'
from common.images import required_rental_images
for image in required_rental_images():
    print(image)
PY
    )
    mapfile -t CATALOG_IMAGES < <(
        PYTHONPATH="$REPO_ROOT" python3 - <<'PY'
from common.images import rental_image_catalog
for image in rental_image_catalog():
    print(image)
PY
    )
    mapfile -t CANARY_IMAGE_LIST < <(
        PYTHONPATH="$REPO_ROOT" python3 - <<'PY'
from common.images import canary_image_catalog
for image in canary_image_catalog():
    print(image)
PY
    )
    if [[ ${#REQUIRED_IMAGES[@]} -eq 0 ]]; then
        err "image catalog is empty; check common/images.py or NODEXO_REQUIRED_RENTAL_IMAGES"
        exit 1
    fi
    if [[ ${#CANARY_IMAGE_LIST[@]} -eq 0 ]]; then
        err "canary image catalog is empty; check common/images.py or CANARY_IMAGES"
        exit 1
    fi
}

require_root_or_sudo() {
    if [[ $EUID -ne 0 ]] && ! command -v sudo >/dev/null 2>&1; then
        err "Need root or sudo for system package install. Re-run as root or install sudo."
        exit 1
    fi
}

sh_run() {  # run with sudo if not root
    if [[ $EUID -eq 0 ]]; then "$@"; else sudo "$@"; fi
}

package_installed() {
    dpkg-query -W -f='${db:Status-Abbrev}' "$1" 2>/dev/null | grep -q '^ii'
}

sysbox_configured() {
    [[ -x /usr/bin/sysbox-runc ]] && package_installed sysbox-ce
}

prepare_docker_for_sysbox_install() {
    if sysbox_configured; then
        return 0
    fi
    if ! command -v docker >/dev/null 2>&1; then
        return 0
    fi

    local container_count
    container_count="$(docker ps -aq 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${container_count:-0}" == "0" ]]; then
        return 0
    fi
    if [[ "${NODEXO_SYSBOX_ALLOW_DOCKER_RESTART:-0}" != "1" ]]; then
        err "Installing Sysbox may reconfigure/restart Docker, but $container_count Docker container(s) exist."
        err "Remove containers first, or set NODEXO_SYSBOX_ALLOW_DOCKER_RESTART=1 if this is an empty testnet/calibration host."
        exit 1
    fi
    warn "NODEXO_SYSBOX_ALLOW_DOCKER_RESTART=1 set; removing $container_count Docker container(s) before Sysbox install."
    docker ps -a --format '  {{.ID}} {{.Names}} {{.Status}}'
    sh_run docker rm -f $(docker ps -aq)
}

ensure_sysbox() {
    if sysbox_configured; then
        ok "sysbox-runc already installed"
        return 0
    fi

    local arch suffix checksum
    arch="$(dpkg --print-architecture)"
    case "$arch" in
        amd64)
            suffix="linux_amd64"
            checksum="$SYSBOX_AMD64_SHA256"
            ;;
        arm64)
            suffix="linux_arm64"
            checksum="${NODEXO_SYSBOX_ARM64_SHA256:-}"
            ;;
        *)
            if [[ "${NODEXO_ALLOW_MISSING_SYSBOX_FOR_CALIBRATION:-0}" == "1" ]]; then
                warn "Sysbox package install is unsupported for architecture '$arch'; continuing calibration/proof-only."
                return 0
            fi
            err "Sysbox package install is unsupported for architecture '$arch'."
            exit 1
            ;;
    esac

    if command -v snap >/dev/null 2>&1 && snap list docker >/dev/null 2>&1; then
        err "Docker installed via snap is not supported by Sysbox. Install Docker via apt/native packages, then re-run setup."
        exit 1
    fi

    prepare_docker_for_sysbox_install

    local deb_url deb_path tmpdir
    deb_url="${NODEXO_SYSBOX_DEB_URL:-https://downloads.nestybox.com/sysbox/releases/v${SYSBOX_VERSION}/sysbox-ce_${SYSBOX_VERSION}-0.${suffix}.deb}"
    tmpdir="$(mktemp -d)"
    deb_path="$tmpdir/sysbox-ce.deb"
    log "Installing sysbox-runc ${SYSBOX_VERSION} from Nestybox package..."
    curl -fL --retry 3 --connect-timeout 15 -o "$deb_path" "$deb_url"
    if [[ -n "$checksum" ]]; then
        printf '%s  %s\n' "$checksum" "$deb_path" | sha256sum -c -
    else
        warn "No checksum configured for Sysbox ${SYSBOX_VERSION} $suffix; set NODEXO_SYSBOX_${arch^^}_SHA256 for pinned verification."
    fi
    sh_run apt-get update -qq
    sh_run apt-get install -y -qq jq
    sh_run apt-get install -y -qq "$deb_path"
    rm -rf "$tmpdir"

    if [[ ! -x /usr/bin/sysbox-runc ]]; then
        err "Sysbox package installed but /usr/bin/sysbox-runc is missing."
        exit 1
    fi
    if command -v systemctl >/dev/null 2>&1; then
        sh_run systemctl is-active --quiet sysbox || {
            err "Sysbox service is not active after install."
            sh_run systemctl status sysbox -n 30 --no-pager || true
            exit 1
        }
    fi
    docker info --format '{{json .Runtimes}}' | grep -q 'sysbox-runc' || {
        err "Docker does not report the sysbox-runc runtime after install."
        exit 1
    }
    ok "sysbox-runc installed and visible to Docker"
}

ensure_system_packages() {
    local packages=(
        ca-certificates
        curl
        git
        ninja-build
        python3
        python3-venv
        python3-pip
    )

    # Many production GPU hosts already use Docker CE from Docker's upstream
    # repo. Installing Ubuntu's docker.io on top conflicts with docker-ce, so
    # only ask apt for docker.io when no docker CLI is present at all.
    if command -v docker >/dev/null 2>&1; then
        ok "Docker already installed: $(docker --version)"
    else
        packages+=(docker.io)
    fi

    log "Installing system packages..."
    sh_run apt-get update -qq
    sh_run apt-get install -y -qq --no-install-recommends "${packages[@]}"
    ok "System packages installed"
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
}

ensure_clock_sync() {
    if [[ "${NODEXO_SKIP_CLOCK_SYNC_CHECK:-0}" == "1" ]]; then
        warn "NODEXO_SKIP_CLOCK_SYNC_CHECK=1 set; skipping system clock synchronization check."
        return 0
    fi
    if ! command -v timedatectl >/dev/null 2>&1; then
        warn "timedatectl unavailable; cannot verify system clock synchronization."
        warn "Signed miner traffic requires host clock skew below the validator freshness window."
        return 0
    fi
    if clock_synchronized; then
        ok "System clock synchronized"
        return 0
    fi

    warn "System clock is not synchronized; signed validator traffic will fail if clock skew is large."
    warn "Installing persistent NTP configuration before miner startup."
    configure_timesyncd_ntp
    sh_run timedatectl set-ntp true || true
    sh_run systemctl restart systemd-timesyncd 2>/dev/null || true
    for _ in {1..8}; do
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
    for _ in {1..10}; do
        sleep 2
        if clock_synchronized; then
            ok "System clock synchronized"
            return 0
        fi
    done

    err "System clock is still not synchronized."
    err "Fix NTP/chrony before starting the miner, or set NODEXO_SKIP_CLOCK_SYNC_CHECK=1 only if the host clock is managed externally."
    exit 1
}

ensure_pm2() {
    command -v pm2 >/dev/null 2>&1 && return 0
    if $PROBE_ONLY; then
        warn "pm2 not installed; --probe-only will not install it."
        return 1
    fi
    log "Installing PM2..."
    if ! command -v npm >/dev/null 2>&1; then
        sh_run apt-get update -qq
        sh_run apt-get install -y -qq --no-install-recommends nodejs npm
    fi
    sh_run npm install -g pm2
    ok "PM2 installed"
}

validate_start_config() {
    [[ -n "$MINER_WALLET" ]] || { err "--start requires --wallet or NODEXO_MINER_WALLET"; exit 1; }
    [[ -n "$MINER_HOTKEY" ]] || { err "--start requires --hotkey or NODEXO_MINER_HOTKEY"; exit 1; }
    [[ -n "$MINER_NETUID" ]] || {
        err "--start requires --netuid or a known --subtensor-network (finney/test)"
        exit 1
    }
    [[ -n "$MINER_CHAIN_CONFIG" && -f "$MINER_CHAIN_CONFIG" ]] || {
        err "--start requires a readable chain config."
        err "Expected one of: chain_config_mainnet.json, chain_config_testnet.json, chain_config.json"
        err "Pass --chain-config only for custom deployments."
        exit 1
    }
    [[ -n "$MINER_ENDPOINT" ]] || { err "--start requires --endpoint or NODEXO_MINER_ENDPOINT"; exit 1; }
    if [[ ! -f "$HOME/.bittensor/wallets/$MINER_WALLET/hotkeys/$MINER_HOTKEY" ]]; then
        err "Bittensor hotkey not found: ~/.bittensor/wallets/$MINER_WALLET/hotkeys/$MINER_HOTKEY"
        err "Create/import the wallet before using --start."
        exit 1
    fi
}

write_miner_launcher() {
    validate_start_config
    mkdir -p "$(dirname "$MINER_LAUNCHER_PATH")"
    cat > "$MINER_LAUNCHER_PATH" <<EOF
#!/usr/bin/env bash
# Auto-generated by scripts/setup_miner.sh.
set -euo pipefail

cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/zkgemm/cuda"
export PYTHONUNBUFFERED=1
export NODEXO_REQUIRE_NATIVE_ZKGEMM="\${NODEXO_REQUIRE_NATIVE_ZKGEMM:-1}"
export NODEXO_DATA_DIR="\${NODEXO_DATA_DIR:-$HOME/.nodexo}"
export PROOF_EPOCH_BLOCKS="\${PROOF_EPOCH_BLOCKS:-15}"
export NODEXO_SUBNET_CONFIG_URL="\${NODEXO_SUBNET_CONFIG_URL:-$MINER_SUBNET_CONFIG_URL}"
export NODEXO_SUBNET_CONFIG_REFRESH_SECONDS="\${NODEXO_SUBNET_CONFIG_REFRESH_SECONDS:-600}"
export NODEXO_SUBNET_CONFIG_CACHE_PATH="\${NODEXO_SUBNET_CONFIG_CACHE_PATH:-\$NODEXO_DATA_DIR/subnet_config_cache.json}"
export NODEXO_SUBTENSOR_ENDPOINT="\${NODEXO_SUBTENSOR_ENDPOINT:-$MINER_SUBTENSOR_ENDPOINT}"
export NODEXO_EVM_RPC_URL="\${NODEXO_EVM_RPC_URL:-$MINER_EVM_RPC_URL}"
export EVM_RPC_TIMEOUT_SECONDS="\${EVM_RPC_TIMEOUT_SECONDS:-$MINER_EVM_RPC_TIMEOUT_SECONDS}"
export NODEXO_MONITOR_IMAGE="\${NODEXO_MONITOR_IMAGE:-$MINER_MONITOR_IMAGE}"
export MINER_AUTO_UPDATE="\${MINER_AUTO_UPDATE:-$MINER_AUTO_UPDATE}"
export NODEXO_ALLOW_UNCALIBRATED_GPU="\${NODEXO_ALLOW_UNCALIBRATED_GPU:-$MINER_ALLOW_UNCALIBRATED_GPU}"
export NODEXO_PROOF_ONLY_CALIBRATION="\${NODEXO_PROOF_ONLY_CALIBRATION:-$MINER_PROOF_ONLY_CALIBRATION}"
export PROOF_WORKER_TIMEOUT_SECONDS="\${PROOF_WORKER_TIMEOUT_SECONDS:-$MINER_PROOF_WORKER_TIMEOUT_SECONDS}"
export PROOF_SUBMIT_SAFETY_SECONDS="\${PROOF_SUBMIT_SAFETY_SECONDS:-$MINER_PROOF_SUBMIT_SAFETY_SECONDS}"
export NODEXO_VALIDATOR_POST_TIMEOUT_SECONDS="\${NODEXO_VALIDATOR_POST_TIMEOUT_SECONDS:-$MINER_VALIDATOR_POST_TIMEOUT_SECONDS}"
export NODEXO_VALIDATOR_CONNECT_TIMEOUT_SECONDS="\${NODEXO_VALIDATOR_CONNECT_TIMEOUT_SECONDS:-$MINER_VALIDATOR_CONNECT_TIMEOUT_SECONDS}"
export NODEXO_VALIDATOR_BROADCAST_DEADLINE_SECONDS="\${NODEXO_VALIDATOR_BROADCAST_DEADLINE_SECONDS:-$MINER_VALIDATOR_BROADCAST_DEADLINE_SECONDS}"
export NODEXO_VALIDATOR_POST_MAX_ATTEMPTS="\${NODEXO_VALIDATOR_POST_MAX_ATTEMPTS:-$MINER_VALIDATOR_POST_MAX_ATTEMPTS}"
export MINER_RENTAL_PORT_PREFLIGHT="\${MINER_RENTAL_PORT_PREFLIGHT:-$MINER_RENTAL_PORT_PREFLIGHT_MODE}"
export MINER_RENTAL_PORT_PREFLIGHT_SCOPE="\${MINER_RENTAL_PORT_PREFLIGHT_SCOPE:-$MINER_RENTAL_PORT_PREFLIGHT_SCOPE_MODE}"
export MINER_RENTAL_IMAGE_PREFLIGHT="\${MINER_RENTAL_IMAGE_PREFLIGHT:-$MINER_RENTAL_IMAGE_PREFLIGHT_MODE}"
export MINER_OPTIONAL_IMAGE_PREFLIGHT="\${MINER_OPTIONAL_IMAGE_PREFLIGHT:-$MINER_OPTIONAL_IMAGE_PREFLIGHT_MODE}"
export MINER_MIN_RENTAL_STORAGE_GB="\${MINER_MIN_RENTAL_STORAGE_GB:-$MIN_RENTAL_STORAGE_GB}"
export MINER_RENTAL_STORAGE_HEADROOM_GB="\${MINER_RENTAL_STORAGE_HEADROOM_GB:-$RENTAL_STORAGE_HEADROOM_GB}"
export NODEXO_ALLOWED_VALIDATOR_HOTKEYS="\${NODEXO_ALLOWED_VALIDATOR_HOTKEYS:-$MINER_ALLOWED_VALIDATORS}"
export NODEXO_STRICT_ALLOWLIST="\${NODEXO_STRICT_ALLOWLIST:-$MINER_STRICT_ALLOWLIST}"

args=(
  --wallet "$MINER_WALLET"
  --hotkey "$MINER_HOTKEY"
  --subtensor-network "$MINER_SUBTENSOR_NETWORK"
  --port "$MINER_PORT"
  --bind-host "$MINER_BIND_HOST"
  --endpoint "$MINER_ENDPOINT"
  --rental-port-start "$MINER_RENTAL_PORT_START"
  --rental-port-end "$MINER_RENTAL_PORT_END"
)
if [[ "$MINER_SUBTENSOR_NETWORK" != "test" && "$MINER_SUBTENSOR_NETWORK" != "finney" ]]; then
  args+=(--netuid "$MINER_NETUID" --chain-config "$MINER_CHAIN_CONFIG")
fi
if [[ -n "\${NODEXO_SUBTENSOR_ENDPOINT:-}" ]]; then
  args+=(--subtensor-endpoint "\$NODEXO_SUBTENSOR_ENDPOINT")
fi

exec "$VENV_DIR/bin/python" -m neurons.miner.miner "\${args[@]}"
EOF
    chmod 755 "$MINER_LAUNCHER_PATH"
    ok "Miner launcher written: $MINER_LAUNCHER_PATH"
}

start_miner_pm2() {
    ensure_pm2
    write_miner_launcher
    log "Starting miner under PM2 as $PM2_NAME..."
    pm2 delete "$PM2_NAME" >/dev/null 2>&1 || true
    pm2 start "$MINER_LAUNCHER_PATH" --name "$PM2_NAME" --interpreter bash
    pm2 save >/dev/null 2>&1 || warn "pm2 save failed; process may not survive reboot until pm2 startup is configured"
    ok "PM2 miner process started: $PM2_NAME"
}

if $LAUNCHER_ONLY; then
    write_miner_launcher
    exit 0
fi

docker_root_dir() {
    sh_run docker info -f '{{.DockerRootDir}}' 2>/dev/null || printf '/var/lib/docker\n'
}

docker_storage_free_gb() {
    local root
    root="$(docker_root_dir)"
    df -BG --output=avail "$root" 2>/dev/null | awk 'NR==2 {gsub(/G/, "", $1); print int($1)}'
}

rental_storage_reserve_gb() {
    printf '%s\n' "$((MIN_RENTAL_STORAGE_GB + RENTAL_STORAGE_HEADROOM_GB))"
}

image_pull_reserve_gb() {
    local img=$1
    PYTHONPATH="$REPO_ROOT" python3 - "$img" <<'PY'
import sys
from common.images import image_pull_reserve_gb
print(image_pull_reserve_gb(sys.argv[1]))
PY
}

zkgemm_ext_suffix() {
    "$VENV_DIR/bin/python" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
PY
}

verify_sha256() {
    local file=$1
    local expected=${2:-}
    if [[ -z "$expected" ]]; then
        return 0
    fi
    local actual
    actual="$(sha256sum "$file" | awk '{print $1}')"
    if [[ "${actual,,}" != "${expected,,}" ]]; then
        err "SHA-256 mismatch for $(basename "$file")"
        err "expected: $expected"
        err "actual:   $actual"
        return 1
    fi
}

download_zkgemm_artifact() {
    local url=$1
    local expected_sha=${2:-}
    local target=$3
    local tmp
    tmp="$(mktemp)"
    curl -fsSL "$url" -o "$tmp"
    verify_sha256 "$tmp" "$expected_sha"
    rm -f "$(dirname "$target")"/zkgemm_cuda*.so
    mv "$tmp" "$target"
    chmod 755 "$target" || true
}

install_zkgemm_wheel_path() {
    local wheel_path=$1
    local expected_sha=${2:-}
    verify_sha256 "$wheel_path" "$expected_sha"
    rm -f "$REPO_ROOT"/zkgemm/cuda/zkgemm_cuda*.so
    "$VENV_DIR/bin/pip" install -q --force-reinstall --no-deps "$wheel_path"
}

install_zkgemm_wheel_url() {
    local url=$1
    local expected_sha=${2:-}
    local filename tmp_dir tmp
    filename="${url%%[\?#]*}"
    filename="${filename##*/}"
    if [[ "$filename" != *.whl ]]; then
        err "Wheel artifact URL must end in .whl: $url"
        return 1
    fi
    tmp_dir="$(mktemp -d)"
    tmp="$tmp_dir/$filename"
    curl -fsSL "$url" -o "$tmp"
    verify_sha256 "$tmp" "$expected_sha"
    install_zkgemm_wheel_path "$tmp"
    rm -rf "$tmp_dir"
}

select_zkgemm_manifest_artifact() {
    local manifest_path=$1
    "$VENV_DIR/bin/python" - "$manifest_path" <<'PY'
import json
import platform
import sys
from pathlib import Path

try:
    import torch
except Exception as exc:
    raise SystemExit(f"torch import failed while selecting zkgemm artifact: {exc}")

manifest = json.loads(Path(sys.argv[1]).read_text())
artifacts = manifest.get("artifacts", manifest if isinstance(manifest, list) else [])
if not isinstance(artifacts, list):
    raise SystemExit("zkgemm manifest must contain an artifacts list")

py_abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
machine = platform.machine().lower()
if machine in {"amd64", "x86_64"}:
    platform_tag = "linux_x86_64"
else:
    platform_tag = f"{sys.platform}_{machine}"

torch_version = str(torch.__version__).split("+", 1)[0]
torch_major_minor = ".".join(torch_version.split(".")[:2])
cuda_version = torch.version.cuda or ""
cuda_tag = "cu" + cuda_version.replace(".", "") if cuda_version else ""

def norm(value):
    return str(value or "").strip()

def artifact_kind(artifact):
    kind = norm(artifact.get("kind") or artifact.get("type") or artifact.get("format")).lower()
    if kind in {"wheel", "whl"}:
        return "wheel"
    if kind in {"so", "shared_object", "shared-library", "shared_library"}:
        return "so"
    filename = norm(artifact.get("filename") or artifact.get("name") or artifact.get("url")).lower()
    if filename.endswith(".whl"):
        return "wheel"
    return "so"

def matches(artifact):
    platform_value = norm(artifact.get("platform") or artifact.get("platform_tag") or platform_tag)
    if platform_value and platform_value != platform_tag:
        return False

    python_value = norm(artifact.get("python_abi") or artifact.get("python") or artifact.get("abi"))
    if python_value and python_value != py_abi:
        return False

    torch_value = norm(artifact.get("torch") or artifact.get("torch_version"))
    if torch_value:
        torch_value = torch_value.removeprefix("torch")
        if not (torch_version == torch_value or torch_version.startswith(torch_value + ".") or torch_major_minor == torch_value):
            return False

    cuda_value = norm(artifact.get("cuda") or artifact.get("cuda_tag") or artifact.get("torch_cuda"))
    if cuda_value:
        cuda_value = cuda_value.replace(".", "")
        wanted = {cuda_tag, cuda_version.replace(".", "")}
        if cuda_value not in wanted:
            return False

    return True

matches_for_host = [
    artifact for artifact in artifacts
    if isinstance(artifact, dict) and matches(artifact)
]

for artifact in sorted(matches_for_host, key=lambda item: 0 if artifact_kind(item) == "wheel" else 1):
    url = norm(artifact.get("url"))
    sha = norm(artifact.get("sha256"))
    if not url:
        raise SystemExit("matching zkgemm artifact has no url")
    print(artifact_kind(artifact))
    print(url)
    print(sha)
    raise SystemExit(0)

available = [
    {
        "platform": a.get("platform") or a.get("platform_tag"),
        "python": a.get("python_abi") or a.get("python") or a.get("abi"),
        "torch": a.get("torch") or a.get("torch_version"),
        "cuda": a.get("cuda") or a.get("cuda_tag") or a.get("torch_cuda"),
    }
    for a in artifacts
    if isinstance(a, dict)
]
raise SystemExit(
    "no zkgemm artifact matches "
    f"platform={platform_tag} python={py_abi} torch={torch_major_minor} cuda={cuda_tag}; "
    f"available={available}"
)
PY
}

install_zkgemm_artifact() {
    local suffix target tmp selected
    suffix="$(zkgemm_ext_suffix)"
    target="$REPO_ROOT/zkgemm/cuda/zkgemm_cuda${suffix}"
    mkdir -p "$REPO_ROOT/zkgemm/cuda"

    if [[ -n "${ZKGEMM_CUDA_SO_PATH:-}" ]]; then
        log "Installing shipped zkgemm_cuda artifact from $ZKGEMM_CUDA_SO_PATH"
        cp "$ZKGEMM_CUDA_SO_PATH" "$target"
        verify_sha256 "$target" "${ZKGEMM_CUDA_SHA256:-}"
        chmod 755 "$target" || true
        return 0
    fi

    if [[ -n "${ZKGEMM_CUDA_WHEEL_PATH:-}" ]]; then
        log "Installing shipped zkgemm_cuda wheel from $ZKGEMM_CUDA_WHEEL_PATH"
        install_zkgemm_wheel_path "$ZKGEMM_CUDA_WHEEL_PATH" "${ZKGEMM_CUDA_SHA256:-${ZKGEMM_CUDA_WHEEL_SHA256:-}}"
        return 0
    fi

    if [[ -n "${ZKGEMM_CUDA_MANIFEST_PATH:-}" || -n "${ZKGEMM_CUDA_MANIFEST_URL:-}" ]]; then
        local manifest_path manifest_tmp kind url sha
        manifest_tmp=""
        if [[ -n "${ZKGEMM_CUDA_MANIFEST_PATH:-}" ]]; then
            manifest_path="$ZKGEMM_CUDA_MANIFEST_PATH"
        else
            log "Downloading zkgemm artifact manifest"
            manifest_tmp="$(mktemp)"
            curl -fsSL "$ZKGEMM_CUDA_MANIFEST_URL" -o "$manifest_tmp"
            manifest_path="$manifest_tmp"
        fi

        mapfile -t selected < <(select_zkgemm_manifest_artifact "$manifest_path")
        rm -f "$manifest_tmp"
        if [[ "${#selected[@]}" -lt 1 || -z "${selected[0]}" ]]; then
            err "No zkgemm artifact selected from manifest"
            return 1
        fi
        kind="${selected[0]}"
        url="${selected[1]:-}"
        sha="${selected[2]:-}"
        if [[ -z "$url" ]]; then
            err "No zkgemm artifact URL selected from manifest"
            return 1
        fi
        if [[ "$kind" == "wheel" ]]; then
            log "Installing zkgemm_cuda wheel from manifest"
            install_zkgemm_wheel_url "$url" "$sha"
        else
            log "Downloading zkgemm_cuda shared object from manifest"
            download_zkgemm_artifact "$url" "$sha" "$target"
        fi
        return 0
    fi

    if [[ -n "${ZKGEMM_CUDA_WHEEL_URL:-}" ]]; then
        log "Downloading shipped zkgemm_cuda wheel"
        install_zkgemm_wheel_url "$ZKGEMM_CUDA_WHEEL_URL" "${ZKGEMM_CUDA_SHA256:-${ZKGEMM_CUDA_WHEEL_SHA256:-}}"
        return 0
    fi

    if [[ -n "${ZKGEMM_CUDA_SO_URL:-}" ]]; then
        log "Downloading shipped zkgemm_cuda artifact"
        download_zkgemm_artifact "$ZKGEMM_CUDA_SO_URL" "${ZKGEMM_CUDA_SHA256:-}" "$target"
        return 0
    fi

    return 1
}

zkgemm_artifact_update_needed() {
    local suffix target manifest_path manifest_tmp selected kind sha actual
    suffix="$(zkgemm_ext_suffix)"
    target="$REPO_ROOT/zkgemm/cuda/zkgemm_cuda${suffix}"

    if [[ "${ZKGEMM_CUDA_FORCE_INSTALL:-0}" == "1" ]] || $ZKGEMM_ONLY; then
        return 0
    fi

    if [[ ! -f "$target" ]]; then
        return 0
    fi

    if [[ -n "${ZKGEMM_CUDA_MANIFEST_PATH:-}" || -n "${ZKGEMM_CUDA_MANIFEST_URL:-}" ]]; then
        manifest_tmp=""
        if [[ -n "${ZKGEMM_CUDA_MANIFEST_PATH:-}" ]]; then
            manifest_path="$ZKGEMM_CUDA_MANIFEST_PATH"
        else
            manifest_tmp="$(mktemp)"
            curl -fsSL "$ZKGEMM_CUDA_MANIFEST_URL" -o "$manifest_tmp"
            manifest_path="$manifest_tmp"
        fi

        mapfile -t selected < <(select_zkgemm_manifest_artifact "$manifest_path")
        rm -f "$manifest_tmp"
        kind="${selected[0]:-}"
        sha="${selected[2]:-}"

        # Wheels install into site-packages, so the source-tree .so is not the
        # authoritative object to compare. Raw .so manifests are checksum-gated.
        if [[ "$kind" != "so" || -z "$sha" ]]; then
            return 1
        fi

        actual="$(sha256sum "$target" | awk '{print $1}')"
        [[ "${actual,,}" != "${sha,,}" ]]
        return $?
    fi

    return 1
}

verify_zkgemm_extension() {
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        warn "Python venv missing; cannot validate zkgemm_cuda yet."
        return 1
    fi
    local args=()
    if [[ "${ZKGEMM_CUDA_SMOKE:-1}" != "0" ]]; then
        args+=(--cuda-smoke)
    fi
    "$VENV_DIR/bin/python" "$REPO_ROOT/scripts/verify_zkgemm_cuda.py" "${args[@]}"
}

# Distro guard. We use apt-get + Debian-style docker packages, so a
# fresh CentOS / Fedora / Arch host fails halfway. Fail loudly and
# early with a clear message instead.
if ! command -v apt-get >/dev/null 2>&1; then
    err "setup_miner.sh assumes a Debian/Ubuntu host (uses apt-get)."
    err "Adapt the system-package + nvidia-toolkit + docker.io install"
    err "steps for your distribution, or run --probe-only on this box."
    exit 1
fi

# ── 1. nvidia-smi present ────────────────────────────────────────────────────
if ! $ZKGEMM_ONLY; then
    log "Checking nvidia-smi..."
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        err "nvidia-smi not found. Install the NVIDIA driver before running this script."
        exit 1
    fi
    nvidia-smi -L | head -8
    ok "GPU(s) visible on host"

    if $PROBE_ONLY; then
        log "Probe-only mode — skipping installs"
    fi
else
    log "ZkGEMM-only mode — skipping host GPU, Docker, image, and PM2 setup"
fi

# ── 2. System packages ───────────────────────────────────────────────────────
if ! $PROBE_ONLY && ! $ZKGEMM_ONLY; then
    require_root_or_sudo
    prepare_docker_for_sysbox_install
    ensure_system_packages
    ensure_clock_sync

    # ── 3. NVIDIA container toolkit (docker --gpus all) ──────────────────────
    log "Configuring NVIDIA container toolkit..."
    if ! package_installed nvidia-container-toolkit; then
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
            sh_run gpg --dearmor --batch --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sh_run tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
        sh_run apt-get update -qq
        sh_run apt-get install -y -qq nvidia-container-toolkit
        sh_run nvidia-ctk runtime configure --runtime=docker
        sh_run systemctl restart docker
    fi
    ok "NVIDIA container toolkit configured"

    # ── 4. Sysbox-runc (required for rentable miners) ────────────────────────
    if [[ "${NODEXO_ALLOW_MISSING_SYSBOX_FOR_CALIBRATION:-0}" == "1" ]]; then
        if [[ -x /usr/bin/sysbox-runc ]]; then
            ok "sysbox-runc already installed"
        else
            warn "NODEXO_ALLOW_MISSING_SYSBOX_FOR_CALIBRATION=1 set; continuing proof-only without Sysbox."
            warn "Validator scoring will keep this host out of rental flow."
        fi
    else
        ensure_sysbox
    fi
fi

# ── 5. Python venv ───────────────────────────────────────────────────────────
if [[ -d "$VENV_DIR" && ! -x "$VENV_DIR/bin/pip" ]] && ! $PROBE_ONLY; then
    warn "Python venv at $VENV_DIR is incomplete; recreating it."
    rm -rf "$VENV_DIR"
fi
if [[ ! -d "$VENV_DIR" ]] && ! $PROBE_ONLY; then
    log "Creating Python venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi
if [[ -d "$VENV_DIR" ]] && ! $PROBE_ONLY; then
    log "Installing project Python deps into venv..."
    "$VENV_DIR/bin/pip" install -q --no-cache-dir --upgrade pip
    if [[ -f "$REPO_ROOT/requirements.txt" ]]; then
        "$VENV_DIR/bin/pip" install -q --no-cache-dir -r "$REPO_ROOT/requirements.txt"
    elif [[ -f "$REPO_ROOT/pyproject.toml" ]]; then
        "$VENV_DIR/bin/pip" install -q --no-cache-dir -e "$REPO_ROOT"
    else
        warn "No requirements.txt or pyproject.toml — skipping pip install."
    fi
    if [[ -f "$REPO_ROOT/pyproject.toml" ]]; then
        "$VENV_DIR/bin/pip" install -q --no-cache-dir --no-deps -e "$REPO_ROOT"
    fi
    "$VENV_DIR/bin/pip" cache purge >/dev/null 2>&1 || true
    ok "Python venv ready at $VENV_DIR"
fi

# ── 5. Native ZkGEMM proof extension ─────────────────────────────────────────
# Production miners receive a prebuilt zkgemm_cuda extension. It is ABI-bound to
# Python, PyTorch, CUDA, and architecture, so a copied local developer .so is not
# acceptable. Validate it here and fail before registration/proofs start.
if $SKIP_ZKGEMM_CHECK; then
    warn "--skip-zkgemm-check set; miner may fail proof generation if zkgemm_cuda is stale."
elif [[ -d "$VENV_DIR" ]]; then
    if zkgemm_artifact_update_needed; then
        log "Installing current zkgemm_cuda artifact"
        if install_zkgemm_artifact && verify_zkgemm_extension; then
            ok "zkgemm_cuda artifact installed and validated"
        else
            err "zkgemm_cuda artifact install failed."
            err "Provide a shipped artifact with ZKGEMM_CUDA_SO_PATH=/path/to/zkgemm_cuda*.so"
            err "or ZKGEMM_CUDA_WHEEL_PATH=/path/to/nodexo_zkgemm_cuda.whl"
            err "or ZKGEMM_CUDA_MANIFEST_URL=$DEFAULT_ZKGEMM_CUDA_MANIFEST_URL and re-run setup_miner.sh."
            exit 1
        fi
    else
        log "Validating shipped zkgemm_cuda extension..."
        if verify_zkgemm_extension; then
            ok "zkgemm_cuda extension validates on this host"
        elif install_zkgemm_artifact && verify_zkgemm_extension; then
            ok "zkgemm_cuda artifact installed and validated"
        else
            err "zkgemm_cuda is missing or incompatible with this host."
            err "Provide a shipped artifact with ZKGEMM_CUDA_SO_PATH=/path/to/zkgemm_cuda*.so"
            err "or ZKGEMM_CUDA_WHEEL_PATH=/path/to/nodexo_zkgemm_cuda.whl"
            err "or ZKGEMM_CUDA_MANIFEST_URL=$DEFAULT_ZKGEMM_CUDA_MANIFEST_URL and re-run setup_miner.sh."
            exit 1
        fi
    fi
fi

if $ZKGEMM_ONLY; then
    ok "ZkGEMM artifact install check completed"
    exit 0
fi

# ── 6. Image pre-pull ────────────────────────────────────────────────────────
pull_one() {
    local img=$1
    local mode=${2:-required}
    if sh_run docker image inspect "$img" >/dev/null 2>&1; then
        ok "image present: $img"
        return 0
    fi
    local reserve free_before free_after
    reserve="$(rental_storage_reserve_gb)"
    if [[ "$mode" == "optional" ]]; then
        reserve="$((reserve + $(image_pull_reserve_gb "$img")))"
    fi
    free_before="$(docker_storage_free_gb)"
    if [[ -n "$free_before" && "$free_before" -lt "$reserve" ]]; then
        warn "skipping $img; Docker storage has ${free_before}GB free, reserve is ${reserve}GB"
        return 2
    fi
    log "pulling $img..."
    if sh_run docker pull "$img"; then
        free_after="$(docker_storage_free_gb)"
        if [[ -n "$free_after" && "$free_after" -lt "$reserve" ]]; then
            warn "removing $img; pull left only ${free_after}GB free, reserve is ${reserve}GB"
            sh_run docker image rm "$img" >/dev/null 2>&1 || true
            return 2
        fi
        ok "pulled: $img"
    else
        err "failed to pull $img"
        return 1
    fi
}

ensure_monitor_image() {
    if [[ "${NODEXO_REQUIRE_MONITOR:-1}" != "1" ]]; then
        warn "NODEXO_REQUIRE_MONITOR=0; skipping monitor image preparation."
        return 0
    fi
    if sh_run docker image inspect "$MINER_MONITOR_IMAGE" >/dev/null 2>&1; then
        ok "monitor image present: $MINER_MONITOR_IMAGE"
        return 0
    fi
    if [[ "$MINER_MONITOR_IMAGE" == "nodexo-monitor:dev" || "$MINER_MONITOR_IMAGE" != */* ]]; then
        log "Building monitor image: $MINER_MONITOR_IMAGE"
        sh_run docker build \
            -t "$MINER_MONITOR_IMAGE" \
            -f "$REPO_ROOT/neurons/monitor/Dockerfile" \
            "$REPO_ROOT"
        ok "monitor image built: $MINER_MONITOR_IMAGE"
        return 0
    fi
    log "Pulling monitor image: $MINER_MONITOR_IMAGE"
    sh_run docker pull "$MINER_MONITOR_IMAGE"
    ok "monitor image pulled: $MINER_MONITOR_IMAGE"
}

# Add the invoking user to the docker group so subsequent commands
# (and the eventual miner daemon under PM2) don't need sudo for every docker
# call. The user needs to re-login or `newgrp docker` for the change to apply to
# their current shell; existing service managers may need a restart.
if ! $PROBE_ONLY && [[ $EUID -ne 0 ]] && [[ -n "${SUDO_USER:-}" || -n "${USER:-}" ]]; then
    DOCKER_USER="${SUDO_USER:-$USER}"
    if id -nG "$DOCKER_USER" 2>/dev/null | grep -qw docker; then
        ok "user $DOCKER_USER already in docker group"
    else
        log "adding $DOCKER_USER to docker group..."
        sh_run usermod -aG docker "$DOCKER_USER" || \
            warn "could not add $DOCKER_USER to docker group; sudo will be needed for docker"
        warn "log out + back in (or run 'newgrp docker') for docker group to take effect"
    fi
fi

if ! $PROBE_ONLY && ! $ZKGEMM_ONLY; then
    ensure_monitor_image
fi

load_image_catalog

if $SKIP_IMAGES; then
    warn "--skip-images set; not pulling required warm rental images."
    warn "This host may be excluded from fast checkout until images are cached."
elif ! $PROBE_ONLY; then
    log "Pre-pulling required warm rental images..."
    missing_required=()
    for img in "${REQUIRED_IMAGES[@]}"; do
        if ! pull_one "$img"; then
            missing_required+=("$img")
        fi
    done
    if [[ ${#missing_required[@]} -gt 0 ]]; then
        err "required warm rental image(s) are not cached:"
        for img in "${missing_required[@]}"; do
            err "  - $img"
        done
        err "Free Docker storage, check registry access, or set NODEXO_REQUIRED_RENTAL_IMAGES to the curated image set this host can actually serve."
        exit 1
    fi
    if [[ "${SETUP_PULL_OPTIONAL_IMAGES:-1}" != "0" ]]; then
        log "Pre-pulling optional catalog images while storage reserve allows..."
        for img in "${CATALOG_IMAGES[@]}"; do
            skip=false
            for required in "${REQUIRED_IMAGES[@]}"; do
                [[ "$img" == "$required" ]] && skip=true && break
            done
            $skip && continue
            pull_one "$img" optional || warn "optional catalog image not cached: $img"
        done
    else
        log "Optional catalog image pre-pull disabled by SETUP_PULL_OPTIONAL_IMAGES=0."
    fi
fi

# ── 7. Verification probe ────────────────────────────────────────────────────
log "Probing docker GPU access (mirrors the validator's canary path)..."
# This matches the monitor runtime: --gpus all with NVIDIA_DRIVER_CAPABILITIES.
# NVML inside the container requires compute,utility capabilities.
if sh_run docker run --rm --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    "${CANARY_IMAGE_LIST[0]}" \
    sh -lc 'PY=$(command -v /opt/conda/bin/python3 || command -v python3 || command -v python); "$PY" -c "import torch; print(\"cuda_available=\", torch.cuda.is_available(), \"count=\", torch.cuda.device_count())"' 2>/dev/null; then
    ok "docker + GPU + torch.cuda works inside container"
else
    warn "docker GPU probe failed — canary will not function."
    warn "Common causes: nvidia-container-toolkit not installed, docker daemon not restarted after toolkit install, GPU permissions."
fi

if $START_MINER; then
    start_miner_pm2
fi

# ── Next steps ───────────────────────────────────────────────────────────────
if $START_MINER; then
    cat <<EOF

==============================================================================
Setup complete. Miner was started under PM2.

  Status:
       pm2 status $PM2_NAME
       pm2 logs $PM2_NAME

  Launcher:
       $MINER_LAUNCHER_PATH
==============================================================================
EOF
else
    cat <<EOF

==============================================================================
Setup complete. Next steps:

  1. Create or import a Bittensor wallet:
       .venv/bin/btcli wallet new_coldkey --wallet-name nodexo_miner
       .venv/bin/btcli wallet new_hotkey  --wallet-name nodexo_miner --hotkey default

  2. Register on the subnet.

     Public testnet preview:
       .venv/bin/btcli subnet register --wallet-name nodexo_miner --hotkey default \\
         --netuid 468 --network test

     Mainnet:
       .venv/bin/btcli subnet register --wallet-name nodexo_miner --hotkey default \\
         --netuid 106 --network finney

  3. Fund your EVM mirror address. The miner logs this address on first start;
     it is derived from the hotkey seed and is not the hotkey SS58 address.
     From the repo root, the helper can transfer TAO from the coldkey to that
     mirror address:
       .venv/bin/nodexo --wallet nodexo_miner --hotkey default \\
         --subtensor-network test fund --amount 0.05 --yes

  4. Start with existing wallet/config in one command. This writes a PM2
     launcher and starts the miner.

     Public testnet preview, direct public miner API:
       bash scripts/setup_miner.sh --start \\
         --wallet nodexo_miner --hotkey default \\
         --subtensor-network test \\
         --endpoint http://YOUR_PUBLIC_IP:8091

     Hardened public testnet preview behind nginx:
       bash scripts/setup_endpoint_proxy.sh --role miner --public-port 8091
       bash scripts/setup_miner.sh --start \\
         --wallet nodexo_miner --hotkey default \\
         --subtensor-network test \\
         --port 18091 --bind-host 127.0.0.1 \\
         --endpoint http://YOUR_PUBLIC_IP:8091

     Mainnet:
       bash scripts/setup_miner.sh --start \\
         --wallet nodexo_miner --hotkey default \\
         --subtensor-network finney \\
         --endpoint https://YOUR_PUBLIC_MINER_DOMAIN

     To use a private/local RPC for the selected network, add:
       --subtensor-endpoint ws://127.0.0.1:9944

     Then watch:
       pm2 logs miner-nodexo

     For production operations, put nginx in front of the miner API, bind the
     Python backend to loopback, then copy and edit ecosystem.config.example.cjs:
       bash scripts/setup_endpoint_proxy.sh --role miner --public-port 8091
       cp ecosystem.config.example.cjs ecosystem.config.cjs
       pm2 start ecosystem.config.cjs --only miner-nodexo

The miner auto-registers on the ComputeRegistry contract on first start
(register_executor) and renews its 24h lease on every restart.
==============================================================================
EOF
fi
