#!/usr/bin/env bash
# =============================================================================
# Nodexo one-command installer
# =============================================================================
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/nodexo-ai/nodexo/main/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/nodexo-ai/nodexo/main/install.sh | bash -s -- --validator
# =============================================================================

set -euo pipefail

ROLE="miner"
RUN_SETUP=true
DEFAULT_REPO_URL="https://github.com/nodexo-ai/nodexo.git"
REPO_URL="${NODEXO_REPO_URL:-$DEFAULT_REPO_URL}"
REPO_URL_EXPLICIT=false
if [[ -n "${NODEXO_REPO_URL:-}" ]]; then
    REPO_URL_EXPLICIT=true
fi
REPO_REF="${NODEXO_REPO_REF:-main}"
INSTALL_DIR="${NODEXO_INSTALL_DIR:-}"
SETUP_ARGS=()

usage() {
    cat <<'USAGE'
Nodexo installer

Usage:
  install.sh [--miner|--validator] [--repo URL] [--ref REF] [--dir PATH] [--no-setup] [setup args...]

Examples:
  curl -fsSL https://raw.githubusercontent.com/nodexo-ai/nodexo/main/install.sh | bash
  curl -fsSL https://raw.githubusercontent.com/nodexo-ai/nodexo/main/install.sh | bash -s -- --validator

  # Testnet miner start: netuid 468, chain_config_testnet.json
  curl -fsSL https://raw.githubusercontent.com/nodexo-ai/nodexo/main/install.sh | bash -s -- \
    --miner --start --wallet nodexo_miner --hotkey default \
    --subtensor-network test --endpoint http://YOUR_PUBLIC_IP:8091

  # Mainnet miner start: netuid 106, chain_config_mainnet.json
  curl -fsSL https://raw.githubusercontent.com/nodexo-ai/nodexo/main/install.sh | bash -s -- \
    --miner --start --wallet nodexo_miner --hotkey default \
    --subtensor-network finney --endpoint https://YOUR_PUBLIC_MINER_DOMAIN

Environment:
  NODEXO_REPO_URL       Override release repo URL.
  NODEXO_REPO_REF       Git ref to check out. Default: main.
  NODEXO_INSTALL_DIR    Install directory. Default: /opt/nodexo as root, otherwise ~/nodexo.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --miner) ROLE="miner"; shift ;;
        --validator) ROLE="validator"; shift ;;
        --repo) REPO_URL="${2:?--repo requires a value}"; REPO_URL_EXPLICIT=true; shift 2 ;;
        --ref) REPO_REF="${2:?--ref requires a value}"; shift 2 ;;
        --dir) INSTALL_DIR="${2:?--dir requires a value}"; shift 2 ;;
        --no-setup) RUN_SETUP=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) SETUP_ARGS+=("$1"); shift ;;
    esac
done

log()  { printf '\033[1;36m[nodexo]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[err]\033[0m %s\n' "$*" >&2; }

sh_run() {
    if [[ $EUID -eq 0 ]]; then "$@"; else sudo "$@"; fi
}

ensure_git() {
    command -v git >/dev/null 2>&1 && return 0
    if command -v apt-get >/dev/null 2>&1; then
        log "Installing git..."
        sh_run apt-get update -qq
        sh_run apt-get install -y -qq --no-install-recommends git ca-certificates
        return 0
    fi
    err "git is required. Install git or run from an existing Nodexo checkout."
    exit 1
}

default_install_dir() {
    if [[ -n "$INSTALL_DIR" ]]; then
        printf '%s\n' "$INSTALL_DIR"
    elif [[ -d /workspace && -w /workspace ]]; then
        printf '/workspace/nodexo\n'
    elif [[ $EUID -eq 0 ]]; then
        printf '/opt/nodexo\n'
    else
        printf '%s/nodexo\n' "$HOME"
    fi
}

is_nodexo_checkout() {
    [[ -f pyproject.toml && -f scripts/setup_miner.sh && -f scripts/setup_validator.sh ]] \
        && grep -qi "nodexo" pyproject.toml 2>/dev/null
}

local_file_repo_path() {
    case "$REPO_URL" in
        file:///*) printf '%s\n' "${REPO_URL#file://}" ;;
        *) return 1 ;;
    esac
}

assert_local_file_repo_clean() {
    local path=""
    path="$(local_file_repo_path 2>/dev/null || true)"
    [[ -n "$path" && -d "$path/.git" ]] || return 0

    if [[ -n "$(git -C "$path" status --porcelain)" ]]; then
        err "Local file repo has uncommitted changes: $path"
        err "Git clone/fetch would ignore dirty working-tree files."
        err "Commit the local repository first, or rsync the working tree directly and run setup scripts there."
        exit 1
    fi
}

update_existing_checkout() {
    log "Updating existing Nodexo checkout: $REPO_DIR"

    local current_remote=""
    current_remote="$(git -C "$REPO_DIR" remote get-url origin 2>/dev/null || true)"
    if [[ -z "$current_remote" ]]; then
        git -C "$REPO_DIR" remote add origin "$REPO_URL"
    elif [[ "$current_remote" != "$REPO_URL" ]]; then
        log "Setting origin to $REPO_URL"
        git -C "$REPO_DIR" remote set-url origin "$REPO_URL"
    fi

    git -C "$REPO_DIR" fetch --prune origin "$REPO_REF"
    if [[ -n "$(git -C "$REPO_DIR" status --porcelain --untracked-files=no)" ]]; then
        warn "Discarding local tracked changes in $REPO_DIR to match requested release ref."
    fi
    git -C "$REPO_DIR" checkout -B nodexo-install FETCH_HEAD
    git -C "$REPO_DIR" reset --hard FETCH_HEAD >/dev/null
}

echo
echo "============================================================"
echo "  Nodexo installer ($ROLE)"
echo "============================================================"
echo

if ! $REPO_URL_EXPLICIT && [[ -z "$INSTALL_DIR" ]] && is_nodexo_checkout; then
    REPO_DIR="$(pwd)"
    log "Using current Nodexo checkout: $REPO_DIR"
else
    ensure_git
    assert_local_file_repo_clean
    REPO_DIR="$(default_install_dir)"
    if [[ -d "$REPO_DIR/.git" ]]; then
        update_existing_checkout
    elif [[ -d "$REPO_DIR" && -f "$REPO_DIR/pyproject.toml" ]]; then
        log "Using existing directory: $REPO_DIR"
    else
        log "Cloning Nodexo release repo to $REPO_DIR"
        mkdir -p "$(dirname "$REPO_DIR")"
        git clone --branch "$REPO_REF" "$REPO_URL" "$REPO_DIR"
    fi
fi

cd "$REPO_DIR"

if ! $RUN_SETUP; then
    ok "Repository ready at $REPO_DIR"
    exit 0
fi

if [[ "$ROLE" == "validator" ]]; then
    log "Running validator setup..."
    bash scripts/setup_validator.sh "${SETUP_ARGS[@]}"
else
    log "Running miner setup..."
    bash scripts/setup_miner.sh "${SETUP_ARGS[@]}"
fi

cat <<EOF

============================================================
Nodexo $ROLE setup finished.

Repository:
  $REPO_DIR

Next docs:
  docs/MINER_QUICKSTART.md
  docs/VALIDATOR_QUICKSTART.md
  docs/ENDPOINT_SECURITY.md
  docs/CLI_REFERENCE.md
============================================================
EOF
