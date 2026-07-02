#!/usr/bin/env bash
# =============================================================================
# Configure nginx as the public endpoint for a Nodexo miner or validator.
#
# The Python daemon should bind to 127.0.0.1 and nginx should be the only public
# listener for the API port. This keeps body limits, timeouts, route filtering,
# and TLS termination in one place.
#
# Examples:
#   bash scripts/setup_endpoint_proxy.sh --role miner --public-port 8091
#   bash scripts/setup_endpoint_proxy.sh --role validator --public-port 9443
#   # Defaults proxy public 8091 -> 127.0.0.1:18091 for miners and
#   # public 9443 -> 127.0.0.1:19443 for validators.
#   bash scripts/setup_endpoint_proxy.sh --role miner --tls existing \
#     --server-name gpu1.example.com --public-port 443 \
#     --cert-path /etc/letsencrypt/live/gpu1.example.com/fullchain.pem \
#     --key-path /etc/letsencrypt/live/gpu1.example.com/privkey.pem
# =============================================================================

set -euo pipefail

ROLE="miner"
PUBLIC_PORT=""
BACKEND_HOST="127.0.0.1"
BACKEND_PORT=""
RECEIPT_BACKEND_PORT=""
SERVER_NAME="_"
TLS_MODE="off"  # off | existing | self-signed
CERT_PATH=""
KEY_PATH=""

usage() {
    sed -n '2,31p' "$0" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --role) ROLE="${2:?--role requires miner or validator}"; shift 2 ;;
        --public-port) PUBLIC_PORT="${2:?--public-port requires a value}"; shift 2 ;;
        --backend-host) BACKEND_HOST="${2:?--backend-host requires a value}"; shift 2 ;;
        --backend-port) BACKEND_PORT="${2:?--backend-port requires a value}"; shift 2 ;;
        --receipt-backend-port) RECEIPT_BACKEND_PORT="${2:?--receipt-backend-port requires a value}"; shift 2 ;;
        --server-name) SERVER_NAME="${2:?--server-name requires a value}"; shift 2 ;;
        --tls) TLS_MODE="${2:?--tls requires off, existing, or self-signed}"; shift 2 ;;
        --cert-path) CERT_PATH="${2:?--cert-path requires a value}"; shift 2 ;;
        --key-path) KEY_PATH="${2:?--key-path requires a value}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; usage; exit 2 ;;
    esac
done

case "$ROLE" in
    miner|validator) ;;
    *) echo "--role must be miner or validator" >&2; exit 2 ;;
esac
case "$TLS_MODE" in
    off|existing|self-signed) ;;
    *) echo "--tls must be off, existing, or self-signed" >&2; exit 2 ;;
esac

default_public_port() {
    if [[ "$ROLE" == "miner" ]]; then printf '8091\n'; else printf '9443\n'; fi
}

default_backend_port() {
    if [[ "$ROLE" == "miner" ]]; then printf '18091\n'; else printf '19443\n'; fi
}

if [[ -z "$BACKEND_PORT" ]]; then
    BACKEND_PORT="$(default_backend_port)"
fi
if [[ -z "$PUBLIC_PORT" ]]; then
    PUBLIC_PORT="$(default_public_port)"
fi
if [[ "$ROLE" == "validator" && -z "$RECEIPT_BACKEND_PORT" ]]; then
    RECEIPT_BACKEND_PORT="19444"
fi

case "$BACKEND_HOST" in
    127.*|localhost|::1)
        if [[ "$BACKEND_PORT" == "$PUBLIC_PORT" ]]; then
            echo "backend and public ports must differ when nginx proxies to loopback." >&2
            echo "Use the default backend port, or pass --backend-port 18091 for miners / 19443 for validators." >&2
            exit 2
        fi
        ;;
esac

sh_run() {
    if [[ $EUID -eq 0 ]]; then "$@"; else sudo "$@"; fi
}

log() { printf '\033[1;36m[proxy]\033[0m %s\n' "$*"; }
ok()  { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }

if ! command -v nginx >/dev/null 2>&1; then
    log "Installing nginx..."
    sh_run apt-get update -qq
    sh_run apt-get install -y -qq --no-install-recommends nginx openssl
fi

CERT_DIR="/etc/nginx/ssl"
if [[ "$TLS_MODE" == "self-signed" ]]; then
    CERT_PATH="$CERT_DIR/nodexo-${ROLE}.crt"
    KEY_PATH="$CERT_DIR/nodexo-${ROLE}.key"
    sh_run mkdir -p "$CERT_DIR"
    if [[ ! -f "$CERT_PATH" || ! -f "$KEY_PATH" ]]; then
        log "Generating self-signed certificate for $SERVER_NAME"
        sh_run openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$KEY_PATH" \
            -out "$CERT_PATH" \
            -subj "/CN=$SERVER_NAME" >/dev/null 2>&1
    fi
fi

if [[ "$TLS_MODE" == "existing" || "$TLS_MODE" == "self-signed" ]]; then
    [[ -f "$CERT_PATH" ]] || { echo "certificate not found: $CERT_PATH" >&2; exit 1; }
    [[ -f "$KEY_PATH" ]] || { echo "private key not found: $KEY_PATH" >&2; exit 1; }
fi

target_dir="/etc/nginx/sites-available"
enabled_dir="/etc/nginx/sites-enabled"
if [[ ! -d "$target_dir" || ! -d "$enabled_dir" ]]; then
    target_dir="/etc/nginx/conf.d"
    enabled_dir=""
fi

conf_name="nodexo-${ROLE}.conf"
conf_path="$target_dir/$conf_name"
limits_conf_path="/etc/nginx/conf.d/nodexo-limits.conf"

listen_line="listen ${PUBLIC_PORT};"
ssl_lines=""
if [[ "$TLS_MODE" == "existing" || "$TLS_MODE" == "self-signed" ]]; then
    listen_line="listen ${PUBLIC_PORT} ssl http2;"
    ssl_lines=$(cat <<EOF
    ssl_certificate $CERT_PATH;
    ssl_certificate_key $KEY_PATH;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;
EOF
)
fi

proxy_common=$(cat <<'EOF'
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
EOF
)

write_limit_profile() {
    local profile=${1:-endpoint}
    case "$profile" in
        endpoint)
            printf '        limit_req zone=nodexo_endpoint_per_ip burst=300 nodelay;\n'
            printf '        limit_conn nodexo_conn_per_ip 100;\n'
            ;;
        receipt)
            printf '        limit_req zone=nodexo_receipt_per_ip burst=1500 nodelay;\n'
            printf '        limit_conn nodexo_conn_per_ip 200;\n'
            ;;
        chain_context)
            printf '        limit_req zone=nodexo_context_per_ip burst=300 nodelay;\n'
            printf '        limit_conn nodexo_conn_per_ip 100;\n'
            ;;
        none)
            ;;
        *)
            echo "unknown limit profile: $profile" >&2
            exit 2
            ;;
    esac
}

write_proxy_location() {
    local path=$1
    local modifier=${2:-}
    local target_port=${3:-$BACKEND_PORT}
    local limit_profile=${4:-endpoint}
    if [[ -n "$modifier" ]]; then
        printf '    location %s %s {\n' "$modifier" "$path"
    else
        printf '    location %s {\n' "$path"
    fi
    write_limit_profile "$limit_profile"
    printf '        proxy_pass http://%s:%s;\n' "$BACKEND_HOST" "$target_port"
    printf '%s\n' "$proxy_common"
    printf '    }\n'
}

write_receipt_ingress_location() {
    local target_port=${1:-$RECEIPT_BACKEND_PORT}
    printf '    location = /proofs/receipt {\n'
    write_limit_profile "receipt"
    printf '        client_max_body_size 8k;\n'
    printf '        proxy_pass http://%s:%s;\n' "$BACKEND_HOST" "$target_port"
    printf '%s\n' "$proxy_common"
    printf '    }\n'
}

limits_tmp="$(mktemp)"
cat <<'EOF' > "$limits_tmp"
# Shared Nodexo public endpoint guardrails. These are intentionally generous for
# protocol traffic; application-level signatures and replay checks remain the
# authority for accepting requests.
limit_req_zone $binary_remote_addr zone=nodexo_endpoint_per_ip:10m rate=100r/s;
limit_req_zone $binary_remote_addr zone=nodexo_receipt_per_ip:10m rate=200r/s;
limit_req_zone $binary_remote_addr zone=nodexo_context_per_ip:10m rate=100r/s;
limit_conn_zone $binary_remote_addr zone=nodexo_conn_per_ip:10m;
EOF

log "Writing nginx limits $limits_conf_path"
sh_run mkdir -p "$(dirname "$limits_conf_path")"
sh_run cp "$limits_tmp" "$limits_conf_path"
rm -f "$limits_tmp"

tmp="$(mktemp)"
{
    cat <<EOF
server {
    $listen_line
    server_name $SERVER_NAME;
$ssl_lines
    client_max_body_size 50m;
    limit_req_status 429;
    limit_conn_status 429;
    add_header X-Content-Type-Options nosniff always;
    add_header Referrer-Policy no-referrer always;
EOF

    if [[ "$ROLE" == "validator" ]]; then
        # Validator public endpoint: signed miner ingress plus harmless status.
        # Rent/admin/browse routes should be reached through the web app or
        # operator network, not through this public listener.
        write_proxy_location "/health" "="
        write_proxy_location "/version" "="
        write_proxy_location "/subnet-config" "="
        write_proxy_location "/heartbeat" "="
        write_proxy_location "/monitor/report" "="
        write_proxy_location "/chain/context" "=" "$BACKEND_PORT" "chain_context"
        write_receipt_ingress_location "$RECEIPT_BACKEND_PORT"
        write_proxy_location "/proofs/"
        cat <<'EOF'
    location / {
        return 404;
    }
EOF
    else
        # Miner endpoint: public challenge/status plus validator-signed control
        # routes. The miner daemon still enforces SR25519 auth on container
        # lifecycle routes.
        write_proxy_location "/"
    fi

    cat <<'EOF'
}
EOF
} > "$tmp"

log "Writing nginx config $conf_path"
sh_run mkdir -p "$target_dir"
sh_run cp "$tmp" "$conf_path"
rm -f "$tmp"

if [[ -n "$enabled_dir" ]]; then
    sh_run mkdir -p "$enabled_dir"
    sh_run ln -sf "$conf_path" "$enabled_dir/$conf_name"
fi

if sh_run nginx -t; then
    if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files nginx.service >/dev/null 2>&1; then
        sh_run systemctl reload nginx 2>/dev/null || sh_run systemctl restart nginx
    else
        if pgrep -x nginx >/dev/null 2>&1; then
            sh_run nginx -s reload
        else
            sh_run nginx
        fi
    fi
fi

scheme="http"
if [[ "$TLS_MODE" != "off" ]]; then
    scheme="https"
fi

ok "nginx proxy ready for $ROLE"
advertise_host="$SERVER_NAME"
if [[ "$advertise_host" == "_" ]]; then
    advertise_host="<YOUR_PUBLIC_IP_OR_DOMAIN>"
fi
cat <<EOF

Backend:
  http://$BACKEND_HOST:$BACKEND_PORT
$(if [[ "$ROLE" == "validator" ]]; then printf '  receipt ingress: http://%s:%s\n' "$BACKEND_HOST" "$RECEIPT_BACKEND_PORT"; fi)

Public endpoint:
  $scheme://$advertise_host:$PUBLIC_PORT

Set the daemon to bind locally:
  ${ROLE^^}_BIND_HOST=$BACKEND_HOST
  ${ROLE^^}_PORT=$BACKEND_PORT

Daemon args:
  --bind-host $BACKEND_HOST --port $BACKEND_PORT

Advertise this endpoint:
  ${ROLE^^}_ENDPOINT=$scheme://$advertise_host:$PUBLIC_PORT
EOF
