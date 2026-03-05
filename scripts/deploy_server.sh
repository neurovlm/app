#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SERVER_NAME="${SERVER_NAME:-neurovlm.ucsd.edu}"
RUST_PORT="${RUST_PORT:-8081}"
LLM_PORT="${LLM_PORT:-8002}"
STUDY_PORT="${STUDY_PORT:-8003}"
RUN_USER="${RUN_USER:-${SUDO_USER:-$(id -un)}}"
RUN_GROUP="${RUN_GROUP:-$(id -gn "${RUN_USER}")}"

NGINX_TEMPLATE="${REPO_DIR}/deploy/nginx/neurovlm.conf.template"
SERVICE_TEMPLATE="${REPO_DIR}/deploy/systemd/neurovlm.service.template"

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

escape_sed() {
    printf '%s' "$1" | sed -e 's/[\\/&]/\\&/g'
}

require_cmd cargo
require_cmd sudo
require_cmd nginx
require_cmd systemctl

if [[ ! -x "${REPO_DIR}/.venv/bin/python" ]]; then
    echo "Missing virtualenv python at ${REPO_DIR}/.venv/bin/python" >&2
    exit 1
fi

for required in \
    "static/models/specter2_traced.pt" \
    "static/models/decoder_traced.pt" \
    "static/models/aligner_traced.pt" \
    "static/models/latent_text_specter2_adhoc_query.safetensors"; do
    if [[ ! -f "${REPO_DIR}/${required}" ]]; then
        echo "Missing required file: ${REPO_DIR}/${required}" >&2
        exit 1
    fi
done

if [[ ! -f "${NGINX_TEMPLATE}" ]]; then
    echo "Missing nginx template: ${NGINX_TEMPLATE}" >&2
    exit 1
fi
if [[ ! -f "${SERVICE_TEMPLATE}" ]]; then
    echo "Missing systemd template: ${SERVICE_TEMPLATE}" >&2
    exit 1
fi

echo "Building release binary..."
(
    cd "${REPO_DIR}"
    LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 cargo build --release
)

TMP_NGINX="$(mktemp)"
TMP_SERVICE="$(mktemp)"
cleanup() {
    rm -f "${TMP_NGINX}" "${TMP_SERVICE}"
}
trap cleanup EXIT

REPO_DIR_ESCAPED="$(escape_sed "${REPO_DIR}")"
SERVER_NAME_ESCAPED="$(escape_sed "${SERVER_NAME}")"
RUN_USER_ESCAPED="$(escape_sed "${RUN_USER}")"
RUN_GROUP_ESCAPED="$(escape_sed "${RUN_GROUP}")"
RUST_PORT_ESCAPED="$(escape_sed "${RUST_PORT}")"
LLM_PORT_ESCAPED="$(escape_sed "${LLM_PORT}")"
STUDY_PORT_ESCAPED="$(escape_sed "${STUDY_PORT}")"

sed \
    -e "s/__REPO_DIR__/${REPO_DIR_ESCAPED}/g" \
    -e "s/__SERVER_NAME__/${SERVER_NAME_ESCAPED}/g" \
    -e "s/__RUST_PORT__/${RUST_PORT_ESCAPED}/g" \
    -e "s/__LLM_PORT__/${LLM_PORT_ESCAPED}/g" \
    -e "s/__STUDY_PORT__/${STUDY_PORT_ESCAPED}/g" \
    "${NGINX_TEMPLATE}" > "${TMP_NGINX}"

sed \
    -e "s/__REPO_DIR__/${REPO_DIR_ESCAPED}/g" \
    -e "s/__RUN_USER__/${RUN_USER_ESCAPED}/g" \
    -e "s/__RUN_GROUP__/${RUN_GROUP_ESCAPED}/g" \
    "${SERVICE_TEMPLATE}" > "${TMP_SERVICE}"

echo "Installing nginx and systemd configs..."
sudo install -m 644 "${TMP_NGINX}" /etc/nginx/sites-available/neurovlm
sudo ln -sfn /etc/nginx/sites-available/neurovlm /etc/nginx/sites-enabled/neurovlm
sudo install -m 644 "${TMP_SERVICE}" /etc/systemd/system/neurovlm.service

echo "Validating nginx config..."
sudo nginx -t

echo "Restarting services..."
sudo systemctl daemon-reload
sudo systemctl enable --now neurovlm
sudo systemctl restart neurovlm
sudo systemctl reload nginx

echo "Done."
echo "NeuroVLM should be reachable at https://${SERVER_NAME}"
echo "Service status: sudo systemctl status neurovlm --no-pager"
