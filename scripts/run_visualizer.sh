#!/usr/bin/env bash
set -euo pipefail

# Waldo Vision Visualizer launcher
# - Builds and runs the visual_tester with the web feature
# - Exports NAT/ICE env for WebRTC
# - Starts the app in the background and checks health

DEFAULT_VIDEO="assets/Monroe_Walking.mp4"
DEFAULT_OUTPUT="/tmp/out.mp4"
DEFAULT_BIND="127.0.0.1:3001"
DEFAULT_DOMAIN="dev.thegonzalez.design"
DEFAULT_ICE_RANGE="30000-30100"
LOG_FILE="/tmp/waldo_visualizer.log"
PID_FILE="/tmp/waldo_visualizer.pid"

usage() {
  cat << USAGE
Usage: $0 [command] [options]

Commands:
  start       Start the visualizer server
  stop        Stop the visualizer server
  restart     Restart the visualizer server
  status      Show server status
  logs        Tail server logs

Options (for start/restart):
  -v <path>   Input video (default: ${DEFAULT_VIDEO})
  -o <path>   Output mp4 (default: ${DEFAULT_OUTPUT})
  -b <addr>   Bind address (default: ${DEFAULT_BIND})
  -d <host>   Public domain for health check (default: ${DEFAULT_DOMAIN})
  -n <ip>     NAT public IP (default: auto-detect)
  -r <range>  ICE UDP port range, e.g. 30000-30100 (default: ${DEFAULT_ICE_RANGE})

Examples:
  $0 start -v assets/Monroe_Walking.mp4 -o /tmp/out.mp4 -b 127.0.0.1:3001
  $0 logs
USAGE
}

detect_public_ip() {
  # Try instance metadata first, then ifconfig.me
  local ip
  ip=$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/public-ipv4 || true)
  if [[ -z "${ip}" ]]; then
    ip=$(curl -s --max-time 2 https://ifconfig.me || true)
  fi
  echo "${ip}"
}

is_running() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid=$(cat "${PID_FILE}" 2>/dev/null || echo "")
    if [[ -n "${pid}" && -d "/proc/${pid}" ]]; then
      return 0
    fi
  fi
  return 1
}

start_server() {
  local video="${1}" output="${2}" bind_addr="${3}" domain="${4}" nat_ip="${5}" ice_range="${6}"

  echo "[waldo] Building visual_tester (release, web feature)…"
  cargo build -q -p visual_tester --features web --release

  echo "[waldo] Stopping any existing instance…"
  $0 stop || true

  if [[ -z "${nat_ip}" || "${nat_ip}" == "auto" ]]; then
    nat_ip=$(detect_public_ip)
  fi

  : "${nat_ip:=unset}"
  : "${ice_range:=${DEFAULT_ICE_RANGE}}"

  echo "[waldo] Starting server: bind=${bind_addr}, NAT_IP=${nat_ip}, ICE_RANGE=${ice_range}"
  (
    export WV_NAT_IP="${nat_ip}"
    export WV_ICE_PORT_RANGE="${ice_range}"
    exec target/release/visual_tester --serve "${bind_addr}" "${video}" "${output}"
  ) >"${LOG_FILE}" 2>&1 &

  echo $! > "${PID_FILE}"
  sleep 1

  echo "[waldo] Waiting for health…"
  local tries=0
  while (( tries < 30 )); do
    if curl -sk --max-time 1 "https://${domain}/healthz" >/dev/null 2>&1; then
      echo "[waldo] Health OK at https://${domain}/healthz"
      break
    fi
    # fallback to local if domain not routed
    if curl -s --max-time 1 "http://127.0.0.1:${bind_addr##*:}/healthz" >/dev/null 2>&1; then
      echo "[waldo] Local health OK at http://127.0.0.1:${bind_addr##*:}/healthz"
      break
    fi
    sleep 1; tries=$((tries+1))
  done

  if (( tries >= 30 )); then
    echo "[waldo] Health check did not succeed (continuing). See logs: $0 logs" >&2
  fi

  echo "[waldo] Server PID: $(cat "${PID_FILE}")"
  echo "[waldo] Open your browser: https://${domain} (then click Play)"
}

stop_server() {
  if is_running; then
    local pid
    pid=$(cat "${PID_FILE}")
    echo "[waldo] Stopping PID ${pid}…"
    kill "${pid}" || true
    sleep 1
    if [[ -d "/proc/${pid}" ]]; then
      echo "[waldo] Killing PID ${pid}…"
      kill -9 "${pid}" || true
      sleep 1
    fi
    rm -f "${PID_FILE}"
  else
    echo "[waldo] Not running"
  fi
}

show_status() {
  if is_running; then
    echo "[waldo] Running (PID $(cat "${PID_FILE}"))"
  else
    echo "[waldo] Not running"
  fi
}

tail_logs() {
  if [[ -f "${LOG_FILE}" ]]; then
    tail -n 200 -f "${LOG_FILE}"
  else
    echo "[waldo] No log file yet: ${LOG_FILE}"
  fi
}

cmd=${1:-}
shift || true

case "${cmd}" in
  start)
    VIDEO="${DEFAULT_VIDEO}"; OUTPUT="${DEFAULT_OUTPUT}"; BIND="${DEFAULT_BIND}"; DOMAIN="${DEFAULT_DOMAIN}"; NAT_IP="auto"; ICE_RANGE="${DEFAULT_ICE_RANGE}"
    while getopts ":v:o:b:d:n:r:" opt; do
      case $opt in
        v) VIDEO="$OPTARG" ;;
        o) OUTPUT="$OPTARG" ;;
        b) BIND="$OPTARG" ;;
        d) DOMAIN="$OPTARG" ;;
        n) NAT_IP="$OPTARG" ;;
        r) ICE_RANGE="$OPTARG" ;;
        *) usage; exit 1 ;;
      esac
    done
    start_server "${VIDEO}" "${OUTPUT}" "${BIND}" "${DOMAIN}" "${NAT_IP}" "${ICE_RANGE}"
    ;;
  stop)
    stop_server
    ;;
  restart)
    $0 stop || true
    shift || true
    $0 start "$@"
    ;;
  status)
    show_status
    ;;
  logs)
    tail_logs
    ;;
  *)
    usage
    ;;
esac

