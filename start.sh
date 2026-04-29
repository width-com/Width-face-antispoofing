#!/bin/bash
# FLIP Face Anti-Spoofing Service - gunicorn + Flask
# Usage:
#   bash start.sh
#   bash start.sh --daemon
#   bash start.sh --foreground
#   PORT=5010 bash start.sh
#   bash start.sh 5010
#   bash start.sh --foreground 5010

set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_BIN:-/home/ubuntu/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-fas}"
HOST="${HOST:-0.0.0.0}"
GUNICORN_WORKERS="${GUNICORN_WORKERS:-1}"
GUNICORN_THREADS="${GUNICORN_THREADS:-8}"
GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-180}"
ENABLE_TUNNEL="${ENABLE_TUNNEL:-0}"
RUN_MODE="${RUN_MODE:-daemon}"
PORT="${PORT:-5010}"

PIDFILE="$WORK_DIR/.server.pid"
TUNNEL_PIDFILE="$WORK_DIR/.tunnel.pid"
TUNNEL_LOG="$WORK_DIR/.tunnel.log"
SERVER_LOG="$WORK_DIR/.server.log"

usage() {
    cat <<EOF
Usage:
  bash start.sh [--daemon|-d] [PORT]
  bash start.sh [--foreground|-f] [PORT]

Examples:
  bash start.sh
  bash start.sh --daemon
  bash start.sh --foreground
  bash start.sh 5010
  bash start.sh --foreground 5010
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -d|--daemon)
            RUN_MODE="daemon"
            shift
            ;;
        -f|--foreground)
            RUN_MODE="foreground"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            PORT="$1"
            shift
            ;;
    esac
done

kill_pidfile() {
    local pidfile="$1"
    if [ -f "$pidfile" ]; then
        local old_pid
        old_pid="$(cat "$pidfile")"
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "[*] Killing previous process (PID $old_pid)..."
            kill "$old_pid" 2>/dev/null || true
            sleep 1
        fi
        rm -f "$pidfile"
    fi
}

kill_pidfile "$PIDFILE"
kill_pidfile "$TUNNEL_PIDFILE"
fuser -k "${PORT}/tcp" 2>/dev/null || true
pkill -f "cloudflared.*tunnel.*${PORT}" 2>/dev/null || true
sleep 1

if [ ! -x "$CONDA_BIN" ]; then
    echo "[!] conda not found: $CONDA_BIN"
    exit 1
fi

cd "$WORK_DIR"

echo "[*] Starting FLIP server with gunicorn on ${HOST}:${PORT} ..."
if [ "$RUN_MODE" = "foreground" ]; then
    echo "[*] Running in foreground mode"
    exec "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" gunicorn \
        --bind "${HOST}:${PORT}" \
        --workers "$GUNICORN_WORKERS" \
        --threads "$GUNICORN_THREADS" \
        --timeout "$GUNICORN_TIMEOUT" \
        --access-logfile - \
        --error-logfile - \
        --capture-output \
        server:app
fi

nohup "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" gunicorn \
    --bind "${HOST}:${PORT}" \
    --workers "$GUNICORN_WORKERS" \
    --threads "$GUNICORN_THREADS" \
    --timeout "$GUNICORN_TIMEOUT" \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    server:app \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PIDFILE"
echo "[+] Server PID: $SERVER_PID"

echo -n "[*] Waiting for server"
for i in $(seq 1 180); do
    if curl -s -o /dev/null "http://127.0.0.1:${PORT}/"; then
        echo ""
        echo "[+] Server ready!"
        break
    fi
    echo -n "."
    sleep 1
done

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "[!] Server failed to start. Logs:"
    tail -50 "$SERVER_LOG"
    exit 1
fi

TUNNEL_URL=""
if [ "$ENABLE_TUNNEL" = "1" ]; then
    echo "[*] Starting Cloudflare tunnel..."
    rm -f "$TUNNEL_LOG"
    nohup cloudflared tunnel --url "http://127.0.0.1:${PORT}" > "$TUNNEL_LOG" 2>&1 &
    TUNNEL_PID=$!
    echo "$TUNNEL_PID" > "$TUNNEL_PIDFILE"
    echo "[+] Tunnel PID: $TUNNEL_PID"

    echo -n "[*] Waiting for tunnel URL"
    for i in $(seq 1 30); do
        TUNNEL_URL="$(grep -oE 'https://[a-z0-9]+-[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1)"
        if [ -n "$TUNNEL_URL" ]; then
            echo ""
            break
        fi
        echo -n "."
        sleep 1
    done
fi

echo ""
echo "========================================"
echo "  FLIP Face Anti-Spoofing Service"
echo "========================================"
echo "  Local:  http://127.0.0.1:${PORT}"
if [ -n "$TUNNEL_URL" ]; then
    echo "  Public: ${TUNNEL_URL}"
fi
echo "  API:    POST /predict"
echo "  Web UI: GET  /"
echo "  Env:    ${CONDA_ENV}"
echo "  Server PID: $SERVER_PID"
echo "========================================"
