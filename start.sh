#!/bin/bash
# FLIP Face Anti-Spoofing Service - One-click start
# Usage: bash start.sh

set -e

WORK_DIR="/home/yman/workspace/FLIP"
CONDA_ENV="fas"
PORT=8000
PIDFILE="$WORK_DIR/.server.pid"
TUNNEL_PIDFILE="$WORK_DIR/.tunnel.pid"
TUNNEL_LOG="$WORK_DIR/.tunnel.log"

# ---- Kill previous processes ----
if [ -f "$PIDFILE" ]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[*] Killing previous server (PID $OLD_PID)..."
        kill -9 "$OLD_PID" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$PIDFILE"
fi

if [ -f "$TUNNEL_PIDFILE" ]; then
    OLD_PID=$(cat "$TUNNEL_PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[*] Killing previous tunnel (PID $OLD_PID)..."
        kill -9 "$OLD_PID" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$TUNNEL_PIDFILE"
fi

# Also kill any leftover processes on the port
fuser -k ${PORT}/tcp 2>/dev/null || true
pkill -f "cloudflared.*tunnel.*${PORT}" 2>/dev/null || true
sleep 1

# ---- Activate conda ----
eval "$(/home/yman/anaconda3/bin/conda shell.bash hook)"
conda activate "$CONDA_ENV"

export LD_LIBRARY_PATH=/home/yman/anaconda3/envs/fas/lib/python3.7/site-packages/nvidia/cudnn/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH

cd "$WORK_DIR"

# ---- Start server in background ----
echo "[*] Starting FLIP server on port $PORT ..."
nohup python server.py > .server.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > "$PIDFILE"
echo "[+] Server PID: $SERVER_PID"

# Wait for server to be ready
echo -n "[*] Waiting for server"
for i in $(seq 1 120); do
    if curl -s -o /dev/null http://localhost:${PORT}/predict 2>/dev/null; then
        echo ""
        echo "[+] Server ready!"
        break
    fi
    echo -n "."
    sleep 1
done

# Verify server is running
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "[!] Server failed to start. Logs:"
    tail -20 .server.log
    exit 1
fi

# ---- Start cloudflared tunnel in background ----
echo "[*] Starting Cloudflare tunnel..."
rm -f "$TUNNEL_LOG"
nohup cloudflared tunnel --url http://localhost:${PORT} > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!
echo $TUNNEL_PID > "$TUNNEL_PIDFILE"
echo "[+] Tunnel PID: $TUNNEL_PID"

# Wait for tunnel URL
echo -n "[*] Waiting for tunnel URL"
TUNNEL_URL=""
for i in $(seq 1 30); do
    TUNNEL_URL=$(grep -oE 'https://[a-z0-9]+-[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1)
    if [ -n "$TUNNEL_URL" ]; then
        echo ""
        break
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "========================================"
echo "  FLIP Face Anti-Spoofing Service"
echo "========================================"
echo "  Local:  http://localhost:${PORT}"
if [ -n "$TUNNEL_URL" ]; then
    echo "  Public: ${TUNNEL_URL}"
else
    echo "  Public: (tunnel still connecting, check .tunnel.log)"
fi
echo ""
echo "  API:    POST /predict"
echo "  Web UI: GET  /"
echo ""
echo "  Server PID: $SERVER_PID"
echo "  Tunnel PID: $TUNNEL_PID"
echo "========================================"
