@echo off
REM FLIP Face Anti-Spoofing Service - One-click start (Windows)
REM Starts WSL server + Windows cloudflared tunnel

echo [*] Killing previous processes...
wsl -d Ubuntu-24.04 -- bash -c "pkill -9 -f 'python server.py' 2>/dev/null; fuser -k 8000/tcp 2>/dev/null; exit 0"
taskkill /F /IM cloudflared.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo [*] Starting FLIP server in WSL...
start /B wsl -d Ubuntu-24.04 -- bash -ic "conda activate fas && export LD_LIBRARY_PATH=/home/yman/anaconda3/envs/fas/lib/python3.7/site-packages/nvidia/cudnn/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH && cd /home/yman/workspace/FLIP && python server.py"

echo [*] Waiting for server to be ready...
:wait_server
timeout /t 2 /nobreak >nul
curl -s -o nul http://localhost:8000/static/index.html
if %errorlevel% neq 0 goto wait_server
echo [+] Server ready!

echo [*] Starting Cloudflare tunnel...
start /B cloudflared tunnel --url http://localhost:8000

timeout /t 10 /nobreak >nul
echo.
echo ========================================
echo   FLIP Face Anti-Spoofing Service
echo ========================================
echo   Local:  http://localhost:8000
echo   Public: (check cloudflared window for URL)
echo   API:    POST /predict
echo   Web UI: GET  /
echo ========================================
pause
