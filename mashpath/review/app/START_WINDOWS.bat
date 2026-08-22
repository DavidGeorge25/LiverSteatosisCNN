@echo off
cd /d "%~dp0"
where python >nul 2>nul || (
  start "" "index.html"
  echo Opened index.html directly.
  pause
  exit /b 0
)
start "" http://127.0.0.1:8642/index.html
python -m http.server 8642 --bind 127.0.0.1
pause
