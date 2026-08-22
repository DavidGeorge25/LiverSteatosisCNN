#!/bin/bash
# Double-click this. It opens the study in your browser.
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then
  open "index.html"
  echo "Opened index.html directly."
  read -r -p "Press return to close."
  exit 0
fi
PORT=8642
"$PY" -m http.server $PORT --bind 127.0.0.1 >/dev/null 2>&1 &
SRV=$!
sleep 1
open "http://127.0.0.1:$PORT/index.html"
echo
echo "The study is open in your browser."
echo "Leave this window OPEN while you work. Closing it stops the app;"
echo "it does not lose anything -- your results file is written as you go."
echo
trap "kill $SRV 2>/dev/null" EXIT
wait $SRV
