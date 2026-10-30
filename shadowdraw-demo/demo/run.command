#!/bin/bash
# ShadowDraw demo launcher for macOS.
# Double-click this file in Finder to start the demo.
# (If macOS blocks it the first time: right-click -> Open -> Open.)

cd "$(dirname "$0")" || exit 1

# pick a Python 3
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info[0]==3 else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3 was not found. Install it from https://www.python.org/downloads/ and try again."
  read -n 1 -s -r -p "Press any key to close."
  exit 1
fi

# install dependencies only if missing
if ! "$PY" -c "import numpy, PIL" >/dev/null 2>&1; then
  echo "Installing dependencies (numpy, Pillow) ..."
  "$PY" -m pip install --quiet numpy Pillow
fi

PORT=9000
# open the browser a moment after the server comes up
( sleep 2; open "http://127.0.0.1:$PORT/" ) &

echo "Starting ShadowDraw demo on http://127.0.0.1:$PORT/"
echo "Close this window (or press Ctrl+C) to stop."
"$PY" server.py "$PORT"
