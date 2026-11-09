#!/bin/bash
# SketchFade launcher for macOS. Double-click this file in Finder to start.
# (If macOS blocks it the first time: right-click -> Open -> Open.)

cd "$(dirname "$0")" || exit 1

PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info[0]==3 else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3 was not found. Install it from https://www.python.org/downloads/"
  read -n 1 -s -r -p "Press any key to close."
  exit 1
fi

if ! "$PY" -c "import numpy, PIL, diffusers, peft" >/dev/null 2>&1; then
  echo "Installing dependencies ..."
  "$PY" -m pip install --quiet -r requirements.txt
fi

PORT=9100
( sleep 3; open "http://127.0.0.1:$PORT/" ) &

echo "Starting SketchFade on http://127.0.0.1:$PORT/"
echo "Close this window (or press Ctrl+C) to stop."
"$PY" server.py "$PORT"
